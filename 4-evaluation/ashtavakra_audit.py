import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path
from safetensors.torch import load_file

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    VOCAB_SIZE, N_LAYER, N_HEAD, N_EMBD, CONTEXT_LENGTH = config.VOCAB_SIZE, config.LAYERS, config.HEADS, config.EMBED_DIM, config.CONTEXT_LENGTH
    CHECKPOINT_DIR, TOKENIZER_MODEL = config.PT_CHECKPOINT_DIR, config.TOKENIZER_DIR / "sutra_tokenizer.model"
    DROPOUT = 0.3
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

def get_target_checkpoint():
    """Prioritizes CLI args, then latest epoch, then interrupt."""
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1])
    
    checkpoints = sorted(list(CHECKPOINT_DIR.glob("epoch_*.safetensors")), 
                        key=lambda p: int(p.stem.split("_")[1]))
    if checkpoints:
        return checkpoints[-1]
    
    interrupt = CHECKPOINT_DIR / "interrupt.safetensors"
    return interrupt if interrupt.exists() else None

CHECKPOINT_PATH = get_target_checkpoint()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"

# --- SageGPT Architecture (Must Match Training Engine) ---
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

def precompute_freqs_cis(dim, end):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_, xk_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)), torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.n_embd = n_head, n_embd
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH))
        self.resid_drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, L, D = x.shape
        xq, xk, xv = self.wq(x).view(B, L, self.n_head, -1), self.wk(x).view(B, L, self.n_head, -1), self.wv(x).view(B, L, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        y = F.scaled_dot_product_attention(
            xq.transpose(1, 2), 
            xk.transpose(1, 2), 
            xv, 
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )
        return self.resid_drop(self.wo(y.transpose(1, 2).contiguous().view(B, L, D)))

class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.resid_drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2, self.attn = RMSNorm(n_embd), RMSNorm(n_embd), MultiHeadAttention(n_embd, n_head)
        self.mlp = SwiGLU(n_embd)
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

class SageGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm, self.output = RMSNorm(n_embd), nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, x):
        x = self.drop(self.tok_emb(x))
        for layer in self.layers: x = layer(x)
        return self.output(self.norm(x))

# --- Evaluation Helpers ---
def audit_generate(model, sp, prompt, max_tokens=25, temp=0.7):
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
    return sp.decode(x[0].tolist())

def main():
    print(f"\n🐚 ASHTAVAKRA AUDIT V2.2: 8-BEND FULL DIAGNOSTIC 🐚")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    
    if not CHECKPOINT_PATH or not CHECKPOINT_PATH.exists(): return print(f"❌ Checkpoint missing in {CHECKPOINT_DIR}")
    model.load_state_dict(load_file(str(CHECKPOINT_PATH)))
    
    # [Bend Name, Prompt, Keyword to check for "STRAIGHT" status]
    bends = [
        ("1. Phonetic Stability", "ॐ", "नमः"),
        ("2. Invocation", "असतो मा", "सद्गमय"),
        ("3. Case Inflection", "राम", "ः"),
        ("4. Sandhi Logic", "नर", "इन्द्र"),
        ("5. Concept Flow", "यथा नद्यः", "समुद्रे"),
        ("6. Verse Sequence", "ईशा वास्य", "सर्वं"),
        ("7. Orthography", "कृष्", "ण"),
        ("8. The Atman Test", "तत्त्वमसि", "श्वेतकेतो")
    ]

    print(f"{'BEND (TEST)':<25} | {'STATUS':<10} | {'PREVIEW'}")
    print("-" * 95)

    score = 0
    for name, prompt, target in bends:
        gen = audit_generate(model, sp, prompt)
        status = "STRAIGHT" if target in gen else "CROOKED"
        if status == "STRAIGHT": score += 1
        print(f"{name:<25} | {status:<10} | {gen.replace(chr(10), ' ')[:50]}")

    print("-" * 95)
    print(f"🕸️ Vedic Scorecard: {score}/8 Bends Straightened")

if __name__ == "__main__":
    main()