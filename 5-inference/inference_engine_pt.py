import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# --- Configuration (STRICT ALIGNMENT WITH train_engine_pt.py) ---
VOCAB_SIZE = 8000
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
CONTEXT_LENGTH = 256
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    CHECKPOINT_DIR = project_root / "3-model/pt/checkpoints"
    TOKENIZER_MODEL = project_root / "2-tokenizer/sutra_tokenizer.model"

# --- Model Architecture (EXACT REPLICA of train_engine_pt.py) ---
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout_p = DROPOUT
        freqs_cis = precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, L, self.n_head, -1)
        k = k.view(B, L, self.n_head, -1)
        v = v.view(B, L, self.n_head, -1).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:L])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # PyTorch SDPA without explicit masks speeds up inference massively
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x)) 
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# --- Dual-Mode Inference Logic ---
def sample_top_p(logits, temperature=0.8, top_p=0.9, repetition_penalty=1.2, seen_tokens=[]):
    if seen_tokens:
        mask_np = np.zeros(logits.shape[-1], dtype=np.float32)
        mask_np[list(set(seen_tokens))] = repetition_penalty
        mask_pt = torch.tensor(mask_np, device=device).type_as(logits)
        logits = logits - mask_pt

    logits = logits / temperature
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift to keep first token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))
    
    probs = F.softmax(sorted_logits, dim=-1)
    sorted_sampled_idx = torch.multinomial(probs, num_samples=1)
    
    sampled_token = sorted_indices.gather(-1, sorted_sampled_idx)
    return sampled_token.item()

def generate(model, tokenizer, prompt):
    print(f"\n>> {prompt}")
    
    print("\n[RAW]")
    ids_raw = tokenizer.encode(prompt)
    x_raw = torch.tensor([ids_raw], dtype=torch.long, device=DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
        for _ in range(50):
            logits = model(x_raw[:, -CONTEXT_LENGTH:])[:, -1, :]
            token_id = torch.argmax(logits, dim=-1).item()
            print(tokenizer.decode([token_id]), end="", flush=True)
            x_raw = torch.cat([x_raw, torch.tensor([[token_id]], device=DEVICE)], dim=1)
    print()

    print(f"\n[SAMPLED] (Top-P: 0.9, Temp: 0.8, RepPenalty: 1.2)")
    ids_sampled = tokenizer.encode(prompt)
    x_sampled = torch.tensor([ids_sampled], dtype=torch.long, device=DEVICE)
    seen_tokens = ids_sampled.copy()
    
    global device
    device = DEVICE
    
    with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
        for _ in range(100):
            logits = model(x_sampled[:, -CONTEXT_LENGTH:])[:, -1, :]
            logits_1d = logits[0]
            
            token_id = sample_top_p(logits_1d, seen_tokens=seen_tokens)
            print(tokenizer.decode([token_id]), end="", flush=True)
            
            seen_tokens.append(token_id)
            x_sampled = torch.cat([x_sampled, torch.tensor([[token_id]], device=DEVICE)], dim=1)
    print("\n")

def main():
    print(f"SanskritGPT PyTorch Inference Engine V2 on {DEVICE.upper()}")
    
    if not TOKENIZER_MODEL.exists():
        print(f"❌ Tokenizer not found at {TOKENIZER_MODEL}")
        return

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    
    if DEVICE == "cuda" and hasattr(torch, "compile"):
        print("Compiling model for ultra-fast generation...")
        model = torch.compile(model)
    
    possible_ckpts = [CHECKPOINT_DIR / "interrupt_save.safetensors"]
    possible_ckpts.extend(sorted(CHECKPOINT_DIR.glob("epoch_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]), reverse=True))
    
    loaded_path = None
    for p in possible_ckpts:
        if p.exists():
            loaded_path = p
            break
            
    if not loaded_path:
        print(f"❌ No PT checkpoints found in {CHECKPOINT_DIR}")
        return

    try:
        model.load_state_dict(load_file(str(loaded_path)))
        model.eval()
        print(f"✅ Loaded weights from {loaded_path.name}")
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    print("\nReady. Type 'exit' or 'q' to quit.")
    while True:
        try:
            p = input("Prompt (Sanskrit) > ")
            if p.lower() in ['q', 'exit']: break
            if p.strip(): generate(model, sp, p)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
