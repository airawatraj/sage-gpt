import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    VOCAB_SIZE = config.VOCAB_SIZE
    N_LAYER = config.LAYERS
    N_HEAD = config.HEADS
    N_EMBD = config.EMBED_DIM
    CONTEXT_LENGTH = config.CONTEXT_LENGTH
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    print("❌ Error: config.py not found.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SageGPT Architecture (Exact Replica of DGX Engine) ---

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
    def forward(self, x):
        B, L, D = x.shape
        xq, xk, xv = self.wq(x).view(B, L, self.n_head, -1), self.wk(x).view(B, L, self.n_head, -1), self.wv(x).view(B, L, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        y = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, L, D))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2, self.attn = RMSNorm(n_embd), RMSNorm(n_embd), MultiHeadAttention(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.SiLU(), nn.Linear(4 * n_embd, n_embd, bias=False))
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

class SageGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm, self.output = RMSNorm(n_embd), nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, x):
        x = self.tok_emb(x)
        for layer in self.layers: x = layer(x)
        return self.output(self.norm(x))

# --- Sampling Logic ---

def sample_top_p(logits, temperature=0.8, top_p=0.9, repetition_penalty=1.5, seen_tokens=[]):
    if seen_tokens:
        for tid in set(seen_tokens):
            logits[tid] /= repetition_penalty
    logits = logits / max(temperature, 1e-5)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return sorted_indices[next_token].item()

def generate(model, sp, prompt):
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    seen_tokens = ids.copy()
    print(f"\nSutra-GPT >> {prompt}", end="", flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(100):
            logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
            token_id = sample_top_p(logits[0], seen_tokens=seen_tokens)
            if token_id == sp.eos_id(): break 
            print(sp.decode([token_id]), end="", flush=True)
            seen_tokens.append(token_id)
            x = torch.cat([x, torch.tensor([[token_id]], device=DEVICE)], dim=1)
    print("\n")

def main():
    print(f"--- 🕉️ SAGE-GPT INFUSION ENGINE (DGX) ---")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    
    ckpts = sorted(list(CHECKPOINT_DIR.glob("epoch_*.safetensors")), key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    if not ckpts: return print("❌ No checkpoints found.")
    
    # CRITICAL: Load weights BEFORE compiling
    print(f"📂 Loading Weights: {ckpts[0].name}")
    model.load_state_dict(load_file(str(ckpts[0])))
    
    if hasattr(torch, "compile"):
        print("🚀 Compiling for GB10 Acceleration...")
        model = torch.compile(model)

    while True:
        try:
            p = input("🕉️  Sutra Prompt > ")
            if p.lower() in ['q', 'exit']: break
            if p.strip(): generate(model, sp, p)
        except (KeyboardInterrupt, EOFError): break

if __name__ == "__main__":
    main()