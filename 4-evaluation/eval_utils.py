"""
eval_utils.py — Shared utilities for all Sage-GPT evaluation scripts.

Centralises:
  • Config loading
  • get_target_checkpoint()
  • load_model_from_checkpoint()
  • Full SageGPT eval-mode architecture (no KV-cache)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file

# ── Project root on path ───────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    VOCAB_SIZE      = config.VOCAB_SIZE
    N_LAYER         = config.LAYERS
    N_HEAD          = config.HEADS
    N_EMBD          = config.EMBED_DIM
    CONTEXT_LENGTH  = config.CONTEXT_LENGTH
    DROPOUT         = config.DROPOUT
    CHECKPOINT_DIR  = config.PT_CHECKPOINT_DIR
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    print("❌ eval_utils: config.py not found.")
    sys.exit(1)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


# ── Checkpoint resolution ──────────────────────────────────────────────────────
def get_target_checkpoint(checkpoint_dir: Path = None) -> Path | None:
    """
    Priority: CLI arg → latest epoch_*.safetensors → interrupt.safetensors
    """
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1])
    checkpoints = sorted(
        ckpt_dir.glob("epoch_*.safetensors"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if checkpoints:
        return checkpoints[-1]
    interrupt = ckpt_dir / "interrupt.safetensors"
    return interrupt if interrupt.exists() else None


# ── Safe model loader (handles weight-tying) ──────────────────────────────────
def load_model_from_checkpoint(ckpt_path: Path, device: str) -> "SageGPT":
    """
    Build and load a SageGPT eval model.
    output.weight is tied to tok_emb.weight so it is absent from the safetensors
    file — strict=False is correct and expected here.
    """
    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(device)
    missing, unexpected = model.load_state_dict(load_file(str(ckpt_path)), strict=False)
    real_missing = set(missing) - {"output.weight"}
    if real_missing:
        raise RuntimeError(f"❌ Unexpected missing keys in checkpoint: {real_missing}")
    if unexpected:
        print(f"⚠️  Unexpected extra keys (ignored): {set(unexpected)}")
    return model


# ── SageGPT Eval Architecture (no KV-cache) ───────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def precompute_freqs_cis(dim: int, end: int):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    return (
        torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq),
        torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk),
    )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.n_embd = n_head, n_embd
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH))
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, L, D = x.shape
        xq = self.wq(x).view(B, L, self.n_head, -1)
        xk = self.wk(x).view(B, L, self.n_head, -1)
        xv = self.wv(x).view(B, L, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        y = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True,
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
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.mlp  = SwiGLU(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SageGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop    = nn.Dropout(DROPOUT)
        self.layers  = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm    = RMSNorm(n_embd)
        self.output  = nn.Linear(n_embd, vocab_size, bias=False)
        self.output.weight = self.tok_emb.weight  # Weight tying

    def forward(self, x):
        x = self.drop(self.tok_emb(x))
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))
