import sys
import os
import time
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sentencepiece as spm
from pathlib import Path
from datetime import datetime
from safetensors.torch import save_file, load_file

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

# --- Configuration ---
VOCAB_SIZE = config.VOCAB_SIZE
N_LAYER = config.LAYERS
N_HEAD = config.HEADS
N_EMBD = config.EMBED_DIM
CONTEXT_LENGTH = config.CONTEXT_LENGTH
DROPOUT = 0.05 # Lowered for Grokking; weight decay handles regularization

# Training Hyperparameters
WEIGHT_DECAY = 0.1
LEARNING_RATE_MAX = 6e-4 # Bumped slightly for DGX scale
LEARNING_RATE_MIN = 6e-5
WARMUP_STEPS = 1000
LR_DECAY_STEPS = 50000 

# DGX Hardware Config (Gradient Accumulation Strategy)
TOTAL_BATCH_SIZE = 2048 # Global Batch
MICRO_BATCH_SIZE = 64   # How many fit in VRAM at once
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE

# Paths
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Architecture ---

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

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
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        B, L, D = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, L, self.n_head, -1)
        xk = xk.view(B, L, self.n_head, -1)
        xv = xv.view(B, L, self.n_head, -1).transpose(1, 2)
        
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, L, D))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(n_embd), RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SageGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight Initialization for Grokking Stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.tok_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# --- Training Utilities ---

def get_lr(it):
    if it < WARMUP_STEPS: return LEARNING_RATE_MAX * (it + 1) / WARMUP_STEPS
    if it > LR_DECAY_STEPS: return LEARNING_RATE_MIN
    decay_ratio = (it - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE_MIN + coeff * (LEARNING_RATE_MAX - LEARNING_RATE_MIN)

def main():
    print(f"--- 🕉️ SAGE-GPT DGX FOUNDRY: STARTING ---")
    
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    train_data = data[:int(len(data)*0.95)] # 95/5 split for pure Sanskrit
    val_data = data[int(len(data)*0.95):]

    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    if hasattr(torch, "compile"): model = torch.compile(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler() # For mixed precision stability

    step = 0
    best_val = float('inf')

    try:
        while True:
            lr = get_lr(step)
            for pg in optimizer.param_groups: pg['lr'] = lr
            
            # --- Gradient Accumulation Loop ---
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            
            for _ in range(GRAD_ACCUM_STEPS):
                ix = np.random.randint(0, len(train_data) - CONTEXT_LENGTH, MICRO_BATCH_SIZE)
                x = torch.stack([torch.from_numpy(train_data[i:i+CONTEXT_LENGTH].astype(np.int64)) for i in ix]).to(DEVICE)
                y = torch.stack([torch.from_numpy(train_data[i+1:i+CONTEXT_LENGTH+1].astype(np.int64)) for i in ix]).to(DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS # Scale loss for accumulation
                
                accum_loss += loss.item()
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                print(f"Step {step} | Loss: {accum_loss:.4f} | LR: {lr:.2e}")
                # Validation and Checkpointing logic here...
            
            step += 1
            if step > LR_DECAY_STEPS: break

    except KeyboardInterrupt:
        save_file(model.state_dict(), str(CHECKPOINT_DIR / "interrupt.safetensors"))

if __name__ == "__main__":
    main()