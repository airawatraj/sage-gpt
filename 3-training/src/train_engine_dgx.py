"""
Sage-GPT-48M-Sutra (NVIDIA DGX Spark)
Author: 73985767+airawatraj@users.noreply.github.com
Description: High-performance Transformer SLM trained from scratch on pure Sanskrit.
"""

import sys
import os
import subprocess
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

# Import modular pruning logic
from prune_checkpoints import prune_checkpoints

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

# --- Configuration (Verified Sutra Specs) ---
VOCAB_SIZE = config.VOCAB_SIZE
N_LAYER = config.LAYERS
N_HEAD = config.HEADS
N_EMBD = config.EMBED_DIM
CONTEXT_LENGTH = config.CONTEXT_LENGTH

# Regularization — single source of truth from config.py
DROPOUT = config.DROPOUT

# Training Hyperparameters
WEIGHT_DECAY = 0.05
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 6e-5
WARMUP_STEPS = 150

# LR schedule: cosine decays MAX→MIN over this many steps.
# Training continues beyond this at LEARNING_RATE_MIN until MAX_STEPS.
LR_DECAY_STEPS = 1500

# Full Run Control — None = infinite (Ctrl+C triggers emergency save & clean exit).
MAX_STEPS = None

# Thermal Safety — home DGX Spark operation
THERMAL_LIMIT_C     = 83   # °C — pause training above this GPU temperature
THERMAL_SLEEP_S     = 30   # seconds to sleep when throttling
THERMAL_CHECK_EVERY = 100  # check GPU temp every N steps

# DGX Hardware Config (Gradient Accumulation)
TOTAL_BATCH_SIZE = 256 
MICRO_BATCH_SIZE = 64   
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE

# Pathing
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"

DEVICE = "cuda"

# --- Model Architecture (RMSNorm, RoPE, FlashAttention) ---

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
        self.n_head, self.n_embd = n_head, n_embd
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH))
        
        # ADDED DROPOUT MODULE
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, L, D = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = xq.view(B, L, self.n_head, -1), xk.view(B, L, self.n_head, -1)
        xv = xv.view(B, L, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        
        # ADDED INTERNAL DROPOUT TO SDPA
        y = F.scaled_dot_product_attention(
            xq.transpose(1, 2), 
            xk.transpose(1, 2), 
            xv, 
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )
        
        # APPLIED RESIDUAL DROPOUT
        return self.resid_drop(self.wo(y.transpose(1, 2).contiguous().view(B, L, D)))

class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        
        # ADDED DROPOUT MODULE
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        # APPLIED RESIDUAL DROPOUT
        return self.resid_drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(n_embd), RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.mlp = SwiGLU(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class SageGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # ADDED EMBEDDING DROPOUT
        self.drop = nn.Dropout(DROPOUT)
        
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm, self.output = RMSNorm(n_embd), nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.output.weight = self.tok_emb.weight
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): torch.nn.init.normal_(m.weight, 0, 0.02)
            
    def forward(self, x, targets=None):
        # APPLIED DROPOUT TO EMBEDDINGS
        x = self.drop(self.tok_emb(x))
        
        for layer in self.layers: x = layer(x)
        logits = self.output(self.norm(x))
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1) if targets is not None else (logits, None)

# --- Training Utilities ---

def get_lr(it):
    if it < WARMUP_STEPS: return LEARNING_RATE_MAX * (it + 1) / WARMUP_STEPS
    if it > LR_DECAY_STEPS: return LEARNING_RATE_MIN
    coeff = 0.5 * (1.0 + math.cos(math.pi * (it - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)))
    return LEARNING_RATE_MIN + coeff * (LEARNING_RATE_MAX - LEARNING_RATE_MIN)

def get_batch(data, ctx_len, batch_size):
    ix = torch.randint(0, len(data) - ctx_len, (batch_size,)).numpy()
    offsets = np.arange(ctx_len)
    x_idx = ix[:, None] + offsets
    y_idx = ix[:, None] + offsets + 1
    
    x = torch.from_numpy(data[x_idx].astype(np.int64)).to(DEVICE)
    y = torch.from_numpy(data[y_idx].astype(np.int64)).to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, ctx):
    model.eval()
    losses = []
    for _ in range(50):
        x, y = get_batch(data, ctx, 32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

def get_gpu_temp():
    """Query GPU temperature via nvidia-smi. Returns 0 if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0  # Assume safe if nvidia-smi is unavailable

def main():
    print(f"--- 🕉️ SAGE-GPT DGX FOUNDRY: INITIALIZING ---")
    
    # Memory Load the Purified Corpus into Shared LPDDR5X (Zero Copy on GB10)
    data = np.fromfile(DATA_PATH, dtype=np.uint16)
    train_data, val_data = data[:int(len(data)*0.95)], data[int(len(data)*0.95):]

    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    if hasattr(torch, "compile"): 
        print("🚀 Compiling Graph for GB10 Acceleration...")
        model = torch.compile(model)
    
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=LEARNING_RATE_MAX, betas=(0.9, 0.95))

    step, tokens_processed, epoch = 0, 0, 0
    best_v_loss = float('inf')
    gap_exceeded_steps = 0
    
    # Auto-Resume Logic
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt.safetensors"
    interrupt_state = CHECKPOINT_DIR / "interrupt_state.pt"
    
    if interrupt_ckpt.exists():
        print(f"🚑 Rescue Mission: Auto-Resuming from Emergency Save ({interrupt_ckpt.name})...")
        model_state = load_file(str(interrupt_ckpt))
        
        # --- SAFETENSORS TIE-WEIGHT RESTORE ---
        if 'tok_emb.weight' in model_state and 'output.weight' not in model_state:
            model_state['output.weight'] = model_state['tok_emb.weight']
            
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
            
        if interrupt_state.exists():
            checkpoint = torch.load(str(interrupt_state), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            epoch = checkpoint['epoch']
            tokens_processed = checkpoint['tokens_processed']
            best_v_loss = checkpoint.get('best_v_loss', float('inf'))
            print(f"✅ Emergency State restored successfully: Epoch {epoch}, Step {step}")
        else:
            print("⚠️ Emergency Optimizer state missing. Model loaded, starting fresh Adam moments.")
    else:
        ckpts = sorted(list(CHECKPOINT_DIR.glob("epoch_*.safetensors")), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            latest = ckpts[-1]
            epoch_num = int(latest.stem.split("_")[1])
            print(f"🔄 Auto-Resuming from {latest.name}...")
            
            # Load Model Weights
            model_state = load_file(str(latest))
            
            # --- SAFETENSORS TIE-WEIGHT RESTORE ---
            if 'tok_emb.weight' in model_state and 'output.weight' not in model_state:
                model_state['output.weight'] = model_state['tok_emb.weight']
                
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
                
            # Load Optimizer State
            state_path = CHECKPOINT_DIR / f"epoch_{epoch_num}_state.pt"
            if state_path.exists():
                checkpoint = torch.load(str(state_path), map_location="cpu", weights_only=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                step = checkpoint['step']
                epoch = checkpoint['epoch']
                tokens_processed = checkpoint['tokens_processed']
                best_v_loss = checkpoint.get('best_v_loss', float('inf'))
                print(f"✅ State restored successfully: Epoch {epoch}, Step {step}")
            else:
                print("⚠️ Optimizer state missing. Model loaded, starting fresh Adam moments.")
                epoch = epoch_num

    tokens_per_epoch = len(train_data)
    mem = 0.0  # Initialised here; refreshed at step%20 and each val check
    t0 = time.time()

    try:
        while True:
            lr = get_lr(step)
            for pg in optimizer.param_groups: pg['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            
            # Gradient Accumulation to hit Global Batch Size
            for _ in range(GRAD_ACCUM_STEPS):
                x, y = get_batch(train_data, CONTEXT_LENGTH, MICRO_BATCH_SIZE)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS
                
                accum_loss += loss.item()
                loss.backward()
                
                # Prevent VRAM Leak: Force deletion of tensors to free graph memory immediately
                del x, y, loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            step_tokens = TOTAL_BATCH_SIZE * CONTEXT_LENGTH
            tokens_processed += step_tokens
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tps = step_tokens / dt 
            
            # Verbose Terminal Logging
            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                exact_epoch = (step * step_tokens) / tokens_per_epoch
                print(f"[DGX Spark] [Step {step:5d}] Loss: {accum_loss:.4f} | LR: {lr:.2e} | {tps/1e6:.2f}M tok/s | Mem: {mem:.1f}GB | Epoch: {exact_epoch:.2f}")
                
                if tps < 90000:
                    print(f"⚠️ HARDWARE GUARD: TPS dropped below 0.09M baseline ({tps/1e6:.2f}M tok/s)")
                if mem > 14.0 or mem < 7.0:
                    print(f"⚠️ HARDWARE GUARD: VRAM deviation detected ({mem:.1f}GB)")

            # CSV Logging & Validation
            if step % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9  # Always fresh for CSV
                v_loss = estimate_loss(model, val_data, CONTEXT_LENGTH)
                gap = v_loss - accum_loss
                print(f"\n🌟 VAL REPORT | Val Loss: {v_loss:.4f} | Gap: {gap:.4f}\n")
                
                if v_loss < 2.5:
                    print("🚨 PHASE SHIFT DETECTED: SAGE-GPT IS GROKKING SANSKRIT!")
                
                if gap > 2.5:
                    gap_exceeded_steps += 50
                    if gap_exceeded_steps >= 500:
                        print("🚨 WARNING: OVERFITTING GAP EXCEEDED 2.5 FOR 500 STEPS!")
                else:
                    gap_exceeded_steps = 0
                
                if v_loss < best_v_loss:
                    best_v_loss = v_loss
                    best_path = CHECKPOINT_DIR / "best_grok_model.safetensors"
                    state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                    
                    # --- SAFETENSORS TIE-WEIGHT FIX ---
                    if 'output.weight' in state_dict and 'tok_emb.weight' in state_dict:
                        if state_dict['output.weight'].data_ptr() == state_dict['tok_emb.weight'].data_ptr():
                            del state_dict['output.weight']
                    
                    save_file(state_dict, str(best_path))
                    print(f"🌟 New Best Model Saved (v_loss: {v_loss:.4f}): {best_path}")

                header = ["Timestamp", "Step", "Train_Loss", "Val_Loss", "LR", "TPS", "Mem_GB"]
                file_exists = LOG_FILE.exists()
                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists: writer.writerow(header)
                    writer.writerow([datetime.now().isoformat(), step, f"{accum_loss:.4f}", f"{v_loss:.4f}", f"{lr:.2e}", f"{tps:.0f}", f"{mem:.2f}"])

            # Checkpointing & Auto-Pruning
            is_epoch_boundary = False
            while tokens_processed >= tokens_per_epoch:
                epoch += 1
                tokens_processed -= tokens_per_epoch
                is_epoch_boundary = True

            if is_epoch_boundary or (step > 0 and step % 500 == 0):
                existing_ckpts = [int(p.stem.split("_")[1]) for p in CHECKPOINT_DIR.glob("epoch_*.safetensors") if len(p.stem.split("_")) > 1 and p.stem.split("_")[1].isdigit()]
                save_idx = max(existing_ckpts) + 1 if existing_ckpts else 1
                save_path = CHECKPOINT_DIR / f"epoch_{save_idx}.safetensors"
                
                # Unwrap compiled model for clean saving
                state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                
                # --- SAFETENSORS TIE-WEIGHT FIX ---
                if 'output.weight' in state_dict and 'tok_emb.weight' in state_dict:
                    if state_dict['output.weight'].data_ptr() == state_dict['tok_emb.weight'].data_ptr():
                        del state_dict['output.weight']
                
                save_file(state_dict, str(save_path))
                
                # Save Optimizer and Engine State natively
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'tokens_processed': tokens_processed,
                    'best_v_loss': best_v_loss
                }, str(CHECKPOINT_DIR / f"epoch_{save_idx}_state.pt"))
                
                print(f"💾 Checkpoint Saved: {save_path}")
                prune_checkpoints(keep=10) # Keep disk lean
            
            # Thermal Safety — throttle if GPU is running too hot
            if step % THERMAL_CHECK_EVERY == 0:
                gpu_temp = get_gpu_temp()
                if gpu_temp >= THERMAL_LIMIT_C:
                    print(f"🌡️  THERMAL GUARD: GPU at {gpu_temp}°C — pausing {THERMAL_SLEEP_S}s to cool down...")
                    time.sleep(THERMAL_SLEEP_S)

            step += 1
            if MAX_STEPS is not None and step >= MAX_STEPS:
                print(f"✅ MAX_STEPS ({MAX_STEPS:,}) reached. Training complete.")
                break
            
    except KeyboardInterrupt:
        print("\n--- 🛑 SUTRA FOUNDRY HALTED: EMERGENCY SAVE ---")
        state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
        
        # --- SAFETENSORS TIE-WEIGHT FIX ---
        if 'output.weight' in state_dict and 'tok_emb.weight' in state_dict:
            if state_dict['output.weight'].data_ptr() == state_dict['tok_emb.weight'].data_ptr():
                del state_dict['output.weight']
                
        save_file(state_dict, str(CHECKPOINT_DIR / "interrupt.safetensors"))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'step': step,
            'epoch': epoch,
            'tokens_processed': tokens_processed,
            'best_v_loss': best_v_loss
        }, str(CHECKPOINT_DIR / "interrupt_state.pt"))

if __name__ == "__main__":
    main()