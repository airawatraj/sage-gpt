"""
train_engine_dgx.py — Main training loop for Sage-GPT on DGX Spark
Patched for better grokking (higher weight decay + gradual regularization)
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
DROPOUT = config.DROPOUT

# ==================== KEY CHANGES FOR GROKKING ====================
WEIGHT_DECAY = 0.08          # Increased from 0.05
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 6e-5
WARMUP_STEPS = 150
LR_DECAY_STEPS = 2500        # Longer decay for smoother transition
MAX_STEPS = None

THERMAL_LIMIT_C = 75
THERMAL_SLEEP_S = 30
THERMAL_CHECK_EVERY = 100

TOTAL_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE

# Paths
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
DEVICE = "cuda"

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
        self.n_head, self.n_embd = n_head, n_embd
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH))
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, L, D = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = xq.view(B, L, self.n_head, -1), xk.view(B, L, self.n_head, -1)
        xv = xv.view(B, L, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:L])
        y = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv,
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
        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm, self.output = RMSNorm(n_embd), nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.output.weight = self.tok_emb.weight

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x, targets=None):
        x = self.drop(self.tok_emb(x))
        for layer in self.layers:
            x = layer(x)
        logits = self.output(self.norm(x))
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1)
            return logits, loss
        return logits, None

# --- Training Utilities ---
def get_lr(it):
    if it < WARMUP_STEPS:
        return LEARNING_RATE_MAX * (it + 1) / WARMUP_STEPS
    if it > LR_DECAY_STEPS:
        return LEARNING_RATE_MIN
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
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip().split("\n")[0])
    except:
        return 0

def main():
    print(f"--- 🕉️ SAGE-GPT DGX FOUNDRY: INITIALIZING ---")
    
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

    # Auto-Resume Logic (unchanged)
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt.safetensors"
    interrupt_state = CHECKPOINT_DIR / "interrupt_state.pt"
    
    if interrupt_ckpt.exists():
        print(f"🚑 Auto-Resuming from Emergency Save...")
        model_state = load_file(str(interrupt_ckpt))
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
            print(f"✅ State restored: Epoch {epoch}, Step {step}")

    tokens_per_epoch = len(train_data)
    t0 = time.time()

    try:
        while True:
            lr = get_lr(step)
            for pg in optimizer.param_groups: 
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(GRAD_ACCUM_STEPS):
                x, y = get_batch(train_data, CONTEXT_LENGTH, MICRO_BATCH_SIZE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS
                accum_loss += loss.item()
                loss.backward()
                del x, y, loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_tokens = TOTAL_BATCH_SIZE * CONTEXT_LENGTH
            tokens_processed += step_tokens
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tps = step_tokens / dt

            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                exact_epoch = (step * step_tokens) / tokens_per_epoch
                print(f"[DGX Spark] [Step {step:5d}] Loss: {accum_loss:.4f} | LR: {lr:.2e} | {tps/1e6:.2f}M tok/s | Mem: {mem:.1f}GB | Epoch: {exact_epoch:.2f}")

            if step % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                v_loss = estimate_loss(model, val_data, CONTEXT_LENGTH)
                gap = v_loss - accum_loss
                print(f"\n🌟 VAL REPORT | Val Loss: {v_loss:.4f} | Gap: {gap:.4f}")

                if v_loss < best_v_loss:
                    if best_v_loss != float('inf') and (best_v_loss - v_loss) > 0.15:
                        print("🚨 PHASE SHIFT DETECTED: STRONG VALIDATION IMPROVEMENT!")
                    best_v_loss = v_loss
                    best_path = CHECKPOINT_DIR / "best_grok_model.safetensors"
                    state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                    if 'output.weight' in state_dict and 'tok_emb.weight' in state_dict:
                        if state_dict['output.weight'].data_ptr() == state_dict['tok_emb.weight'].data_ptr():
                            del state_dict['output.weight']
                    save_file(state_dict, str(best_path))
                    print(f"🌟 New Best Model Saved (v_loss: {v_loss:.4f})")

                if gap > 2.5:
                    gap_exceeded_steps += 50
                    if gap_exceeded_steps >= 500:
                        print("🚨 WARNING: Large generalization gap persisting!")
                else:
                    gap_exceeded_steps = 0

                # Logging to CSV
                header = ["Timestamp", "Step", "Train_Loss", "Val_Loss", "LR", "TPS", "Mem_GB"]
                file_exists = LOG_FILE.exists()
                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(header)
                    writer.writerow([datetime.now().isoformat(), step, f"{accum_loss:.4f}", f"{v_loss:.4f}", f"{lr:.2e}", f"{tps:.0f}", f"{mem:.2f}"])

            # Checkpointing
            is_epoch_boundary = False
            while tokens_processed >= tokens_per_epoch:
                epoch += 1
                tokens_processed -= tokens_per_epoch
                is_epoch_boundary = True

            if is_epoch_boundary or (step > 0 and step % 500 == 0):
                if is_epoch_boundary:
                    save_path = CHECKPOINT_DIR / f"epoch_{epoch}.safetensors"
                    state_path = CHECKPOINT_DIR / f"epoch_{epoch}_state.pt"
                else:
                    save_path = CHECKPOINT_DIR / f"step_{step}.safetensors"
                    state_path = CHECKPOINT_DIR / f"step_{step}_state.pt"

                state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                if 'output.weight' in state_dict and 'tok_emb.weight' in state_dict:
                    if state_dict['output.weight'].data_ptr() == state_dict['tok_emb.weight'].data_ptr():
                        del state_dict['output.weight']
                save_file(state_dict, str(save_path))

                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'tokens_processed': tokens_processed,
                    'best_v_loss': best_v_loss
                }, str(state_path))
                print(f"💾 Checkpoint Saved: {save_path}")
                prune_checkpoints(keep=10)

            # Thermal Guard
            if step % THERMAL_CHECK_EVERY == 0:
                gpu_temp = get_gpu_temp()
                if gpu_temp >= THERMAL_LIMIT_C:
                    print(f"🌡️ THERMAL GUARD: GPU at {gpu_temp}°C — pausing {THERMAL_SLEEP_S}s")
                    time.sleep(THERMAL_SLEEP_S)

            # Gradual regularization boost
            if step > 52000 and step % 5000 == 0:
                for pg in optimizer.param_groups:
                    if 'weight_decay' in pg and pg['weight_decay'] < 0.12:
                        pg['weight_decay'] = min(0.12, pg['weight_decay'] * 1.05)
                        print(f"🔧 Increased weight decay to {pg['weight_decay']:.3f}")

            step += 1
            if MAX_STEPS is not None and step >= MAX_STEPS:
                print(f"✅ MAX_STEPS reached.")
                break

    except KeyboardInterrupt:
        print("\n--- 🛑 SUTRA FOUNDRY HALTED: EMERGENCY SAVE ---")
        state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
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