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
from datetime import datetime, timezone, timedelta
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
VOCAB_SIZE = 8000
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
CONTEXT_LENGTH = 256
DROPOUT = 0.1

# Training
WEIGHT_DECAY = 0.1
LEARNING_RATE_MAX = 3e-4
LEARNING_RATE_MIN = 3e-5
WARMUP_STEPS = 2000
LR_DECAY_STEPS = 100000

# Governor (AEDT UTC+11)
AEDT_OFFSET = timezone(timedelta(hours=11))
STEALTH_BATCH_SIZE = 4
FACTORY_BATCH_SIZE = 128
STEALTH_SLEEP = 0.2
MEMORY_LIMIT_BYTES = 100 * 1024 * 1024 * 1024 # 100 GB for GB10

# Paths
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history_pt.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
MODE_OVERRIDE_FILE = project_root / "MODE_OVERRIDE.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Setup ---
file_exists = list(LOG_FILE.exists() for _ in [0]) 
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        writer.writerow(["Timestamp", "Step", "Epoch", "Train_Loss", "Val_Loss", "Mode", "Memory_GB", "Batch_Size", "Tokens_Per_Sec", "LR", "Val_Plateau_Count"])

def log_metrics(step, epoch, train_loss, val_loss, mode, mem_gb, batch_size, tps, lr, val_plateau_count):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        writer.writerow([datetime.now(AEDT_OFFSET).isoformat(), step, epoch, f"{train_loss:.4f}", val_str, mode, f"{mem_gb:.2f}", batch_size, f"{tps:.2f}", f"{lr:.2e}", val_plateau_count])

# --- Model Architecture (PyTorch Variant) ---

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

# --- Helper Logic ---

def get_lr(it):
    if it < WARMUP_STEPS:
        return LEARNING_RATE_MAX * (it + 1) / WARMUP_STEPS
    if it > LR_DECAY_STEPS:
        return LEARNING_RATE_MIN
    decay_ratio = (it - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE_MIN + coeff * (LEARNING_RATE_MAX - LEARNING_RATE_MIN)

def get_governor_state(current_batch_size, override_status=None):
    if DEVICE == "cuda":
        active_mem = torch.cuda.memory_allocated()
    else:
        active_mem = 0
        
    if active_mem > MEMORY_LIMIT_BYTES:
        print(f"[SAFETY] Memory {active_mem/1024**3:.2f}GB > Limit. Dropping Batch Size.")
        new_bs = max(1, current_batch_size // 2)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return new_bs, 0.1, "SAFETY_RECOVERY"

    if override_status == "FACTORY":
        return FACTORY_BATCH_SIZE, 0.0, "FACTORY (OVERRIDE)"
    elif override_status == "STEALTH":
        return STEALTH_BATCH_SIZE, STEALTH_SLEEP, "STEALTH (OVERRIDE)"

    now = datetime.now(AEDT_OFFSET)
    hour = now.hour
    is_work_hours = 9 <= hour < 18
    
    if is_work_hours:
        return STEALTH_BATCH_SIZE, STEALTH_SLEEP, "STEALTH"
    else:
        return FACTORY_BATCH_SIZE, 0.0, "FACTORY"

def generate_cooing(model, tokenizer):
    model.eval()
    input_ids = tokenizer.encode('ॐ ') 
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    tokens = [t for t in input_ids]
    
    with torch.no_grad():
        for _ in range(32):
            logits = model(x[:, -CONTEXT_LENGTH:])
            logits = logits[:, -1, :]
            # Sample using multinomial
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(token)
            x = torch.cat([x, torch.tensor([[token]], device=DEVICE)], dim=1)
    
    decoded = tokenizer.decode(tokens)
    print(f"\n[SAGE-COO-PT]: {decoded}\n")
    model.train()

def get_last_step():
    if not LOG_FILE.exists():
        return 0
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            if not lines:
                return 0
            last_line = lines[-1].strip()
            if not last_line:
                last_line = lines[-2].strip() if len(lines) > 1 else ""
            if not last_line:
                 return 0
            row = next(csv.reader([last_line]))
            if row[1] == "Step":
                return 0
            return int(row[1])
    except Exception as e:
        return 0

def get_latest_checkpoint():
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt_save.safetensors"
    if interrupt_ckpt.exists():
        return interrupt_ckpt
    checkpoints = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
    if not checkpoints:
        return None
    try:
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        return latest
    except ValueError:
        return None

# --- Training Loop ---

def main():
    print(f"Initializing PyTorch Engine (AEDT Aware) on {DEVICE.upper()}...")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {DATA_PATH}")
    
    data_map = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    split_idx = int(len(data_map) * 0.9)
    train_data = data_map[:split_idx]
    val_data = data_map[split_idx:]
    
    print(f"Corpus Mapped. Total: {len(data_map)/1e6:.2f}M tokens.")
    
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    assert sp.vocab_size() == VOCAB_SIZE, f"Shape Mismatch! Expected {VOCAB_SIZE}, got {sp.vocab_size()}"
    
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    
    # Enable torch compile if available and supported (linux/CUDA typically)
    if DEVICE == "cuda" and hasattr(torch, "compile"):
        print("Compiling model for GB10 Acceleration...")
        model = torch.compile(model)
        
    model.train()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params/1e6:.2f}M")
    
    start_step = 0
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        try:
            print(f"Loading weights from {latest_ckpt.name}...")
            state_dict = load_file(str(latest_ckpt))
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
    
    last_step_from_csv = get_last_step()
    if last_step_from_csv > 0:
        start_step = last_step_from_csv
        print(f"Resumed Step Count: {start_step}")
    else:
        print("[RESUME] Starting from Step 0 (No history found).")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY)
    
    def estimate_loss():
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
            for _ in range(10):
                ix = np.random.randint(0, len(val_data) - CONTEXT_LENGTH, STEALTH_BATCH_SIZE) 
                x_np = np.stack([val_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int64)
                y_np = np.stack([val_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int64)
                x = torch.tensor(x_np, device=DEVICE)
                y = torch.tensor(y_np, device=DEVICE)
                logits = model(x)
                l = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                losses.append(l.item())
        model.train()
        return sum(losses) / len(losses)

    step = start_step 
    epoch = 0 
    tokens_processed = 0
    last_coo_time = time.time()
    
    best_val_loss = float('inf')
    val_plateau_count = 0
    
    override_status = None
    if MODE_OVERRIDE_FILE.exists():
        try:
            content = MODE_OVERRIDE_FILE.read_text().strip().upper()
            if content in ["FACTORY", "STEALTH"]:
                override_status = content
        except Exception:
            pass

    batch_size, sleep_time, mode = get_governor_state(STEALTH_BATCH_SIZE, override_status)
    iter_start = time.time()
    
    try:
        while True:
            if step % 20 == 0:
                new_override = None
                if MODE_OVERRIDE_FILE.exists():
                    try:
                        new_override = MODE_OVERRIDE_FILE.read_text().strip().upper()
                        if new_override not in ["FACTORY", "STEALTH"]: new_override = None
                    except Exception: pass
                
                if new_override != override_status:
                    print(f"\n⚠️ TRANSITIONING TO [{new_override if new_override else 'AUTO'}] ⚠️")
                    state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                    save_file(state_dict, str(CHECKPOINT_DIR / "interrupt_save.safetensors"))
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                    override_status = new_override

            target_bs, sleep_time, mode = get_governor_state(batch_size, override_status)
            batch_size = target_bs 
            
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            accum_steps = max(1, FACTORY_BATCH_SIZE // batch_size)
            total_loss = 0.0

            optimizer.zero_grad()
            
            for _ in range(accum_steps):
                ix = np.random.randint(0, len(train_data) - CONTEXT_LENGTH, batch_size)
                x_np = np.stack([train_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int64)
                y_np = np.stack([train_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int64)
                
                x = torch.tensor(x_np, device=DEVICE)
                y = torch.tensor(y_np, device=DEVICE)
                
                # Use bfloat16 mixed precision since DGX GB10 natively crushes it
                with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                    
                scaled_loss = loss / accum_steps
                scaled_loss.backward()
                total_loss += loss.item() / accum_steps
                
                if sleep_time > 0: time.sleep(sleep_time)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            dt = time.time() - iter_start
            iter_start = time.time()
            effective_tokens = FACTORY_BATCH_SIZE * CONTEXT_LENGTH
            tokens_processed += effective_tokens
            
            if step % 500 == 0:
                val_loss = estimate_loss()
                if val_loss < (best_val_loss - 0.001):
                    best_val_loss = val_loss
                    val_plateau_count = 0
                else:
                    val_plateau_count += 1
                
                mem_gb = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
                tps = effective_tokens / dt
                print(f"[Step {step}] {mode} | TrLoss:{total_loss:.4f} | ValLoss:{val_loss:.4f} | LR:{lr:.2e} | Mem:{mem_gb:.1f}GB | Plat: {val_plateau_count}")
                log_metrics(step, epoch, total_loss, val_loss, mode, mem_gb, FACTORY_BATCH_SIZE, tps, lr, val_plateau_count)
            elif step % 20 == 0:
                tps = effective_tokens / dt
                print(f"[Step {step}] {mode} | Loss:{total_loss:.4f} | LR:{lr:.2e} | {tps:.0f} tok/s")

            if time.time() - last_coo_time > 1800: 
                generate_cooing(model, sp)
                last_coo_time = time.time()
            
            if tokens_processed >= len(train_data) or step % 2000 == 0:
                epoch += 1
                tokens_processed = 0
                ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.safetensors"
                state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                save_file(state_dict, str(ckpt_path))
                
            step += 1
            
    except KeyboardInterrupt:
        print("\nPaused.")
        state_dict = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
        save_file(state_dict, str(CHECKPOINT_DIR / "interrupt_save.safetensors"))

if __name__ == "__main__":
    main()
