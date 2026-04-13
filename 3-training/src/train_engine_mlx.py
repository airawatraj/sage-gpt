import sys
import os
import time
import math
import csv
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map # Added tree_map for gradient accumulation
import sentencepiece as spm
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
MEMORY_LIMIT_BYTES = 12 * 1024 * 1024 * 1024  
LOWER_MEMORY_LIMIT = 8 * 1024 * 1024 * 1024  

# Paths
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
MODE_OVERRIDE_FILE = project_root / "MODE_OVERRIDE.txt"

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

# --- Model Architecture ---

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head, traditional=True)
        self.dropout = nn.Dropout(DROPOUT)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        
        q = self.rope(q)
        k = self.rope(k)
        
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            att = att + mask
        att = mx.softmax(att, axis=-1)
        att = self.dropout(att)
        
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
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
    def __call__(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.dropout = nn.Dropout(DROPOUT)
    def __call__(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.mlp(self.ln2(x)) 
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
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
    active_mem = mx.get_active_memory()
    if active_mem > MEMORY_LIMIT_BYTES:
        print(f"[SAFETY] Memory {active_mem/1024**3:.2f}GB > Limit. Dropping Batch Size.")
        new_bs = max(1, current_batch_size // 2)
        mx.clear_cache()
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
    input_ids = tokenizer.encode('ॐ ') 
    x = mx.array([input_ids], dtype=mx.uint32)
    tokens = [t for t in input_ids]
    
    for _ in range(32):
        logits = model(x[:, -CONTEXT_LENGTH:])
        logits = logits[:, -1, :]
        token = mx.random.categorical(logits, num_samples=1).item()
        tokens.append(token)
        x = mx.concatenate([x, mx.array([[token]])], axis=1)
    
    decoded = tokenizer.decode(tokens)
    print(f"\n[SAGE-COO]: {decoded}\n")

def get_last_step():
    if not LOG_FILE.exists():
        print(f"[RESUME] Log file not found at {LOG_FILE}")
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
            if row[1] == "Step": # FIXED: Safely ignore the CSV header
                return 0
                
            step_val = int(row[1])
            print(f"[RESUME] Found last step in log: {step_val}")
            return step_val

    except Exception as e:
        print(f"[WARN] Failed to read last step from log: {e}")
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
    print(f"Initializing MLX Engine (AEDT Aware)...")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {DATA_PATH}")
    
    data_map = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    split_idx = int(len(data_map) * 0.9)
    train_data = data_map[:split_idx]
    val_data = data_map[split_idx:]
    
    print(f"Corpus Mapped. Total: {len(data_map)/1e6:.2f}M tokens.")
    print(f"Split: Train={len(train_data)/1e6:.2f}M, Val={len(val_data)/1e6:.2f}M")
    
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    # Vocabulary Alignment & Tokenizer Safety Check
    assert sp.vocab_size() == VOCAB_SIZE, f"Shape Mismatch! Expected {VOCAB_SIZE}, got {sp.vocab_size()}"
    byte_fallback_active = sp.piece_to_id('<0x00>') != sp.unk_id() if hasattr(sp, 'piece_to_id') else "Unknown"
    print(f"[TOKENIZER] byte_fallback active: {byte_fallback_active}")
    
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    mx.eval(model.parameters())
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model Params: {params/1e6:.2f}M")
    
    start_step = 0
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        try:
            print(f"Loading weights from {latest_ckpt.name}...")
            model.load_weights(str(latest_ckpt))
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
    
    last_step_from_csv = get_last_step()
    if last_step_from_csv > 0:
        start_step = last_step_from_csv
        print(f"Resumed Step Count: {start_step}")
    else:
        print("[RESUME] Starting from Step 0 (No history found).")
    
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY)
    
    def loss_fn(model, x, y):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    def estimate_loss():
        losses = []
        for _ in range(10):
            ix = np.random.randint(0, len(val_data) - CONTEXT_LENGTH, STEALTH_BATCH_SIZE) 
            x_np = np.stack([val_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int32)
            y_np = np.stack([val_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int32)
            x = mx.array(x_np)
            y = mx.array(y_np)
            logits = model(x)
            l = nn.losses.cross_entropy(logits, y)
            losses.append(mx.mean(l).item())
        return sum(losses) / len(losses)

    step = start_step 
    print(f"[DEBUG] Loop initialized with Step: {step}")
    epoch = 0 
    tokens_processed = 0
    last_coo_time = time.time()
    
    # Validation Plateau Tracking
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
    print(f"Resuming from Step {step} in {mode} Mode (Micro-BS={batch_size}, Effective-BS={FACTORY_BATCH_SIZE})")
    
    iter_start = time.time()
    print("Starting Infinite Training Loop...")
    
    try:
        while True:
            # FIXED: Check override every 20 steps, enforce uppercase
            if step % 20 == 0:
                new_override = None
                if MODE_OVERRIDE_FILE.exists():
                    try:
                        content = MODE_OVERRIDE_FILE.read_text().strip().upper()
                        if content in ["FACTORY", "STEALTH"]:
                            new_override = content
                    except Exception as e:
                        print(f"[WARN] Failed to read override file: {e}")
                
                if new_override != override_status:
                    print(f"\n⚠️ SOVEREIGN OVERRIDE: TRANSITIONING TO [{new_override if new_override else 'AUTO'}] ⚠️")
                    msg = f"Transitioning to {new_override}" if new_override else "Returning to Auto Governor"
                    print(f"[Governor] {msg}. Saving safety checkpoint...")
                    model.save_weights(str(CHECKPOINT_DIR / "interrupt_save.safetensors"))
                    mx.clear_cache()
                    print("[Governor] Metal Cache Cleared.")
                    override_status = new_override

            target_bs, sleep_time, mode = get_governor_state(batch_size, override_status)
            batch_size = target_bs 
            
            lr = get_lr(step)
            optimizer.learning_rate = lr

            # --- GRADIENT ACCUMULATION BLOCK ---
            # Calculates how many micro-batches are needed to hit a simulated batch of 128
            accum_steps = max(1, FACTORY_BATCH_SIZE // batch_size)
            accumulated_grads = None
            total_loss = 0.0

            for _ in range(accum_steps):
                ix = np.random.randint(0, len(train_data) - CONTEXT_LENGTH, batch_size)
                x_np = np.stack([train_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int32)
                y_np = np.stack([train_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int32)
                
                x = mx.array(x_np)
                y = mx.array(y_np)
                
                loss, grads = loss_and_grad_fn(model, x, y)
                
                scaled_loss = loss / accum_steps
                total_loss += scaled_loss.item()
                
                # Scale the gradients for this micro-batch
                scaled_grads = tree_map(lambda g: g / accum_steps, grads)
                
                if accumulated_grads is None:
                    accumulated_grads = scaled_grads
                else:
                    accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, scaled_grads)
                
                # Sleep is applied per micro-batch to keep thermals low in STEALTH
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Eager evaluation in STEALTH to prevent graph build-up
                if accum_steps > 1:
                    mx.eval(loss, accumulated_grads)

            # Apply the fully accumulated gradients (Mathematically identical to 1x 128 Batch)
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state) 
            
            # Logging Logic
            dt = time.time() - iter_start
            iter_start = time.time()
            effective_tokens = FACTORY_BATCH_SIZE * CONTEXT_LENGTH
            tokens_processed += effective_tokens
            
            if step % 500 == 0:
                val_loss = estimate_loss()
                
                # Plateau Detection Logic
                if val_loss < (best_val_loss - 0.001):
                    best_val_loss = val_loss
                    val_plateau_count = 0
                else:
                    val_plateau_count += 1
                
                mem_gb = mx.get_active_memory() / 1024**3
                tps = effective_tokens / dt
                print(f"[Step {step}] {mode} | Eff_BS:{FACTORY_BATCH_SIZE} (Micro:{batch_size}x{accum_steps}) | TrLoss:{total_loss:.4f} | ValLoss:{val_loss:.4f} | LR:{lr:.2e} | Mem:{mem_gb:.1f}GB | Plateau: {val_plateau_count}")
                
                if val_plateau_count >= 5:
                    print(f"[SAGE-ARCH]: Validation Plateau Detected. Entering Memorization Phase.")
                
                log_metrics(step, epoch, total_loss, val_loss, mode, mem_gb, FACTORY_BATCH_SIZE, tps, lr, val_plateau_count)
            elif step % 20 == 0:
                mem_gb = mx.get_active_memory() / 1024**3
                tps = effective_tokens / dt
                print(f"[Step {step}] {mode} | Eff_BS:{FACTORY_BATCH_SIZE} | Loss:{total_loss:.4f} | LR:{lr:.2e} | {tps:.0f} tok/s")

            if time.time() - last_coo_time > 1800: 
                generate_cooing(model, sp)
                last_coo_time = time.time()
            
            if tokens_processed >= len(train_data) or step % 2000 == 0:
                epoch += 1
                tokens_processed = 0
                ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.safetensors"
                print(f"Saving Epoch {epoch} checkpoint to {ckpt_path}...")
                model.save_weights(str(ckpt_path))
                
            step += 1
            
    except KeyboardInterrupt:
        print("\nPaused.")
        model.save_weights(str(CHECKPOINT_DIR / "interrupt_save.safetensors"))

if __name__ == "__main__":
    main()