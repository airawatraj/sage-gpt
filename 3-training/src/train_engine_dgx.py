"""
train_engine_dgx.py - Main training loop for Sage-GPT on DGX Spark.

Patch intent:
- Keep the current architecture and checkpoint format intact.
- Make train/validation gap apples-to-apples by evaluating both in model.eval().
- Log raw cross-entropy separately from label-smoothed training loss.
- Save best_grok_model only after a larger deterministic validation check.
- Keep old interrupt/epoch/step checkpoint resume compatibility.
"""

import csv
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sentencepiece as spm  # Kept because this environment already depends on it.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import load_file, save_file

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

# --- Configuration ---
VOCAB_SIZE = config.VOCAB_SIZE
N_LAYER = config.LAYERS
N_HEAD = config.HEADS
N_EMBD = config.EMBED_DIM
CONTEXT_LENGTH = config.CONTEXT_LENGTH
DROPOUT = config.DROPOUT

WEIGHT_DECAY = 0.08
LEARNING_RATE_MAX = 2e-5
LEARNING_RATE_MIN = 2e-5
WARMUP_STEPS = 150
LR_DECAY_STEPS = 2500
MAX_STEPS = None

LABEL_SMOOTHING = 0.1
TOTAL_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
GRAD_CLIP_NORM = 1.0

# Fast eval is used for routine monitoring. Full eval is used before replacing best.
EVAL_EVERY_STEPS = 50
EVAL_BATCH_SIZE = 32
EVAL_BATCHES_FAST = 50
EVAL_BATCHES_FULL = 500
EVAL_SEED_TRAIN = 4242
EVAL_SEED_VAL = 1337
BEST_MIN_DELTA = 0.001

CHECKPOINT_EVERY_STEPS = 500
THERMAL_LIMIT_C = 75
THERMAL_SLEEP_S = 30
THERMAL_CHECK_EVERY = 100

DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"

DEVICE = "cuda"

LOG_HEADER = [
    "Timestamp",
    "Step",
    "Train_Loss",
    "Val_Loss",
    "Gap",
    "Live_Train_Loss",
    "Train_Smoothed_Loss",
    "Val_Smoothed_Loss",
    "LR",
    "TPS",
    "Mem_GB",
    "Eval_Batches",
    "Best_Val_Raw_Loss",
]


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
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH))
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        xq = self.wq(x).view(batch_size, seq_len, self.n_head, -1)
        xk = self.wk(x).view(batch_size, seq_len, self.n_head, -1)
        xv = self.wv(x).view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[:seq_len])
        y = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True,
        )
        return self.resid_drop(self.wo(y.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)))


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
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.output.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, label_smoothing=LABEL_SMOOTHING):
        x = self.drop(self.tok_emb(x))
        for layer in self.layers:
            x = layer(x)
        logits = self.output(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            label_smoothing=label_smoothing,
        )
        return logits, loss


# --- Training Utilities ---
def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE_MAX * (step + 1) / WARMUP_STEPS
    if step > LR_DECAY_STEPS:
        return LEARNING_RATE_MIN
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * (step - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS))
    )
    return LEARNING_RATE_MIN + coeff * (LEARNING_RATE_MAX - LEARNING_RATE_MIN)


def get_batch(data, ctx_len, batch_size, rng=None):
    if len(data) <= ctx_len + 1:
        raise ValueError(f"Dataset has {len(data)} tokens, but context length is {ctx_len}.")
    high = len(data) - ctx_len
    if rng is None:
        ix = np.random.randint(0, high, size=(batch_size,))
    else:
        ix = rng.integers(0, high, size=(batch_size,))
    offsets = np.arange(ctx_len)
    x_idx = ix[:, None] + offsets
    y_idx = ix[:, None] + offsets + 1
    x = torch.from_numpy(data[x_idx].astype(np.int64)).to(DEVICE, non_blocking=True)
    y = torch.from_numpy(data[y_idx].astype(np.int64)).to(DEVICE, non_blocking=True)
    return x, y


def compute_ce_losses(logits, targets):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    raw = F.cross_entropy(flat_logits, flat_targets, label_smoothing=0.0)
    smoothed = F.cross_entropy(flat_logits, flat_targets, label_smoothing=LABEL_SMOOTHING)
    return raw, smoothed


@torch.no_grad()
def estimate_loss(model, data, ctx, num_batches, batch_size, seed):
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    raw_losses = []
    smoothed_losses = []

    for _ in range(num_batches):
        x, y = get_batch(data, ctx, batch_size, rng=rng)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x, None)
            raw_loss, smoothed_loss = compute_ce_losses(logits, y)
        raw_losses.append(raw_loss.item())
        smoothed_losses.append(smoothed_loss.item())
        del x, y, logits, raw_loss, smoothed_loss

    if was_training:
        model.train()

    raw_mean = float(np.mean(raw_losses))
    smooth_mean = float(np.mean(smoothed_losses))
    return {
        "raw": raw_mean,
        "smoothed": smooth_mean,
        "raw_ppl": float(math.exp(raw_mean)) if raw_mean < 20 else float("inf"),
        "smoothed_ppl": float(math.exp(smooth_mean)) if smooth_mean < 20 else float("inf"),
    }


def get_gpu_temp():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def get_base_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def checkpoint_state_dict(model):
    state_dict = get_base_model(model).state_dict()
    if "output.weight" in state_dict and "tok_emb.weight" in state_dict:
        if state_dict["output.weight"].data_ptr() == state_dict["tok_emb.weight"].data_ptr():
            state_dict = dict(state_dict)
            del state_dict["output.weight"]
    return state_dict


def load_model_weights(model, ckpt_path):
    model_state = load_file(str(ckpt_path))
    if "tok_emb.weight" in model_state and "output.weight" not in model_state:
        model_state["output.weight"] = model_state["tok_emb.weight"]
    get_base_model(model).load_state_dict(model_state)


def save_model_only(model, path):
    save_file(checkpoint_state_dict(model), str(path))


def save_training_checkpoint(model, optimizer, step, epoch, tokens_processed, best_val_raw_loss, save_path, state_path):
    save_model_only(model, save_path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "tokens_processed": tokens_processed,
            "best_val_raw_loss": best_val_raw_loss,
            "best_v_loss": best_val_raw_loss,  # Backward-compatible key.
        },
        str(state_path),
    )


def ensure_log_header():
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="") as f:
            first_line = f.readline().strip()
        if first_line:
            existing_header = first_line.split(",")
            if existing_header != LOG_HEADER:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                legacy_path = LOG_FILE.with_name(f"{LOG_FILE.stem}.legacy_{timestamp}{LOG_FILE.suffix}")
                LOG_FILE.rename(legacy_path)
                print(f"Archived legacy training log to {legacy_path}")

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(LOG_HEADER)


def append_log_row(
    step,
    train_raw,
    val_raw,
    gap,
    live_train_loss,
    train_smoothed,
    val_smoothed,
    lr,
    tps,
    mem_gb,
    eval_batches,
    best_val_raw_loss,
):
    ensure_log_header()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(),
                step,
                f"{train_raw:.6f}",
                f"{val_raw:.6f}",
                f"{gap:.6f}",
                f"{live_train_loss:.6f}",
                f"{train_smoothed:.6f}",
                f"{val_smoothed:.6f}",
                f"{lr:.8e}",
                f"{tps:.0f}",
                f"{mem_gb:.2f}",
                eval_batches,
                f"{best_val_raw_loss:.6f}" if math.isfinite(best_val_raw_loss) else "inf",
            ]
        )


def maybe_save_best(model, val_raw_fast, best_val_raw_loss, step):
    if not math.isfinite(best_val_raw_loss) or val_raw_fast < best_val_raw_loss - BEST_MIN_DELTA:
        print("Candidate best detected. Running larger deterministic validation...")
        full_val = estimate_loss(
            model,
            val_data_global,
            CONTEXT_LENGTH,
            EVAL_BATCHES_FULL,
            EVAL_BATCH_SIZE,
            EVAL_SEED_VAL,
        )
        print(
            f"FULL VAL | Raw CE: {full_val['raw']:.4f} | "
            f"PPL: {full_val['raw_ppl']:.2f} | batches: {EVAL_BATCHES_FULL}"
        )
        if not math.isfinite(best_val_raw_loss) or full_val["raw"] < best_val_raw_loss - BEST_MIN_DELTA:
            best_path = CHECKPOINT_DIR / "best_grok_model.safetensors"
            save_model_only(model, best_path)
            print(f"New best model saved: {best_path} | raw_val_ce={full_val['raw']:.4f} | step={step}")
            return full_val["raw"], full_val
        print("Candidate rejected after full validation; keeping previous best.")
        return best_val_raw_loss, full_val
    return best_val_raw_loss, None


# This global is set in main() so maybe_save_best keeps its call signature small.
val_data_global = None


def main():
    global val_data_global

    print("--- SAGE-GPT DGX FOUNDRY: INITIALIZING ---")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DGX training script, but torch.cuda.is_available() is false.")

    data = np.fromfile(DATA_PATH, dtype=np.uint16)
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    val_data_global = val_data

    print(f"Data loaded: train_tokens={len(train_data):,} | val_tokens={len(val_data):,}")

    model = SageGPT(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    if hasattr(torch, "compile"):
        print("Compiling graph for acceleration...")
        model = torch.compile(model)

    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
    nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE_MAX,
        betas=(0.9, 0.95),
    )

    step = 0
    tokens_processed = 0
    epoch = 0
    best_val_raw_loss = float("inf")
    gap_exceeded_steps = 0

    interrupt_ckpt = CHECKPOINT_DIR / "interrupt.safetensors"
    interrupt_state = CHECKPOINT_DIR / "interrupt_state.pt"
    if interrupt_ckpt.exists():
        print("Auto-resuming from emergency save...")
        load_model_weights(model, interrupt_ckpt)
        if interrupt_state.exists():
            checkpoint = torch.load(str(interrupt_state), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = int(checkpoint.get("step", 0))
            epoch = int(checkpoint.get("epoch", 0))
            tokens_processed = int(checkpoint.get("tokens_processed", 0))
            best_val_raw_loss = float(checkpoint.get("best_val_raw_loss", float("inf")))
            print(f"State restored: epoch={epoch} | step={step} | best_val_raw={best_val_raw_loss}")
        else:
            print("Warning: interrupt model found, but interrupt_state.pt was missing.")

    tokens_per_epoch = len(train_data)
    ensure_log_header()
    t0 = time.time()

    try:
        while True:
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            live_train_loss = 0.0

            for _ in range(GRAD_ACCUM_STEPS):
                x, y = get_batch(train_data, CONTEXT_LENGTH, MICRO_BATCH_SIZE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y, label_smoothing=LABEL_SMOOTHING)
                    loss = loss / GRAD_ACCUM_STEPS
                live_train_loss += loss.item()
                loss.backward()
                del x, y, loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            step_tokens = TOTAL_BATCH_SIZE * CONTEXT_LENGTH
            tokens_processed += step_tokens
            is_epoch_boundary = False
            while tokens_processed >= tokens_per_epoch:
                epoch += 1
                tokens_processed -= tokens_per_epoch
                is_epoch_boundary = True

            t1 = time.time()
            dt = max(t1 - t0, 1e-9)
            t0 = t1
            tps = step_tokens / dt
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            exact_epoch = epoch + (tokens_processed / tokens_per_epoch)

            if step % 20 == 0:
                print(
                    f"[DGX Spark] [Step {step:6d}] "
                    f"LiveLoss: {live_train_loss:.4f} | LR: {lr:.2e} | "
                    f"{tps / 1e6:.2f}M tok/s | Mem: {mem_gb:.1f}GB | Epoch: {exact_epoch:.2f}"
                )

            if step % EVAL_EVERY_STEPS == 0:
                train_eval = estimate_loss(
                    model,
                    train_data,
                    CONTEXT_LENGTH,
                    EVAL_BATCHES_FAST,
                    EVAL_BATCH_SIZE,
                    EVAL_SEED_TRAIN,
                )
                val_eval = estimate_loss(
                    model,
                    val_data,
                    CONTEXT_LENGTH,
                    EVAL_BATCHES_FAST,
                    EVAL_BATCH_SIZE,
                    EVAL_SEED_VAL,
                )
                gap = val_eval["raw"] - train_eval["raw"]

                print(
                    "\nVAL REPORT | "
                    f"Train Raw CE: {train_eval['raw']:.4f} | "
                    f"Val Raw CE: {val_eval['raw']:.4f} | "
                    f"Gap: {gap:.4f} | "
                    f"Val PPL: {val_eval['raw_ppl']:.2f}"
                )
                print(
                    f"SMOOTHED | Train: {train_eval['smoothed']:.4f} | "
                    f"Val: {val_eval['smoothed']:.4f} | Live train: {live_train_loss:.4f}\n"
                )

                best_val_raw_loss, _ = maybe_save_best(model, val_eval["raw"], best_val_raw_loss, step)

                if gap > 2.5:
                    gap_exceeded_steps += EVAL_EVERY_STEPS
                    if gap_exceeded_steps >= 500:
                        print("Warning: large eval-mode generalization gap persisting.")
                else:
                    gap_exceeded_steps = 0

                append_log_row(
                    step=step,
                    train_raw=train_eval["raw"],
                    val_raw=val_eval["raw"],
                    gap=gap,
                    live_train_loss=live_train_loss,
                    train_smoothed=train_eval["smoothed"],
                    val_smoothed=val_eval["smoothed"],
                    lr=lr,
                    tps=tps,
                    mem_gb=mem_gb,
                    eval_batches=EVAL_BATCHES_FAST,
                    best_val_raw_loss=best_val_raw_loss,
                )

            if is_epoch_boundary or (step > 0 and step % CHECKPOINT_EVERY_STEPS == 0):
                if is_epoch_boundary:
                    save_path = CHECKPOINT_DIR / f"epoch_{epoch}.safetensors"
                    state_path = CHECKPOINT_DIR / f"epoch_{epoch}_state.pt"
                else:
                    save_path = CHECKPOINT_DIR / f"step_{step}.safetensors"
                    state_path = CHECKPOINT_DIR / f"step_{step}_state.pt"

                save_training_checkpoint(
                    model,
                    optimizer,
                    step,
                    epoch,
                    tokens_processed,
                    best_val_raw_loss,
                    save_path,
                    state_path,
                )
                print(f"Checkpoint saved: {save_path}")
                prune_checkpoints(keep=10, keep_steps=4)

            if step % THERMAL_CHECK_EVERY == 0:
                gpu_temp = get_gpu_temp()
                if gpu_temp >= THERMAL_LIMIT_C:
                    print(f"THERMAL GUARD: GPU at {gpu_temp}C; pausing {THERMAL_SLEEP_S}s")
                    time.sleep(THERMAL_SLEEP_S)

            if step > 52000 and step % 5000 == 0:
                for param_group in optimizer.param_groups:
                    if "weight_decay" in param_group and param_group["weight_decay"] > 0.0:
                        if param_group["weight_decay"] < 0.12:
                            param_group["weight_decay"] = min(0.12, param_group["weight_decay"] * 1.05)
                            print(f"Increased weight decay to {param_group['weight_decay']:.3f}")

            step += 1
            if MAX_STEPS is not None and step >= MAX_STEPS:
                print("MAX_STEPS reached.")
                break

    except KeyboardInterrupt:
        print("\n--- SUTRA FOUNDRY HALTED: EMERGENCY SAVE ---")
        save_training_checkpoint(
            model,
            optimizer,
            step,
            epoch,
            tokens_processed,
            best_val_raw_loss,
            CHECKPOINT_DIR / "interrupt.safetensors",
            CHECKPOINT_DIR / "interrupt_state.pt",
        )
        print("Emergency checkpoint saved.")


if __name__ == "__main__":
    main()
