import sys
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
import numpy as np
from pathlib import Path

# --- Configuration (STRICT ALIGNMENT WITH train_engine_mlx.py) ---
VOCAB_SIZE = 8000
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
CONTEXT_LENGTH = 256
DROPOUT = 0.1

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    # Fallback if config not found
    CHECKPOINT_DIR = project_root / "3-model/mlx/checkpoints"
    TOKENIZER_MODEL = project_root / "2-tokenizer/sutra_tokenizer.model"

# --- Model Architecture (EXACT REPLICA of train_engine_mlx.py) ---
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
        x_emb = self.embedding(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x_emb.dtype)
        x = x_emb
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# --- Dual-Mode Inference Logic ---
def sample_top_p(logits, temperature=0.8, top_p=0.9, repetition_penalty=1.2, seen_tokens=[]):
    """
    Applies custom top-p (nucleus) sampling, additive repetition penalty, and temperature scaling.
    Expected logits shape: 1D array of shape [vocab_size].
    """
    # 1) Additive repetition penalty
    if seen_tokens:
        mask_np = np.zeros(logits.shape[-1], dtype=np.float32)
        mask_np[list(set(seen_tokens))] = repetition_penalty
        mask_mx = mx.array(mask_np)
        logits = logits - mask_mx

    # 2) Temperature scaling
    logits = logits / temperature
    
    # 3) Nucleus (Top-P) sampling
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = logits[sorted_indices]
    
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right to keep the first token that crosses the threshold
    shift = mx.concatenate([
        mx.zeros_like(sorted_indices_to_remove[:1]),
        sorted_indices_to_remove[:-1]
    ], axis=-1)
    
    sorted_logits = mx.where(shift, mx.array(-1e9), sorted_logits)
    
    # 4) Sample using categorical
    sorted_sampled_idx = mx.random.categorical(sorted_logits, num_samples=1)
    
    # Map back to original index
    sampled_token = sorted_indices[sorted_sampled_idx]
    return sampled_token

def generate(model, tokenizer, prompt):
    print(f"\n>> {prompt}")
    
    # --- MODE 1 (RAW) ---
    print("\n[RAW]")
    ids_raw = tokenizer.encode(prompt)
    x_raw = mx.array([ids_raw], dtype=mx.uint32)
    for _ in range(50):
        logits = model(x_raw[:, -CONTEXT_LENGTH:])[:, -1, :]
        token = mx.argmax(logits, axis=-1, keepdims=True)
        token_id = token.item()
        print(tokenizer.decode([token_id]), end="", flush=True)
        x_raw = mx.concatenate([x_raw, token], axis=1)
    print()

    # --- MODE 2 (SAMPLED) ---
    temp = 0.8
    top_p = 0.9
    rep_penalty = 1.2
    
    print(f"\n[SAMPLED] (Top-P: {top_p}, Temp: {temp}, RepPenalty: {rep_penalty})")
    ids_sampled = tokenizer.encode(prompt)
    x_sampled = mx.array([ids_sampled], dtype=mx.uint32)
    seen_tokens = ids_sampled.copy()
    
    for _ in range(100):
        logits = model(x_sampled[:, -CONTEXT_LENGTH:])[:, -1, :]
        logits_1d = logits[0]  # Squeeze the batch dim
        
        token = sample_top_p(
            logits_1d, 
            temperature=temp, 
            top_p=top_p, 
            repetition_penalty=rep_penalty, 
            seen_tokens=seen_tokens
        )
        token_id = token.item()
        print(tokenizer.decode([token_id]), end="", flush=True)
        
        # update state
        seen_tokens.append(token_id)
        
        token_2d = mx.array([[token_id]], dtype=mx.uint32)
        x_sampled = mx.concatenate([x_sampled, token_2d], axis=1)
    print("\n")

def main():
    print("SanskritGPT Inference Engine V2 (Dual-Mode)")
    
    if not TOKENIZER_MODEL.exists():
        print(f"❌ Tokenizer not found at {TOKENIZER_MODEL}")
        return

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    
    # Try different potential checkpoint locations
    possible_ckpts = [
        CHECKPOINT_DIR / "interrupt_save.safetensors",
    ]
    # Add numbered epochs
    possible_ckpts.extend(sorted(CHECKPOINT_DIR.glob("epoch_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]), reverse=True))
    
    loaded_path = None
    for p in possible_ckpts:
        if p.exists():
            loaded_path = p
            break
            
    if not loaded_path:
        print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
        return

    try:
        model.load_weights(str(loaded_path))
        mx.eval(model.parameters()) # memory safety
        print(f"✅ Loaded weights from {loaded_path.name}")
        print(f"✅ Architecture Match: VOCAB={VOCAB_SIZE}, CTX={CONTEXT_LENGTH}, Dropout={DROPOUT}")
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    print("\nReady. Type 'exit' or 'q' to quit.")
    # Quick sanity check for auto-run purposes in development
    # To run a quick test programmatically, you can pipe "test_prompt\nexit"
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
