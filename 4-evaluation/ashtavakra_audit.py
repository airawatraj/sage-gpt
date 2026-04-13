import sys
import os
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
import numpy as np
from pathlib import Path

# --- Configuration & Paths ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    CHECKPOINT_DIR = project_root / "3-model/mlx/checkpoints"
    TOKENIZER_MODEL = project_root / "2-tokenizer/sutra_tokenizer.model"

INTERRUPT_SAVE = CHECKPOINT_DIR / "interrupt_save.safetensors"

# Architecture Alignment
VOCAB_SIZE = 8000
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
CONTEXT_LENGTH = 256
DROPOUT = 0.1

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
    logits = logits / max(temperature, 1e-5)
    
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

# --- Updated Audit Generating Logic ---
def audit_generate(model, sp, prompt, max_tokens=30, temp=0.8, use_sampler=True):
    ids = sp.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    seen_tokens = ids.copy()

    for _ in range(max_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
        logits_1d = logits[0]
        
        if use_sampler:
            token = sample_top_p(logits_1d, temperature=temp, top_p=0.9, repetition_penalty=1.5, seen_tokens=seen_tokens)
        else:
            token = mx.argmax(logits_1d, axis=-1, keepdims=True)
            
        token_id = token.item()
        seen_tokens.append(token_id)
        
        token_2d = mx.array([[token_id]], dtype=mx.uint32)
        x = mx.concatenate([x, token_2d], axis=1)
        
    return sp.decode(seen_tokens)

def check_stutter(text):
    if len(text) < 8: return False
    
    # Ignore sacred repetitions
    ignored_grams = ["ॐ", "नमः", "॥", "।"]
    
    for i in range(len(text) - 4):
        gram = text[i:i+4]
        # Only check stutter if it's not made entirely of ignored characters/words
        is_ignored = any(ig in gram for ig in ignored_grams)     
        if not is_ignored and text.count(gram) > 2:
            return True
            
    return False

# --- Updated Bends ---
def bend_3_vibhakti(model, sp):
    """3. Case Inflection: Summing probability of shattered Visarga IDs [259, 263]."""
    prompt = "राम"
    ids = sp.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    logits = model(x)[:, -1, :][0] # Get the logits for the next token
    probs = mx.softmax(logits)
    
    # Evaluate at the indices for ':' and 'ः'
    visarga_prob = probs[259].item() + probs[263].item()
    if visarga_prob > 0.3:
        return "STRAIGHT", f"P(ः fragments) = {visarga_prob:.2f}"
    return "CROOKED", f"P(ः fragments) = {visarga_prob:.2f} (Low)"

def main():
    print("\n🐚 ASHTAVAKRA AUDIT V2: SAMPLED DIAGNOSTIC 🐚")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    
    # Check if weights exist
    if not INTERRUPT_SAVE.exists():
        print(f"❌ Weights not found at {INTERRUPT_SAVE}")
        return
        
    model.load_weights(str(INTERRUPT_SAVE))
    mx.eval(model.parameters())

    print(f"{'BEND (TEST)':<30} | {'STATUS':<10} | {'DETAILS':<45}")
    print("-" * 90)

    score = 0
    total_bends = 8

    # --- Bend 1: Phonetic Stability ---
    gen1 = audit_generate(model, sp, "ॐ", max_tokens=30, temp=0.8)
    status1 = "WOBBLY" if check_stutter(gen1) else "STRAIGHT"
    if not any('\u0900' <= char <= '\u097F' for char in gen1.replace("ॐ", "")):
         status1 = "CROOKED" 
    if status1 == "STRAIGHT": score += 1
    details1 = gen1.replace('\n', ' ')
    print(f"{'1. Phonetic Stability':<30} | {status1:<10} | {details1[:43]}")

    # --- Bend 2: Invocation ---
    gen2 = audit_generate(model, sp, "असतो मा", max_tokens=15, temp=0.1, use_sampler=False)
    status2 = "STRAIGHT" if "सद्गमय" in gen2 else "CROOKED"
    if status2 == "STRAIGHT": score += 1
    details2 = gen2.replace('\n', ' ')
    print(f"{'2. Invocation':<30} | {status2:<10} | {details2[:43]}")

    # --- Bend 3: Case Inflection ---
    status3, details3 = bend_3_vibhakti(model, sp)
    if status3 == "STRAIGHT": score += 1
    details3 = details3.replace('\n', ' ')
    print(f"{'3. Case Inflection':<30} | {status3:<10} | {details3[:43]}")
    
    # --- Bend 4: Sandhi ---
    gen4 = audit_generate(model, sp, "नर", max_tokens=15, temp=0.1, use_sampler=False)
    status4 = "STRAIGHT" if ("इन्द्र" in gen4 or "ेन्द्र" in gen4) else "CROOKED"
    if status4 == "STRAIGHT": score += 1
    details4 = gen4.replace('\n', ' ')
    print(f"{'4. Name Synthesis':<30} | {status4:<10} | {details4[:43]}")
    
    # --- Bend 5: Concept Flow ---
    gen5 = audit_generate(model, sp, "यथा नद्यः", max_tokens=25, temp=0.8)
    status5 = "STRAIGHT" if "समुद्रे" in gen5 else "CROOKED"
    if status5 == "STRAIGHT": score += 1
    details5 = gen5.replace('\n', ' ')
    print(f"{'5. Concept Flow':<30} | {status5:<10} | {details5[:43]}")

    # --- Bend 6: Verse Sequence ---
    gen6 = audit_generate(model, sp, "ईशा वास्यमिदं", max_tokens=20, temp=0.1, use_sampler=False)
    status6 = "STRAIGHT" if "सर्वं" in gen6 else "CROOKED"
    if status6 == "STRAIGHT": score += 1
    details6 = gen6.replace('\n', ' ')
    print(f"{'6. Verse Sequence':<30} | {status6:<10} | {details6[:43]}")
    
    # --- Bend 7: Orthography ---
    gen7 = audit_generate(model, sp, "कृष्", max_tokens=10, temp=0.1, use_sampler=False)
    status7 = "STRAIGHT" if "ण" in gen7 else "CROOKED"
    if status7 == "STRAIGHT": score += 1
    details7 = gen7.replace('\n', ' ')
    print(f"{'7. Orthography':<30} | {status7:<10} | {details7[:43]}")
    
    # --- Bend 8: The Atman Test ---
    gen8 = audit_generate(model, sp, "तत्त्वमसि", max_tokens=20, temp=0.8)
    status8 = "STRAIGHT" if "श्वेतकेतो" in gen8 else "CROOKED"
    if status8 == "STRAIGHT": score += 1
    details8 = gen8.replace('\n', ' ')
    print(f"{'8. The Atman Test':<30} | {status8:<10} | {details8[:43]}")

    # --- Final Scorecard ---
    print("-" * 90)
    print(f"🕸️  Vedic Scorecard: {score}/{total_bends} Bends Straightened")

if __name__ == "__main__":
    main()