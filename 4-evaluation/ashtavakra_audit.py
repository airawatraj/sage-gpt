import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# --- Configuration & Paths ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    CHECKPOINT_DIR = project_root / "3-model/pt/checkpoints"
    TOKENIZER_MODEL = project_root / "2-tokenizer/sutra_tokenizer.model"

INTERRUPT_SAVE = CHECKPOINT_DIR / "interrupt.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture Alignment
VOCAB_SIZE = 8000
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
CONTEXT_LENGTH = 256
DROPOUT = 0.1

# --- PyTorch Architecture Replication ---
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
        freqs_cis = precompute_freqs_cis(n_embd // n_head, CONTEXT_LENGTH * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, L, self.n_head, -1)
        k = k.view(B, L, self.n_head, -1)
        v = v.view(B, L, self.n_head, -1).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:L].to(x.device))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
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

# --- Dual-Mode Inference Logic ---
def sample_top_p(logits, temperature=0.8, top_p=0.9, repetition_penalty=1.2, seen_tokens=None):
    if seen_tokens:
        mask = torch.zeros_like(logits)
        mask[list(set(seen_tokens))] = repetition_penalty
        logits = logits - mask

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

def audit_generate(model, sp, prompt, max_tokens=30, temp=0.8, use_sampler=True):
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    seen_tokens = ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
            logits_1d = logits[0]
            
            if use_sampler:
                token_id = sample_top_p(logits_1d, temperature=temp, top_p=0.9, repetition_penalty=1.5, seen_tokens=seen_tokens)
            else:
                token_id = torch.argmax(logits_1d, dim=-1).item()
                
            seen_tokens.append(token_id)
            x = torch.cat([x, torch.tensor([[token_id]], device=DEVICE)], dim=1)
            
    return sp.decode(seen_tokens)

def check_stutter(text):
    if len(text) < 8: return False
    ignored_grams = ["ॐ", "नमः", "॥", "।"]
    for i in range(len(text) - 4):
        gram = text[i:i+4]
        is_ignored = any(ig in gram for ig in ignored_grams)     
        if not is_ignored and text.count(gram) > 2:
            return True
    return False

def bend_3_vibhakti(model, sp):
    prompt = "राम"
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(x)[:, -1, :][0]
        probs = F.softmax(logits, dim=-1)
        visarga_prob = probs[259].item() + probs[263].item()
    if visarga_prob > 0.3:
        return "STRAIGHT", f"P(ः fragments) = {visarga_prob:.2f}"
    return "CROOKED", f"P(ः fragments) = {visarga_prob:.2f} (Low)"

def main():
    print(f"\n🐚 ASHTAVAKRA AUDIT V2: PT DGX DIAGNOSTIC ({DEVICE}) 🐚")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD).to(DEVICE)
    
    if not INTERRUPT_SAVE.exists():
        print(f"❌ Weights not found at {INTERRUPT_SAVE}")
        return
        
    state_dict = load_file(str(INTERRUPT_SAVE))
    model.load_state_dict(state_dict)
    model.eval()

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
    print(f"{'1. Phonetic Stability':<30} | {status1:<10} | {gen1.replace(chr(10), ' ')[:43]}")

    # --- Bend 2: Invocation ---
    gen2 = audit_generate(model, sp, "असतो मा", max_tokens=15, temp=0.1, use_sampler=False)
    status2 = "STRAIGHT" if "सद्गमय" in gen2 else "CROOKED"
    if status2 == "STRAIGHT": score += 1
    print(f"{'2. Invocation':<30} | {status2:<10} | {gen2.replace(chr(10), ' ')[:43]}")

    # --- Bend 3: Case Inflection ---
    status3, details3 = bend_3_vibhakti(model, sp)
    if status3 == "STRAIGHT": score += 1
    print(f"{'3. Case Inflection':<30} | {status3:<10} | {details3.replace(chr(10), ' ')[:43]}")
    
    # --- Bend 4: Sandhi ---
    gen4 = audit_generate(model, sp, "नर", max_tokens=15, temp=0.1, use_sampler=False)
    status4 = "STRAIGHT" if ("इन्द्र" in gen4 or "ेन्द्र" in gen4) else "CROOKED"
    if status4 == "STRAIGHT": score += 1
    print(f"{'4. Name Synthesis':<30} | {status4:<10} | {gen4.replace(chr(10), ' ')[:43]}")
    
    # --- Bend 5: Concept Flow ---
    gen5 = audit_generate(model, sp, "यथा नद्यः", max_tokens=25, temp=0.8)
    status5 = "STRAIGHT" if "समुद्रे" in gen5 else "CROOKED"
    if status5 == "STRAIGHT": score += 1
    print(f"{'5. Concept Flow':<30} | {status5:<10} | {gen5.replace(chr(10), ' ')[:43]}")

    # --- Bend 6: Verse Sequence ---
    gen6 = audit_generate(model, sp, "ईशा वास्यमिदं", max_tokens=20, temp=0.1, use_sampler=False)
    status6 = "STRAIGHT" if "सर्वं" in gen6 else "CROOKED"
    if status6 == "STRAIGHT": score += 1
    print(f"{'6. Verse Sequence':<30} | {status6:<10} | {gen6.replace(chr(10), ' ')[:43]}")
    
    # --- Bend 7: Orthography ---
    gen7 = audit_generate(model, sp, "कृष्", max_tokens=10, temp=0.1, use_sampler=False)
    status7 = "STRAIGHT" if "ण" in gen7 else "CROOKED"
    if status7 == "STRAIGHT": score += 1
    print(f"{'7. Orthography':<30} | {status7:<10} | {gen7.replace(chr(10), ' ')[:43]}")
    
    # --- Bend 8: The Atman Test ---
    gen8 = audit_generate(model, sp, "तत्त्वमसि", max_tokens=20, temp=0.8)
    status8 = "STRAIGHT" if "श्वेतकेतो" in gen8 else "CROOKED"
    if status8 == "STRAIGHT": score += 1
    print(f"{'8. The Atman Test':<30} | {status8:<10} | {gen8.replace(chr(10), ' ')[:43]}")

    # --- Final Scorecard ---
    print("-" * 90)
    print(f"🕸️  Vedic Scorecard: {score}/{total_bends} Bends Straightened")

if __name__ == "__main__":
    main()