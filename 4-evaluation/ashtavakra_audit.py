"""
ashtavakra_audit.py — Enhanced 8-Bend Sanskrit generation audit for Sage-GPT.
"""
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from eval_utils import (
    CONTEXT_LENGTH, TOKENIZER_MODEL, DEVICE,
    get_target_checkpoint, load_model_from_checkpoint,
)

CHECKPOINT_PATH = get_target_checkpoint()

# ── Generation helper ─────────────────────────────────────────────────────────
def audit_generate(model, sp, prompt: str, max_tokens: int = 30, temp: float = 0.6) -> str:
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
            # More stable sampling for audit
            if temp < 0.3:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
    return sp.decode(x[0].tolist())

# ── Main audit ────────────────────────────────────────────────────────────────
def main():
    print(f"\n🐚 ASHTAVAKRA AUDIT V2.4: 8-BEND FULL DIAGNOSTIC 🐚")
    if not CHECKPOINT_PATH or not CHECKPOINT_PATH.exists():
        return print(f"❌ Checkpoint missing. Run training first.")

    print(f"📂 Checkpoint: {CHECKPOINT_PATH.name} | Device: {DEVICE}")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    bends = [
        ("1. Phonetic Stability", "ॐ", "नमः"),
        ("2. Invocation", "असतो मा", "सद्गमय"),
        ("3. Case Inflection", "राम", "ः"),
        ("4. Sandhi Logic", "नर", "इन्द्र"),
        ("5. Concept Flow", "यथा नद्यः", "समुद्रे"),
        ("6. Verse Sequence", "ईशा वास्य", "सर्वं"),
        ("7. Orthography", "कृष्", "ण"),
        ("8. The Atman Test", "तत्त्वमसि", "श्वेतकेतो"),
    ]

    print(f"\n{'BEND (TEST)':<25} | {'STATUS':<10} | {'PREVIEW'}")
    print("-" * 95)
    score = 0
    for name, prompt, target in bends:
        gen = audit_generate(model, sp, prompt)
        status = "STRAIGHT" if target in gen else "CROOKED"
        if status == "STRAIGHT":
            score += 1
        preview = gen.replace("\n", " ")[:60]
        print(f"{name:<25} | {status:<10} | {preview}")

    print("-" * 95)
    if score >= 6:
        print(f"🕉️  Vedic Score: {score}/8  →  🎉 GROKKED! Strong generalization.")
    elif score >= 4:
        print(f"🕉️  Vedic Score: {score}/8  →  Good progress. Circuits forming.")
    else:
        print(f"🕉️  Vedic Score: {score}/8  →  Still in circuit formation phase.")
    print()

if __name__ == "__main__":
    main()