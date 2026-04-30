"""
ashtavakra_audit.py — 8-Bend Sanskrit generation audit for Sage-GPT.
Architecture and checkpoint loading live in eval_utils.py.
"""

import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from eval_utils import (
    CONTEXT_LENGTH, TOKENIZER_MODEL, DEVICE,
    get_target_checkpoint, load_model_from_checkpoint,
)

CHECKPOINT_PATH = get_target_checkpoint()


# ── Generation helper ─────────────────────────────────────────────────────────
def audit_generate(model, sp, prompt: str, max_tokens: int = 25, temp: float = 0.7) -> str:
    ids = sp.encode(prompt)
    x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits     = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
            probs      = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x          = torch.cat([x, next_token], dim=1)
    return sp.decode(x[0].tolist())


# ── Main audit ────────────────────────────────────────────────────────────────
def main():
    print(f"\n🐚 ASHTAVAKRA AUDIT V2.3: 8-BEND FULL DIAGNOSTIC 🐚")

    if not CHECKPOINT_PATH or not CHECKPOINT_PATH.exists():
        return print(f"❌ Checkpoint missing. Run training first.")

    print(f"📂 Checkpoint: {CHECKPOINT_PATH.name}  |  Device: {DEVICE}")

    sp    = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    bends = [
        ("1. Phonetic Stability", "ॐ",            "नमः"),
        ("2. Invocation",         "असतो मा",       "सद्गमय"),
        ("3. Case Inflection",    "राम",            "ः"),
        ("4. Sandhi Logic",       "नर",             "इन्द्र"),
        ("5. Concept Flow",       "यथा नद्यः",      "समुद्रे"),
        ("6. Verse Sequence",     "ईशा वास्य",      "सर्वं"),
        ("7. Orthography",        "कृष्",           "ण"),
        ("8. The Atman Test",     "तत्त्वमसि",      "श्वेतकेतो"),
    ]

    print(f"\n{'BEND (TEST)':<25} | {'STATUS':<10} | {'PREVIEW'}")
    print("-" * 95)

    score = 0
    for name, prompt, target in bends:
        gen    = audit_generate(model, sp, prompt)
        status = "STRAIGHT" if target in gen else "CROOKED"
        if status == "STRAIGHT":
            score += 1
        print(f"{name:<25} | {status:<10} | {gen.replace(chr(10), ' ')[:50]}")

    print("-" * 95)
    print(f"🕸️  Vedic Scorecard: {score}/8 Bends Straightened\n")


if __name__ == "__main__":
    main()