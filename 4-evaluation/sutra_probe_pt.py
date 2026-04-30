"""
sutra_probe_pt.py — Deep structural + health inspection of a Sage-GPT checkpoint.
get_target_checkpoint() is shared from eval_utils.py.
"""

import sys
import torch
from pathlib import Path
from safetensors.torch import load_file

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from eval_utils import CHECKPOINT_DIR, get_target_checkpoint

try:
    import config
    EXPECTED_LAYERS = config.LAYERS
    EXPECTED_DIM    = config.EMBED_DIM
    EXPECTED_VOCAB  = config.VOCAB_SIZE
except ImportError:
    print("❌ Critical: config.py not found.")
    sys.exit(1)


def main():
    print(f"\n🔎 [SAGE-PROBE] Initializing Deep Checkpoint Inspection...")
    ckpt_path = get_target_checkpoint()

    if not ckpt_path or not ckpt_path.exists():
        print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
        return

    print(f"📂 Target: {ckpt_path.name}")
    print("-" * 60)

    try:
        weights = load_file(str(ckpt_path))

        # 1. Architecture Identity Check
        has_emb    = "tok_emb.weight" in weights
        layer_keys = [k for k in weights if "layers." in k]
        if not layer_keys:
            print("⚠️  No 'layers' found — checkpoint may use old 'blocks' schema.")
            layer_keys = [k for k in weights if "blocks." in k]

        layer_indices = []
        for k in layer_keys:
            try:
                layer_indices.append(int(k.split(".")[1]))
            except (IndexError, ValueError):
                continue

        found_layers = max(layer_indices) + 1 if layer_indices else 0
        emb_shape    = weights["tok_emb.weight"].shape if has_emb else (0, 0)

        print(f"💎 Identity: SageGPT Transformer")
        print(f"📏 Layers:   {found_layers} / {EXPECTED_LAYERS} expected")
        print(f"🌌 Vocab:    {emb_shape[0]} / {EXPECTED_VOCAB} expected")
        print(f"🧬 D_Model:  {emb_shape[1]} / {EXPECTED_DIM} expected")

        # 2. Mathematical Health Check
        print("-" * 60)
        print("🧠 Mechanistic Health Audit:")

        nan_found   = False
        inf_found   = False
        total_params = 0

        for key, tensor in weights.items():
            if tensor.is_complex():
                # freqs_cis (RoPE buffer) is complex64 — a precomputed constant,
                # not a trainable weight. Skip to avoid spurious UserWarning.
                continue
            t = tensor.float()
            total_params += t.numel()
            if torch.isnan(t).any(): nan_found = True
            if torch.isinf(t).any(): inf_found = True

        health = "💚 HEALTHY" if not (nan_found or inf_found) else "💔 CORRUPTED"
        print(f"   Status:   {health}")
        print(f"   NaNs:     {'Found! ❌' if nan_found else 'None ✅'}")
        print(f"   Infs:     {'Found! ❌' if inf_found else 'None ✅'}")
        print(f"   Params:   {total_params / 1e6:.2f}M")

        # 3. Consistency check
        if found_layers != EXPECTED_LAYERS or emb_shape[1] != EXPECTED_DIM:
            print(f"\n🚨 ARCHITECTURE MISMATCH!")
            print(f"   Config expects {EXPECTED_LAYERS}L/{EXPECTED_DIM}D "
                  f"but checkpoint shows {found_layers}L/{emb_shape[1]}D.")
        else:
            print("\n✅ Checkpoint perfectly aligned with config.py")

    except Exception as e:
        print(f"💥 Error during probe: {e}")


if __name__ == "__main__":
    main()