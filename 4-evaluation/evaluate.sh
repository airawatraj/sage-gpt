#!/bin/bash
set -euo pipefail

CHECKPOINT="${1:-3-model/pt/checkpoints/best_grok_model.safetensors}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "[SAGE-ARCH] ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "[SAGE-ARCH] Using checkpoint: $CHECKPOINT"

echo "[SAGE-ARCH] 🔬 Probe: Architecture & Health Check..."
uv run python3 4-evaluation/sutra_probe_pt.py "$CHECKPOINT"

echo "[SAGE-ARCH] 📊 Building Weight Norm History + Plot..."
uv run python3 4-evaluation/build_norm_history.py

echo "[SAGE-ARCH] 📉 Generating Generalisation Gap Plot..."
uv run python3 4-evaluation/generalisation_gap_monitor.py

echo "[SAGE-ARCH] 🐚 Running Ashtavakra Audit..."
uv run python3 4-evaluation/ashtavakra_audit.py "$CHECKPOINT"

echo "[SAGE-ARCH] 🕉️  Running Inference Engine..."
uv run python3 5-inference/inference_engine_pt.py <<'PROMPTS'
यथा नद्यः
उत्तिष्ठत
तत्त्वमसि
असतो मा
ईशा वास्यमिदं
कृष्णः
ॐ
रामः
ॐ नमः
अग्निमीळे
exit
PROMPTS
