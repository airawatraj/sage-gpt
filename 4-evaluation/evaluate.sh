#!/bin/bash
set -e

echo "[SAGE-ARCH] Running Mechanistic Weight Norm Inspection..."
uv run python3 4-evaluation/inspect_norms.py

echo "[SAGE-ARCH] Visualizing Weight Norm Trajectories..."
uv run python3 4-evaluation/plot_norms.py

echo "[SAGE-ARCH] Generating Generalisation Gap Plot..."
uv run python3 4-evaluation/generalisation_gap_monitor.py

echo "[SAGE-ARCH] Running the Ashtavakra Audit..."
uv run python3 4-evaluation/ashtavakra_audit.py

echo "[SAGE-ARCH] Running Inference Engine..."
uv run python3 5-inference/inference_engine_mlx_v2.py <<EOF
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
EOF