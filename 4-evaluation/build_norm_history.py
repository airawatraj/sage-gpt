"""
build_norm_history.py
─────────────────────
All-in-one replacement for the bash loop + inspect + plot pipeline.
Deps: torch, safetensors, matplotlib  (NO pandas required)

Usage:
    python 4-evaluation/build_norm_history.py
    python 4-evaluation/build_norm_history.py --plot-only   # skip re-scanning, just re-plot
"""

import sys
import csv
import argparse
import torch
from pathlib import Path
from datetime import datetime
from safetensors.torch import load_file

# ── Paths ──────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    LOG_DIR        = config.LOG_DIR / "evaluation"
except ImportError:
    print("❌ config.py not found — using relative paths.")
    CHECKPOINT_DIR = project_root / "3-model" / "pt" / "checkpoints"
    LOG_DIR        = project_root / "6-logs" / "evaluation"

LOG_FILE    = LOG_DIR / "norm_tracking_pt.csv"
OUTPUT_PLOT = LOG_DIR / "norm_history.png"
HEADER      = ["Timestamp", "Checkpoint", "Avg_Attn_Norm", "Avg_MLP_Norm", "Peak_Value"]


# ── 1. Inspect a single checkpoint ────────────────────────────────────────────
def inspect_checkpoint(ckpt_path: Path) -> dict:
    weights = load_file(str(ckpt_path))
    attn_norms, mlp_norms, peak_val = [], [], 0.0

    for key, tensor in sorted(weights.items()):
        if "weight" not in key:
            continue
        if "attn" not in key and "mlp" not in key:
            continue
        t = tensor.float()
        l2 = torch.linalg.norm(t).item()
        peak_val = max(peak_val, torch.max(torch.abs(t)).item())
        if "attn" in key:
            attn_norms.append(l2)
        else:
            mlp_norms.append(l2)

    return {
        "avg_attn": sum(attn_norms) / len(attn_norms) if attn_norms else 0.0,
        "avg_mlp":  sum(mlp_norms)  / len(mlp_norms)  if mlp_norms  else 0.0,
        "peak":     peak_val,
    }


# ── 2. Load existing CSV into a dict keyed by checkpoint name ─────────────────
def load_csv() -> dict:
    rows = {}
    if not LOG_FILE.exists():
        return rows
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["Checkpoint"]] = row
    return rows


# ── 3. Append a new row (or skip if already present) ─────────────────────────
def append_row(ckpt_name: str, stats: dict, existing: dict):
    if ckpt_name in existing:
        print(f"  ↩  {ckpt_name} already logged — skipping.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(HEADER)
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ckpt_name,
            f"{stats['avg_attn']:.4f}",
            f"{stats['avg_mlp']:.4f}",
            f"{stats['peak']:.4f}",
        ])
    print(f"  ✅ {ckpt_name}  attn={stats['avg_attn']:.4f}  mlp={stats['avg_mlp']:.4f}  peak={stats['peak']:.4f}")


# ── 4. Plot from CSV (pure csv + matplotlib, no pandas) ───────────────────────
def plot():
    if not LOG_FILE.exists():
        print("⚠️  No CSV found — nothing to plot.")
        return

    rows = []
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            try:
                epoch = int("".join(filter(str.isdigit, row["Checkpoint"])) or "0")
                rows.append({
                    "epoch":    epoch,
                    "avg_attn": float(row["Avg_Attn_Norm"]),
                    "avg_mlp":  float(row["Avg_MLP_Norm"]),
                    "peak":     float(row["Peak_Value"]),
                    "ckpt":     row["Checkpoint"],
                })
            except (ValueError, KeyError):
                continue  # skip malformed rows

    # Deduplicate by checkpoint name (keep last), then sort by epoch
    seen, deduped = set(), []
    for r in reversed(rows):
        if r["ckpt"] not in seen:
            seen.add(r["ckpt"])
            deduped.append(r)
    rows = sorted(deduped, key=lambda r: r["epoch"])

    if len(rows) < 2:
        print(f"⚠️  Only {len(rows)} unique epoch(s) logged.")
        print(f"   Epochs present: {[r['epoch'] for r in rows]}")
        print("   Run without --plot-only to scan all checkpoints first.")
        return

    import matplotlib
    matplotlib.use("Agg")          # headless — no display needed on DGX
    import matplotlib.pyplot as plt

    epochs    = [r["epoch"]    for r in rows]
    avg_attn  = [r["avg_attn"] for r in rows]
    avg_mlp   = [r["avg_mlp"]  for r in rows]
    peak_vals = [r["peak"]     for r in rows]

    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(16, 9))

    ln1 = ax1.plot(epochs, avg_attn,  label="Avg Attention L2",     color="#8FBC8F", linewidth=2.5, marker="o", markersize=6)
    ln2 = ax1.plot(epochs, avg_mlp,   label="Avg MLP L2",           color="#F4A460", linewidth=2.5, marker="s", markersize=6)
    ax1.set_xlabel("Epochs",          fontsize=12, color="#AAAAAA")
    ax1.set_ylabel("Macro L2 Norm",   fontsize=12, color="#AAAAAA")

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, peak_vals, label="Global Peak Intensity", color="#87CEFA", linewidth=1.8, linestyle="--", marker="v", alpha=0.7)
    ax2.set_ylabel("Peak Weight Intensity", fontsize=12, color="#87CEFA")

    lns  = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", framealpha=0.1)

    plt.title(
        f"SAGE-GPT MECHANISTIC CONTRACTIONS\nEpochs {epochs[0]} → {epochs[-1]}  ({len(rows)} checkpoints)",
        fontsize=16, pad=20, fontweight="bold",
    )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\n✅ Plot saved → {OUTPUT_PLOT}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Skip scanning; just re-plot existing CSV.")
    args = parser.parse_args()

    if not args.plot_only:
        checkpoints = sorted(
            CHECKPOINT_DIR.glob("epoch_*.safetensors"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not checkpoints:
            print(f"❌ No epoch_*.safetensors found in {CHECKPOINT_DIR}")
            sys.exit(1)

        print(f"\n🔬 Found {len(checkpoints)} checkpoint(s) in {CHECKPOINT_DIR}")
        existing = load_csv()

        for ckpt in checkpoints:
            stats = inspect_checkpoint(ckpt)
            append_row(ckpt.name, stats, existing)
            existing[ckpt.name] = {}   # mark as done for this run

    print("\n📊 Plotting norm history...")
    plot()


if __name__ == "__main__":
    main()
