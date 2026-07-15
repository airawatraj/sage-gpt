"""build_norm_history.py - rebuild current active checkpoint norm history."""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config

    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    LOG_DIR = config.LOG_DIR / "evaluation"
except ImportError:
    CHECKPOINT_DIR = project_root / "3-model" / "pt" / "checkpoints"
    LOG_DIR = project_root / "6-logs" / "evaluation"

LOG_FILE = LOG_DIR / "norm_tracking_pt.csv"
OUTPUT_PLOT = LOG_DIR / "norm_history.png"
HEADER = ["Timestamp", "Checkpoint", "Epoch", "Avg_Attn_Norm", "Avg_MLP_Norm", "Peak_Value"]


def checkpoint_epoch(path):
    match = re.match(r"epoch_(\d+)\.safetensors$", path.name)
    if not match:
        return None
    return int(match.group(1))


def inspect_checkpoint(ckpt_path):
    weights = load_file(str(ckpt_path))
    attn_norms = []
    mlp_norms = []
    peak_val = 0.0

    for key, tensor in sorted(weights.items()):
        key_l = key.lower()
        if "weight" not in key_l:
            continue

        is_attn = "attn" in key_l or "attention" in key_l
        is_mlp = "mlp" in key_l or "ffn" in key_l or "feed_forward" in key_l

        if not is_attn and not is_mlp:
            continue

        t = tensor.float()
        l2 = torch.linalg.norm(t).item()
        peak_val = max(peak_val, torch.max(torch.abs(t)).item())

        if is_attn:
            attn_norms.append(l2)
        elif is_mlp:
            mlp_norms.append(l2)

    return {
        "avg_attn": sum(attn_norms) / len(attn_norms) if attn_norms else 0.0,
        "avg_mlp": sum(mlp_norms) / len(mlp_norms) if mlp_norms else 0.0,
        "peak": peak_val,
    }


def scan_active_epoch_checkpoints():
    checkpoints = []
    for path in CHECKPOINT_DIR.glob("epoch_*.safetensors"):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            checkpoints.append((epoch, path))

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def write_csv(rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["checkpoint"],
                    row["epoch"],
                    f"{row['avg_attn']:.6f}",
                    f"{row['avg_mlp']:.6f}",
                    f"{row['peak']:.6f}",
                ]
            )


def read_csv():
    if not LOG_FILE.exists():
        return []

    rows = []
    with LOG_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "checkpoint": row["Checkpoint"],
                        "epoch": int(row["Epoch"]),
                        "avg_attn": float(row["Avg_Attn_Norm"]),
                        "avg_mlp": float(row["Avg_MLP_Norm"]),
                        "peak": float(row["Peak_Value"]),
                    }
                )
            except (KeyError, ValueError):
                continue

    rows.sort(key=lambda row: row["epoch"])
    return rows


def set_tight_ylim(ax, values, pad_ratio=0.10):
    vals = [v for v in values if v > 0]
    if not vals:
        return
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        hi = lo + 1e-6
    pad = (hi - lo) * pad_ratio
    ax.set_ylim(lo - pad, hi + pad)


def plot():
    rows = read_csv()
    if len(rows) < 2:
        print(f"Need at least 2 logged checkpoints to plot. Found {len(rows)}.")
        return

    epochs = [r["epoch"] for r in rows]
    avg_attn = [r["avg_attn"] for r in rows]
    avg_mlp = [r["avg_mlp"] for r in rows]
    peak_vals = [r["peak"] for r in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    fig, ax1 = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0B0B0B")
    ax1.set_facecolor("#0B0B0B")
    ax1.grid(True, alpha=0.16, linewidth=0.8)

    ln1 = ax1.plot(
        epochs,
        avg_attn,
        label="Avg Attention L2",
        color="#8FBC8F",
        linewidth=2.5,
        marker="o",
        markersize=5,
    )
    ln2 = ax1.plot(
        epochs,
        avg_mlp,
        label="Avg MLP L2",
        color="#F4A460",
        linewidth=2.5,
        marker="s",
        markersize=5,
    )

    ax1.set_xlabel("Epochs", fontsize=12, color="#D8D8D8")
    ax1.set_ylabel("Macro L2 norm", fontsize=12, color="#D8D8D8")
    ax1.tick_params(colors="#D8D8D8")
    set_tight_ylim(ax1, avg_attn + avg_mlp)

    ax2 = ax1.twinx()
    ln3 = ax2.plot(
        epochs,
        peak_vals,
        label="Global Peak Intensity",
        color="#87CEFA",
        linewidth=1.8,
        linestyle="--",
        marker="v",
        markersize=5,
        alpha=0.80,
    )
    ax2.set_ylabel("Peak weight intensity", fontsize=12, color="#87CEFA")
    ax2.tick_params(colors="#87CEFA")
    set_tight_ylim(ax2, peak_vals)

    lines = ln1 + ln2 + ln3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.18)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#A0A0A0")

    plt.title(
        "SAGE-GPT Mechanistic Contractions\n"
        f"Active epoch checkpoints only | Epochs {epochs[0]} to {epochs[-1]} | {len(rows)} checkpoints",
        fontsize=16,
        pad=20,
        fontweight="bold",
        color="#F2F2F2",
    )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Norm history saved: {OUTPUT_PLOT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Only replot existing CSV.")
    args = parser.parse_args()

    if not args.plot_only:
        checkpoints = scan_active_epoch_checkpoints()
        if not checkpoints:
            print(f"No active epoch checkpoints found in {CHECKPOINT_DIR}")
            sys.exit(1)

        print(f"Scanning {len(checkpoints)} active epoch checkpoint(s) from {CHECKPOINT_DIR}")
        rows = []
        for epoch, ckpt in checkpoints:
            stats = inspect_checkpoint(ckpt)
            rows.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "checkpoint": ckpt.name,
                    "epoch": epoch,
                    "avg_attn": stats["avg_attn"],
                    "avg_mlp": stats["avg_mlp"],
                    "peak": stats["peak"],
                }
            )
            print(
                f"[OK] {ckpt.name} "
                f"attn={stats['avg_attn']:.4f} "
                f"mlp={stats['avg_mlp']:.4f} "
                f"peak={stats['peak']:.4f}"
            )

        write_csv(rows)

    plot()


if __name__ == "__main__":
    main()
