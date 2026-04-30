"""
generalisation_gap_monitor.py — Plots train/val loss curves and generalisation gap.
Rewritten without pandas; uses stdlib csv + matplotlib only.
"""

import sys
import csv
import math
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    LOG_FILE    = config.LOG_DIR / "training" / "training_history_dgx.csv"
    OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "generalisation_gap.png"
    WARMUP_STEPS = 150  # Matches train_engine_dgx.py
except ImportError:
    print("❌ Critical: config.py not found.")
    sys.exit(1)


# ── Rolling variance (ddof=1, min_periods=1) — replaces pandas rolling().var() ──
def rolling_var(data: list[float], window: int = 10) -> list[float]:
    result = []
    for i in range(len(data)):
        chunk = data[max(0, i - window + 1): i + 1]
        n = len(chunk)
        if n < 2:
            result.append(0.0)
        else:
            mean = sum(chunk) / n
            result.append(sum((v - mean) ** 2 for v in chunk) / (n - 1))
    return result


def plot_curves():
    if not LOG_FILE.exists():
        print(f"⚠️  [SAGE-GAP] No log found at {LOG_FILE}. Waiting for next engine flush...")
        return

    # 1. Load CSV with stdlib
    rows = []
    try:
        with open(LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    step       = float(row["Step"])
                    train_loss = float(row["Train_Loss"])
                    val_loss   = float(row["Val_Loss"])
                    if math.isnan(train_loss) or math.isnan(val_loss):
                        continue
                    rows.append({"step": step, "train": train_loss, "val": val_loss})
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Sort by step (handles out-of-order appends)
    rows.sort(key=lambda r: r["step"])

    if len(rows) < 2:
        print("⚠️  [SAGE-GAP] Not enough data points to plot.")
        return

    steps       = [r["step"]  for r in rows]
    train_loss  = [r["train"] for r in rows]
    val_loss    = [r["val"]   for r in rows]
    gap         = [v - t for v, t in zip(val_loss, train_loss)]
    train_var   = rolling_var(train_loss, window=10)

    latest_tr   = train_loss[-1]
    latest_val  = val_loss[-1]
    latest_gap  = gap[-1]

    # 2. Grokking detection
    if len(rows) >= 10:
        recent_val_avg = sum(val_loss[-10:-1]) / 9
        if recent_val_avg > 0:
            drop_pct = (recent_val_avg - latest_val) / recent_val_avg
            if drop_pct > 0.05:
                print(f"\n\a\033[1;31m🔥 [SAGE-GROK-DETECTED] PHASE TRANSITION ALERT!")
                print(f"Validation loss dropped {drop_pct*100:.2f}% vs recent average.")
                print(f"Generalization Gap narrowing: {latest_gap:.4f}\033[0m\n")

    # 3. Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: Loss landscape
    ax1.plot(steps, train_loss, label="Train Loss (Memorization)",
             color="#8FBC8F", linewidth=1.5, alpha=0.8)
    ax1.plot(steps, val_loss,   label="Val Loss (Generalization)",
             color="#F4A460", linewidth=2.5, alpha=0.9)
    ax1.axvline(x=WARMUP_STEPS, color="#555555", linestyle="--",
                label="Warmup End", alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12, labelpad=10)
    ax1.set_title(
        f"SAGE-GPT Generalization Monitor\n"
        f"Gap: {latest_gap:.4f} | Steps: {int(steps[-1])}",
        fontsize=16, pad=20, fontweight="bold",
    )
    ax1.legend(loc="upper right", framealpha=0.1)
    ax1.grid(True, which="both", linestyle="-", alpha=0.05)

    # Bottom: Gap + turbulence
    ax2.fill_between(steps, gap, color="#FF6347", alpha=0.15, label="Generalization Gap")
    ax2.plot(steps, gap, color="#FF6347", linewidth=1.2, alpha=0.7)

    ax3 = ax2.twinx()
    ax3.plot(steps, train_var, color="#87CEFA", label="Loss Turbulence",
             linewidth=1, linestyle=":", alpha=0.6)
    ax3.set_yscale("log")
    ax3.set_ylabel("Turbulence", color="#87CEFA", fontsize=10)

    ax2.set_xlabel("Global Training Steps", fontsize=12, labelpad=10)
    ax2.set_ylabel("The Gap", fontsize=12, labelpad=10)
    ax2.legend(loc="upper left", framealpha=0.1)

    # 4. Save
    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150, facecolor="#0B0B0B")
    print(f"📊 [SUCCESS] Gap monitor rendered → {OUTPUT_PLOT}")


if __name__ == "__main__":
    plot_curves()