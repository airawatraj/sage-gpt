"""
generalisation_gap_monitor.py — Enhanced plotter with better grokking detection.
"""
import sys
import csv
import math
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
    OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "generalisation_gap.png"
    WARMUP_STEPS = 150
except ImportError:
    print("❌ Critical: config.py not found.")
    sys.exit(1)

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
        print(f"⚠️ No log found at {LOG_FILE}.")
        return

    rows = []
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = float(row["Step"])
                train_loss = float(row["Train_Loss"])
                val_loss = float(row["Val_Loss"])
                if math.isnan(train_loss) or math.isnan(val_loss):
                    continue
                rows.append({"step": step, "train": train_loss, "val": val_loss})
            except:
                continue

    rows.sort(key=lambda r: r["step"])
    if len(rows) < 10:
        print("⚠️ Not enough data points.")
        return

    steps = [r["step"] for r in rows]
    train_loss = [r["train"] for r in rows]
    val_loss = [r["val"] for r in rows]
    gap = [v - t for v, t in zip(val_loss, train_loss)]
    train_var = rolling_var(train_loss, window=10)

    latest_gap = gap[-1]
    latest_val = val_loss[-1]

    # ── Enhanced Grokking Detection ─────────────────────────────────────
    if len(rows) >= 30:
        recent_val_avg = sum(val_loss[-20:]) / 20
        drop_pct = (recent_val_avg - latest_val) / recent_val_avg if recent_val_avg > 0 else 0

        if drop_pct > 0.05 or (latest_gap > -0.20 and abs(latest_gap) < 0.25):
            print(f"\n\033[1;32m🔥 PHASE SHIFT DETECTED! Gap narrowing + Val drop\033[0m")

        if latest_val < 2.15 and abs(latest_gap) < 0.18:
            print(f"\n\033[1;33m🎉 GROKKING ACHIEVED! Strong generalization detected.\033[0m")

    # ── Plot ────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(steps, train_loss, label="Train Loss (Memorization)", color="#8FBC8F", linewidth=1.5)
    ax1.plot(steps, val_loss, label="Val Loss (Generalization)", color="#F4A460", linewidth=2.5)
    ax1.axvline(x=WARMUP_STEPS, color="#555555", linestyle="--", label="Warmup End")
    ax1.set_yscale("log")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title(f"SAGE-GPT Generalization Monitor\nGap: {latest_gap:.4f} | Steps: {int(steps[-1])}",
                  fontsize=16, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.1)

    # Bottom: Gap
    ax2.plot(steps, gap, color="#FF6347", linewidth=1.5, label="Generalization Gap")
    ax2.axhline(y=0, color="#666666", linestyle="--", alpha=0.6)
    ax2.fill_between(steps, gap, color="#FF6347", alpha=0.15)

    ax3 = ax2.twinx()
    ax3.plot(steps, train_var, color="#87CEFA", linestyle=":", linewidth=1, label="Turbulence")
    ax3.set_yscale("log")

    ax2.set_xlabel("Global Training Steps")
    ax2.set_ylabel("The Gap")
    ax3.set_ylabel("Turbulence", color="#87CEFA")

    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=180, facecolor="#0B0B0B")
    print(f"📊 [SUCCESS] Gap monitor rendered → {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_curves()