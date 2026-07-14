"""
generalisation_gap_monitor.py - plot train/validation evaluation loss.

Compatible with both the legacy Sage-GPT CSV and the patched CSV. With the
patched trainer, Train_Loss and Val_Loss are eval-mode raw cross-entropy, so the
gap is apples-to-apples.
"""

import csv
import math
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config

    LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
    OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "generalisation_gap.png"
    WARMUP_STEPS = 150
except ImportError:
    print("Critical: config.py not found.")
    sys.exit(1)


def _to_float(value, default=None):
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def rolling_var(data: list[float], window: int = 10) -> list[float]:
    result = []
    for i in range(len(data)):
        chunk = data[max(0, i - window + 1): i + 1]
        if len(chunk) < 2:
            result.append(0.0)
            continue
        mean = sum(chunk) / len(chunk)
        result.append(sum((v - mean) ** 2 for v in chunk) / (len(chunk) - 1))
    return result


def load_rows():
    if not LOG_FILE.exists():
        print(f"No log found at {LOG_FILE}.")
        return []

    rows = []
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _to_float(row.get("Step"))
            train_loss = _to_float(row.get("Train_Loss"))
            val_loss = _to_float(row.get("Val_Loss"))
            live_train_loss = _to_float(row.get("Live_Train_Loss"))
            gap = _to_float(row.get("Gap"))

            if step is None or train_loss is None or val_loss is None:
                continue
            if gap is None:
                gap = val_loss - train_loss

            rows.append(
                {
                    "step": step,
                    "train": train_loss,
                    "val": val_loss,
                    "gap": gap,
                    "live_train": live_train_loss,
                }
            )

    rows.sort(key=lambda r: r["step"])
    return rows


def print_status(rows):
    latest = rows[-1]
    latest_ppl = math.exp(latest["val"]) if latest["val"] < 20 else float("inf")
    print(
        f"Latest: step={int(latest['step'])} | "
        f"train_ce={latest['train']:.4f} | val_ce={latest['val']:.4f} | "
        f"gap={latest['gap']:.4f} | val_ppl={latest_ppl:.2f}"
    )

    if len(rows) >= 20:
        recent = rows[-20:]
        recent_best = min(r["val"] for r in recent)
        recent_avg = sum(r["val"] for r in recent) / len(recent)
        print(f"Recent 20 evals: best_val_ce={recent_best:.4f} | avg_val_ce={recent_avg:.4f}")

    if len(rows) >= 30:
        prev = rows[-30:-10]
        recent = rows[-10:]
        prev_avg = sum(r["val"] for r in prev) / len(prev)
        recent_avg = sum(r["val"] for r in recent) / len(recent)
        delta_pct = (prev_avg - recent_avg) / prev_avg if prev_avg > 0 else 0.0
        if delta_pct > 0.02:
            print(f"Validation is improving: recent avg down {delta_pct * 100:.2f}%.")
        elif abs(delta_pct) <= 0.005:
            print("Validation appears plateaued over the latest window.")
        else:
            print(f"Validation has worsened: recent avg up {-delta_pct * 100:.2f}%.")


def plot_curves():
    rows = load_rows()
    if len(rows) < 3:
        print("Not enough data points.")
        return

    steps = [r["step"] for r in rows]
    train_loss = [r["train"] for r in rows]
    val_loss = [r["val"] for r in rows]
    gap = [r["gap"] for r in rows]
    live_train = [r["live_train"] for r in rows]
    train_var = rolling_var(train_loss, window=10)

    latest_gap = gap[-1]
    latest_val = val_loss[-1]
    latest_ppl = math.exp(latest_val) if latest_val < 20 else float("inf")

    print_status(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(steps, train_loss, label="Train eval CE", color="#8FBC8F", linewidth=1.5)
    ax1.plot(steps, val_loss, label="Val eval CE", color="#F4A460", linewidth=2.2)

    if any(v is not None for v in live_train):
        live_steps = [s for s, v in zip(steps, live_train) if v is not None]
        live_values = [v for v in live_train if v is not None]
        if live_steps:
            ax1.plot(live_steps, live_values, label="Live train smoothed CE", color="#AAAAAA", alpha=0.35, linewidth=1.0)

    ax1.axvline(x=WARMUP_STEPS, color="#555555", linestyle="--", label="Warmup End")
    ax1.set_yscale("log")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title(
        "SAGE-GPT Generalization Monitor\n"
        f"Gap: {latest_gap:.4f} | Val PPL: {latest_ppl:.2f} | Steps: {int(steps[-1])}",
        fontsize=16,
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.1)

    ax2.plot(steps, gap, color="#FF6347", linewidth=1.5, label="Val CE - train CE")
    ax2.axhline(y=0, color="#666666", linestyle="--", alpha=0.6)
    ax2.fill_between(steps, gap, color="#FF6347", alpha=0.15)

    ax3 = ax2.twinx()
    safe_var_steps = []
    safe_var = []
    for s, v in zip(steps, train_var):
        if v > 0:
            safe_var_steps.append(s)
            safe_var.append(v)
    if safe_var:
        ax3.plot(safe_var_steps, safe_var, color="#87CEFA", linestyle=":", linewidth=1, label="Train CE turbulence")
        ax3.set_yscale("log")
    ax2.set_xlabel("Global training steps")
    ax2.set_ylabel("Generalization gap")
    ax3.set_ylabel("Turbulence", color="#87CEFA")

    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=180, facecolor="#0B0B0B")
    print(f"Gap monitor rendered: {OUTPUT_PLOT}")


if __name__ == "__main__":
    plot_curves()
