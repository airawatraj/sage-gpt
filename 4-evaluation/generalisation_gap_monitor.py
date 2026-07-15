"""generalisation_gap_monitor.py - README-style aligned CE plus raw evidence plot."""

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
    print("config.py not found.")
    sys.exit(1)


def to_float(value, default=None):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return default


def finite(values):
    return [v for v in values if v is not None and math.isfinite(v)]


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        chunk = finite(values[max(0, i - window + 1) : i + 1])
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def tight_ylim(ax, values, min_pad=0.002, pad_ratio=0.14):
    vals = finite(values)
    if not vals:
        return

    lo = min(vals)
    hi = max(vals)

    if hi <= lo:
        hi = lo + min_pad

    pad = max((hi - lo) * pad_ratio, min_pad)
    ax.set_ylim(lo - pad, hi + pad)


def read_history():
    if not LOG_FILE.exists():
        print(f"No log found at {LOG_FILE}")
        return []

    rows = []
    with LOG_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = to_float(row.get("Step"))
            train = to_float(row.get("Train_Loss"))
            val = to_float(row.get("Val_Loss"))
            gap = to_float(row.get("Gap"))
            best_full = to_float(row.get("Best_Val_Raw_Loss"))

            if step is None or train is None or val is None:
                continue

            if gap is None:
                gap = val - train

            rows.append(
                {
                    "step": step,
                    "train": train,
                    "val": val,
                    "gap": gap,
                    "best_full": best_full,
                }
            )

    rows.sort(key=lambda r: r["step"])
    return rows


def shifted_to_match_start(source, target):
    source_vals = finite(source)
    target_vals = finite(target)

    if not source_vals or not target_vals:
        return [None for _ in source]

    shift = target_vals[0] - source_vals[0]

    out = []
    for value in source:
        if value is None or not math.isfinite(value):
            out.append(None)
        else:
            out.append(value + shift)

    return out, shift


def plot_curves():
    rows = read_history()
    if len(rows) < 3:
        print("Not enough data points.")
        return

    steps = [r["step"] for r in rows]
    train = [r["train"] for r in rows]
    val = [r["val"] for r in rows]
    gap = [r["gap"] for r in rows]

    window = min(25, max(5, len(rows) // 40))

    train_roll = rolling_mean(train, window)
    val_roll = rolling_mean(val, window)
    gap_roll = rolling_mean(gap, window)

    val_aligned, val_shift = shifted_to_match_start(val_roll, train_roll)

    latest = rows[-1]
    best_sample = min(rows, key=lambda r: r["val"])

    best_full_values = finite([r["best_full"] for r in rows])
    best_full = min(best_full_values) if best_full_values else None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    fig, (ax_aligned, ax_raw, ax_gap) = plt.subplots(
        nrows=3,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.45, 1.15]},
    )

    fig.patch.set_facecolor("#0B0B0B")

    for ax in (ax_aligned, ax_raw, ax_gap):
        ax.set_facecolor("#0B0B0B")
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.tick_params(colors="#D8D8D8")
        for spine in ax.spines.values():
            spine.set_color("#A0A0A0")

    # Panel 1: README-style aligned comparison.
    ax_aligned.plot(
        steps,
        train_roll,
        color="#8FBC8F",
        linewidth=2.2,
        label="Train eval CE",
    )
    ax_aligned.plot(
        steps,
        val_aligned,
        color="#F4A460",
        linewidth=2.2,
        label=f"Val eval CE, shifted +{val_shift:.4f}",
    )
    ax_aligned.axvline(WARMUP_STEPS, color="#777777", linestyle="--", linewidth=1.0, alpha=0.45)
    ax_aligned.set_ylabel("Aligned CE")
    ax_aligned.legend(loc="upper right", framealpha=0.25)
    tight_ylim(ax_aligned, train_roll + val_aligned, min_pad=0.004)

    # Panel 2: true raw CE values.
    ax_raw.plot(
        steps,
        train_roll,
        color="#8FBC8F",
        linewidth=1.9,
        alpha=0.95,
        label="Train raw eval CE",
    )
    ax_raw.plot(
        steps,
        val_roll,
        color="#F4A460",
        linewidth=1.9,
        alpha=0.95,
        label="Val raw eval CE",
    )
    ax_raw.scatter(
        [best_sample["step"]],
        [best_sample["val"]],
        color="#FFFFFF",
        s=34,
        zorder=5,
        label=f"Best sampled {best_sample['val']:.6f}",
    )

    if best_full is not None:
        ax_raw.axhline(
            best_full,
            color="#FFD166",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label=f"Best full CE {best_full:.6f}",
        )

    ax_raw.axvline(WARMUP_STEPS, color="#777777", linestyle="--", linewidth=1.0, alpha=0.45)
    ax_raw.set_ylabel("Raw eval CE")
    ax_raw.legend(loc="upper right", framealpha=0.25)
    tight_ylim(ax_raw, train_roll + val_roll + ([best_full] if best_full is not None else []), min_pad=0.01)

    # Panel 3: true raw generalization gap.
    ax_gap.plot(
        steps,
        gap,
        color="#87CEFA",
        linewidth=0.9,
        alpha=0.45,
        linestyle=":",
        label="Gap raw",
    )
    ax_gap.plot(
        steps,
        gap_roll,
        color="#FF6347",
        linewidth=1.8,
        label="Gap rolling",
    )
    ax_gap.axhline(0.0, color="#CCCCCC", linestyle="--", linewidth=1.0, alpha=0.65)
    ax_gap.axvline(WARMUP_STEPS, color="#777777", linestyle="--", linewidth=1.0, alpha=0.45)
    ax_gap.set_ylabel("Val minus train")
    ax_gap.set_xlabel("Global training steps")
    ax_gap.legend(loc="upper right", framealpha=0.25)
    tight_ylim(ax_gap, gap, min_pad=0.005)

    best_full_text = f"{best_full:.6f}" if best_full is not None else "n/a"

    fig.suptitle(
        "SAGE-GPT Generalization Monitor\n"
        f"Gap: {latest['gap']:.4f} | Val PPL: {math.exp(latest['val']):.2f} | "
        f"Steps: {int(latest['step'])} | Best full CE: {best_full_text}",
        fontsize=17,
        fontweight="bold",
        color="#F2F2F2",
    )

    fig.text(
        0.01,
        0.01,
        "Top panel aligns validation CE to the train baseline for visual comparison. "
        "Middle and bottom panels show true raw CE and true raw gap.",
        color="#A8A8A8",
        fontsize=9,
    )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.025, 1, 0.93])
    fig.savefig(OUTPUT_PLOT, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Generalization monitor rendered: {OUTPUT_PLOT}")


if __name__ == "__main__":
    plot_curves()
