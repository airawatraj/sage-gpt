import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup strict paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

LOG_FILE = config.LOG_DIR / "training" / "training_history.csv"
OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "generalisation_gap.png"

# Hyperparameters
WARMUP_STEPS = 2000

def plot_curves():
    if not LOG_FILE.exists():
        print(f"[SAGE-ARCH] Error: No log found at {LOG_FILE}")
        return

    # Read and clean data
    df = pd.read_csv(LOG_FILE)
    df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
    df['Train_Loss'] = pd.to_numeric(df['Train_Loss'], errors='coerce')
    df['Val_Loss'] = pd.to_numeric(df['Val_Loss'], errors='coerce')
    
    # Sort by step to align appended data
    df = df.sort_values(by='Step')

    train_df = df.dropna(subset=['Train_Loss']).copy()
    val_df = df.dropna(subset=['Val_Loss'])

    if train_df.empty or val_df.empty:
        print("[SAGE-ARCH] Not enough data to plot.")
        return

    train_df['Train_Loss_Var'] = train_df['Train_Loss'].rolling(window=50, min_periods=1).var()

    latest_tr = train_df['Train_Loss'].iloc[-1]
    latest_val = val_df['Val_Loss'].iloc[-1]
    
    # Calculate grokking detection trigger
    # Only if we have at least 6 validation points
    if len(val_df) >= 6:
        recent_val_avg = val_df['Val_Loss'].iloc[-6:-1].mean()
        drop_pct = (recent_val_avg - latest_val) / recent_val_avg
        
        if drop_pct > 0.05:
            # \a chimes the terminal bell, \033[1;31m sets bold red text
            print(f"\a\033[1;31m[SAGE-GROK-DETECTED] ALARM: Validation loss dropped by {drop_pct*100:.2f}%! Phase transition initiated.\033[0m")
            print(f"\033[1;31mRecent Avg: {recent_val_avg:.4f} -> Latest Val: {latest_val:.4f}\033[0m")

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot curves
    ax1.plot(train_df['Step'], train_df['Train_Loss'], label='Train Loss', color='#8FBC8F', linewidth=1.5, alpha=0.9)
    ax1.plot(val_df['Step'], val_df['Val_Loss'], label='Val Loss', color='#F4A460', linewidth=2.0, alpha=0.9)
    
    # Warmup Marker
    ax1.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', label='Warmup End', alpha=0.6)

    # Log Scale
    ax1.set_yscale('log')
    
    ax1.set_title(f"SAGE-GPT Learning Curves (Grokking Regime)\nLatest Tr: {latest_tr:.4f} | Val: {latest_val:.4f}", fontsize=14, pad=15)
    # Move x-axis label to the bottom subplot
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12)
    # Remove x-axis tick labels for top subplot to keep it clean (optional, but requested layout is minimal)
    # Actually, keep it simple. Let's just set the grid and legend.
    ax1.grid(True, which="both", linestyle='-', alpha=0.15)
    ax1.legend(loc='upper right', framealpha=0.3)
    
    # Variance plot
    ax2.plot(train_df['Step'], train_df['Train_Loss_Var'], label='Training Loss Turbulence (50-Step Rolling Variance)', color='#FF6347', linewidth=1.5, alpha=0.9)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle='-', alpha=0.15)
    ax2.legend(loc='upper right', framealpha=0.3)
    
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, facecolor=fig.get_facecolor())
    print(f"[SAGE-ARCH] Gap monitor rendered to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_curves()
