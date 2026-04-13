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

LOG_FILE = config.LOG_DIR / "evaluation" / "norm_tracking.csv"
OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "norm_history.png"

def plot_norms():
    if not LOG_FILE.exists():
        print(f"[SAGE-ARCH] Error: No log found at {LOG_FILE}")
        return

    # Read data
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"[SAGE-ARCH] Error reading CSV: {e}")
        return

    if df.empty:
        print("[SAGE-ARCH] Not enough data to plot.")
        return

    # Convert specific columns to numeric, dropping NaNs or handling bad parsing
    df['Average Attention L2 Norm'] = pd.to_numeric(df['Average Attention L2 Norm'], errors='coerce')
    df['Average MLP L2 Norm'] = pd.to_numeric(df['Average MLP L2 Norm'], errors='coerce')
    df['Block 0 QKV Peak Norm'] = pd.to_numeric(df['Block 0 QKV Peak Norm'], errors='coerce')

    # Convert Timestamp to datetime for sequential plotting if steps aren't explicitly tracked
    # We can just plot against row index or time, row index (representing checkpoints) is straightforward
    df = df.reset_index()
    
    # Drop rows where all of our target columns are NaN
    df = df.dropna(subset=['Average Attention L2 Norm', 'Average MLP L2 Norm', 'Block 0 QKV Peak Norm'], how='all')

    if df.empty:
        print("[SAGE-ARCH] Not enough valid quantitative data to plot.")
        return

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot curves
    ax.plot(df.index, df['Average Attention L2 Norm'], label='Avg Attn L2 Norm', color='#8FBC8F', linewidth=2.0, alpha=0.9, marker='o')
    ax.plot(df.index, df['Average MLP L2 Norm'], label='Avg MLP L2 Norm', color='#F4A460', linewidth=2.0, alpha=0.9, marker='o')
    ax.plot(df.index, df['Block 0 QKV Peak Norm'], label='Block 0 QKV Peak Norm', color='#87CEFA', linewidth=2.0, alpha=0.9, marker='o')
    
    # Extract latest values for the title if available
    latest_attn = df['Average Attention L2 Norm'].dropna().iloc[-1] if not df['Average Attention L2 Norm'].dropna().empty else float('nan')
    latest_mlp = df['Average MLP L2 Norm'].dropna().iloc[-1] if not df['Average MLP L2 Norm'].dropna().empty else float('nan')

    ax.set_title(f"SAGE-GPT Weight Norm Micro-Contractions\nLatest Attn: {latest_attn:.4f} | MLP: {latest_mlp:.4f}", fontsize=14, pad=15)
    
    # Use checkpoint names or index for x-axis
    ax.set_xlabel('Checkpoint Tracking Index', fontsize=12)
    ax.set_ylabel('Norm Magnitude', fontsize=12)
    
    ax.grid(True, which="both", linestyle='-', alpha=0.15)
    ax.legend(loc='upper right', framealpha=0.3)
    
    # Ensure output directory exists before saving
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, facecolor=fig.get_facecolor())
    print(f"[SAGE-ARCH] Norm visualization rendered to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_norms()
