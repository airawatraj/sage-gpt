import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Strict Path Alignment ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    LOG_FILE = config.LOG_DIR / "evaluation" / "norm_tracking_pt.csv"
    OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "norm_history.png"
except ImportError:
    LOG_FILE = Path("6-logs/evaluation/norm_tracking_pt.csv")
    OUTPUT_PLOT = Path("6-logs/evaluation/norm_history.png")

def plot_norms():
    if not LOG_FILE.exists():
        print(f"⚠️  No data found at {LOG_FILE}. Run the inspector first.")
        return

    # 1. Load and Clean
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("⚠️  CSV is empty.")
        return

    # 2. Sort by Epoch so the lines flow correctly
    df['Epoch'] = df['Checkpoint'].str.extract(r'(\d+)').astype(float)
    df = df.sort_values(by='Epoch').reset_index(drop=True)

    # 3. Visualization Setup
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(16, 9))
    
    # Left Axis: L2 Norms
    ln1 = ax1.plot(df['Epoch'], df['Avg_Attn_Norm'], label='Avg Attention L2', 
                   color='#8FBC8F', linewidth=2.5, marker='o', markersize=6)
    ln2 = ax1.plot(df['Epoch'], df['Avg_MLP_Norm'], label='Avg MLP L2', 
                   color='#F4A460', linewidth=2.5, marker='s', markersize=6)
    
    ax1.set_xlabel('Epochs', fontsize=12, color='#AAAAAA')
    ax1.set_ylabel('Macro L2 Norm', fontsize=12, color='#AAAAAA')
    
    # Right Axis: Peak Intensity (Stability)
    ax2 = ax1.twinx()
    ln3 = ax2.plot(df['Epoch'], df['Peak_Value'], label='Global Peak Intensity', 
                   color='#87CEFA', linewidth=1.8, linestyle='--', marker='v', alpha=0.7)
    ax2.set_ylabel('Peak Weight Intensity', fontsize=12, color='#87CEFA')

    # Metadata
    plt.title(f"SAGE-GPT MECHANISTIC CONTRACTIONS\nLatest Epoch: {int(df['Epoch'].iloc[-1])}", 
              fontsize=16, pad=20, fontweight='bold')
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', framealpha=0.1)
    
    # 4. Save the actual PNG
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ [SUCCESS] PNG rendered to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_norms()