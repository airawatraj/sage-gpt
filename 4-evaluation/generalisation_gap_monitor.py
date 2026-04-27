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
    # Synchronized with train_engine_dgx.py output
    LOG_FILE = config.LOG_DIR / "training" / "training_history_dgx.csv"
    OUTPUT_PLOT = config.LOG_DIR / "evaluation" / "generalisation_gap.png"
    WARMUP_STEPS = 150 # Matches train_engine_dgx.py
except ImportError:
    print("❌ Critical: config.py not found. Engine cannot resolve paths.")
    sys.exit(1)

def plot_curves():
    if not LOG_FILE.exists():
        print(f"⚠️  [SAGE-GAP] No log found at {LOG_FILE}. Waiting for next engine flush...")
        return

    # 1. Load and Clean (The "No Shortcuts" Data Hygiene)
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Ensure numeric types for all critical columns
    cols_to_fix = ['Step', 'Train_Loss', 'Val_Loss']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by step to ensure a continuous line if data was appended out of order
    df = df.sort_values(by='Step').dropna(subset=['Train_Loss', 'Val_Loss'])

    if len(df) < 2:
        print("⚠️  [SAGE-GAP] Not enough data points to calculate variance or gap.")
        return

    # 2. Calculate Mechanistic Metrics
    # Turbulence: Rolling variance identifies 'instability' before a grokking shift
    df['Train_Var'] = df['Train_Loss'].rolling(window=10, min_periods=1).var()
    # The Gap: Quantifying the distance between memorization and generalization
    df['Gap'] = df['Val_Loss'] - df['Train_Loss']

    latest_tr = df['Train_Loss'].iloc[-1]
    latest_val = df['Val_Loss'].iloc[-1]
    latest_gap = df['Gap'].iloc[-1]
    
    # 3. Grokking Detection (The Phase Transition Alarm)
    if len(df) >= 10:
        recent_val_avg = df['Val_Loss'].iloc[-10:-1].mean()
        drop_pct = (recent_val_avg - latest_val) / recent_val_avg
        
        if drop_pct > 0.05:
            print(f"\n\a\033[1;31m🔥 [SAGE-GROK-DETECTED] PHASE TRANSITION ALERT!")
            print(f"Validation loss dropped by {drop_pct*100:.2f}% relative to recent average.")
            print(f"Generalization Gap is narrowing: {latest_gap:.4f}\033[0m\n")

    # 4. Professional Visualization
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Top Plot: The Loss Landscape ---
    ax1.plot(df['Step'], df['Train_Loss'], label='Train Loss (Memorization)', 
             color='#8FBC8F', linewidth=1.5, alpha=0.8)
    ax1.plot(df['Step'], df['Val_Loss'], label='Val Loss (Generalization)', 
             color='#F4A460', linewidth=2.5, alpha=0.9)
    
    # Mark the end of the warmup phase
    ax1.axvline(x=WARMUP_STEPS, color='#555555', linestyle='--', label='Warmup End', alpha=0.5)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12, labelpad=10)
    ax1.set_title(f"SAGE-GPT Generalization Monitor\nGap: {latest_gap:.4f} | Steps: {int(df['Step'].iloc[-1])}", 
                  fontsize=16, pad=20, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.1)
    ax1.grid(True, which="both", linestyle='-', alpha=0.05)

    # --- Bottom Plot: The Gap & Turbulence ---
    # Shaded region for the Generalization Gap
    ax2.fill_between(df['Step'], df['Gap'], color='#FF6347', alpha=0.15, label='Generalization Gap')
    ax2.plot(df['Step'], df['Gap'], color='#FF6347', linewidth=1.2, alpha=0.7)
    
    # Secondary axis for Loss Turbulence (Variance)
    ax3 = ax2.twinx()
    ax3.plot(df['Step'], df['Train_Var'], color='#87CEFA', label='Loss Turbulence', 
             linewidth=1, linestyle=':', alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_ylabel('Turbulence', color='#87CEFA', fontsize=10)
    
    ax2.set_xlabel('Global Training Steps', fontsize=12, labelpad=10)
    ax2.set_ylabel('The Gap', fontsize=12, labelpad=10)
    ax2.legend(loc='upper left', framealpha=0.1)
    
    # 5. Final Rendering
    plt.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, facecolor='#0B0B0B')
    print(f"📊 [SUCCESS] Production gap monitor rendered to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_curves()