import sys
import torch
from pathlib import Path
from safetensors.torch import load_file
import csv
from datetime import datetime

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    LOG_DIR = config.LOG_DIR / "evaluation"
except ImportError:
    print("Error: config.py not found. Using relative paths.")
    CHECKPOINT_DIR = Path("3-model/pt/checkpoints")
    LOG_DIR = Path("6-logs/evaluation")

def main():
    # Allow passing a specific checkpoint path as an argument
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        # Default to the most recent healthy epoch
        ckpts = sorted(list(CHECKPOINT_DIR.glob("epoch_*.safetensors")), 
                       key=lambda x: int(x.stem.split('_')[1]))
        if not ckpts:
            print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
            sys.exit(1)
        checkpoint_path = ckpts[-1]
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"\n🔬 INSPECTING MECHANISTIC NORMS: {checkpoint_path.name}")
    print("-" * 70)
    weights = load_file(str(checkpoint_path))
    
    attn_norms = []
    mlp_norms = []
    peak_val = 0.0
    
    # SageGPT naming: layers.X.attn.wq.weight, layers.X.mlp.0.weight, etc.
    for key, tensor in sorted(weights.items()):
        if "weight" in key and ("attn" in key or "mlp" in key):
            t_float = tensor.float()
            l2_norm = torch.linalg.norm(t_float).item()
            
            # Track peak absolute value for training stability check
            current_peak = torch.max(torch.abs(t_float)).item()
            peak_val = max(peak_val, current_peak)

            print(f"{key:.<55} {l2_norm:.4f}")
            
            if "attn" in key:
                attn_norms.append(l2_norm)
            elif "mlp" in key:
                mlp_norms.append(l2_norm)

    avg_attn = sum(attn_norms) / len(attn_norms) if attn_norms else 0.0
    avg_mlp = sum(mlp_norms) / len(mlp_norms) if mlp_norms else 0.0

    print("-" * 70)
    print(f"📊 SUMMARY STATS")
    print(f"Avg Attention L2: {avg_attn:.4f}")
    print(f"Avg MLP L2:       {avg_mlp:.4f}")
    print(f"Global Peak Val:  {peak_val:.4f}")
    print("-" * 70)

    # --- CSV Logging ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "norm_tracking_pt.csv"
    
    header = ["Timestamp", "Checkpoint", "Avg_Attn_Norm", "Avg_MLP_Norm", "Peak_Value"]
    file_exists = log_file.exists()

    # Deduplication guard: skip if this checkpoint was already logged.
    if file_exists:
        with open(log_file, mode='r', newline='') as f:
            logged = [row[1] for row in csv.reader(f) if row]
        if checkpoint_path.name in logged:
            print(f"ℹ️  Checkpoint '{checkpoint_path.name}' already logged — skipping duplicate row.")
            return

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            checkpoint_path.name, 
            f"{avg_attn:.4f}", 
            f"{avg_mlp:.4f}", 
            f"{peak_val:.4f}"
        ])
    print(f"✅ Metrics appended to {log_file}")

if __name__ == "__main__":
    main()