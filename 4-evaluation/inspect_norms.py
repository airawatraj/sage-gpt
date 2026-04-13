import mlx.core as mx
import sys
from pathlib import Path
import csv
from datetime import datetime

def main():
    # Define the path to the specified checkpoint
    checkpoint_path = Path("3-model/mlx/checkpoints/interrupt_save.safetensors")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    weights = mx.load(str(checkpoint_path))
    
    print("\nL2 Norms of Attention and MLP Layer Weight Matrices:")
    print("-" * 60)
    
    # Track statistics
    attn_norms = []
    mlp_norms = []
    block_0_qkv_peak_norm = None
    
    # Calculate and log L2 norm for targeted parameters
    for key, tensor in sorted(weights.items()):
        if "attn" in key or "mlp" in key:
            if "weight" in key:
                # Compute the L2 norm for the weight matrix
                l2_norm = mx.sqrt(mx.sum(mx.square(tensor))).item()
                print(f"{key:.<50} {l2_norm:.4f}")
                
                if "attn" in key:
                    attn_norms.append(l2_norm)
                elif "mlp" in key:
                    mlp_norms.append(l2_norm)
                
                if key == "blocks.0.attn.qkv.weight":
                    block_0_qkv_peak_norm = mx.max(mx.abs(tensor)).item()

    print("-" * 60)
    avg_attn_norm = sum(attn_norms) / len(attn_norms) if attn_norms else 0.0
    avg_mlp_norm = sum(mlp_norms) / len(mlp_norms) if mlp_norms else 0.0

    if attn_norms:
        print(f"Average Attention Weight L2 Norm: {avg_attn_norm:.4f}")
    if mlp_norms:
        print(f"Average MLP Weight L2 Norm:       {avg_mlp_norm:.4f}")
    if block_0_qkv_peak_norm is not None:
        print(f"Block 0 QKV Peak Norm:            {block_0_qkv_peak_norm:.4f}")

    log_dir = Path("6-logs/evaluation")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "norm_tracking.csv"
    
    file_exists = log_file.exists()
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Checkpoint Name", "Average Attention L2 Norm", "Average MLP L2 Norm", "Block 0 QKV Peak Norm"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_name = checkpoint_path.name
        
        row = [
            timestamp, 
            checkpoint_name, 
            f"{avg_attn_norm:.4f}", 
            f"{avg_mlp_norm:.4f}", 
            f"{block_0_qkv_peak_norm:.4f}" if block_0_qkv_peak_norm is not None else "N/A"
        ]
        writer.writerow(row)

if __name__ == "__main__":
    main()
