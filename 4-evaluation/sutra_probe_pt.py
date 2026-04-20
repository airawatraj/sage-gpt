import sys
import torch
from pathlib import Path
from safetensors.torch import load_file

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
except ImportError:
    CHECKPOINT_DIR = project_root / "3-model/pt/checkpoints"

def get_latest_checkpoint():
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt.safetensors"
    if interrupt_ckpt.exists():
        return interrupt_ckpt
    checkpoints = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
    if not checkpoints:
        return None
    try:
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        return latest
    except ValueError:
        return None

def main():
    print(f"🔎 Scanning for PyTorch checkpoints in {CHECKPOINT_DIR}")
    checkpoint_path = get_latest_checkpoint()
    
    if not checkpoint_path:
         print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
         return

    try:
        weights = load_file(str(checkpoint_path))
        print(f"✅ Loaded weights from {checkpoint_path}")
        print(f"Total keys: {len(weights)}")
        
        if "embedding.weight" in weights:
            print(f"embedding.weight shape: {weights['embedding.weight'].shape}")
        
        max_layer = -1
        for key in weights.keys():
            if "blocks." in key:
                try:
                    layer_num = int(key.split(".")[1])
                    max_layer = max(max_layer, layer_num)
                except:
                    pass
        print(f"Max Block Index found: {max_layer} (Implies {max_layer+1} layers)")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()
