import mlx.core as mx
import sys
from pathlib import Path

# Setup paths to ensure we can properly import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
except ImportError:
    CHECKPOINT_DIR = Path("../3-model/mlx/checkpoints")

def get_latest_checkpoint():
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt_save.safetensors"
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
    print(f"🔎 Scanning for checkpoints in {CHECKPOINT_DIR}")
    checkpoint_path = get_latest_checkpoint()
    
    if not checkpoint_path:
         print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
         return

    try:
        weights = mx.load(str(checkpoint_path))
        print(f"✅ Loaded weights from {checkpoint_path}")
        print(f"Total keys: {len(weights)}")
        
        # 1. Check Embedding Weight Shape
        if "embedding.weight" in weights:
            print(f"embedding.weight shape: {weights['embedding.weight'].shape}")
        elif "token_embedding.weight" in weights:
            print(f"token_embedding.weight shape: {weights['token_embedding.weight'].shape}")
        else:
            print("No embedding weight found!")

        # 2. Check Layer Count
        max_layer = -1
        for key in weights.keys():
            if "blocks." in key:
                try:
                    layer_num = int(key.split(".")[1])
                    max_layer = max(max_layer, layer_num)
                except:
                    pass
        print(f"Max Block Index found: {max_layer} (Implies {max_layer+1} layers)")

        # 3. Check for Position Embeddings
        if "position_embedding.weight" in weights:
            print("Found position_embedding.weight (Learned Pos Emb)")
        else:
            print("No position_embedding.weight found (Likely RoPE)")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()
