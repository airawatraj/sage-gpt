import sys
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# --- Strict Path Alignment ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
    EXPECTED_LAYERS = config.LAYERS
    EXPECTED_DIM = config.EMBED_DIM
    EXPECTED_VOCAB = config.VOCAB_SIZE
except ImportError:
    print("❌ Critical: config.py not found.")
    sys.exit(1)

def get_target_checkpoint():
    """Prioritizes CLI args, then latest epoch, then interrupt."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    
    checkpoints = sorted(list(CHECKPOINT_DIR.glob("epoch_*.safetensors")), 
                        key=lambda p: int(p.stem.split("_")[1]))
    if checkpoints:
        return checkpoints[-1]
    
    interrupt = CHECKPOINT_DIR / "interrupt.safetensors"
    return interrupt if interrupt.exists() else None

def main():
    print(f"\n🔎 [SAGE-PROBE] Initializing Deep Checkpoint Inspection...")
    ckpt_path = get_target_checkpoint()
    
    if not ckpt_path or not ckpt_path.exists():
         print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
         return

    print(f"📂 Target: {ckpt_path.name}")
    print("-" * 60)

    try:
        weights = load_file(str(ckpt_path))
        
        # 1. Architecture Identity Check
        # SageGPT uses 'tok_emb.weight' and 'layers.X...'
        has_emb = "tok_emb.weight" in weights
        layer_keys = [k for k in weights.keys() if "layers." in k]
        
        if not layer_keys:
            print("⚠️  Warning: No 'layers' found. Checkpoint might use old 'blocks' schema.")
            layer_keys = [k for k in weights.keys() if "blocks." in k]
        
        # Parse Max Layer Index
        layer_indices = []
        for k in layer_keys:
            try:
                layer_indices.append(int(k.split(".")[1]))
            except: continue
        
        found_layers = max(layer_indices) + 1 if layer_indices else 0
        
        # 2. Dimensional Validation
        emb_shape = weights["tok_emb.weight"].shape if has_emb else (0, 0)
        
        print(f"💎 Identity: SageGPT Transformer")
        print(f"📏 Layers:   {found_layers} / {EXPECTED_LAYERS} expected")
        print(f"🌌 Vocab:    {emb_shape[0]} / {EXPECTED_VOCAB} expected")
        print(f"🧬 D_Model:  {emb_shape[1]} / {EXPECTED_DIM} expected")
        
        # 3. Mathematical Health Check (The 'No Shortcuts' Audit)
        print("-" * 60)
        print("🧠 Mechanistic Health Audit:")
        
        nan_found = False
        inf_found = False
        total_params = 0
        
        for key, tensor in weights.items():
            t_float = tensor.float()
            total_params += t_float.numel()
            if torch.isnan(t_float).any(): nan_found = True
            if torch.isinf(t_float).any(): inf_found = True
            
        health_status = "💚 HEALTHY" if not (nan_found or inf_found) else "💔 CORRUPTED"
        print(f"   Status:   {health_status}")
        print(f"   NaNs:     {'Found! ❌' if nan_found else 'None ✅'}")
        print(f"   Infs:     {'Found! ❌' if inf_found else 'None ✅'}")
        print(f"   Params:   {total_params/1e6:.2f}M")
        
        # 4. Consistency Warning
        if found_layers != EXPECTED_LAYERS or emb_shape[1] != EXPECTED_DIM:
            print("\n🚨 ARCHITECTURE MISMATCH DETECTED!")
            print(f"   Config expects {EXPECTED_LAYERS}L/{EXPECTED_DIM}D but file shows {found_layers}L/{emb_shape[1]}D.")
        else:
            print("\n✅ Checkpoint is perfectly aligned with current config.py")

    except Exception as e:
        print(f"💥 Error during probe: {e}")

if __name__ == "__main__":
    main()