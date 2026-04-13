import sys
import subprocess
from pathlib import Path

def main():
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        print("[SAGE-LAUNCHER] Nvidia CUDA Environment Detected. Launching High-Performance PyTorch Engine...")
        script = Path("3-training/src/train_engine_pt.py")
    else:
        print("[SAGE-LAUNCHER] No CUDA Detected. Defaulting to Apple Silicon MLX Engine...")
        script = Path("3-training/src/train_engine_mlx.py")

    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Engine exited: {e.returncode}")

if __name__ == "__main__":
    main()
