import sys
import subprocess
from pathlib import Path

def main():
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        print("❌ Error: NVIDIA DGX environment (CUDA) is strictly required for inference.")
        sys.exit(1)

    print("[SAGE-LAUNCHER] Nvidia CUDA Environment Detected. Launching High-Performance PyTorch Engine...")
    script = Path("5-inference/inference_engine_pt.py")

    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Engine exited: {e.returncode}")

if __name__ == "__main__":
    main()
