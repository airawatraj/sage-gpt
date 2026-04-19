import sys
import subprocess
from pathlib import Path

def main():
    print("[SAGE-LAUNCHER] Launching Highly Optimized DGX Training Engine...")
    script = Path("3-training/src/train_engine_dgx.py")

    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Engine exited: {e.returncode}")

if __name__ == "__main__":
    main()
