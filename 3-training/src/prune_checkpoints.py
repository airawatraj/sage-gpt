import sys
from pathlib import Path

# Setup paths to import the project's config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
except ImportError:
    print("Error: config.py not found. Please check your path structure.")
    sys.exit(1)

def prune_checkpoints(keep=5):
    if not CHECKPOINT_DIR.exists():
        print(f"Directory not found: {CHECKPOINT_DIR}")
        return

    print(f"Scanning {CHECKPOINT_DIR} for old epochs...")
    
    # 1. Target only standard epoch files 
    epoch_files = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
    
    if not epoch_files:
        print("No epoch checkpoints found to prune.")
        return

    # 2. Sort them mathematically by the epoch number in the filename
    try:
        epoch_files.sort(key=lambda p: int(p.stem.split("_")[1]))
    except ValueError as e:
        print(f"Error parsing epoch numbers: {e}")
        return

    # 3. Identify files to delete
    if len(epoch_files) <= keep:
        print(f"Found {len(epoch_files)} epochs. Nothing to delete (keeping {keep}).")
        return

    files_to_delete = epoch_files[:-keep]
    files_to_keep = epoch_files[-keep:]

    # 4. Execute deletion safely 
    for file_path in files_to_delete:
        print(f"Deleting old epoch: {file_path.name}")
        file_path.unlink()

    print(f"\nPruning complete! Retained the {len(files_to_keep)} latest epochs.")

if __name__ == "__main__":
    prune_checkpoints(keep=5)