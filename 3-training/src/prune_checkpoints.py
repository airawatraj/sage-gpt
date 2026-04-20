import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
except ImportError:
    CHECKPOINT_DIR = None

def prune_checkpoints(keep=10, verbose=False):
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    epoch_files = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
    if not epoch_files or len(epoch_files) <= keep:
        return

    # Sort mathematically by epoch number
    epoch_files.sort(key=lambda p: int(p.stem.split("_")[1]))

    files_to_delete = epoch_files[:-keep]

    for file_path in files_to_delete:
        if verbose: print(f"🗑️ Pruning old epoch: {file_path.name}")
        file_path.unlink()
        
        # Prune associated optimizer state file
        state_file = CHECKPOINT_DIR / f"{file_path.stem}_state.pt"
        if state_file.exists():
            state_file.unlink()

if __name__ == "__main__":
    # Manual run remains verbose
    prune_checkpoints(keep=5, verbose=True)