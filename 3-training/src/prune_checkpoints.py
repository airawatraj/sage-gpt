"""
prune_checkpoints.py - checkpoint pruning for Sage-GPT.

Keeps recent epoch checkpoints and recent step checkpoints, while preserving
special recovery/artifact files such as best_grok_model and interrupt saves.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config

    CHECKPOINT_DIR = config.PT_CHECKPOINT_DIR
except ImportError:
    CHECKPOINT_DIR = None

PROTECTED_STEMS = {
    "best_grok_model",
    "interrupt",
    "latest",
}


def _numeric_suffix(path: Path, prefix: str) -> int:
    marker = f"{prefix}_"
    if not path.stem.startswith(marker):
        return -1
    suffix = path.stem[len(marker):]
    return int(suffix) if suffix.isdigit() else -1


def _delete_checkpoint_pair(model_path: Path, verbose: bool = False) -> None:
    if model_path.stem in PROTECTED_STEMS:
        return

    if model_path.exists():
        if verbose:
            print(f"Pruning checkpoint: {model_path.name}")
        model_path.unlink()

    state_path = model_path.with_name(f"{model_path.stem}_state.pt")
    if state_path.exists():
        if verbose:
            print(f"Pruning checkpoint state: {state_path.name}")
        state_path.unlink()


def _prune_group(prefix: str, keep: int, verbose: bool = False) -> None:
    if keep < 0:
        raise ValueError("keep must be >= 0")
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    files = [
        path
        for path in CHECKPOINT_DIR.glob(f"{prefix}_*.safetensors")
        if _numeric_suffix(path, prefix) >= 0 and path.stem not in PROTECTED_STEMS
    ]
    files.sort(key=lambda p: _numeric_suffix(p, prefix))

    if len(files) <= keep:
        return

    for model_path in files[: len(files) - keep]:
        _delete_checkpoint_pair(model_path, verbose=verbose)


def prune_orphan_states(verbose: bool = False) -> None:
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    for state_path in CHECKPOINT_DIR.glob("*_state.pt"):
        stem = state_path.stem[:-6] if state_path.stem.endswith("_state") else state_path.stem
        if stem in {"interrupt", "latest", "best_grok_model"}:
            continue
        model_path = CHECKPOINT_DIR / f"{stem}.safetensors"
        if not model_path.exists():
            if verbose:
                print(f"Pruning orphan state: {state_path.name}")
            state_path.unlink()


def prune_checkpoints(keep: int = 10, keep_steps: int = 4, verbose: bool = False) -> None:
    """
    Prune checkpoint files.

    Args:
        keep: Number of latest epoch_*.safetensors checkpoints to keep.
        keep_steps: Number of latest step_*.safetensors checkpoints to keep.
        verbose: Print each deleted file.
    """
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    _prune_group("epoch", keep=keep, verbose=verbose)
    _prune_group("step", keep=keep_steps, verbose=verbose)
    prune_orphan_states(verbose=verbose)


if __name__ == "__main__":
    prune_checkpoints(keep=10, keep_steps=4, verbose=True)
