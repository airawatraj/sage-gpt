"""
prune_checkpoints.py - safe checkpoint pruning for Sage-GPT.

Prunes by file modification time, not numeric epoch/step suffix.

Why:
A fresh resume-from-best polish run may restart counters at epoch_1 / step_0
while old long-run checkpoints are epoch_480 / step_126500. Numeric pruning
would incorrectly delete the fresh polish checkpoints. Modification-time pruning
keeps the most recent files regardless of counter reset.

Protected recovery files are never pruned.
"""

from __future__ import annotations

import argparse
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

PROTECTED_PREFIXES = (
    "best_grok_model",
    "interrupt",
    "latest",
)


def _is_protected(path: Path) -> bool:
    """Return True when a checkpoint or state file must never be pruned."""
    stem = path.stem

    if stem.endswith("_state"):
        stem = stem[: -len("_state")]

    if stem in PROTECTED_STEMS:
        return True

    return any(stem.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _numeric_suffix(path: Path, prefix: str) -> int:
    """Return numeric suffix for matching checkpoint names, or -1."""
    marker = f"{prefix}_"
    if not path.stem.startswith(marker):
        return -1
    suffix = path.stem[len(marker):]
    return int(suffix) if suffix.isdigit() else -1


def _mtime_ns(path: Path) -> int:
    """Return file modification time in nanoseconds."""
    return path.stat().st_mtime_ns


def _delete_file(path: Path, *, dry_run: bool, verbose: bool) -> None:
    if not path.exists():
        return

    action = "Would prune" if dry_run else "Pruning"
    if verbose or dry_run:
        print(f"{action}: {path.name}")

    if not dry_run:
        path.unlink()


def _delete_checkpoint_pair(model_path: Path, *, dry_run: bool = False, verbose: bool = False) -> None:
    if _is_protected(model_path):
        return

    _delete_file(model_path, dry_run=dry_run, verbose=verbose)

    state_path = model_path.with_name(f"{model_path.stem}_state.pt")
    if state_path.exists() and not _is_protected(state_path):
        _delete_file(state_path, dry_run=dry_run, verbose=verbose)


def _collect_group(prefix: str) -> list[Path]:
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return []

    files = []
    for path in CHECKPOINT_DIR.glob(f"{prefix}_*.safetensors"):
        if _numeric_suffix(path, prefix) < 0:
            continue
        if _is_protected(path):
            continue
        files.append(path)

    return files


def _prune_group(prefix: str, keep: int, *, dry_run: bool = False, verbose: bool = False) -> None:
    if keep < 0:
        raise ValueError("keep must be >= 0")
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    files = _collect_group(prefix)

    # Keep newest by modification time. Use the numeric suffix only as a stable
    # tie-breaker for files written at nearly the same time.
    files.sort(key=lambda p: (_mtime_ns(p), _numeric_suffix(p, prefix)), reverse=True)

    to_delete = files[keep:]
    for model_path in to_delete:
        _delete_checkpoint_pair(model_path, dry_run=dry_run, verbose=verbose)


def prune_orphan_states(*, dry_run: bool = False, verbose: bool = False) -> None:
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    for state_path in CHECKPOINT_DIR.glob("*_state.pt"):
        if _is_protected(state_path):
            continue

        stem = state_path.stem
        if stem.endswith("_state"):
            stem = stem[: -len("_state")]

        model_path = CHECKPOINT_DIR / f"{stem}.safetensors"
        if not model_path.exists():
            _delete_file(state_path, dry_run=dry_run, verbose=verbose)


def prune_checkpoints(
    keep: int = 10,
    keep_steps: int = 4,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Prune checkpoint files safely.

    Args:
        keep: Number of newest epoch_*.safetensors checkpoints to keep by mtime.
        keep_steps: Number of newest step_*.safetensors checkpoints to keep by mtime.
        verbose: Print each deleted file.
        dry_run: Print what would be deleted without deleting.
    """
    if not CHECKPOINT_DIR or not CHECKPOINT_DIR.exists():
        return

    _prune_group("epoch", keep=keep, dry_run=dry_run, verbose=verbose)
    _prune_group("step", keep=keep_steps, dry_run=dry_run, verbose=verbose)
    prune_orphan_states(dry_run=dry_run, verbose=verbose)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely prune Sage-GPT checkpoints by modification time.")
    parser.add_argument("--keep", type=int, default=10, help="Newest epoch checkpoints to keep by mtime.")
    parser.add_argument("--keep-steps", type=int, default=4, help="Newest step checkpoints to keep by mtime.")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be pruned without deleting.")
    parser.add_argument("--verbose", action="store_true", help="Print pruned files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prune_checkpoints(
        keep=args.keep,
        keep_steps=args.keep_steps,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
