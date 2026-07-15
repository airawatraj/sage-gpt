#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="docs/experiments/2026-07-15-polish-lr2e5"
HISTORY_SRC="6-logs/training/training_history_dgx.csv"

if [ ! -f "$HISTORY_SRC" ]; then
  echo "ERROR: missing history CSV: $HISTORY_SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

uv run python3 - <<'PY'
from pathlib import Path
import csv
import json
import hashlib
import subprocess
from datetime import datetime

out_dir = Path("docs/experiments/2026-07-15-polish-lr2e5")
out_dir.mkdir(parents=True, exist_ok=True)

history_src = Path("6-logs/training/training_history_dgx.csv")
history_dst = out_dir / "training_history_dgx.polish_lr2e5.csv"
history_dst.write_text(history_src.read_text())

rows = list(csv.DictReader(history_src.open()))
if not rows:
    raise SystemExit("History CSV has no rows")

best_sample = min(rows, key=lambda r: float(r["Val_Loss"]))
last = rows[-1]

def git_cmd(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def sha256_file(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

checkpoint_paths = [
    "3-model/pt/checkpoints/best_grok_model.safetensors",
    "3-model/pt/checkpoints/interrupt.safetensors",
    "3-model/pt/checkpoints/interrupt_state.pt",
    "3-model/pt/run_archives/polish_lr2e5_step3150/best_grok_model.full_rawce_1.185209.safetensors",
    "3-model/pt/run_archives/polish_lr2e5_step3150/interrupt_step3150.safetensors",
    "3-model/pt/run_archives/polish_lr2e5_step3150/interrupt_state_step3150.pt",
]

hashes = []
for path in checkpoint_paths:
    digest = sha256_file(path)
    if digest:
        hashes.append({"path": path, "sha256": digest})

summary = {
    "experiment": "polish_lr2e5_after_eval_metrics_fix",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "branch": git_cmd("branch", "--show-current"),
    "commit": git_cmd("rev-parse", "HEAD"),
    "main_commit_at_eval_fix_merge": "194e1d6",
    "baseline_before_polish": {
        "full_val_raw_ce": 1.185215,
        "note": "Tracked baseline before LR 2e-5 polish.",
    },
    "polish_result": {
        "lr": "2e-5 flat",
        "stopped_epoch": 12,
        "stopped_step": 3181,
        "best_confirmed_full_val_raw_ce": float(last["Best_Val_Raw_Loss"]),
        "best_sampled_step": int(best_sample["Step"]),
        "best_sampled_train_raw_ce": float(best_sample["Train_Loss"]),
        "best_sampled_val_raw_ce": float(best_sample["Val_Loss"]),
        "best_sampled_gap": float(best_sample["Gap"]),
        "last_step": int(last["Step"]),
        "last_val_raw_ce": float(last["Val_Loss"]),
    },
    "conclusion": [
        "Flat 2e-5 polish improved sampled validation compared with the 6e-5 plateau.",
        "Confirmed full-validation raw CE improved only slightly, from about 1.185215 to about 1.185209.",
        "Treat the polish LR change as experimental until qualitative evaluation confirms better generations.",
        "Do not merge the full polish branch into main yet.",
    ],
    "checkpoint_policy": {
        "git_tracks_checkpoints": False,
        "note": "GitHub stores experiment metadata and hashes only. Exact model recovery requires external checkpoint backup.",
    },
    "checkpoint_hashes": hashes,
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

hash_lines = [f'{item["sha256"]}  {item["path"]}' for item in hashes]
(out_dir / "checkpoint_hashes.sha256").write_text("\n".join(hash_lines) + "\n")

readme = f"""# Sage-GPT experiment ledger: eval-fix polish LR 2e-5

Date: 2026-07-15

Branch: `{summary["branch"]}`

Commit: `{summary["commit"]}`

## Purpose

Test whether a flat `2e-5` polish run from `best_grok_model.safetensors` improves after the eval-mode raw CE metrics fix.

## Starting point

The run resumed from `best_grok_model.safetensors` with no optimizer state.

The optimizer state was intentionally reset by removing `interrupt_state.pt`.

Baseline to beat:

```text
Full validation raw CE: 1.185215
Validation PPL: about 3.27
```

## Result

Confirmed best full-validation raw CE:

```text
{float(last["Best_Val_Raw_Loss"]):.12f}
```

Best sampled validation row:

```text
Step: {best_sample["Step"]}
Train raw CE: {float(best_sample["Train_Loss"]):.6f}
Val raw CE: {float(best_sample["Val_Loss"]):.6f}
Gap: {float(best_sample["Gap"]):.6f}
```

Final recorded row:

```text
Step: {last["Step"]}
Train raw CE: {float(last["Train_Loss"]):.6f}
Val raw CE: {float(last["Val_Loss"]):.6f}
Gap: {float(last["Gap"]):.6f}
Best full val raw CE: {float(last["Best_Val_Raw_Loss"]):.12f}
```

## Interpretation

Flat `2e-5` polish improved sampled validation compared with the `6e-5` plateau.

The confirmed full-validation improvement was tiny:

```text
About 1.185215 to {float(last["Best_Val_Raw_Loss"]):.12f}
```

This is useful but not enough to merge the whole polish branch into `main` yet.

## Important checkpoint note

The polish run restarted step and epoch counters from 0 because it resumed from `best_grok_model.safetensors` without `interrupt_state.pt`.

Numbered polish checkpoints such as `epoch_12.safetensors` and `step_3000.safetensors` were likely pruned because older run checkpoints had much larger epoch and step numbers.

The latest polish state is preserved locally in:

```text
3-model/pt/checkpoints/interrupt.safetensors
3-model/pt/checkpoints/interrupt_state.pt
```

and archived locally in:

```text
3-model/pt/run_archives/polish_lr2e5_step3150/
```

## Recovery limitation

Checkpoints are not tracked in GitHub.

This ledger preserves metrics, conclusions, and checkpoint hashes. It does not preserve the model weights.

For exact recovery after a DGX reset, copy `.safetensors` and `.pt` checkpoint files to external storage.

## Files in this ledger

```text
README.md
summary.json
training_history_dgx.polish_lr2e5.csv
checkpoint_hashes.sha256
```
"""

(out_dir / "README.md").write_text(readme)

print("Wrote experiment ledger:")
for path in sorted(out_dir.iterdir()):
    print(f"  {path}")
PY

echo "Done. Ledger directory: $OUT_DIR"
