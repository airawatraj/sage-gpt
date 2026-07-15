# Sage-GPT golden recovery guide

Date: 2026-07-15

Purpose: make the repo recoverable after a DGX reset, firmware issue, disk loss, or failed experiment. GitHub stores code and experiment decisions. External backup stores model/data artifacts.

## Recovery goal

After a fresh machine reset, you should be able to:

1. Clone the repo.
2. Install dependencies with `uv`.
3. Restore tokenizer, data, and checkpoint artifacts from external backup.
4. Run evaluation against the restored best checkpoint.
5. Resume training only from the recommended recipe.

## Current known-good code state

Stable main branch:

```text
main
```

Important tags:

```text
v0.1.0-grok-plateau-123700
v0.1.2-polish-lr2e5-ledger
v0.1.3-polish-lr2e5-eval-notes
```

Important experiment branch:

```text
polish-lr-2e-5
```

Do not merge the full `polish-lr-2e-5` branch into `main` unless the flat `2e-5` training config is intentionally promoted.

## Current model decision

Metric winner:

```text
3-model/pt/checkpoints/best_grok_model.safetensors
```

Best confirmed full validation raw CE:

```text
1.185208922624588
```

Latest polish state:

```text
3-model/pt/checkpoints/interrupt.safetensors
3-model/pt/checkpoints/interrupt_state.pt
```

Evaluation conclusion:

```text
best_grok_model.safetensors = metric winner
interrupt.safetensors = qualitative candidate only
polish-lr-2e-5 = experimental branch
```

Reason: latest polish interrupt scored 1/8 on Ashtavakra, but showed contamination-like output. Do not promote it solely on that score.

## Files that must be backed up outside GitHub

GitHub does not track checkpoints or large artifacts. External backup must include:

```text
3-model/pt/checkpoints/best_grok_model.safetensors
3-model/pt/checkpoints/interrupt.safetensors
3-model/pt/checkpoints/interrupt_state.pt
3-model/pt/run_archives/polish_lr2e5_step3150/
```

Also back up tokenizer artifacts:

```text
2-tokenizer/
```

Also back up purified or tokenized data artifacts needed to avoid rerunning the full data pipeline:

```text
1-data/
3-model/pt/
```

If disk space is tight, prioritize:

```text
best_grok_model.safetensors
interrupt.safetensors
interrupt_state.pt
SentencePiece model and vocab
final purified corpus
final tokenized train/val files
```

## External backup verification

After copying artifacts to external storage, verify with hashes.

Example backup location:

```text
/home/airawatraj/sage-gpt-backups/2026-07-15-golden/
```

Create hashes:

```bash
cd /home/airawatraj/sage-gpt-backups/2026-07-15-golden
find . -type f \( -name "*.safetensors" -o -name "*.pt" -o -name "*.model" -o -name "*.vocab" -o -name "*.csv" -o -name "*.md" -o -name "*.json" -o -name "*.sha256" \) -print0 | sort -z | xargs -0 sha256sum > BACKUP_SHA256SUMS.txt
```

Verify on the destination:

```bash
cd /path/to/copied/2026-07-15-golden
sha256sum -c BACKUP_SHA256SUMS.txt
```

Expected result: all files report `OK`.

## Fresh recovery from factory reset

### 1. Clone repo

```bash
git clone https://github.com/airawatraj/sage-gpt.git
cd sage-gpt
git checkout main
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Restore artifacts

Copy external backup artifacts back into their expected repo paths.

Minimum checkpoint restore:

```text
3-model/pt/checkpoints/best_grok_model.safetensors
3-model/pt/checkpoints/interrupt.safetensors
3-model/pt/checkpoints/interrupt_state.pt
```

Restore tokenizer and data artifacts before training or inference.

### 4. Verify environment

```bash
uv run python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

### 5. Verify checkpoint health

```bash
uv run python3 4-evaluation/sutra_probe_pt.py 3-model/pt/checkpoints/best_grok_model.safetensors
```

Expected:

```text
Status: healthy
Layers: 6 / 6
Vocab: 8192 / 8192
D_Model: 256 / 256
NaNs: none
Infs: none
```

### 6. Run full evaluation

```bash
bash 4-evaluation/evaluate.sh 3-model/pt/checkpoints/best_grok_model.safetensors
```

## Resume rule

Resume training only after checking the experiment ledger:

```text
docs/experiments/2026-07-15-polish-lr2e5/
```

Do not resume blind long runs from the latest interrupt without knowing whether it is the metric winner.

Current recommended default:

```text
Use best_grok_model.safetensors as the metric winner.
Treat interrupt.safetensors as a qualitative candidate only.
```

## Before the next overnight training run

Fix these first:

1. Pruning behavior for restart-from-best runs.
2. Validation split quality.
3. Contamination audit for Hindi-like output.
4. Checkpoint bake-off with fixed prompts and fixed seeds.

## Known pruning issue

The polish run restarted step and epoch counters from 0 because it resumed from `best_grok_model.safetensors` without `interrupt_state.pt`.

Old checkpoints had much larger numbers, such as epochs 471 to 480. Low-number polish checkpoints such as `epoch_12.safetensors` and `step_3000.safetensors` were pruned.

Before another restart-from-best experiment, use one of these approaches:

1. Separate checkpoint directory for each experiment.
2. Continue global step and epoch numbering from the prior run.
3. Update pruning to protect checkpoints from the active experiment.

## Current next best training direction

Do not continue blind LR polishing.

Recommended next work:

1. Keep `main` as the stable workflow branch.
2. Add a document-level shuffled train/validation split.
3. Add a contamination audit over purified data and model generations.
4. Add fixed-seed checkpoint bake-off prompts.
5. Run one deliberate next overnight training attempt only after the above are in place.

## What GitHub preserves

GitHub preserves:

```text
code
training scripts
evaluation scripts
experiment ledgers
metrics CSV snapshots
checkpoint hashes
recovery instructions
```

GitHub does not preserve:

```text
model weights
optimizer state
large data artifacts
large tokenizer/data outputs if ignored
```

External backup is required for exact recovery.