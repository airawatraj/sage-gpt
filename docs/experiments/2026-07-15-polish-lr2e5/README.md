# Sage-GPT experiment ledger: eval-fix polish LR 2e-5

Date: 2026-07-15

Branch: `polish-lr-2e-5`

Commit: `9d3d0ffe42d2b14681b849661073ee8edbbd593c`

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
1.185209000000
```

Best sampled validation row:

```text
Step: 3050
Train raw CE: 1.515505
Val raw CE: 1.187649
Gap: -0.327856
```

Final recorded row:

```text
Step: 3150
Train raw CE: 1.516083
Val raw CE: 1.188328
Gap: -0.327755
Best full val raw CE: 1.185209000000
```

## Interpretation

Flat `2e-5` polish improved sampled validation compared with the `6e-5` plateau.

The confirmed full-validation improvement was tiny:

```text
About 1.185215 to 1.185209000000
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
