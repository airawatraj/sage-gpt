# Evaluation notes

Date: 2026-07-15

## Checkpoint: best_grok_model.safetensors

Probe result:

```text
Status: healthy
Layers: 6 / 6
Vocab: 8192 / 8192
D_Model: 256 / 256
Params: 7.21M
NaNs: none
Infs: none
```

Ashtavakra result:

```text
Vedic Score: 0/8
Conclusion: still in circuit formation phase
```

Qualitative generation:

The model produces Sanskrit-like continuations, but output is still repetitive and structurally unstable.

## Checkpoint: interrupt.safetensors

Ashtavakra result:

```text
Vedic Score: 1/8
Passed: Orthography
```

Important qualitative note:

The interrupt checkpoint produced contamination-like text in the audit preview:

```text
आपकोनमस्कारारहै
```

This means the interrupt checkpoint should not be promoted as the winner solely because it scored 1/8.

## Plot notes

The generalisation gap plot is visually sparse because the active CSV contains only the short polish run, steps 0 to 3150. Train raw CE and validation raw CE are nearly flat, while live train smoothed CE uses a much higher range.

The plot is not missing data, but it is not very informative for this short polish window.

The norm history plot reflects older high-number checkpoints through epoch 480. It does not meaningfully represent the polish run because low-number polish checkpoints were pruned.

## Decision

Do not merge the full `polish-lr-2e-5` branch into `main`.

Keep:

```text
best_grok_model.safetensors = metric winner
interrupt.safetensors = qualitative candidate only
polish-lr-2e-5 = experimental branch
```