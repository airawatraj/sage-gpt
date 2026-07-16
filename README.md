# 🕉️ SageGPT - Sanskrit Small Language Model (SLM)

> *"To find the Sutra in the Signal."*

**Sage-GPT** is a Sanskrit-only Small Language Model (SLM) trained **from scratch** on a specialised Sanskrit corpus. The active tokenized training stream is a 139 MB binary containing **72.8M SentencePiece model-token IDs**, split into **69.2M training tokens** and **3.6M validation tokens**. It is not a fine-tuned version of an existing LLM.

The goal is not to compete with large general-purpose LLMs. The goal is to test whether a compact, carefully trained model can learn Sanskrit-specific structure, including morphology, sandhi patterns, verse-like continuation, and domain vocabulary, from a focused Sanskrit corpus.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/model-7.2M%20SLM-purple)
![Training](https://img.shields.io/badge/training-from%20scratch-orange)
![Dataset](https://img.shields.io/badge/data-72.8M%20model%20tokens-blue)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Context](https://img.shields.io/badge/context-1024-lightgrey)

Sanskrit is structurally unlike most modern high-resource languages. It has complex *sandhi* rules, rich case inflection across 8 cases, strict metrical structure in verse, and a philosophical vocabulary that often does not map cleanly into English-centric model spaces. General-purpose LLMs can generate Sanskrit-looking text, but small domain-specific models make the training dynamics easier to inspect and reason about.

---

## 🏛️ Architecture

A compact decoder-only Transformer, sized for repeated experimentation on a home DGX Spark setup while preserving enough capacity to learn Sanskrit morphology and sequence structure.

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Parameters** | ~7.2M | Small enough for fast iteration, large enough for Sanskrit structure learning |
| **Depth** | 6 Layers, 8 Attention Heads | Hierarchical capacity for morphology and phrase-level dependencies |
| **Dimensions** | 256 Embedding Dim | Efficient representation size for a focused Sanskrit corpus |
| **Context Window** | 1024 Tokens | Handles shlokas, prose passages, and chapter fragments in one pass |
| **Feed-Forward** | SwiGLU | Gated activation, smoother training dynamics |
| **Positional Encoding** | RoPE | Rotary embeddings for relative position awareness |
| **Normalisation** | RMSNorm | Lightweight normalization without mean-centering |
| **Weight Tying** | `output.weight = tok_emb.weight` | Shared input/output embeddings reduce parameters and stabilize output projection |
| **Tokenizer** | SentencePiece Unigram, 8,192 vocab | Reduces fragmentation in Sanskrit compounds compared with simple byte or BPE setups |
| **Precision** | `bfloat16` via `torch.autocast` | Uses DGX Spark hardware efficiently with stable dynamic range |

---

## 🚦 Execution Pipeline

### Step 1 - Deep Data Purification

The purification stage transforms raw Sanskrit source material into a cleaner training corpus. The pipeline is designed to reduce OCR noise, obvious vernacular contamination, malformed Unicode, and non-Sanskrit fragments while preserving Devanagari structure and Vedic markers where possible.

This is a conservative filtering pipeline, but it should not be interpreted as a formal guarantee of zero contamination. Corpus quality is continuously audited and improved.

Main features:

- **OCR fallback** for scanned manuscripts when fast text extraction is unavailable
- **Devanagari normalization** using strict Unicode cleanup
- **Nukta and vernacular marker filtering** to reduce Hindi, Marathi, Pali, Prakrit, and OCR-adjacent noise
- **Large image safety** for high-resolution manuscript scans
- **Swara protection** for Sanskrit and Vedic marks where retained by the source
- **Audit-friendly output** for downstream inspection and correction

```bash
uv run python3 1-data/05-scripts/visuddhiv5.py
```

![Visuddhi Run](assets/visuddhi-run.png)

### Step 2 - Refine Corpus

A second filtering pass removes additional noisy patterns, repeated fragments, and likely vernacular residues. The goal is to improve Sanskrit density without overclaiming perfect linguistic separation.

```bash
uv run python3 1-data/05-scripts/refine_corpus.py
```

![Refine Corpus](assets/refine-corpus.png)

### Step 3 - Sutra Tokenization

Trains a SentencePiece Unigram tokenizer with an 8,192 token vocabulary and writes `1-data/03-tokenized/corpus.bin`. The current binary is 139 MB and contains 72.8M `uint16` SentencePiece token IDs.

```bash
uv run python3 2-tokenizer/sutra_tokenizer.py
```

![Tokenization](assets/sutra-tokenizer.png)

### Step 4 - DGX Training

`train_dgx.py` is the entry point. It launches `3-training/src/train_engine_dgx.py`, which runs the full training loop with `torch.compile`, fused SDPA, gradient accumulation, checkpoint pruning, thermal guardrails, and emergency-save recovery.

```bash
uv run python3 train_dgx.py
```

![Sage Training on DGX Spark](assets/sage-gpt-trainnig-DGX.png)

---

## 🔥 Training Configuration

Current public training marker:

```bash
git checkout v0.1.4-best-restart-lr2e5
```

This recipe is designed for continuing from the best checkpoint with a fresh optimizer state. It uses a flat `2e-5` learning rate after warmup to avoid high-LR shock when restarting from `best_grok_model.safetensors`.

| Hyperparameter | Value | Purpose |
| :--- | :--- | :--- |
| **Optimizer** | AdamW, β₁=0.9, β₂=0.95 | Standard transformer optimizer |
| **Learning Rate** | Flat `2e-5` after warmup | Stable best-checkpoint restart recipe |
| **Warmup** | 150 steps | Avoids abrupt optimizer shock |
| **Weight Decay** | 0.05 | Regularises weight matrices |
| **Dropout** | 0.1 | Applied to embeddings, attention, and MLP residuals |
| **Label Smoothing** | 0.1 | Reduces overconfident wrong predictions |
| **Gradient Clipping** | 1.0 | Guards against gradient spikes |
| **Batch Size** | 256 global / 64 micro | 4-step gradient accumulation |
| **Training Duration** | Indefinite, `MAX_STEPS = None` | Ctrl+C triggers emergency save and clean exit |
| **Thermal Guard** | Pause at 75°C | DGX Spark thermal safety |

Current loaded split: **69.2M train tokens** and **3.6M validation tokens**.

### Training Signal Strategy

The patched trainer logs apples-to-apples evaluation metrics:

| Column | Meaning |
| :--- | :--- |
| `Train_Loss` | Train split raw cross-entropy in eval mode |
| `Val_Loss` | Validation split raw cross-entropy in eval mode |
| `Gap` | `Val_Loss - Train_Loss` |
| `Live_Train_Loss` | Live training-batch loss for operational monitoring |
| `Best_Val_Raw_Loss` | Deterministic full-validation best used for checkpoint promotion |

The validation split is currently easier than the train split, so a stable negative gap is expected. The key evidence is whether validation raw CE and deterministic full-validation best continue improving without instability.

---

## 🛡️ Linguistic Guardrails

| Guardrail | Implementation | Goal |
| :--- | :--- | :--- |
| **Unicode Normalisation** | NFKC-focused cleanup | Reduce malformed Devanagari and inconsistent ligature forms |
| **Language Filtering** | Marker-based rejection and corpus audit | Reduce obvious Hindi, Marathi, Pali, Prakrit, and OCR residues |
| **OCR Recovery** | Tesseract-assisted fallback where needed | Recover text from scanned Sanskrit manuscripts |
| **Vedic Integrity** | Swara-aware preservation | Avoid dropping Sanskrit and Vedic accent marks unnecessarily |
| **Corpus Auditability** | Scripted logs and measurable filters | Make contamination and repetition visible rather than assumed away |

These guardrails are best-effort engineering controls. They improve corpus quality, but they do not claim perfect linguistic purity.

---

## 📊 Evaluation & Mechanistic Suite

All evaluation scripts share a common foundation: `eval_utils.py`. It centralises the model architecture, checkpoint resolution, and weight-tied loading logic.

| Shared Module | Role |
| :--- | :--- |
| `eval_utils.py` | SageGPT eval-mode model, `get_target_checkpoint()`, `load_model_from_checkpoint()` |

### 1. Checkpoint Probe

Architecture shape validation and NaN/Inf health check. Run this first.

```bash
uv run python3 4-evaluation/sutra_probe_pt.py
```

![Sutra Probe](assets/sutra-probe.png)

### 2. Weight Norm History

Rebuilds norm history from active `epoch_*.safetensors` checkpoints in `3-model/pt/checkpoints/`, computes L2 norms of Attention and MLP weight matrices, writes `norm_tracking_pt.csv`, and renders `norm_history.png`.

This avoids mixing stale archived checkpoints into current-run evidence.

```bash
uv run python3 4-evaluation/build_norm_history.py
uv run python3 4-evaluation/build_norm_history.py --plot-only
```

![Norm History](assets/norm_history.png)
<sub><em>
<strong>Norm History notes.</strong>
<strong>Epoch checkpoint</strong> means a model snapshot saved at an epoch boundary, for example <code>epoch_217.safetensors</code>.
<strong>Active checkpoints only</strong> means the plot uses checkpoints currently in <code>3-model/pt/checkpoints/</code>, not archived older runs.
<strong>Attention L2 norm</strong> is the average L2 magnitude of attention weight matrices, tracking how attention weights evolve during training.
<strong>MLP L2 norm</strong> is the average L2 magnitude of feed-forward or MLP weight matrices, tracking how dense transformation layers evolve.
<strong>Global peak intensity</strong> is the largest absolute weight value found in the inspected weight matrices, useful for spotting weight spikes or instability.
<strong>Mechanistic contractions</strong> is a diagnostic view of whether weight magnitudes are consolidating, drifting, or becoming unstable during training.
<strong>Weight consolidation</strong> means gradual, stable movement in weight norms rather than sudden spikes or erratic changes.
The generalisation plot is the main model-quality signal; the norm plot is a supporting stability diagnostic.
</em></sub>

### 3. Generalisation Gap Monitor

Renders an evidence plot from `6-logs/training/training_history_dgx.csv`.

| Panel | Meaning |
| :--- | :--- |
| **Aligned CE** | Validation CE shifted to the train baseline for visual comparison |
| **Raw CE** | True train raw CE, true validation raw CE, sampled best, and deterministic full-validation best |
| **Gap** | True `Val_Loss - Train_Loss` over time |

```bash
uv run python3 4-evaluation/generalisation_gap_monitor.py
```

![Generalisation Gap](assets/generalisation_gap.png)
<sub><em>
<strong>Generalisation Gap Monitor notes.</strong>
<strong>CE</strong> means cross-entropy loss, lower is better.
<strong>Train eval CE</strong> is loss measured on the training split while the model is in evaluation mode, so it is not the noisy live batch loss.
<strong>Val eval CE</strong> is loss measured on the validation split while the model is in evaluation mode, and is the main sampled validation signal.
<strong>Aligned CE</strong> is a visual comparison view where validation CE is shifted to the train baseline so train and validation curve shapes can be compared.
<strong>Raw CE</strong> shows the true unshifted cross-entropy values and is the scientific evidence panel.
<strong>Val shifted</strong> means validation CE was moved only for visual alignment; the raw CE panel still shows the real value.
<strong>Best sampled CE</strong> is the lowest validation CE seen during regular sampled validation checks.
<strong>Best full CE</strong> is the deterministic full-validation CE used to promote <code>best_grok_model.safetensors</code>.
<strong>Gap</strong> means <code>Val_Loss - Train_Loss</code>; a negative gap means validation loss is lower than train loss.
<strong>Stable negative gap</strong> means the validation split is easier than the train split, so the gap staying below zero is expected for this run.
<strong>Val PPL</strong> means validation perplexity, calculated as <code>exp(Val CE)</code>, lower is better.
<strong>Steps</strong> are optimizer training steps completed.
</em></sub>


### 4. The Ashtavakra Audit

An 8-bend Sanskrit generative consistency test, named after the sage Ashtavakra. Each bend is a Sanskrit prompt tested against an expected token in the completion: **STRAIGHT** if present, **CROOKED** if not.

This is a heuristic generation audit, not a formal benchmark.

| Bend | Prompt | Expected | Tests |
| :--- | :--- | :--- | :--- |
| 1. Phonetic Stability | `ॐ` | `नमः` | Basic phoneme chaining |
| 2. Invocation | `असतो मा` | `सद्गमय` | Brhadaranyaka Upanishad mantra |
| 3. Case Inflection | `राम` | `ः` | Nominative visarga suffix |
| 4. Sandhi Logic | `नर` | `इन्द्र` | Compound word formation |
| 5. Concept Flow | `यथा नद्यः` | `समुद्रे` | River-to-ocean Upanishad metaphor |
| 6. Verse Sequence | `ईशा वास्य` | `सर्वं` | Ishavasyopanishad opening |
| 7. Orthography | `कृष्` | `ण` | Conjunct completion |
| 8. The Atman Test | `तत्त्वमसि` | `श्वेतकेतो` | Chandogya Upanishad dialogue |

```bash
uv run python3 4-evaluation/ashtavakra_audit.py
```

![Ashtavakra Audit](assets/ashtavakra-audit.png)

### Run Full Eval Pipeline

```bash
bash 4-evaluation/evaluate.sh
```

Optional explicit checkpoint:

```bash
bash 4-evaluation/evaluate.sh 3-model/pt/checkpoints/best_grok_model.safetensors
```

Executes in order:

```text
Probe -> Norms -> Gap -> Audit -> Inference
```

---

## 🕉️ Inference

`inference.py` is the entry point. It validates CUDA availability and launches `5-inference/inference_engine_pt.py`, which uses a KV cache for autoregressive generation, top-p sampling, repetition penalty, and `bfloat16` autocast.

```bash
uv run python3 inference.py
```

---

## 🔧 Utilities

**Prune old checkpoints by modification time:**

```bash
uv run python3 3-training/src/prune_checkpoints.py
```

Dry run:

```bash
uv run python3 3-training/src/prune_checkpoints.py --dry-run --verbose
```

---

## 📌 Notes

- Checkpoints and run logs are intentionally kept out of Git.
- Public images under `assets/` are snapshots, not live training artifacts.
- Current live outputs are generated under `6-logs/evaluation/`.
- The model is experimental. Loss curves and audits are evidence for training progress, not proof of Sanskrit understanding.

---

> 🕉️ *Om Tat Sat* (ॐ तत् सत्) - The Absolute is Truth
