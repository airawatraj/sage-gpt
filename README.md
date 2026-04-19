# 🕉️ Sovereign Ancient General Intelligence (SAGE-GPT)

> "To find the Sutra in the Signal."

**Sage-GPT-7.25M (Grokking Phase)** is a Decoder-only Transformer trained from scratch on 56.89M ultra-pure Sanskrit tokens (164.8M characters). Architected to induce grokking (delayed generalization) through high-overfitting regimes.

### Specification Engine
* **Architecture**: 4 Layers, 8 Attention Heads, 256 Embedding Dim, 256 Context Length.
* **Tokenizer**: SentencePiece Unigram (8k Vocab, byte_fallback active, NFKC Strict).
* **Training Engine**: Dual-Backend Architecture with AEDT-governed execution:
  * **NVIDIA DGX Support**: Utilizes PyTorch `bfloat16` Native Mixed Precision, `torch.compile` graph optimization, and FlashAttention for maximum high-performance cluster throughput.

---

## 🚦 Execution Pipeline

### Step 1 & 2: Data Purification & Tokenization
Transforms 14GB raw data into the 56.89M token pure corpus using NFKC normalization and the 8k Sutra Unigram tokenizer.
```bash
uv run python3 1-data/05-scripts/visuddhi.py
```

### Step 3: Train (Cross-Platform Orchestrator)
Initiates the training loop via the root launcher. It automatically detects if you are on an NVIDIA DGX or an Apple device.

For **NVIDIA DGX** (128GB+ VRAM), use the optimized high-throughput launcher without AEDT throttling:
```bash
uv run python3 train_dgx.py
```

### Step 3.5: Multi-Mode Inference
Deploy the interactive dual-mode inference shell on DGX:
```bash
uv run python3 inference.py
```

### Step 4: Generalisation Gap Monitor
Runs passively in a separate terminal to generate a dark-mode, log-scale plot of the grokking divergence (Train vs Validation Loss). The plot is saved to `6-logs/evaluation/generalisation_gap.png`.
```bash
uv run python3 4-evaluation/generalisation_gap_monitor.py
```

### Step 5: Mechanistic Weight Norm Inspection
Calculates L2 norms for the Attention and MLP layer weight matrices from the `interrupt_save.safetensors` checkpoint. Logs these metrics to `6-logs/evaluation/norm_tracking.csv`.
```bash
uv run python3 4-evaluation/inspect_norms.py
```

### Step 6: Visualize Weight Norm Trajectories
Reads the tracked norms and renders a dark-mode plot of the 'Average Attention L2 Norm', 'Average MLP L2 Norm', and 'Block 0 QKV Peak Norm'. The plot is saved to `6-logs/evaluation/norm_history.png`.
```bash
uv run python3 4-evaluation/plot_norms.py
```

### Step 7: Run the Ashtavakra Audit

```bash
uv run python3 4-evaluation/ashtavakra_audit.py
```

# Tip : Run all evaluation scripts together

```bash
./4-evaluation/evaluate.sh
```


---

## 🛡️ Linguistic Guardrails (V4)
Our "Zero-Poison" policy ensures the model trains only on high-fidelity Sanskrit.

| Feature | Implementation | Goal |
| :--- | :--- | :--- |
| **Normalization** | **NFKC** | Prevents shattering of conjuncts/roots (No NFC). |
| **Noise Shield** | Disjoint Stopwords | 100% rejection of Hindi, Marathi, Pali, and Prakrit. |
| **Punctuation** | Danda-Aware | Protects sutras even with trailing '।' or '॥' markers. |
| **Vedic Safety** | Swara Protection | Preservation of Visarga (ः), Anusvara (ं), and Vedic accents. |

---

## ⚙️ Operational Modes (AEDT Aware)
SAGE-GPT adapts training intensity based on the time of day to protect M1 resources.

### ☀️ STEALTH Mode (09:00 - 18:00)
* **Batch Size**: 4 (with 32-step Grad Accumulation).
* **Effective BS**: 128 (Mathematically stable).
* **VRAM**: Optimized for background execution while working.

### 🌙 FACTORY Mode (18:00 - 09:00)
* **Batch Size**: 128 (Direct Metal acceleration).
* **Focus**: High-throughput stochastic exploration.


### Prune Checkpoints

```bash
uv run python3 3-training/src/prune_checkpoints.py
``` 

---
> 🕉️ Om Tat Sat (ॐ तत् सत्) - The Absolute is Truth
