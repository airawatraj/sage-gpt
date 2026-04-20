# Auto-generated config by setup_factory.py
from pathlib import Path

PROJECT_ROOT = Path("/home/airawatraj/code/rawatlabs/sage-gpt").expanduser().resolve()
ROOT_DIR = PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "1-data/01-raw"
PURIFIED_DATA_DIR = PROJECT_ROOT / "1-data/02-purified"
TOKENIZED_DATA_DIR = PROJECT_ROOT / "1-data/03-tokenized"
META_DATA_DIR = PROJECT_ROOT / "1-data/04-meta"
TOKENIZER_DIR = PROJECT_ROOT / "2-tokenizer"
MODEL_DIR = PROJECT_ROOT / "3-model"
TRAINING_SRC_DIR = PROJECT_ROOT / "3-training/src"
LOG_DIR = PROJECT_ROOT / "6-logs"
VENV_DIR = PROJECT_ROOT / ".venv"

PT_CHECKPOINT_DIR = MODEL_DIR / "pt" / "checkpoints"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PURIFIED_DATA_DIR, TOKENIZED_DATA_DIR, META_DATA_DIR, TOKENIZER_DIR, MODEL_DIR, TRAINING_SRC_DIR, LOG_DIR, PT_CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 8.4M Grokking Specs
# MODEL_NAME = "Sage-GPT-7.25M-Grok"
# DATA_STATS = "56.89M Tokens (164.8M Characters)"
# VOCAB_SIZE = 8000
# CONTEXT_LENGTH = 256
# EMBED_DIM = 256
# HEADS = 8
# LAYERS = 4

# DGX-Optimized "Sage-GPT" Specs
MODEL_NAME = "Sage-GPT-48M-Sutra"
DATA_STATS = "132.7M Tokens (224MB Characters)"
VOCAB_SIZE = 16000 
CONTEXT_LENGTH = 1024 
EMBED_DIM = 512
HEADS = 16
LAYERS = 10