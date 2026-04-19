import os
import sys
import sentencepiece as spm
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

# Configuration
VOCAB_SIZE = config.VOCAB_SIZE
MODEL_PREFIX = str(config.TOKENIZER_DIR / "sutra_tokenizer")
CORPUS_FILE_PATH = config.PURIFIED_DATA_DIR / "corpus.txt"
OUTPUT_BIN_FILE = config.TOKENIZED_DATA_DIR / "corpus.bin"

def train_tokenizer():
    print(f"Training SentencePiece tokenizer (Vocab: {VOCAB_SIZE})...")
    
    # DGX Optimization: input_sentence_size prevents loading TBs of text into RAM
    # 10M sentences is usually enough for a robust 8k-32k vocab.
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE_PATH),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        byte_fallback=True,
        normalization_rule_name="nfkc",
        character_coverage=1.0, 
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True,
        input_sentence_size=10000000, 
        shuffle_input_sentence=True
    )
    print(f"\nTokenizer trained: {MODEL_PREFIX}.model")

def encode_corpus():
    print("Encoding corpus into binary format...")
    sp = spm.SentencePieceProcessor()
    if not sp.load(f"{MODEL_PREFIX}.model"):
        raise RuntimeError("Failed to load trained model.")
    
    token_count = 0
    # Use a specific buffer size for DGX NVMe throughput
    with open(OUTPUT_BIN_FILE, "wb") as f_out:
        with open(CORPUS_FILE_PATH, "r", encoding="utf-8") as f_in:
            # tqdm context manager for better visual progress on SSH
            for line in tqdm(f_in, desc="Encoding Tokens"):
                line = line.strip()
                if not line: continue
                
                # Encode line
                ids = sp.encode_as_ids(line)
                # Add a separator token (like <eos>) if your model expects it
                # ids.append(sp.eos_id()) 

                arr = np.array(ids, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_count += len(ids)
    
    print(f"\nEncoding complete: {OUTPUT_BIN_FILE}")
    print(f"Total Tokens in Corpus: {token_count}")

def main():
    # Create directories if they don't exist
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    config.TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CORPUS_FILE_PATH.exists():
        print(f"Error: Corpus file not found at {CORPUS_FILE_PATH}")
        return

    train_tokenizer()
    encode_corpus()
    print("Sutra Tokenization Phase Complete.")

if __name__ == "__main__":
    main()