import os
import sys
import sentencepiece as spm
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: duplicate config.py not found. Please run setup_factory.py first.")
    sys.exit(1)

# Configuration from config.py
VOCAB_SIZE = config.VOCAB_SIZE
# Ensure tokenizer dir is absolute path string for sentencepiece
MODEL_PREFIX = str(config.TOKENIZER_DIR / "sutra_tokenizer")
CORPUS_FILE_PATH = config.PURIFIED_DATA_DIR / "corpus.txt"
OUTPUT_BIN_FILE = config.TOKENIZED_DATA_DIR / "corpus.bin"

def train_tokenizer():
    print(f"Training SentencePiece tokenizer (Vocab: {VOCAB_SIZE})...")
    # Using strict unigram model, bytes fallback for unknown chars, nfkc norm
    # input argument must be string path
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE_PATH),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        byte_fallback=True,
        normalization_rule_name="nfkc",
        character_coverage=1.0, # Purified data should be covered
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True
    )
    print(f"Tokenizer trained and saved to {MODEL_PREFIX}.model")

def encode_corpus():
    print("Encoding corpus into binary format...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    
    # Calculate total size roughly or just append chunks
    # We use uint16 for vocab up to 65535.
    
    # Chunk reading to save RAM
    CHUNK_SIZE = 1000000 # Read lines in batches, not raw bytes, to avoid splitting utf-8 chars
    
    token_count = 0
    with open(OUTPUT_BIN_FILE, "wb") as f_out:
        with open(CORPUS_FILE_PATH, "r", encoding="utf-8") as f_in:
            while True:
                lines = f_in.readlines(CHUNK_SIZE)
                if not lines:
                    break
                
                # Combine lines for encoding, or encode line by line
                # Batch encoding is faster
                text_chunk = "".join(lines)
                
                # Encode chunk
                ids = sp.encode_as_ids(text_chunk)
                
                # Convert to numpy array (uint16 is enough for vocab up to 65535)
                arr = np.array(ids, dtype=np.uint16)
                
                # Write bytes
                f_out.write(arr.tobytes())
                token_count += len(ids)
                
                print(f"Encoded chunk... Total tokens so far: {token_count}", end='\r')
    
    print(f"\nEncoding complete. Saved to {OUTPUT_BIN_FILE}")
    print(f"Total Tokens: {token_count}")

def main():
    print("Starting Sutra Tokenizer Factory...")
    
    if not CORPUS_FILE_PATH.exists():
        print(f"Error: Corpus file not found at {CORPUS_FILE_PATH}")
        return

    # 1. Train
    train_tokenizer()
    
    # 2. Encode
    encode_corpus()
    
    print("Sutra Output Ready.")

if __name__ == "__main__":
    main()
