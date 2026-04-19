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

# Configuration from config.py
VOCAB_SIZE = config.VOCAB_SIZE
MODEL_PREFIX = str(config.TOKENIZER_DIR / "sutra_tokenizer")
CORPUS_FILE_PATH = config.PURIFIED_DATA_DIR / "corpus.txt"
OUTPUT_BIN_FILE = config.TOKENIZED_DATA_DIR / "corpus.bin"

def train_tokenizer():
    print(f"--- SUTRA TOKENIZER TRAINING (Vocab: {VOCAB_SIZE}) ---")
    
    # DGX Optimization: Using explicit special tokens for Sanskrit boundaries
    # <s> = Start of Verse, </s> = End of Verse, <unk> = Unknown, <pad> = Padding
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
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["।", "॥"] # Ensure dandas are kept as single tokens
    )
    print(f"✅ Tokenizer trained and saved to {MODEL_PREFIX}.model")

def encode_corpus():
    print(f"--- BINARY ENCODING: {OUTPUT_BIN_FILE.name} ---")
    sp = spm.SentencePieceProcessor()
    if not sp.load(f"{MODEL_PREFIX}.model"):
        raise RuntimeError("Failed to load trained model.")
    
    token_count = 0
    # Batch size for writing to disk (prevents I/O bottleneck)
    BATCH_SIZE = 10000 
    token_buffer = []

    with open(OUTPUT_BIN_FILE, "wb") as f_out:
        with open(CORPUS_FILE_PATH, "r", encoding="utf-8") as f_in:
            for line in tqdm(f_in, desc="Processing Sanskrit Tokens"):
                line = line.strip()
                if not line: continue
                
                # Wrap each line in BOS/EOS tokens for context awareness
                ids = [sp.bos_id()] + sp.encode_as_ids(line) + [sp.eos_id()]
                token_buffer.extend(ids)
                
                # Write to disk in chunks
                if len(token_buffer) >= BATCH_SIZE:
                    arr = np.array(token_buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    token_count += len(token_buffer)
                    token_buffer = []
            
            # Flush remaining tokens
            if token_buffer:
                arr = np.array(token_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_count += len(token_buffer)
    
    print(f"\n✅ Encoding complete.")
    print(f"📊 Total Tokens: {token_count:,}")
    print(f"💾 Binary Size: {os.path.getsize(OUTPUT_BIN_FILE) / (1024*1024):.2f} MB")

def main():
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    config.TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CORPUS_FILE_PATH.exists():
        print(f"❌ Error: Corpus file not found. Ensure Visuddhi V5 finished.")
        return

    train_tokenizer()
    encode_corpus()
    print("🚀 All systems ready for Sage-GPT Training.")

if __name__ == "__main__":
    main()