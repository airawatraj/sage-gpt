import os
import sys
import re
import unicodedata
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

# Environment-specific imports
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# DGX Path Resolution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    INPUT_DIR = config.RAW_DATA_DIR
    OUTPUT_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"
    LOG_DIR = config.LOG_DIR / "purification"
except ImportError:
    # Fallback to relative paths if config is missing
    INPUT_DIR = Path("./1-data/01-raw")
    OUTPUT_FILE = Path("./1-data/02-purified/corpus.txt")
    LOG_DIR = Path("./6-logs/purification")

# --- LINGUISTIC CONSTANTS (Pre-compiled for DGX Speed) ---
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F\u0902\u0903\s\u200c\u200d\u0964\u0965]+')
NUKTAS_PATTERN = re.compile(r'[\u0958\u0933]')
SIGNATURE_PATTERN = re.compile(r'[\u094D\u0903]')
DANDA_COLLAPSE = re.compile(r'([।॥]\s*){2,}')
M_COLLAPSE = re.compile(r'म्(?:\s+म्)+|म्{2,}')

NOISE_STOPWORDS = {
    # Hindi/Marathi/Nepali noise to be purged
    "है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा",
    "आहे", "आणि", "पूर्ण", "करण्यासाठी", "आहेत", "होता", "होती", "असा", "तसेच", "या", "व", "काही", "झाली",
    "तस्स", "अरहतो", "सम्मा", "णमो", "हो", "यो", "र", "छ", "छन्", "थियो", "गर्नु", "मलाई", "भोक", "लाग्यो"
}
HINDI_STOP = {"है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा"}

def apply_de_echo(text):
    """Collapses duplicate punctuation and terminal markers."""
    text = DANDA_COLLAPSE.sub('॥ ', text)
    text = M_COLLAPSE.sub('म्', text)
    return text

def clean_text_block(text):
    """
    Analyzes text segments and returns (purified_string, statistics).
    Logic tuned for Sanskrit vs. Vernacular Devanagari.
    """
    stats = {
        'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 
        'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0
    }
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[0-9०-९]+', '', text)
    matches = DEVANAGARI_PATTERN.findall(text)
    
    valid_blocks = []
    for m in matches:
        cleaned = re.sub(r'\s+', ' ', m).strip()
        if not cleaned: 
            continue
        
        stats['total_blocks'] += 1
        
        # 1. Purity Filter (>95% Devanagari)
        char_len = len(cleaned)
        devanagari_and_space = len(re.findall(r'[\u0900-\u097F\u0902\u0903\s]', cleaned))
        if (devanagari_and_space / char_len) < 0.95:
            stats['punctuation'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 2. Blacklist Token Check
        words_set = {w.strip("।॥").strip() for w in cleaned.split()}
        if not words_set.isdisjoint({'हे', 'को', 'मा'}):
            stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 3. Sanskrit Marker Density (Halant/Visarga)
        if char_len > 15:
            if not SIGNATURE_PATTERN.search(cleaned):
                stats['low_density'] += 1
                stats['total_discarded'] += 1
                continue
        
        # 4. Stopword Filter (Linguistic Isolation)
        if not words_set.isdisjoint(NOISE_STOPWORDS):
            if not words_set.isdisjoint(HINDI_STOP):
                stats['hindi'] += 1
            else:
                stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue

        # 5. Nukta Check (Modern modification rejection)
        if NUKTAS_PATTERN.search(cleaned):
            continue

        valid_blocks.append(apply_de_echo(cleaned))
            
    return "\n".join(valid_blocks), stats

def process_file_worker(filepath):
    """Worker task for DGX cores. Isolated from global state."""
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    file_stats = {
        'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 
        'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0
    }
    
    try:
        if ext in ['.html', '.htm'] and BeautifulSoup:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
                for script in soup(["script", "style"]): script.decompose()
                content, file_stats = clean_text_block(soup.get_text())
        elif ext == '.pdf' and fitz:
            doc = fitz.open(filepath)
            full_raw = "\n".join([page.get_text() for page in doc])
            content, file_stats = clean_text_block(full_raw)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content, file_stats = clean_text_block(f.read())
    except Exception:
        return None, file_stats

    return content, file_stats

def iter_files(directory, sample=None, manifest=None):
    valid_exts = {'.txt', '.html', '.htm', '.pdf'}
    count = 0
    if manifest:
        with open(manifest, 'r') as f:
            for line in f:
                path = line.strip()
                if os.path.isfile(path):
                    yield path
                    count += 1
                    if sample and count >= sample: break
        return

    for root, _, files in os.walk(directory):
        for fp in files:
            if os.path.splitext(fp)[1].lower() in valid_exts:
                yield os.path.join(root, fp)
                count += 1
                if sample and count >= sample: return

def main():
    parser = argparse.ArgumentParser(description="Visuddhi V4 - DGX Spark Edition")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = LOG_DIR / "dry_run_corpus.txt" if args.dry_run else OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_list = list(iter_files(INPUT_DIR, sample=args.sample, manifest=args.manifest))
    cores = cpu_count()
    
    master_stats = {
        'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 
        'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0
    }
    total_chars = 0
    total_files = 0

    print(f"Visuddhi V4 | DGX Spark")
    print(f"Cores: {cores} | Target: {len(file_list)} files", flush=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with Pool(processes=cores) as pool:
            # Chunksize 50 optimizes DGX IPC (Inter-process communication)
            pbar = tqdm(pool.imap_unordered(process_file_worker, file_list, chunksize=50), total=len(file_list))
            for content, f_stats in pbar:
                for k in master_stats:
                    master_stats[k] += f_stats.get(k, 0)
                
                if content:
                    out_f.write(content + "\n")
                    total_chars += len(content)
                
                total_files += 1
                pbar.set_postfix({"Yield": f"{total_chars/1e6:.2f}MB"})

    # --- FINAL REPORT ---
    rejection_rate = (master_stats['total_discarded'] / max(1, master_stats['total_blocks'])) * 100
    
    print("\n" + "="*45, flush=True)
    print(" LINGUISTIC PURITY REPORT ".center(45, "="), flush=True)
    print(f" Total Files Processed    : {total_files}", flush=True)
    print(f" Pure Sanskrit Yield      : {total_chars / 1e6:.2f} Million Chars", flush=True)
    print(f" Global Rejection Rate    : {rejection_rate:.2f}%", flush=True)
    print("-" * 45, flush=True)
    print(" NOISE BREAKDOWN ", flush=True)
    print(f" Marathi/Nepali Purged    : {master_stats['marathi_nepali']}", flush=True)
    print(f" Hindi/Vernacular Purged  : {master_stats['hindi']}", flush=True)
    print(f" Low Marker Density       : {master_stats['low_density']}", flush=True)
    print(f" Junk/Artifacts Purged    : {master_stats['punctuation']}", flush=True)
    print("-" * 45, flush=True)
    print(f" TOTAL BLOCKS DISCARDED   : {master_stats['total_discarded']}", flush=True)
    print("="*45 + "\n", flush=True)
    print(f"Output saved to: {output_path}", flush=True)

if __name__ == "__main__":
    main()