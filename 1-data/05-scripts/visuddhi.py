import os
import sys
import re
import unicodedata
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

# Maintain imports for DGX environment
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# DGX-Friendly Path Resolution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    INPUT_DIR = config.RAW_DATA_DIR
    OUTPUT_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"
    LOG_DIR = config.ROOT_DIR / "6-logs/purification"
except ImportError:
    # Fallback for standalone DGX runs
    INPUT_DIR = Path("./data/raw")
    OUTPUT_FILE = Path("./data/purified/corpus.txt")
    LOG_DIR = Path("./logs")

# --- LINGUISTIC CONSTANTS (Pre-compiled for Speed) ---
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F\u0902\u0903\s\u200c\u200d\u0964\u0965]+')
NUKTAS_PATTERN = re.compile(r'[\u0958\u0933]')
SIGNATURE_PATTERN = re.compile(r'[\u094D\u0903]')
DANDA_COLLAPSE = re.compile(r'([।॥]\s*){2,}')
M_COLLAPSE = re.compile(r'म्(?:\s+म्)+|म्{2,}')

NOISE_STOPWORDS = {
    "है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा",
    "आहे", "आणि", "पूर्ण", "करण्यासाठी", "आहेत", "होता", "होती", "असा", "तसेच", "या", "व", "काही", "झाली",
    "तस्स", "अरहतो", "सम्मा", "णमो", "हो", "यो", "र", "छ", "छन्", "थियो", "गर्नु", "मलाई", "भोक", "लाग्यो"
}
HINDI_STOP = {"है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा"}

def apply_de_echo(text):
    text = DANDA_COLLAPSE.sub('॥ ', text)
    text = M_COLLAPSE.sub('म्', text)
    return text

def clean_text_block(text):
    """
    Pure functional cleaning. No global state.
    Returns (cleaned_text, stats_dict)
    """
    stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[0-9०-९]+', '', text)
    matches = DEVANAGARI_PATTERN.findall(text)
    
    valid_blocks = []
    for m in matches:
        cleaned = re.sub(r'\s+', ' ', m).strip()
        if not cleaned: continue
        
        stats['total_blocks'] += 1
        
        # 1. Alphabetical Purity
        char_len = len(cleaned)
        devanagari_and_space = len(re.findall(r'[\u0900-\u097F\u0902\u0903\s]', cleaned))
        if (devanagari_and_space / char_len) < 0.95:
            stats['punctuation'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 2. N-Gram Blacklist
        words_set = {w.strip("।॥").strip() for w in cleaned.split()}
        if not words_set.isdisjoint({'हे', 'को', 'मा'}):
            stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 3. Sanskrit Density (Markers like Halant/Visarga)
        if char_len > 15:
            if not SIGNATURE_PATTERN.search(cleaned):
                stats['low_density'] += 1
                stats['total_discarded'] += 1
                continue
        
        # 4. Stopword Filtering
        if not words_set.isdisjoint(NOISE_STOPWORDS):
            if not words_set.isdisjoint(HINDI_STOP):
                stats['hindi'] += 1
            else:
                stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue

        if NUKTAS_PATTERN.search(cleaned):
            continue

        valid_blocks.append(apply_de_echo(cleaned))
            
    return "\n".join(valid_blocks), stats

def process_file_worker(filepath):
    """
    DGX Worker: Isolated processing with no global dependency.
    """
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    file_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    
    try:
        if ext in ['.html', '.htm'] and BeautifulSoup:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
                for script in soup(["script", "style"]): script.decompose()
                content, file_stats = clean_text_block(soup.get_text())
        elif ext == '.pdf' and fitz:
            doc = fitz.open(filepath)
            pages = [page.get_text() for page in doc]
            content, file_stats = clean_text_block("\n".join(pages))
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content, file_stats = clean_text_block(f.read())
    except Exception as e:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = LOG_DIR / "dry_run_corpus.txt" if args.dry_run else OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_generator = list(iter_files(INPUT_DIR, sample=args.sample, manifest=args.manifest))
    # On DGX, use all available cores. 
    # chunksize=50 helps reduce the cost of context switching between the Pool and the Main thread.
    cores = cpu_count()
    
    master_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    total_chars = 0

    print(f"DGX Spark: Processing {len(file_generator)} files on {cores} cores...")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with Pool(processes=cores) as pool:
            # imap_unordered is best for DGX because it doesn't care about file order, maximizing throughput
            pbar = tqdm(pool.imap_unordered(process_file_worker, file_generator, chunksize=50), total=len(file_generator))
            for content, f_stats in pbar:
                for k in master_stats:
                    master_stats[k] += f_stats[k]
                
                if content:
                    out_f.write(content + "\n")
                    total_chars += len(content)
                
                pbar.set_postfix({"Yield": f"{total_chars/1e6:.1f}MB"})

    print(f"\nPipeline Complete. Saved to {output_path}")

if __name__ == "__main__":
    main()