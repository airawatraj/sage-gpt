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

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    convert_from_path = None
    pytesseract = None

# DGX Path Resolution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    # Ensure paths are expanded for the DGX filesystem
    INPUT_DIR = Path(config.RAW_DATA_DIR).expanduser().resolve()
    OUTPUT_FILE = Path(config.PURIFIED_DATA_DIR).expanduser().resolve() / "corpus.txt"
    LOG_DIR = Path(config.LOG_DIR).expanduser().resolve() / "purification"
except ImportError:
    INPUT_DIR = Path("./1-data/01-raw")
    OUTPUT_FILE = Path("./1-data/02-purified/corpus.txt")
    LOG_DIR = Path("./6-logs/purification")

# --- LINGUISTIC CONSTANTS ---
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
    """Purifies blocks of text and returns (purified_string, stats)."""
    stats = {
        'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 
        'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0
    }
    if not text: return "", stats

    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[0-9०-९]+', '', text)
    matches = DEVANAGARI_PATTERN.findall(text)
    
    valid_blocks = []
    for m in matches:
        cleaned = re.sub(r'\s+', ' ', m).strip()
        if not cleaned: continue
        
        stats['total_blocks'] += 1
        
        # 1. Purity Filter
        char_len = len(cleaned)
        dev_and_space = len(re.findall(r'[\u0900-\u097F\u0902\u0903\s]', cleaned))
        if (dev_and_space / char_len) < 0.95:
            stats['punctuation'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 2. Blacklist Tokens
        words_set = {w.strip("।॥").strip() for w in cleaned.split()}
        if not words_set.isdisjoint({'हे', 'को', 'मा'}):
            stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue
            
        # 3. Sanskrit Marker Density
        if char_len > 15:
            if not SIGNATURE_PATTERN.search(cleaned):
                stats['low_density'] += 1
                stats['total_discarded'] += 1
                continue
        
        # 4. Stopword Filter
        if not words_set.isdisjoint(NOISE_STOPWORDS):
            if not words_set.isdisjoint(HINDI_STOP):
                stats['hindi'] += 1
            else:
                stats['marathi_nepali'] += 1
            stats['total_discarded'] += 1
            continue

        if NUKTAS_PATTERN.search(cleaned): continue

        valid_blocks.append(apply_de_echo(cleaned))
            
    return "\n".join(valid_blocks), stats

def process_pdf_file(filepath):
    """Attempts fast text extraction, fallbacks to OCR if page seems empty."""
    if not fitz: return "", {}
    try:
        doc = fitz.open(filepath)
        full_text = []
        master_f_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
        
        for page in doc:
            text = page.get_text().strip()
            
            # Trigger OCR if the page text is suspiciously short (scanned image)
            if len(text) < 50 and pytesseract and convert_from_path:
                images = convert_from_path(filepath, first_page=page.number+1, last_page=page.number+1, dpi=200)
                if images:
                    text = pytesseract.image_to_string(images[0], lang='san+hin')
            
            cleaned, p_stats = clean_text_block(text)
            for k in master_f_stats: master_f_stats[k] += p_stats[k]
            if cleaned: full_text.append(cleaned)
                
        return "\n".join(full_text), master_f_stats
    except Exception:
        return "", {}

def process_file_worker(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    f_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    
    try:
        if ext in ['.html', '.htm'] and BeautifulSoup:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
                for s in soup(["script", "style"]): s.decompose()
                content, f_stats = clean_text_block(soup.get_text())
        elif ext == '.pdf':
            content, f_stats = process_pdf_file(filepath)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content, f_stats = clean_text_block(f.read())
    except Exception:
        pass
    return content, f_stats

def iter_files(directory, sample=None):
    valid_exts = {'.txt', '.html', '.htm', '.pdf'}
    count = 0
    for root, _, files in os.walk(directory):
        for fp in files:
            if os.path.splitext(fp)[1].lower() in valid_exts:
                yield os.path.join(root, fp)
                count += 1
                if sample and count >= sample: return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_list = list(iter_files(INPUT_DIR, sample=args.sample))
    
    # DGX: Adjust processes if you start hitting RAM limits during OCR
    cores = cpu_count()
    
    master_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    total_chars = 0
    total_files = 0

    print(f"--- SAGE-GPT VISUDDHI ENGINE ---")
    print(f"Input: {INPUT_DIR}\nCores: {cores}\nTarget: {len(file_list)} files\n", flush=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        with Pool(processes=cores) as pool:
            pbar = tqdm(pool.imap_unordered(process_file_worker, file_list, chunksize=50), total=len(file_list))
            for content, stats in pbar:
                for k in master_stats: master_stats[k] += stats.get(k, 0)
                if content:
                    out_f.write(content + "\n")
                    total_chars += len(content)
                total_files += 1
                pbar.set_postfix({"Yield": f"{total_chars/1e6:.1f}MB"})

    # --- FINAL PURITY REPORT ---
    denom = max(1, master_stats['total_blocks'])
    rejection_rate = (master_stats['total_discarded'] / denom) * 100
    
    print("\n" + "="*45, flush=True)
    print(" LINGUISTIC PURITY REPORT ".center(45, "="), flush=True)
    print(f" Files Processed    : {total_files}", flush=True)
    print(f" Sanskrit Extracted : {total_chars / 1e6:.2f} Million Characters", flush=True)
    print(f" Rejection Rate     : {rejection_rate:.2f}%", flush=True)
    print("-" * 45, flush=True)
    print(f" Marathi/Nepali     : {master_stats['marathi_nepali']}", flush=True)
    print(f" Hindi/Vernacular   : {master_stats['hindi']}", flush=True)
    print(f" Low Marker Density : {master_stats['low_density']}", flush=True)
    print(f" Junk/Artifacts     : {master_stats['punctuation']}", flush=True)
    print("="*45 + "\n", flush=True)

if __name__ == "__main__":
    main()