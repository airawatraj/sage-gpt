import os
import sys
import re
import unicodedata
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

import config

# Configuration
INPUT_DIR = config.RAW_DATA_DIR
OUTPUT_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"
# Arbitrary length filters removed to protect short sutras

# Telemetry Initialization
REJECTION_STATS = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0}

# Linguistic Regex Filters
# Explicitly includes Anusvara (\u0902) & Visarga (\u0903) within \u0900-\u097F
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F\u0902\u0903\s\u200c\u200d\u0964\u0965]+')
NUKTAS_PATTERN = re.compile(r'[\u0958\u0933]')  # Rejects modern modifications (क़ and ळ)
SIGNATURE_PATTERN = re.compile(r'[\u094D\u0903]')  # Requires Virama (Halant) or Visarga

NOISE_STOPWORDS = {
    # Hindi
    "है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा",
    # Marathi
    "आहे", "आणि", "पूर्ण", "करण्यासाठी", "आहेत", "होता", "होती", "असा", "तसेच", "या", "व", "काही", "झाली",
    # Pali/Prakrit
    "तस्स", "अरहतो", "सम्मा", "णमो",
    # Nepali/Awadhi/Bhojpuri
    "हो", "यो", "र", "छ", "छन्", "थियो", "गर्नु", "मलाई", "भोक", "लाग्यो"
}

def apply_de_echo(text):
    """
    Collapses repeated Dandas or terminal 'म्'.
    """
    text = re.sub(r'([।॥]\s*){2,}', '॥ ', text)
    text = re.sub(r'म्(?:\s+म्)+', 'म्', text)
    text = re.sub(r'म्{2,}', 'म्', text)
    return text

def clean_text_block(text):
    """
    Extracts pure Devanagari blocks and subjects them to strict grammatical filters.
    """
    global REJECTION_STATS
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[0-9०-९]+', '', text)
    matches = DEVANAGARI_PATTERN.findall(text)
    
    valid_blocks = []
    for m in matches:
        cleaned = re.sub(r'\s+', ' ', m).strip()
        
        if len(cleaned) == 0:
            continue
            
        REJECTION_STATS['total_blocks'] += 1
            
        # --- OMEGA GUARD TRIPLE-DEFENSE ---
        
        # 1. Alphabetical Purity (>95% Devanagari to total characters)
        devanagari_count = len(re.findall(r'[\u0900-\u097F\u0902\u0903]', cleaned))
        devanagari_and_space_count = len(re.findall(r'[\u0900-\u097F\u0902\u0903\s]', cleaned))
        if (devanagari_and_space_count / len(cleaned)) < 0.95:
            REJECTION_STATS['punctuation'] += 1
            REJECTION_STATS['total_discarded'] += 1
            continue
            
        # 2. N-Gram Blacklist (Rejecting explicit Marathi/Nepali/Hindi Bigrams/Tokens)
        ngram_blacklist = {'हे', 'को', 'मा'}
        words_set = {w.strip("।॥").strip() for w in cleaned.split()}
        if not words_set.isdisjoint(ngram_blacklist):
            REJECTION_STATS['marathi_nepali'] += 1
            REJECTION_STATS['total_discarded'] += 1
            continue
            
        # 3. Density Analysis (Ratio over Threshold)
        # Reject any block where len(cleaned) > 15 and sanskrit_markers == 0
        if len(cleaned) > 15:
            sanskrit_markers = len(re.findall(r'[\u0903\u094D]', cleaned))
            if sanskrit_markers == 0:
                REJECTION_STATS['low_density'] += 1
                REJECTION_STATS['total_discarded'] += 1
                continue
                
        # --- END OMEGA GUARD ---
        
        # 0. Reject lines with no Devanagari alphabets (Orphaned Punctuation)
        if not any('\u0905' <= char <= '\u0939' for char in cleaned):
            REJECTION_STATS['punctuation'] += 1
            REJECTION_STATS['total_discarded'] += 1
            continue

        # 0.5 Reject Fragments: < 4 words AND no Danda
        if len(cleaned.split()) < 4 and not re.search(r'[।॥]', cleaned):
            REJECTION_STATS['low_density'] += 1
            REJECTION_STATS['total_discarded'] += 1
            continue
        
        # 1. Reject Noise Stopwords (standalone tokens)
        if not words_set.isdisjoint(NOISE_STOPWORDS):
            hindi_stop = {"है", "था", "थी", "थे", "रहा", "रही", "रहे", "ने", "को", "का", "के", "की", "ला", "होना", "गया", "लिए", "में", "से", "हुए", "कयने", "कयती", "तथा"}
            if not words_set.isdisjoint(hindi_stop):
                REJECTION_STATS['hindi'] += 1
            else:
                REJECTION_STATS['marathi_nepali'] += 1
            REJECTION_STATS['total_discarded'] += 1
            continue
            
        # 2. Reject modern vernacular noise
        if NUKTAS_PATTERN.search(cleaned):
            continue
            
        # 3. Apply De-Echo formatting
        cleaned = apply_de_echo(cleaned)
        
        valid_blocks.append(cleaned)
            
    return "\n".join(valid_blocks)

def process_html_file(filepath):
    if not BeautifulSoup:
        return ""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            return clean_text_block(text)
    except Exception as e:
        print(f"[ERROR] Failed to parse HTML: {filepath} ({e})")
        return ""

def process_pdf_file(filepath):
    if not fitz:
        return ""
    try:
        doc = fitz.open(filepath)
        full_text = []
        for page in doc:
            text = page.get_text()
            cleaned = clean_text_block(text)
            if cleaned:
                full_text.append(cleaned)
        return "\n".join(full_text)
    except Exception as e:
        print(f"[ERROR] Failed to parse PDF: {filepath} ({e})")
        return ""

def process_txt_file(filepath):
    try:
        valid_blocks = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                cleaned = clean_text_block(line)
                if cleaned:
                    valid_blocks.append(cleaned)
        return "\n".join(valid_blocks)
    except Exception as e:
        print(f"[ERROR] Failed to parse TXT: {filepath} ({e})")
        return ""

def process_file_worker(filepath):
    """
    Worker function for processing a single file.
    Returns the deeply purified text string.
    """
    global REJECTION_STATS
    REJECTION_STATS = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    if ext in ['.html', '.htm']:
        content = process_html_file(filepath)
    elif ext == '.pdf':
        content = process_pdf_file(filepath)
    elif ext == '.txt':
        content = process_txt_file(filepath)
        
    return content, REJECTION_STATS

def iter_files(directory, sample=None, manifest=None):
    """
    Dynamically yields file paths using os.walk to protect RAM.
    If sample is provided, yields at most `sample` files.
    """
    valid_exts = {'.txt', '.html', '.htm', '.pdf'}
    count = 0
    
    if manifest:
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and os.path.isfile(line):
                    yield line
                    count += 1
                    if sample is not None and count >= sample:
                        return
        return

    for root, _, files in os.walk(directory):
        for fp in files:
            if os.path.splitext(fp)[1].lower() in valid_exts:
                yield os.path.join(root, fp)
                count += 1
                if sample is not None and count >= sample:
                    return

def main():
    parser = argparse.ArgumentParser(description="Visuddhi V4 Linguistic Pipeline")
    parser.add_argument("--sample", type=int, default=None, help="Process only a sample of N files")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest file of raw files to process")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run and write to 6-logs/purification instead of standard output")
    args = parser.parse_args()

    print(f"Starting Visuddhi V4 Linguistic Pipeline")
    print(f"Input Directory: {INPUT_DIR}")
    
    if args.dry_run:
        output_file = config.ROOT_DIR / "6-logs/purification" / "dry_run_corpus.txt"
        print(f"Mode: DRY-RUN")
    else:
        output_file = OUTPUT_FILE
        
    print(f"Output File: {output_file}")
    print(f"Filters: No Nuktas, De-Echo Active (Length & Signature filters removed)")
    
    file_generator = iter_files(INPUT_DIR, sample=args.sample, manifest=args.manifest)
    cores = cpu_count()
    print(f"Firing up {cores} CPU cores for deep language purification...")
    
    total_chars = 0
    total_files = 0
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    master_stats = {'marathi_nepali': 0, 'hindi': 0, 'low_density': 0, 'punctuation': 0, 'total_discarded': 0, 'total_blocks': 0}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with Pool(processes=cores) as pool:
            pbar = tqdm(pool.imap_unordered(process_file_worker, file_generator, chunksize=10), total=args.sample)
            for content, file_stats in pbar:
                for k in master_stats:
                    master_stats[k] += file_stats[k]
                    
                if content:
                    out_f.write(content + "\n")
                    total_chars += len(content)
                total_files += 1
                
                # Update visual telemetry
                top_noise = max((k for k in master_stats if k not in ['total_discarded', 'total_blocks']), key=lambda k: master_stats[k], default='none')
                pbar.set_postfix({
                    "Sa_Ch": f"{total_chars/1e6:.1f}M", 
                    "Soup_Purged": master_stats['total_discarded'],
                    "Top_Noise": top_noise
                })
                
                if args.sample is not None and total_files >= args.sample:
                    pool.terminate()
                    break

    print(f"\nVisuddhi V4 Complete!")
    print(f"Files Processed:   {total_files}")
    print(f"Total Extracted:   {total_chars / 1e6:.2f} Million Characters")
    print(f"Saved to:          {output_file}")
    
    rejection_rate = (master_stats['total_discarded'] / max(1, master_stats['total_blocks'])) * 100
    
    print("\n" + "="*40)
    print(" LINGUISTIC PURITY REPORT".center(40))
    print("="*40)
    print(" --- SANSKRIT YIELD ---")
    print(f" Pure Sanskrit Extracted  : {total_chars / 1e6:.2f} Million Characters")
    print(f" Linguistic Rejection Rate: {rejection_rate:.2f}%")
    print("-" * 40)
    print(" --- NOISE REJECTED ---")
    print(f" Marathi/Nepali Discarded : {master_stats['marathi_nepali']}")
    print(f" Hindi Discarded          : {master_stats['hindi']}")
    print(f" Low Density (Vernacular) : {master_stats['low_density']}")
    print(f" Punctuation/Artifacts    : {master_stats['punctuation']}")
    print("-" * 40)
    print(f" TOTAL SOUP PURGED        : {master_stats['total_discarded']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
