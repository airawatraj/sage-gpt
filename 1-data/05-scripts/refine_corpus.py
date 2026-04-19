import os
import re
import shutil
import sys
from pathlib import Path

# DGX Path Resolution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    import config
    INPUT_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"
    OUTPUT_FILE = config.PURIFIED_DATA_DIR / "corpus_refined.txt"
except ImportError:
    INPUT_FILE = Path('1-data/02-purified/corpus.txt')
    OUTPUT_FILE = Path('1-data/02-purified/corpus_refined.txt')

# Expanded Hindi/Vernacular "Poison" List
# Added common markers like 'और' (and), 'भी' (also), 'था' (was)
target_words = ['है', 'करना', 'सबहि', 'बिप्र', 'कहहु', 'और', 'भी', 'था', 'होता', 'गया']
target_pattern = re.compile(r'(?<![\u0900-\u097F])(?:' + '|'.join(target_words) + r')(?![\u0900-\u097F])')

def refine():
    lines_processed = 0
    lines_kept = 0
    lines_discarded_words = 0
    lines_discarded_awadhi = 0
    total_chars_remaining = 0

    print(f"--- SAGE-GPT LINGUISTIC SCALPEL ---")
    print(f"Targeting: {INPUT_FILE.name}")

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            lines_processed += 1
            stripped = line.strip()
            if not stripped: continue
                
            # 1. Reject specific Hindi noise words
            if target_pattern.search(stripped):
                lines_discarded_words += 1
                continue
                
            # 2. Awadhi Marker Check
            # Sanskrit 'u' usually uses the matra (ु), but Awadhi often uses the full vowel (उ) at word ends.
            # Sanskrit 'hi' (हि) is valid, but having it on 50% of words is a sign of Awadhi poetry (Ramcharitmanas style).
            words = [w.strip('।॥.,?!;:()[]{}') for w in stripped.split()]
            if not words: continue
                
            awadhi_markers = 0
            for w in words:
                # Target: Word-final 'u' vowel or 'hi/hin' vernacular endings
                if w.endswith('उ') or w.endswith('हि') or w.endswith('हिं'):
                    awadhi_markers += 1
            
            # Threshold: 50% is safer for Sanskrit to avoid killing 'hi' (indeed)
            if (awadhi_markers / len(words)) >= 0.5:
                lines_discarded_awadhi += 1
                continue
                
            # Keep line
            outfile.write(stripped + '\n')
            total_chars_remaining += len(stripped)
            lines_kept += 1

    # Atomically replace the original with the refined version
    shutil.move(str(OUTPUT_FILE), str(INPUT_FILE))
    
    print("\n--- REFINEMENT REPORT ---")
    print(f"Processed   : {lines_processed:,}")
    print(f"Hindi Noise : -{lines_discarded_words}")
    print(f"Awadhi/Brij : -{lines_discarded_awadhi}")
    print(f"Final Count : {lines_kept:,} lines")
    print(f"Final Chars : {total_chars_remaining / 1e6:.2f} MB")
    print("-------------------------\n")

if __name__ == "__main__":
    refine()