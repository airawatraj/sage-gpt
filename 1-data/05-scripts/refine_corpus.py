import os
import re
import shutil

input_file = '1-data/02-purified/corpus.txt'
output_file = '1-data/02-purified/corpus_refined.txt'

target_words = ['है', 'करना', 'सबहि', 'बिप्र', 'कहहु']

target_pattern = re.compile(r'\b(?:' + '|'.join(target_words) + r')\b')

lines_processed = 0
lines_kept = 0
lines_discarded_words = 0
lines_discarded_awadhi = 0
total_chars_remaining = 0

print(f"Starting refinement on {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            lines_processed += 1
            stripped = line.strip()
            
            if not stripped:
                continue
                
            # 1. Reject specific noise words
            if target_pattern.search(stripped):
                lines_discarded_words += 1
                continue
                
            # 2. Reject Awadhi markers
            # Endings in 'उ' (\u0909) or 'हि' (\u0939\u093F) or 'हिं' (\u0939\u093F\u0902)
            # Actually, "उ" is U (vowel), "हि" is hi. Let's just match word ending.
            # Easiest way is to clean puntuation then check endings.
            words = [w.strip('।॥.,?!;:()[]{}') for w in stripped.split()]
            if not words:
                continue
                
            awadhi_count = 0
            for w in words:
                if w.endswith('उ') or w.endswith('हि') or w.endswith('हिं'):
                    awadhi_count += 1
            
            if (awadhi_count / len(words)) > 0.4:
                lines_discarded_awadhi += 1
                continue
                
            # Keep line
            outfile.write(stripped + '\n')
            total_chars_remaining += len(stripped)
            lines_kept += 1

    shutil.move(output_file, input_file)
    
    print("\n--- FINAL REFINEMENT REPORT ---")
    print(f"Total Lines Processed: {lines_processed}")
    print(f"Lines Discarded (Hindi words): {lines_discarded_words}")
    print(f"Lines Discarded (Awadhi markers >40%): {lines_discarded_awadhi}")
    print(f"Total Lines Remaining: {lines_kept}")
    print(f"Total Characters Remaining: {total_chars_remaining}")
    print("-------------------------------\n")
    print("Refinement Complete.")

except Exception as e:
    print(f"Error during refinement: {e}")
