import os
import random
import glob

raw_dir = '1-data/01-raw'
all_files = glob.glob(os.path.join(raw_dir, '**', '*'), recursive=True)
valid_files = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(('.txt', '.html', '.htm', '.pdf'))]
sample_files = random.sample(valid_files, min(25, len(valid_files)))

synthetic_texts = [
    'यह आम है', # Hindi
    'तो मुलगा आहे', # Marathi
    'मलाई भोक लाग्यो', # Nepali
    'यो हो', # Nepali
    'यह पुस्तक है' # Hindi
] # 5 lines total

os.makedirs('1-data/01-raw/fortress_tests', exist_ok=True)
poison_file = '1-data/01-raw/fortress_tests/integrated_poison.txt'
with open(poison_file, 'w', encoding='utf-8') as f:
    for line in synthetic_texts:
        f.write(f"{line}\n")

sample_files.append(poison_file)

os.makedirs('6-logs/purification', exist_ok=True)
manifest_path = '6-logs/purification/integrated_stress_manifest.txt'
with open(manifest_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(sample_files))

print(f"Created {manifest_path} with 25 random files + 1 soup poison file (5 total lines).")
