"""
STAGE 2 - IMPROVED VERSION
Use these cells to replace Stage 2 in CAMELBERT_FINETUNING_COLAB.ipynb
"""

# ============================================================================
# CELL 1: Load Preprocessed Data
# ============================================================================

print("[STAGE 2] Loading preprocessed data...\n")

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
import random

# Define paths
data_dir = Path('/content/drive/MyDrive/Khabar-segmentation/data/processed')
preprocessed_dir = data_dir / 'camelbert_preprocessed'
output_dir = Path('/content/drive/MyDrive/Khabar-segmentation/data/camelbert_training')
output_dir.mkdir(parents=True, exist_ok=True)

# Load preprocessed data (high confidence only)
preprocessed_path = preprocessed_dir / 'training_data_high_conf.json'

if not preprocessed_path.exists():
    print(f"[ERROR] Preprocessed data not found at {preprocessed_path}")
    print(f"[ACTION] Make sure you ran: python scripts/preprocess_annotated_data.py")
else:
    with open(preprocessed_path, 'r', encoding='utf-8') as f:
        preprocessed_examples = json.load(f)

    print(f"[OK] Loaded {len(preprocessed_examples)} high-confidence examples\n")

    # Show statistics
    boundary_types = {}
    for ex in preprocessed_examples:
        btype = ex.get('boundary_type', 'unknown')
        boundary_types[btype] = boundary_types.get(btype, 0) + 1

    print("Boundary type distribution:")
    for btype, count in sorted(boundary_types.items(), key=lambda x: -x[1]):
        print(f"  {btype:20s}: {count:3d} ({100*count/len(preprocessed_examples):5.1f}%)")


# ============================================================================
# CELL 2: Define Functions
# ============================================================================

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text."""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # Remove all numerals (Arabic-Indic, Extended, and regular digits)
    text = re.sub(r'[\u0660-\u0669\u06F0-\u06F90-9]', '', text)
    # Normalize alef variants
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # Normalize TA marbuta
    text = text.replace('ة', 'ه')
    return text


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


def create_bio_tags_from_boundaries(full_text: str, isnad_start: int, isnad_end: int) -> Tuple[List[str], List[str]]:
    """
    Create BIO tags from explicit character boundaries.
    Much cleaner than trying to infer them!
    """
    tokens = simple_tokenize(full_text)
    bio_tags = []
    char_pos = 0
    first_isnad_token = True
    first_khabar_token = True

    for token in tokens:
        # Find token position in text
        token_start = full_text.find(token, char_pos)
        if token_start < 0:
            token_start = char_pos
        token_end = token_start + len(token)

        # Assign BIO tag based on explicit boundaries
        if token_start < isnad_end:
            # Part of isnad
            if first_isnad_token:
                bio_tags.append('B-ISNAD')
                first_isnad_token = False
            else:
                bio_tags.append('I-ISNAD')
        else:
            # Part of khabar
            if first_khabar_token:
                bio_tags.append('B-KHABAR')
                first_khabar_token = False
            else:
                bio_tags.append('I-KHABAR')

        char_pos = token_end

    return tokens, bio_tags


print("[OK] Functions defined")


# ============================================================================
# CELL 3: Create Training Examples from Preprocessed Data
# ============================================================================

print("\n[STAGE 2] Creating training examples from preprocessed boundaries...\n")

training_examples = []
failed_count = 0

for i, ex in enumerate(preprocessed_examples):
    full_text = ex['full_text']
    isnad_start = ex['isnad_start']
    isnad_end = ex['isnad_end']

    # Skip if boundaries are invalid
    if isnad_start < 0 or isnad_end <= 0:
        failed_count += 1
        continue

    # Create tokens and BIO tags from explicit boundaries
    tokens, bio_tags = create_bio_tags_from_boundaries(full_text, isnad_start, isnad_end)

    # Validate: tokens and tags must match
    if len(tokens) > 0 and len(tokens) == len(bio_tags):
        training_examples.append({
            'akhbar_num': ex['akhbar_num'],
            'tokens': tokens,
            'ner_tags': bio_tags,
            'text': ' '.join(tokens),
            'confidence': ex.get('confidence', 0.0)
        })
    else:
        failed_count += 1

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(preprocessed_examples)}")

print(f"\n[RESULTS]")
print(f"  Total extracted: {len(training_examples)}")
print(f"  Failed: {failed_count}")
print(f"  Success rate: {100 * len(training_examples) / len(preprocessed_examples):.1f}%\n")

# Show example
if training_examples:
    import random as rnd
    example = training_examples[rnd.randint(0, len(training_examples)-1)]
    print(f"[EXAMPLE #{example['akhbar_num']}]")
    print(f"  Tokens ({len(example['tokens'])}): {' '.join(example['tokens'][:15])}...")
    print(f"  Tags ({len(example['ner_tags'])}): {example['ner_tags'][:15]}...")


# ============================================================================
# CELL 4: Save Training Data as JSONL
# ============================================================================

print("\n[STAGE 2] Saving training data...\n")

# Define label mappings
label2id = {'O': 0, 'B-ISNAD': 1, 'I-ISNAD': 2, 'B-KHABAR': 3, 'I-KHABAR': 4}
id2label = {v: k for k, v in label2id.items()}

# Split data: 70% train, 15% val, 15% test
random.seed(42)
random.shuffle(training_examples)

n_train = int(0.70 * len(training_examples))
n_val = int(0.15 * len(training_examples))

train_examples = training_examples[:n_train]
val_examples = training_examples[n_train:n_train + n_val]
test_examples = training_examples[n_train + n_val:]

print(f"Train: {len(train_examples)} ({100*len(train_examples)/len(training_examples):.1f}%)")
print(f"Val:   {len(val_examples)} ({100*len(val_examples)/len(training_examples):.1f}%)")
print(f"Test:  {len(test_examples)} ({100*len(test_examples)/len(training_examples):.1f}%)\n")

# Save JSONL files
def save_jsonl(examples, path):
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            # Convert tags to IDs
            ner_tag_ids = [label2id[tag] for tag in ex['ner_tags']]
            json_line = {
                'tokens': ex['tokens'],
                'ner_tags': ner_tag_ids
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

save_jsonl(train_examples, output_dir / 'train.jsonl')
save_jsonl(val_examples, output_dir / 'val.jsonl')
save_jsonl(test_examples, output_dir / 'test.jsonl')

# Save label mappings
with open(output_dir / 'label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)

print(f"[OK] Saved training data:")
print(f"  Train: {output_dir}/train.jsonl")
print(f"  Val:   {output_dir}/val.jsonl")
print(f"  Test:  {output_dir}/test.jsonl")
print(f"  Labels: {output_dir}/label_mapping.json")
