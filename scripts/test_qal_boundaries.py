#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text

with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

normalized = normalize_arabic_text(corpus)

# Count total 'قال' occurrences
total_qal = normalized.count('قال')

# Count with word boundaries
valid_qal = 0
first_valid_pos = -1
pos = 0
while True:
    idx = normalized.find('قال', pos)
    if idx < 0:
        break

    before_ok = idx == 0 or normalized[idx - 1] in ' \n\t'
    if (idx + 2) >= len(normalized):
        after_ok = True
    else:
        next_char = normalized[idx + 2]
        after_ok = next_char in ' \n\t،؛.!-—'
        if not after_ok:
            char_code = ord(next_char)
            is_arabic_letter = (0x0600 <= char_code <= 0x06FF)
            after_ok = not is_arabic_letter

    if before_ok and after_ok:
        valid_qal += 1
        if first_valid_pos < 0:
            first_valid_pos = idx

    pos = idx + 1

with open("../results/test_qal_boundaries.txt", 'w', encoding='utf-8') as f:
    f.write(f"TEST: QAL WORD BOUNDARIES\n\n")
    f.write(f"Total qal occurrences (simple count): {total_qal}\n")
    f.write(f"With valid word boundaries: {valid_qal}\n")
    f.write(f"Percentage: {valid_qal/total_qal*100:.1f}%\n")
    f.write(f"First valid position: {first_valid_pos}\n")

print("[+] Written to ../results/test_qal_boundaries.txt")
