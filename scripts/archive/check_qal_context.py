#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import Counter

def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text

with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

normalized = normalize_arabic_text(corpus)

# Check what comes after 'قال'
chars_after = Counter()
chars_before = Counter()

pos = 0
count = 0
while True:
    idx = normalized.find('قال', pos)
    if idx < 0:
        break

    # Character before
    if idx > 0:
        chars_before[normalized[idx - 1]] += 1

    # Character after
    if idx + 2 < len(normalized):
        chars_after[normalized[idx + 2]] += 1

    count += 1
    pos = idx + 1

with open("../results/check_qal_context.txt", 'w', encoding='utf-8') as f:
    f.write(f"ANALYSIS: WHAT COMES BEFORE/AFTER 'قال'\n\n")
    f.write(f"Total qal occurrences: {count}\n\n")

    f.write(f"Top 20 characters BEFORE 'قال':\n")
    for char, cnt in chars_before.most_common(20):
        char_code = ord(char)
        is_space = char in ' \n\t'
        is_arabic = 0x0600 <= char_code <= 0x06FF
        is_punct = char in '،؛.!-—'
        f.write(f"  U+{char_code:04X} ({char!r}): {cnt:4d} (space={is_space}, arabic={is_arabic}, punct={is_punct})\n")

    f.write(f"\nTop 20 characters AFTER 'قال':\n")
    for char, cnt in chars_after.most_common(20):
        char_code = ord(char)
        is_space = char in ' \n\t'
        is_arabic = 0x0600 <= char_code <= 0x06FF
        is_punct = char in '،؛.!-—'
        f.write(f"  U+{char_code:04X} ({char!r}): {cnt:4d} (space={is_space}, arabic={is_arabic}, punct={is_punct})\n")

print("[+] Analysis written to ../results/check_qal_context.txt")
