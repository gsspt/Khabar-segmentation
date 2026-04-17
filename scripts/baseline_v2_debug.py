#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug version of baseline v2 to see where it's failing.
"""

import re
from pathlib import Path

def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text

ISNAD_START_VERBS = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'حدثت', 'حدثوا',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'أخبرت', 'أخبروا',
    'سمعت', 'سمعنا', 'سمعه', 'سمعها', 'سمعهم',
    'روى', 'رويت', 'روينا', 'رواه', 'روا',
    'أنبأ', 'أنبأني', 'أنبأنا', 'أنبأه', 'أنبأتنا',
    'أنشدني', 'أنشدنا',
    'وأخبرنا', 'وحدثنا', 'وسمعت',
}

ISNAD_MIN_LENGTH = 10
ISNAD_MAX_LENGTH = 410

# Load corpus
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

normalized = normalize_arabic_text(corpus)

# Step 1: Find isnad starts
isnad_starts = []
for verb in sorted(ISNAD_START_VERBS, key=len, reverse=True):
    pos = 0
    while pos < len(normalized):
        idx = normalized.find(verb, pos)
        if idx < 0:
            break

        before_ok = idx == 0 or normalized[idx - 1] in ' \n\t'
        if (idx + len(verb)) >= len(normalized):
            after_ok = True
        else:
            next_char = normalized[idx + len(verb)]
            after_ok = next_char in ' \n\t،؛.!-—'
            if not after_ok:
                char_code = ord(next_char)
                is_arabic_letter = (0x0600 <= char_code <= 0x06FF)
                after_ok = not is_arabic_letter

        if before_ok and after_ok:
            isnad_starts.append((idx, verb))

        pos = idx + 1

isnad_starts.sort(key=lambda x: (x[0], -len(x[1])))
unique = []
last_end = -1
for start_pos, verb in isnad_starts:
    if start_pos >= last_end:
        unique.append((start_pos, verb))
        last_end = start_pos + len(verb)

# Step 2: For each start, find the end (قال)
successful = 0
for i, (start_pos, verb) in enumerate(unique[:50]):  # Test first 50
    search_end = min(start_pos + ISNAD_MAX_LENGTH, len(normalized))
    pos = start_pos

    found_qal = False
    while pos < search_end:
        idx = normalized.find('قال', pos)
        if idx < 0 or idx >= search_end:
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
            isnad_len = idx + 2 - start_pos
            if ISNAD_MIN_LENGTH <= isnad_len <= ISNAD_MAX_LENGTH:
                successful += 1
                found_qal = True
                break

        pos = idx + 1

# Write report
with open("../results/baseline_v2_debug.txt", 'w', encoding='utf-8') as f:
    f.write(f"BASELINE V2 DEBUG\n\n")
    f.write(f"Step 1: Find isnad starts\n")
    f.write(f"  Total starts found: {len(unique)}\n\n")
    f.write(f"Step 2: Find isnad ends (qal)\n")
    f.write(f"  Tested first 50 starts\n")
    f.write(f"  Successful (found valid qal): {successful}\n")
    f.write(f"  Success rate: {successful/50*100:.0f}%\n")
    f.write(f"  Estimated total with this rate: {int(len(unique) * successful // 50)}\n\n")
    f.write(f"CONCLUSION:\n")
    if successful > 40:
        f.write("  Very high success rate - most starts should find valid qal\n")
    elif successful > 20:
        f.write("  Good success rate - majority should work\n")
    else:
        f.write("  Low success rate - many starts may fail\n")

print("[+] Debug report written to ../results/baseline_v2_debug.txt")

