#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tester la détection des isnads avec la fonction du baseline v2.
"""

import re
from pathlib import Path

def normalize_arabic_text(text: str) -> str:
    """Normaliser le texte arabe."""
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

def find_all_isnad_starts(text: str):
    """Test version of isnad start detection."""
    normalized = normalize_arabic_text(text)
    isnad_starts = []

    for verb in sorted(ISNAD_START_VERBS, key=len, reverse=True):
        pos = 0
        while True:
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

    unique_starts = []
    last_end = -1
    for start_pos, verb in isnad_starts:
        if start_pos >= last_end:
            unique_starts.append((start_pos, verb))
            last_end = start_pos + len(verb)

    return unique_starts

# Load and test
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

results = find_all_isnad_starts(corpus)

# Save report
with open("../results/test_isnad_detection.txt", 'w', encoding='utf-8') as f:
    f.write(f"ISNAD Detection Test\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"Corpus size: {len(corpus):,} chars\n")
    f.write(f"Isnads found: {len(results)}\n")
    f.write(f"Expected: ~613\n\n")

    if len(results) > 0:
        f.write(f"First 20 results:\n")
        for i, (pos, verb) in enumerate(results[:20]):
            f.write(f"{i+1}. Pos={pos}, Verb={verb}\n")

        f.write(f"\nLast 10 results:\n")
        for i, (pos, verb) in enumerate(results[-10:]):
            f.write(f"{len(results)-9+i}. Pos={pos}, Verb={verb}\n")
    else:
        f.write("ERROR: No isnads found!\n")

print("[+] Report written to ../results/test_isnad_detection.txt")
