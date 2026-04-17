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

# Test: given first occurrence of حدثنا, can we find قال after it?
verb = 'حدثنا'
verb_pos = normalized.find(verb)

with open("../results/test_qal_after_verb.txt", 'w', encoding='utf-8') as f:
    f.write(f"TEST: Find قال after {verb}\n")
    f.write(f"=" * 80 + "\n\n")

    f.write(f"First {verb} at position {verb_pos}\n\n")

    # Search for قال after it
    qal_positions = []
    search_start = verb_pos + len(verb)
    for i in range(10):
        qal_pos = normalized.find('قال', search_start)
        if qal_pos < 0:
            break
        qal_positions.append(qal_pos)
        search_start = qal_pos + 1
        if len(qal_positions) >= 10:
            break

    f.write(f"Next 10 قال positions: {qal_positions}\n\n")

    # Check first one
    if qal_positions:
        first_qal = qal_positions[0]
        length = first_qal + 2 - verb_pos
        f.write(f"First قال at {first_qal}\n")
        f.write(f"Isnad length: {length} chars\n")
        if 50 <= length <= 400:
            f.write("Status: VALID!\n")
        else:
            f.write(f"Status: OUT OF RANGE (need 50-400)\n")

print("[+] Test written")
