#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déboguer la détection des verbes de transmission dans le corpus.
Sauvegarde le résultat dans un fichier pour éviter les problèmes d'encodage.
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

# Load corpus
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

normalized = normalize_arabic_text(corpus)

# Test each verb in original
found_verbs = {}
for verb in sorted(ISNAD_START_VERBS):
    count = corpus.count(verb)
    if count > 0:
        found_verbs[verb] = count

# Test each verb in normalized
found_with_boundaries = {}
for verb in sorted(ISNAD_START_VERBS):
    count = 0
    pos = 0

    while True:
        idx = normalized.find(verb, pos)
        if idx < 0:
            break

        before_ok = idx == 0 or normalized[idx - 1] in ' \n\t'
        after_ok = (idx + len(verb)) >= len(normalized) or normalized[idx + len(verb)] in ' \n\t'

        if before_ok and after_ok:
            count += 1

        pos = idx + 1

    if count > 0:
        found_with_boundaries[verb] = count

# Write report
report = Path("../results/debug_isnad_verbs.txt")
with open(report, 'w', encoding='utf-8') as f:
    f.write("DEBUG: ISNAD VERB DETECTION\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Corpus length: {len(corpus):,} chars\n")
    f.write(f"Total verbs defined: {len(ISNAD_START_VERBS)}\n")
    f.write(f"Verbs found in original: {len(found_verbs)}\n")
    f.write(f"Verbs found with boundaries: {len(found_with_boundaries)}\n\n")

    f.write("VERBS FOUND IN ORIGINAL:\n")
    f.write("-" * 80 + "\n")
    for verb, count in sorted(found_verbs.items(), key=lambda x: -x[1])[:20]:
        f.write(f"{verb}: {count} times\n")

    f.write("\n\nVERBS FOUND WITH BOUNDARIES:\n")
    f.write("-" * 80 + "\n")
    for verb, count in sorted(found_with_boundaries.items(), key=lambda x: -x[1])[:20]:
        f.write(f"{verb}: {count} times\n")

    if len(found_with_boundaries) == 0:
        f.write("\nWARNING: No verbs found with word boundaries!\n")
        f.write("Checking which verbs are in the corpus at all...\n\n")

        for verb in sorted(ISNAD_START_VERBS)[:10]:
            in_corpus = verb in corpus
            f.write(f"{verb}: {in_corpus}\n")

print("[+] Debug report written to: ../results/debug_isnad_verbs.txt")
