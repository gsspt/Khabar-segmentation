#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Afficher des exemples concrets d'erreurs de frontière.
"""

import json
import re
from pathlib import Path
from baseline_isnad_segmentation import (
    normalize_arabic_text, simple_tokenize, is_isnad_verb,
    detect_isnad_span, detect_khabar_end
)


def load_reference_data():
    """Charger le corpus et les boundaries de référence."""
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        boundaries = json.load(f)

    return corpus, boundaries


def segment_akhbars_with_positions(text: str, strict: bool = True):
    """Segmente avec tracking des positions."""
    normalized = normalize_arabic_text(text)
    tokens = simple_tokenize(normalized)

    akhbars = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if is_isnad_verb(token, strict=strict):
            isnad_start, isnad_end = detect_isnad_span(tokens, i)
            khabar_end = detect_khabar_end(tokens, isnad_end)

            akhbars.append({
                'start_token': isnad_start,
                'end_token': khabar_end,
                'tokens': tokens[isnad_start:khabar_end + 1],
            })

            i = khabar_end + 1
        else:
            i += 1

    return akhbars, tokens


def find_text_in_corpus(text_tokens, corpus, start_search=0):
    """Chercher une séquence dans le corpus."""
    pattern = r'\s*'.join(re.escape(token) for token in text_tokens[:min(5, len(text_tokens))])
    match = re.search(pattern, corpus[start_search:])
    if match:
        start_idx = start_search + match.start()
        last_token = text_tokens[-1]
        end_idx = corpus.find(last_token, start_idx)
        if end_idx >= 0:
            return start_idx, end_idx + len(last_token)
    return -1, -1


def main():
    """Analyser et sauvegarder les exemples."""
    print("[*] Chargement du corpus et segmentation...")
    corpus, ref_boundaries = load_reference_data()
    akhbars, tokens = segment_akhbars_with_positions(corpus, strict=True)

    print(f"[*] Corpus: {len(corpus):,} chars")
    print(f"[*] Référence: {len(ref_boundaries)} boundaries")
    print(f"[*] Détectés: {len(akhbars)} boundaries")
    print(f"[*] Recherche des matches...")

    # Chercher des exemples
    current_search_pos = 0
    examples_good = []
    examples_bad = []

    for det_idx, det_akh in enumerate(akhbars[:150]):
        start, end = find_text_in_corpus(det_akh['tokens'], corpus, current_search_pos)

        if start >= 0:
            # Chercher le match le plus proche
            best_ref_idx = -1
            best_error = float('inf')

            for ref_idx, ref in enumerate(ref_boundaries):
                ref_start = ref['start']
                ref_end = ref['end']
                error = abs(start - ref_start) + abs(end - ref_end)

                if error < best_error and error < 500:
                    best_error = error
                    best_ref_idx = ref_idx

            if best_ref_idx >= 0:
                ref = ref_boundaries[best_ref_idx]
                start_error = start - ref['start']
                end_error = end - ref['end']

                example = {
                    'det_idx': det_idx,
                    'ref_idx': best_ref_idx,
                    'det_pos': (start, end),
                    'ref_pos': (ref['start'], ref['end']),
                    'start_error': start_error,
                    'end_error': end_error,
                    'ref_text': corpus[ref['start']:min(ref['end'], ref['start'] + 200)],
                    'det_text': corpus[start:min(end, start + 200)],
                }

                if abs(start_error) <= 20 and abs(end_error) <= 20:
                    examples_good.append(example)
                elif abs(start_error) > 50 or abs(end_error) > 50:
                    examples_bad.append(example)

            current_search_pos = end

    print(f"[*] Bons matches: {len(examples_good)}")
    print(f"[*] Mauvais matches: {len(examples_bad)}")

    # Écrire le rapport dans un fichier
    report_path = Path("../results/boundary_examples.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Exemples de Boundaries: Référence vs Détection\n\n")
        f.write(f"Corpus: {len(corpus):,} chars\n")
        f.write(f"Référence: {len(ref_boundaries)} boundaries\n")
        f.write(f"Détectés: {len(akhbars)} boundaries\n\n")
        f.write(f"Bons matches (erreur ≤ 20): {len(examples_good)}\n")
        f.write(f"Mauvais matches (erreur > 50): {len(examples_bad)}\n\n")

        if examples_good:
            f.write("## Exemples de BON Match\n\n")
            for i, ex in enumerate(examples_good[:5]):
                f.write(f"### Match #{i+1}\n\n")
                f.write(f"- Détecté #{ex['det_idx']}, Référence #{ex['ref_idx']}\n")
                f.write(f"- START error: {ex['start_error']:+d} chars\n")
                f.write(f"- END error: {ex['end_error']:+d} chars\n")
                f.write(f"- Longueur: Ref {ex['ref_pos'][1] - ex['ref_pos'][0]:,} vs Det {ex['det_pos'][1] - ex['det_pos'][0]:,}\n\n")

        if examples_bad:
            f.write("## Exemples de MAUVAIS Match (erreurs importantes)\n\n")
            for i, ex in enumerate(examples_bad[:10]):
                f.write(f"### Match #{i+1}\n\n")
                f.write(f"- Détecté #{ex['det_idx']}, Référence #{ex['ref_idx']}\n")
                f.write(f"- START error: {ex['start_error']:+d} chars\n")
                f.write(f"- END error: {ex['end_error']:+d} chars\n")
                f.write(f"- Longueur: Ref {ex['ref_pos'][1] - ex['ref_pos'][0]:,} vs Det {ex['det_pos'][1] - ex['det_pos'][0]:,}\n")
                f.write(f"- Length error: {(ex['det_pos'][1] - ex['det_pos'][0]) - (ex['ref_pos'][1] - ex['ref_pos'][0]):+d} chars\n\n")

    print(f"[*] Rapport sauvegardé: {report_path}\n")


if __name__ == "__main__":
    main()
