#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligner les akhbars annotés avec le texte OpenITI en utilisant fuzzy matching au niveau des tokens.
Approche : tokeniser par mots, comparer avec difflib.SequenceMatcher, accepter >90% de similarité.
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher


def remove_arabic_diacritics(text: str) -> str:
    """Enlever les diacritiques arabes."""
    arabic_diacritics = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
        '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
        '\u0655', '\u0656', '\u0657', '\u0658', '\u0670',
    ]
    for diacritic in arabic_diacritics:
        text = text.replace(diacritic, '')
    return text


def tokenize_arabic(text: str) -> list:
    """Tokeniser le texte arabe en mots."""
    # Enlever diacritiques d'abord
    text = remove_arabic_diacritics(text)
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Split par espaces et ponctuation
    tokens = re.split(r'[\s\-\.,\?!;:\(\)]+', text)
    # Garder seulement les tokens non-vides
    return [t for t in tokens if t.strip()]


def extract_akhbar_text(akhbar: dict) -> str:
    """Extraire le texte complet d'un akhbar."""
    segments = akhbar.get("content", {}).get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def find_akhbar_in_text(akhbar_tokens: list, text_tokens: list, min_similarity: float = 0.90) -> tuple:
    """
    Chercher une séquence de tokens dans le texte avec fuzzy matching.
    Retourne (found, match_quality, start_idx, end_idx) ou (False, 0, -1, -1)
    """
    akhbar_len = len(akhbar_tokens)

    if akhbar_len == 0:
        return (False, 0, -1, -1)

    best_match = (False, 0, -1, -1)

    # Chercher toutes les fenêtres de la même longueur
    for i in range(len(text_tokens) - akhbar_len + 1):
        window = text_tokens[i:i + akhbar_len]

        # Comparer avec SequenceMatcher
        matcher = SequenceMatcher(None, akhbar_tokens, window)
        ratio = matcher.ratio()

        if ratio > best_match[1]:
            best_match = (ratio >= min_similarity, ratio, i, i + akhbar_len)

    return best_match


def main():
    """Aligner les akhbars avec fuzzy matching au niveau des tokens."""
    # Charger annotations
    with open("data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
        annot = json.load(f)

    # Charger texte nettoyé
    with open("data/processed/0406IbnHabib_cleaned.txt", 'r', encoding='utf-8') as f:
        cleaned_text = f.read()

    # Tokeniser le texte nettoyé
    text_tokens = tokenize_arabic(cleaned_text)
    print(f"Texte nettoyé: {len(text_tokens)} tokens\n")

    print("=" * 70)
    print("ALIGNEMENT FUZZY AU NIVEAU DES TOKENS")
    print("=" * 70 + "\n")

    found_high = 0  # >95%
    found_medium = 0  # 90-95%
    found_low = 0  # <90%
    not_found = []
    boundaries = []  # Pour tracer les limites des akhbars

    akhbars = annot.get("akhbar", [])
    print(f"Alignant {len(akhbars)} akhbars...\n")

    last_end_idx = 0

    for idx, akhbar in enumerate(akhbars):
        akhbar_num = akhbar.get("num", idx + 1)
        akhbar_text = extract_akhbar_text(akhbar)

        if not akhbar_text.strip():
            continue

        # Tokeniser l'akhbar
        akhbar_tokens = tokenize_arabic(akhbar_text)

        if not akhbar_tokens:
            not_found.append((akhbar_num, len(akhbar_text), 0))
            continue

        # Chercher dans le texte
        found, similarity, start_idx, end_idx = find_akhbar_in_text(akhbar_tokens, text_tokens, min_similarity=0.85)

        if found and similarity >= 0.95:
            found_high += 1
            boundaries.append((start_idx, end_idx, akhbar_num))
            if found_high <= 10:
                print(f"[EXCELLENT {similarity:.1%}] Akhbar {akhbar_num}")
        elif found and similarity >= 0.90:
            found_medium += 1
            boundaries.append((start_idx, end_idx, akhbar_num))
            if found_medium <= 5:
                print(f"[GOOD {similarity:.1%}] Akhbar {akhbar_num}")
        elif similarity >= 0.85:  # Presque trouvé
            found_low += 1
            boundaries.append((start_idx, end_idx, akhbar_num))
            print(f"[PARTIAL {similarity:.1%}] Akhbar {akhbar_num}")
        else:
            not_found.append((akhbar_num, len(akhbar_text), similarity))

    print(f"\n{'=' * 70}")
    print(f"RÉSULTATS")
    print(f"{'=' * 70}")
    print(f"\nExcellent (>95%): {found_high}/{len(akhbars)}")
    print(f"Bon (90-95%): {found_medium}/{len(akhbars)}")
    print(f"Partial (85-90%): {found_low}/{len(akhbars)}")
    print(f"Non trouvés (<85%): {len(not_found)}/{len(akhbars)}")

    total_found = found_high + found_medium + found_low
    print(f"\nTOTAL ALIGNÉS: {total_found}/{len(akhbars)} ({100*total_found/len(akhbars):.1f}%)")

    if not_found:
        print(f"\nExemples non trouvés:")
        for num, length, sim in not_found[:5]:
            print(f"  - Akhbar {num} ({length} chars, similarity={sim:.2%})")

    # Sauvegarder les boundaries pour validation future
    if total_found > 100:
        with open("results/akhbar_boundaries.json", 'w', encoding='utf-8') as f:
            json.dump(boundaries, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Boundaries sauvegardés dans results/akhbar_boundaries.json")


if __name__ == "__main__":
    main()
