#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline v2: Fokus auf ISNAD-Erkennung, nicht KHABAR-Ende.

Strategie:
1. Finde ALLE Anfänge von Isnads (Transmissionsverben)
2. Finde das ENDE jedes Isnads (letztes 'قال')
3. Khabar = Text zwischen letztem قال eines Isnads und Beginn des nächsten Isnads

Diese Strategie ermöglicht es uns:
- Nicht zu fragen "wo endet der khabar?" (schwer)
- Sondern "wo beginnt der nächste isnad?" (leicht)
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys


# ============================================================================
# TRANSMISSION VERBS - EXPANDED and REFINED
# ============================================================================

# Verbes de transmission fiables au DÉBUT d'isnad
ISNAD_START_VERBS = {
    # Hadith transmission verbs
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'حدثت', 'حدثوا',

    # Akhbar transmission verbs
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'أخبرت', 'أخبروا',

    # Audio transmission
    'سمعت', 'سمعنا', 'سمعه', 'سمعها', 'سمعهم',

    # Narration
    'روى', 'رويت', 'روينا', 'رواه', 'روا',

    # Reporting
    'أنبأ', 'أنبأني', 'أنبأنا', 'أنبأه', 'أنبأتنا',

    # Other reliable ones from analysis
    'أنشدني', 'أنشدنا',

    # With prefixes
    'وأخبرنا', 'وحدثنا', 'وسمعت',
}

# Markers that END isnads (transition to khabar)
ISNAD_END_MARKERS = {
    'قال',      # "said" - primary marker
    'فقال',     # "and said"
}

# Bounds for isnad length (in chars)
# From analysis: isnads range from 3 to 407, median 129
# But we need minimum to filter out false positives
# 20 chars allows single-word isnad + verb
ISNAD_MIN_LENGTH = 10
ISNAD_MAX_LENGTH = 410


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def normalize_arabic_text(text: str) -> str:
    """Normaliser le texte arabe (diacritiques, variantes)."""
    # Supprimer les diacritiques (tashkeel)
    text = re.sub(r'[\u064B-\u065F]', '', text)

    # Variantes de alif
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # Ha à la fin
    text = text.replace('ة', 'ه')

    return text


def simple_tokenize(text: str) -> List[str]:
    """Tokeniser le texte simplement (sur whitespace)."""
    return text.split()


# ============================================================================
# ISNAD DETECTION - NEW STRATEGY
# ============================================================================

def find_all_isnad_starts(text: str) -> List[Tuple[int, int]]:
    """
    Trouver TOUTES les positions où un isnad commence.

    Retourne une liste de (position_debut, verbe_trouve).
    """
    normalized = normalize_arabic_text(text)
    isnad_starts = []

    # Chercher chaque verbe de transmission
    for verb in sorted(ISNAD_START_VERBS, key=len, reverse=True):  # Longer verbs first
        # Chercher toutes les occurrences
        pos = 0
        while True:
            idx = normalized.find(verb, pos)
            if idx < 0:
                break

            # Vérifier que c'est un mot complet (moins strict)
            # Before: must be start of string or whitespace
            before_ok = idx == 0 or normalized[idx - 1] in ' \n\t'

            # After: must be end of string OR whitespace OR punctuation
            # (not another Arabic letter)
            if (idx + len(verb)) >= len(normalized):
                after_ok = True
            else:
                next_char = normalized[idx + len(verb)]
                # OK if whitespace, punctuation, or most non-letter chars
                after_ok = next_char in ' \n\t،؛.!-—'
                # Also OK if it's not an Arabic letter
                if not after_ok:
                    # Check if next char is an Arabic letter
                    char_code = ord(next_char)
                    is_arabic_letter = (0x0600 <= char_code <= 0x06FF)
                    after_ok = not is_arabic_letter

            if before_ok and after_ok:
                isnad_starts.append((idx, verb))

            pos = idx + 1

    # Remove duplicates and overlaps (longer matches take precedence)
    # Sort by position, then by verb length (descending)
    isnad_starts.sort(key=lambda x: (x[0], -len(x[1])))

    # Remove overlapping matches
    unique_starts = []
    last_end = -1
    for start_pos, verb in isnad_starts:
        if start_pos >= last_end:
            unique_starts.append((start_pos, verb))
            last_end = start_pos + len(verb)

    return unique_starts


def find_isnad_end(text: str, start_pos: int) -> int:
    """
    Trouver la fin d'un isnad en cherchant le PREMIER 'قال' ou ses variantes après start_pos.

    Note: 'قال' dans le corpus est souvent suivi par d'autres lettres (قالت, قالوا, etc)
    So on cherche juste la première occurrence de 'قال'.
    """
    normalized = normalize_arabic_text(text)

    # Chercher 'قال' à partir de start_pos, mais pas trop loin
    search_end = min(start_pos + ISNAD_MAX_LENGTH, len(normalized))

    # Chercher simplement la première occurrence - pas de vérification de frontières
    # car 'قال' est généralement suivi d'autres lettres dans le corpus
    idx = normalized.find('قال', start_pos)

    if idx >= 0 and idx < search_end:
        # Retourner la position après 'قال'
        # Note: peut être 'قالت', 'قالوا', etc, mais on compte jusqu'après la racine 'قال'
        return idx + 2

    return -1


def segment_akhbars_v2(text: str) -> List[Dict]:
    """
    Segmente basée sur la détection d'isnads et transitions vers khbars.

    Stratégie:
    1. Trouver tous les débuts d'isnads
    2. Pour chaque début, trouver la fin (premier 'قال')
    3. Le khabar commence après ce 'قال' et finit au prochain isnad (ou fin du texte)
    """
    isnad_starts = find_all_isnad_starts(text)

    if not isnad_starts:
        return []

    akhbars = []
    covered_up_to = 0  # Track where we've already segmented

    for i, (isnad_start_pos, verb) in enumerate(isnad_starts):
        # Skip if this isnad overlaps with previous one
        if isnad_start_pos < covered_up_to:
            continue

        # Trouver la fin de cet isnad (premier 'قال')
        isnad_end_pos = find_isnad_end(text, isnad_start_pos)

        if isnad_end_pos < 0:
            continue

        # Valider la longueur de l'isnad
        isnad_length = isnad_end_pos - isnad_start_pos
        if isnad_length < ISNAD_MIN_LENGTH or isnad_length > ISNAD_MAX_LENGTH:
            continue

        # Chercher le début du prochain isnad (pour délimiter le khabar)
        khabar_end_pos = len(text)
        for j in range(i + 1, len(isnad_starts)):
            next_start = isnad_starts[j][0]
            if next_start > isnad_end_pos:  # Must be after current isnad ends
                khabar_end_pos = next_start
                break

        # Minimum khabar length
        khabar_length = khabar_end_pos - isnad_end_pos
        if khabar_length < 20:  # At least some content
            continue

        # Extraire isnad et khabar
        isnad_text = text[isnad_start_pos:isnad_end_pos].strip()
        khabar_text = text[isnad_end_pos:khabar_end_pos].strip()

        if not isnad_text or not khabar_text:
            continue

        akhbars.append({
            'isnad': isnad_text,
            'khabar': khabar_text,
            'text': isnad_text + ' ' + khabar_text,
            'start': isnad_start_pos,
            'end': khabar_end_pos,
            'isnad_end': isnad_end_pos,
        })

        covered_up_to = khabar_end_pos

    return akhbars


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_segmentation(akhbars: List[Dict], text: str) -> Dict:
    """Analyser les résultats de segmentation."""
    coverage = sum(ak['end'] - ak['start'] for ak in akhbars)

    isnad_lengths = [ak['isnad_end'] - ak['start'] for ak in akhbars]
    khabar_lengths = [ak['end'] - ak['isnad_end'] for ak in akhbars]

    return {
        'total_akhbars': len(akhbars),
        'coverage_chars': coverage,
        'total_chars': len(text),
        'coverage_pct': (coverage / len(text) * 100) if len(text) > 0 else 0,
        'avg_isnad_length': sum(isnad_lengths) / len(isnad_lengths) if isnad_lengths else 0,
        'avg_khabar_length': sum(khabar_lengths) / len(khabar_lengths) if khabar_lengths else 0,
        'min_isnad': min(isnad_lengths) if isnad_lengths else 0,
        'max_isnad': max(isnad_lengths) if isnad_lengths else 0,
    }


def main():
    """Tester la baseline v2."""
    from pathlib import Path
    import json

    print("=" * 80)
    print("BASELINE V2: ISNAD-FIRST APPROACH")
    print("=" * 80 + "\n")

    # Charger le corpus de référence
    print("[1] Chargement du corpus de référence...")
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        ref_boundaries = json.load(f)

    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Reference: {len(ref_boundaries)} akhbars\n")

    # Exécuter la baseline v2
    print("[2] Execution de la baseline v2...")
    akhbars = segment_akhbars_v2(corpus)
    print(f"    Akhbars détectés: {len(akhbars)}\n")

    # Analyser
    print("[3] Analyse des résultats...")
    analysis = analyze_segmentation(akhbars, corpus)

    diff = len(akhbars) - len(ref_boundaries)
    pct = 100 * diff / len(ref_boundaries)

    print(f"    Reference: {len(ref_boundaries)} akhbars")
    print(f"    Détectés: {len(akhbars)} akhbars")
    print(f"    Écart: {diff:+d} ({pct:+.1f}%)\n")

    print("[4] Statistiques...")
    print(f"    Couverture: {analysis['coverage_chars']:,} / {analysis['total_chars']:,} chars ({analysis['coverage_pct']:.1f}%)")
    print(f"    Longueur isnad moyenne: {analysis['avg_isnad_length']:.0f} chars")
    print(f"    Longueur khabar moyenne: {analysis['avg_khabar_length']:.0f} chars")
    print(f"    Isnad min/max: {analysis['min_isnad']:.0f} / {analysis['max_isnad']:.0f} chars\n")

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80 + "\n")

    if abs(pct) < 10:
        verdict = "[EXCELLENT] L'écart est faible"
    elif abs(pct) < 20:
        verdict = "[BON] L'écart est acceptable"
    elif abs(pct) < 50:
        verdict = "[ACCEPTABLE] L'écart est modéré"
    else:
        verdict = "[FAIBLE] L'écart est important"

    print(verdict)
    print(f"\nDétectés: {len(akhbars)}/{len(ref_boundaries)} ({100*len(akhbars)/len(ref_boundaries):.1f}%)")
    print(f"Écart: {diff:+d} ({pct:+.1f}%)\n")

    # Sauvegarder
    report_path = Path("../results/baseline_v2_results.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Baseline V2: ISNAD-First Approach\n\n")
        f.write("## Stratégie\n")
        f.write("1. Trouver TOUS les débuts d'isnads (verbes de transmission)\n")
        f.write("2. Pour chaque début, trouver la fin (dernier 'قال')\n")
        f.write("3. Khabar = texte entre deux isnads\n\n")

        f.write("## Résultats\n")
        f.write(f"- Akhbars détectés: {len(akhbars)}\n")
        f.write(f"- Akhbars attendus: {len(ref_boundaries)}\n")
        f.write(f"- Écart: {diff:+d} ({pct:+.1f}%)\n\n")

        f.write("## Statistiques\n")
        f.write(f"- Couverture: {analysis['coverage_pct']:.1f}%\n")
        f.write(f"- Longueur isnad moyenne: {analysis['avg_isnad_length']:.0f} chars\n")
        f.write(f"- Longueur khabar moyenne: {analysis['avg_khabar_length']:.0f} chars\n\n")

        f.write(f"## Verdict\n{verdict}\n")

    print(f"Rapport: {report_path}\n")


if __name__ == "__main__":
    main()
