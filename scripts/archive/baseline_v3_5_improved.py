#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline v3.5: IMPROVED WORD BOUNDARY DETECTION

Phase 2.5 improvement based on gap analysis:
- 57% of undetected have verbs ALREADY in v3
- Key issue: word boundary checking is too strict
- Solution: Relax boundary requirements for certain contexts

Expected improvement: 452 → 480-500 akhbars
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys


# ============================================================================
# TRANSMISSION VERBS - SAME AS V3
# ============================================================================

ISNAD_START_VERBS_BASE = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'حدثت', 'حدثوا',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'أخبرت', 'أخبروا',
    'سمعت', 'سمعنا', 'سمعه', 'سمعها', 'سمعهم',
    'روى', 'رويت', 'روينا', 'رواه', 'روا',
    'أنبأ', 'أنبأني', 'أنبأنا', 'أنبأه', 'أنبأتنا',
    'أنشدني', 'أنشدنا',
}

ISNAD_START_VERBS_PREFIXED = {
    'وأخبرنا', 'وأخبرني', 'وأخبره', 'وأخبرها', 'وأخبرهم',
    'وحدثنا', 'وحدثني', 'وحدثه', 'وحدثها', 'وحدثهم',
    'وسمعت', 'وسمعنا', 'وسمعه', 'وسمعها', 'وسمعهم',
    'وروى', 'ورويت', 'وروينا', 'ورواه',
    'وأنبأ', 'وأنبأني', 'وأنبأنا', 'وأنبأه',
    'وأنشدني', 'وأنشدنا',
}

# HIGH-CONFIDENCE MISSING (including ومن which helps)
ISNAD_START_VERBS_MISSING = {
    'وقال',
    'وبهذا',
    'ومنهم',
    'ومنها',
    'ومن',
    'ولبعضهم',
    'وحكى',
    'وبلغني',
}

ISNAD_START_VERBS = ISNAD_START_VERBS_BASE | ISNAD_START_VERBS_PREFIXED | ISNAD_START_VERBS_MISSING

# Bounds for isnad length
ISNAD_MIN_LENGTH = 10
ISNAD_MAX_LENGTH = 700
ISNAD_ABSOLUTE_MAX = 1500


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def normalize_arabic_text(text: str) -> str:
    """Normaliser le texte arabe."""
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


def simple_tokenize(text: str) -> List[str]:
    """Tokeniser simplement."""
    return text.split()


# ============================================================================
# ISNAD DETECTION - RELAXED BOUNDARIES
# ============================================================================

def find_all_isnad_starts(text: str) -> List[Tuple[int, str]]:
    """
    Trouver TOUTES les positions où un isnad commence.

    V3.5 CHANGE: VERY RELAXED word boundary checking
    - Remove boundary checking entirely - just find verb occurrences
    - Accept ALL matches (validation happens later)
    - This recovers verbs that were filtered by strict boundaries
    """
    normalized = normalize_arabic_text(text)
    isnad_starts = []

    # Chercher chaque verbe - NO word boundary checking
    for verb in sorted(ISNAD_START_VERBS, key=len, reverse=True):
        pos = 0
        while True:
            idx = normalized.find(verb, pos)
            if idx < 0:
                break

            # V3.5: NO word boundary checking
            # Accept all matches - validation happens later
            isnad_starts.append((idx, verb))

            pos = idx + 1

    # Remove overlaps
    isnad_starts.sort(key=lambda x: (x[0], -len(x[1])))

    unique_starts = []
    last_end = -1
    for start_pos, verb in isnad_starts:
        if start_pos >= last_end:
            unique_starts.append((start_pos, verb))
            last_end = start_pos + len(verb)

    return unique_starts


def find_isnad_end(text: str, start_pos: int, end_bound: int = -1) -> int:
    """Trouver la fin d'un isnad (premier قال)."""
    normalized = normalize_arabic_text(text)

    if end_bound < 0:
        search_end = min(start_pos + ISNAD_MAX_LENGTH, len(normalized))
    else:
        search_end = min(end_bound, len(normalized))

    idx = normalized.find('قال', start_pos)

    if idx >= 0 and idx < search_end:
        return idx + 2

    return -1


def segment_akhbars_v3_5(text: str) -> List[Dict]:
    """
    Segmente avec détection d'isnads relaxée.

    V3.5: Relaxed word boundary checking allows more isnads to be found,
    especially for verbs that were previously filtered by strict boundaries.
    """
    isnad_starts = find_all_isnad_starts(text)

    if not isnad_starts:
        return []

    akhbars = []
    covered_up_to = 0

    for i, (isnad_start_pos, verb) in enumerate(isnad_starts):
        if isnad_start_pos < covered_up_to:
            continue

        # Find next isnad
        next_isnad_start = len(text)
        for j in range(i + 1, len(isnad_starts)):
            candidate = isnad_starts[j][0]
            if candidate > isnad_start_pos:
                next_isnad_start = candidate
                break

        # Find isnad end
        isnad_end_pos = find_isnad_end(text, isnad_start_pos, end_bound=-1)

        # Fallback for no قال
        if isnad_end_pos < 0:
            fallback_end = min(isnad_start_pos + 250, next_isnad_start)
            isnad_end_pos = fallback_end

            if next_isnad_start - fallback_end < 20:
                continue
        else:
            # Validate length
            isnad_length = isnad_end_pos - isnad_start_pos
            if isnad_length < ISNAD_MIN_LENGTH or isnad_length > ISNAD_ABSOLUTE_MAX:
                continue

        # Khabar bounds
        khabar_end_pos = next_isnad_start
        khabar_length = khabar_end_pos - isnad_end_pos

        if khabar_length < 20:
            continue

        # Extract
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
    """Analyser les résultats."""
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
    """Tester la baseline v3.5."""
    from pathlib import Path
    import json

    print("=" * 80)
    print("BASELINE V3.5: IMPROVED WORD BOUNDARY DETECTION")
    print("=" * 80 + "\n")

    # Charger le corpus
    print("[1] Chargement du corpus...")
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        ref_boundaries = json.load(f)

    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Reference: {len(ref_boundaries)} akhbars\n")

    # Executer v3.5
    print("[2] Execution de la baseline v3.5...")
    akhbars = segment_akhbars_v3_5(corpus)
    print(f"    Akhbars detectes: {len(akhbars)}\n")

    # Analyser
    print("[3] Analyse...")
    analysis = analyze_segmentation(akhbars, corpus)

    diff = len(akhbars) - len(ref_boundaries)
    pct = 100 * diff / len(ref_boundaries)

    print(f"    Reference: {len(ref_boundaries)} akhbars")
    print(f"    Detectes: {len(akhbars)} akhbars")
    print(f"    Ecart: {diff:+d} ({pct:+.1f}%)\n")

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
        verdict = "[EXCELLENT] Ecart faible"
    elif abs(pct) < 20:
        verdict = "[BON] Ecart acceptable"
    elif abs(pct) < 50:
        verdict = "[ACCEPTABLE] Ecart modere"
    else:
        verdict = "[FAIBLE] Ecart important"

    print(verdict)
    print(f"\nDetectes: {len(akhbars)}/{len(ref_boundaries)} ({100*len(akhbars)/len(ref_boundaries):.1f}%)")
    print(f"Ecart: {diff:+d} ({pct:+.1f}%)\n")

    # Sauvegarder
    report_path = Path("../results/baseline_v3_5_results.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Baseline V3.5: Improved Word Boundary Detection\n\n")
        f.write("## Strategie (v3.5 improvements)\n")
        f.write("1. Relaxed word boundary checking - less strict after verb\n")
        f.write("2. Allows more isnads to be detected that were filtered by v3\n")
        f.write("3. Validation still happens via length checks\n\n")

        f.write("## Resultats\n")
        f.write(f"- Akhbars detectes: {len(akhbars)}\n")
        f.write(f"- Akhbars attendus: {len(ref_boundaries)}\n")
        f.write(f"- Ecart: {diff:+d} ({pct:+.1f}%)\n\n")

        f.write("## Statistiques\n")
        f.write(f"- Couverture: {analysis['coverage_pct']:.1f}%\n")
        f.write(f"- Longueur isnad moyenne: {analysis['avg_isnad_length']:.0f} chars\n")
        f.write(f"- Longueur khabar moyenne: {analysis['avg_khabar_length']:.0f} chars\n\n")

        f.write(f"## Verdict\n{verdict}\n")

    print(f"Rapport: {report_path}\n")


if __name__ == "__main__":
    main()
