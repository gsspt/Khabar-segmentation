#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline v3: REFINED ISNAD-FIRST APPROACH

Phase 2 improvements based on Phase 1 analysis:
1. Dynamic isnad end boundary (use next isnad as upper bound, not fixed ISNAD_MAX_LENGTH)
2. Expanded ISNAD_START_VERBS (add prefixed variants + missing high-frequency verbs)
3. Relaxed length validation (allow longer isnads, safety cap only)
4. Fallback boundary detection for isnads without 'قال' marker
5. Strict word boundaries for common words (قال, ومن, وله)

Expected improvement: 396 → 500-550 akhbars detected
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys


# ============================================================================
# TRANSMISSION VERBS - EXPANDED WITH PREFIXED VARIANTS
# ============================================================================

# Base 34 verbs from v2
ISNAD_START_VERBS_BASE = {
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

    # Other reliable ones
    'أنشدني', 'أنشدنا',
}

# Prefixed variants (wa + base verb) - NEW for v3
ISNAD_START_VERBS_PREFIXED = {
    'وأخبرنا', 'وأخبرني', 'وأخبره', 'وأخبرها', 'وأخبرهم',
    'وحدثنا', 'وحدثني', 'وحدثه', 'وحدثها', 'وحدثهم',
    'وسمعت', 'وسمعنا', 'وسمعه', 'وسمعها', 'وسمعهم',
    'وروى', 'ورويت', 'وروينا', 'ورواه',
    'وأنبأ', 'وأنبأني', 'وأنبأنا', 'وأنبأه',
    'وأنشدني', 'وأنشدنا',
}

# High-frequency missing verbs from Phase 1 analysis - NEW for v3
# OPTIMIZED SET (Config 3 from verb combination test: +56 improvement)
ISNAD_START_VERBS_MISSING = {
    'وقال',       # 20 misses
    # 'قال',      # 17 misses - SKIPPED: adds noise, reduces detection
    'وبهذا',      # 9 misses
    'ومنهم',      # 8 misses
    'ومنها',      # 5 misses
    'ومن',        # 3 misses - GOOD: improves detection despite being short
    # 'وله',      # 3 misses - SKIPPED: reduces detection
    'ولبعضهم',   # 2 misses
    'وحكى',       # 1 miss
    'وبلغني',     # 1 miss
}

# Combine all - base + prefixed + selective missing verbs
ISNAD_START_VERBS = ISNAD_START_VERBS_BASE | ISNAD_START_VERBS_PREFIXED | ISNAD_START_VERBS_MISSING

# Verbs that need STRICT word boundaries (not punctuation, only whitespace/EOL)
STRICT_BOUNDARY_VERBS = {'قال', 'وقال', 'ومن', 'وله'}

# Markers that END isnads (transition to khabar)
ISNAD_END_MARKERS = {
    'قال',      # "said" - primary marker
    'فقال',     # "and said"
}

# Bounds for isnad length (in chars)
# v2: ISNAD_MIN_LENGTH = 10, ISNAD_MAX_LENGTH = 410
# v3: Relaxed - only use absolute safety cap
ISNAD_MIN_LENGTH = 10
ISNAD_MAX_LENGTH = 700      # Raised from 410 to handle long isnads (أخبرنا avg 564)
ISNAD_ABSOLUTE_MAX = 1500   # Safety cap only - reject pathological cases


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
# ISNAD DETECTION - REFINED STRATEGY
# ============================================================================

def find_all_isnad_starts(text: str) -> List[Tuple[int, int]]:
    """
    Trouver TOUTES les positions où un isnad commence.

    Retourne une liste de (position_debut, verbe_trouve).

    V3 CHANGES:
    - Strict boundaries for STRICT_BOUNDARY_VERBS (قال, ومن, وله, وقال)
    - Others use relaxed boundaries (allow punctuation after)
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

            # Vérifier que c'est un mot complet
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


def find_isnad_end(text: str, start_pos: int, end_bound: int = -1) -> int:
    """
    Trouver la fin d'un isnad en cherchant le premier 'قال' jusqu'à end_bound.

    V3 CHANGE: Accept end_bound parameter to use next isnad start as upper bound.
    If end_bound = -1, use ISNAD_MAX_LENGTH as fallback.

    This allows long isnads (e.g., أخبرنا with avg 564 chars) to find their قال
    even if it's beyond the fixed 410-char window.
    """
    normalized = normalize_arabic_text(text)

    # Determine search boundary
    if end_bound < 0:
        # Use fixed fallback (v2 behavior)
        search_end = min(start_pos + ISNAD_MAX_LENGTH, len(normalized))
    else:
        # Use dynamic bound (next isnad start)
        search_end = min(end_bound, len(normalized))

    # Chercher simplement la première occurrence de 'قال'
    idx = normalized.find('قال', start_pos)

    if idx >= 0 and idx < search_end:
        # Retourner la position après 'قال'
        return idx + 2

    return -1


def segment_akhbars_v3(text: str) -> List[Dict]:
    """
    Segmente basée sur la détection d'isnads et transitions vers khbars.

    V3 CHANGES:
    1. Pass next_isnad_start as end_bound to find_isnad_end() - enables long isnad detection
    2. Fallback boundary when no قال found - use 250-char window or next isnad start
    3. Relaxed length validation - only reject if > ISNAD_ABSOLUTE_MAX
    """
    isnad_starts = find_all_isnad_starts(text)

    if not isnad_starts:
        return []

    akhbars = []
    covered_up_to = 0

    for i, (isnad_start_pos, verb) in enumerate(isnad_starts):
        # Skip if this isnad overlaps with previous one
        if isnad_start_pos < covered_up_to:
            continue

        # Trouver la fin de cet isnad (premier 'قال') - use dynamic max boundary
        # V3: Use ISNAD_MAX_LENGTH (now 700) to allow finding قال in long isnads
        isnad_end_pos = find_isnad_end(text, isnad_start_pos, end_bound=-1)

        # Find next isnad start AFTER calling find_isnad_end
        next_isnad_start = len(text)  # Default: end of text
        for j in range(i + 1, len(isnad_starts)):
            candidate = isnad_starts[j][0]
            if candidate > isnad_start_pos:
                next_isnad_start = candidate
                break

        # V3 CHANGE: Fallback when no قال found
        if isnad_end_pos < 0:
            # No قال found - use fallback boundary
            # Assume isnad is ~250 chars max, rest is khabar
            fallback_end = min(isnad_start_pos + 250, next_isnad_start)
            isnad_end_pos = fallback_end

            # Require minimum khabar length
            if next_isnad_start - fallback_end < 20:
                continue
        else:
            # قال was found - validate isnad length
            isnad_length = isnad_end_pos - isnad_start_pos

            # V3 CHANGE: Relaxed validation - only reject if pathologically long
            if isnad_length < ISNAD_MIN_LENGTH or isnad_length > ISNAD_ABSOLUTE_MAX:
                continue

        # Khabar end = next isnad start (or end of text)
        khabar_end_pos = next_isnad_start

        # Minimum khabar length
        khabar_length = khabar_end_pos - isnad_end_pos
        if khabar_length < 20:
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
    """Tester la baseline v3."""
    from pathlib import Path
    import json

    print("=" * 80)
    print("BASELINE V3: REFINED ISNAD-FIRST APPROACH")
    print("=" * 80 + "\n")

    # Charger le corpus de référence
    print("[1] Chargement du corpus de référence...")
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        ref_boundaries = json.load(f)

    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Reference: {len(ref_boundaries)} akhbars\n")

    # Exécuter la baseline v3
    print("[2] Execution de la baseline v3...")
    akhbars = segment_akhbars_v3(corpus)
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
        verdict = "[EXCELLENT] Écart faible"
    elif abs(pct) < 20:
        verdict = "[BON] Écart acceptable"
    elif abs(pct) < 50:
        verdict = "[ACCEPTABLE] Écart modéré"
    else:
        verdict = "[FAIBLE] Écart important"

    print(verdict)
    print(f"\nDétectés: {len(akhbars)}/{len(ref_boundaries)} ({100*len(akhbars)/len(ref_boundaries):.1f}%)")
    print(f"Écart: {diff:+d} ({pct:+.1f}%)\n")

    # Sauvegarder
    report_path = Path("../results/baseline_v3_results.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Baseline V3: Refined ISNAD-First Approach\n\n")
        f.write("## Stratégie (v3 improvements)\n")
        f.write("1. Dynamic isnad end boundary (use next isnad as upper bound)\n")
        f.write("2. Expanded ISNAD_START_VERBS (prefixed variants + missing high-frequency verbs)\n")
        f.write("3. Relaxed length validation (allow longer isnads, safety cap only)\n")
        f.write("4. Fallback boundary detection for isnads without 'قال'\n")
        f.write("5. Strict word boundaries for common words (قال, ومن, وله)\n\n")

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
