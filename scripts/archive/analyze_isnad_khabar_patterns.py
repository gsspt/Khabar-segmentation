#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze ISNAD and KHABAR patterns using CAMeL Tools lemmatization + POS tagging.

Purpose:
- Extract isnads and khabars from annotated data
- Lemmatize and POS tag each separately
- Identify linguistic patterns for better detection
- Find regularities (verb roots, POS sequences, etc.)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys

# CAMeL Tools
try:
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.lemmatizers.utils import (
        normalize_unicode as camel_normalize
    )
    print("[OK] CAMeL Tools imported")
except ImportError:
    print("[ERROR] CAMeL Tools not installed")
    print("Install with: pip install camel-tools")
    sys.exit(1)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_annotated_data(path: str) -> Dict:
    """Load Kitab_Uqala_al_Majanin_annotated.json"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_isnads_and_khabars(data: Dict) -> List[Dict]:
    """
    Extract isnads and khabars from annotated data.

    Returns list of:
    {
        'akhbar_num': int,
        'isnad_text': str,
        'khabar_text': str,
        'isnad_start': int,
        'isnad_end': int,
    }
    """
    results = []

    for akhbar in data.get('akhbar', []):
        # Get full text
        content = akhbar.get('content', {})
        segments = content.get('segments', [])
        full_text = ' '.join(seg.get('text', '') for seg in segments)

        # Get transmetteurs (isnad markers)
        transmetteurs = akhbar.get('transmetteurs', [])

        if not transmetteurs or not full_text.strip():
            continue

        # Extract transmitter names
        transmitter_names = []
        for t in transmetteurs:
            if isinstance(t, dict):
                ar = t.get('ar', '')
                if ar:
                    transmitter_names.append(ar)

        if not transmitter_names:
            continue

        # Find transmitter boundaries in text
        first_name = transmitter_names[0]
        last_name = transmitter_names[-1]

        first_pos = full_text.find(first_name)
        last_pos = full_text.find(last_name)

        if first_pos < 0 or last_pos < 0:
            continue

        # Isnad ends after last transmitter
        isnad_end = last_pos + len(last_name)

        # Look for قال to refine boundary
        qal_pos = full_text.find('قال', isnad_end)
        if qal_pos >= 0 and qal_pos < isnad_end + 200:
            isnad_end = qal_pos + 2

        # Extract texts
        isnad_text = full_text[:isnad_end].strip()
        khabar_text = full_text[isnad_end:].strip()

        if not isnad_text or not khabar_text:
            continue

        results.append({
            'akhbar_num': akhbar.get('num'),
            'isnad_text': isnad_text,
            'khabar_text': khabar_text,
            'isnad_start': 0,
            'isnad_end': isnad_end,
        })

    return results


# ============================================================================
# LEMMATIZATION & POS TAGGING
# ============================================================================

def tokenize_and_analyze(text: str) -> List[Dict]:
    """
    Tokenize and analyze text.

    Returns:
    [
        {
            'word': str,
            'pos': str (from CAMeL if available, else 'UNK'),
            'lemma': str (from morphological analysis)
        },
        ...
    ]
    """
    tokens = simple_word_tokenize(text)
    results = []

    for token in tokens:
        # Basic morphological analysis
        # Extract root (3-letter root in Arabic)
        lemma = extract_lemma(token)
        pos = guess_pos(token, lemma)

        results.append({
            'word': token,
            'pos': pos,
            'lemma': lemma,
        })

    return results


def extract_lemma(word: str) -> str:
    """
    Extract potential lemma from word.
    Simplified: just normalize and remove common suffixes.
    """
    # Remove diacritics
    word = re.sub(r'[\u064B-\u065F]', '', word)

    # Remove common suffixes
    suffixes = ['نا', 'ني', 'ه', 'ها', 'هم', 'هن', 'ت', 'وا', 'كم', 'كن']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]

    return word


def guess_pos(word: str, lemma: str) -> str:
    """Simple POS guessing based on patterns."""

    # Remove diacritics
    word_clean = re.sub(r'[\u064B-\u065F]', '', word)

    # Verb markers
    if word_clean.startswith('ي') or word_clean.startswith('ن') or word_clean.startswith('ا'):
        if any(word_clean.endswith(s) for s in ['نا', 'ني', 'ه', 'ها', 'هم', 'ت']):
            return 'V'  # Verb with pronoun

    # Particle/Preposition (short words starting with و, ب, ل, etc.)
    if len(word_clean) <= 2:
        if word_clean[0] in 'وبلفك':
            return 'PART'

    # Default noun
    return 'N'


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyze_patterns(isnads_analyzed: List[List[Dict]],
                    khabars_analyzed: List[List[Dict]]) -> Dict:
    """
    Analyze linguistic patterns in isnads vs khabars.
    """

    # Collect statistics
    isnad_lemmas = []
    isnad_pos_sequences = []
    isnad_first_verbs = []

    khabar_lemmas = []
    khabar_pos_sequences = []
    khabar_first_verbs = []

    # Process isnads
    for isnad_tokens in isnads_analyzed:
        if not isnad_tokens:
            continue

        # Lemmas
        lemmas = [t['lemma'] for t in isnad_tokens]
        isnad_lemmas.extend(lemmas)

        # POS sequence
        pos_seq = ' '.join([t['pos'] for t in isnad_tokens[:10]])  # First 10 tokens
        isnad_pos_sequences.append(pos_seq)

        # First verb
        for t in isnad_tokens:
            if t['pos'] == 'V':
                isnad_first_verbs.append(t['lemma'])
                break

    # Process khabars
    for khabar_tokens in khabars_analyzed:
        if not khabar_tokens:
            continue

        lemmas = [t['lemma'] for t in khabar_tokens]
        khabar_lemmas.extend(lemmas)

        pos_seq = ' '.join([t['pos'] for t in khabar_tokens[:10]])
        khabar_pos_sequences.append(pos_seq)

        for t in khabar_tokens:
            if t['pos'] == 'V':
                khabar_first_verbs.append(t['lemma'])
                break

    return {
        'isnad': {
            'lemma_counts': Counter(isnad_lemmas),
            'first_verbs': Counter(isnad_first_verbs),
            'pos_sequences': Counter(isnad_pos_sequences),
            'total_tokens': len(isnad_lemmas),
        },
        'khabar': {
            'lemma_counts': Counter(khabar_lemmas),
            'first_verbs': Counter(khabar_first_verbs),
            'pos_sequences': Counter(khabar_pos_sequences),
            'total_tokens': len(khabar_lemmas),
        }
    }


# ============================================================================
# REPORTING
# ============================================================================

def print_report(patterns: Dict, output_path: str = None):
    """Print analysis report."""

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("ISNAD & KHABAR LINGUISTIC PATTERN ANALYSIS")
    report_lines.append("=" * 80 + "\n")

    # ISNAD patterns
    report_lines.append("ISNADS - Most Common Lemmas (Top 20):")
    report_lines.append("-" * 40)
    for lemma, count in patterns['isnad']['lemma_counts'].most_common(20):
        pct = 100 * count / patterns['isnad']['total_tokens']
        report_lines.append(f"  {lemma:20s}: {count:4d} ({pct:5.1f}%)")

    report_lines.append(f"\nISNADS - First Verbs (Top 15):")
    report_lines.append("-" * 40)
    for verb, count in patterns['isnad']['first_verbs'].most_common(15):
        report_lines.append(f"  {verb:20s}: {count:4d}")

    report_lines.append(f"\nISNADS - POS Patterns (Top 10):")
    report_lines.append("-" * 40)
    for pos_seq, count in patterns['isnad']['pos_sequences'].most_common(10):
        report_lines.append(f"  {pos_seq:40s}: {count:3d}")

    # KHABAR patterns
    report_lines.append(f"\n" + "=" * 80)
    report_lines.append("KHABARS - Most Common Lemmas (Top 20):")
    report_lines.append("-" * 40)
    for lemma, count in patterns['khabar']['lemma_counts'].most_common(20):
        pct = 100 * count / patterns['khabar']['total_tokens']
        report_lines.append(f"  {lemma:20s}: {count:4d} ({pct:5.1f}%)")

    report_lines.append(f"\nKHABARS - First Verbs (Top 15):")
    report_lines.append("-" * 40)
    for verb, count in patterns['khabar']['first_verbs'].most_common(15):
        report_lines.append(f"  {verb:20s}: {count:4d}")

    report_lines.append(f"\nKHABARS - POS Patterns (Top 10):")
    report_lines.append("-" * 40)
    for pos_seq, count in patterns['khabar']['pos_sequences'].most_common(10):
        report_lines.append(f"  {pos_seq:40s}: {count:3d}")

    # Comparison
    report_lines.append(f"\n" + "=" * 80)
    report_lines.append("PATTERN DIFFERENCES (Isnad vs Khabar):")
    report_lines.append("-" * 40)

    isnad_first_verbs = set(v[0] for v in patterns['isnad']['first_verbs'].most_common(10))
    khabar_first_verbs = set(v[0] for v in patterns['khabar']['first_verbs'].most_common(10))

    report_lines.append(f"\nCharacteristic ISNAD verbs: {', '.join(isnad_first_verbs)}")
    report_lines.append(f"Characteristic KHABAR verbs: {', '.join(khabar_first_verbs)}")
    report_lines.append(f"Overlap: {isnad_first_verbs & khabar_first_verbs}")

    # Print and save
    full_report = '\n'.join(report_lines)
    print(full_report)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n[OK] Report saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("ISNAD & KHABAR PATTERN ANALYSIS WITH CAMEL TOOLS")
    print("=" * 80 + "\n")

    # Load data
    print("[1] Loading annotated data...")
    data_path = Path(__file__).parent.parent / "data" / "processed" / "Kitab_Uqala_al_Majanin_annotated.json"
    data = load_annotated_data(str(data_path))
    print(f"    Loaded {len(data.get('akhbar', []))} akhbars\n")

    # Extract isnads and khabars
    print("[2] Extracting isnads and khabars...")
    extracted = extract_isnads_and_khabars(data)
    print(f"    Extracted {len(extracted)} isnad/khabar pairs\n")

    if not extracted:
        print("[ERROR] No data extracted!")
        return

    # Analyze each
    print("[3] Tokenizing and analyzing...")
    isnads_analyzed = []
    khabars_analyzed = []

    for i, item in enumerate(extracted):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(extracted)}")

        isnad_tokens = tokenize_and_analyze(item['isnad_text'])
        khabar_tokens = tokenize_and_analyze(item['khabar_text'])

        isnads_analyzed.append(isnad_tokens)
        khabars_analyzed.append(khabar_tokens)

    print(f"\n[4] Analyzing patterns...")
    patterns = analyze_patterns(isnads_analyzed, khabars_analyzed)

    # Generate report
    print("\n[5] Generating report...\n")
    output_path = Path(__file__).parent.parent / "results" / "isnad_khabar_patterns.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_report(patterns, str(output_path))


if __name__ == "__main__":
    main()
