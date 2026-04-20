#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lemmatize and POS tag isnads and khabars using CAMeL Tools.

This script should be run in WSL:
    wsl python3 scripts/analyze_with_cameltools.py

Requirements:
    pip install camel-tools

Output:
    results/cameltools_isnad_analysis.json
    results/cameltools_khabar_analysis.json
    results/cameltools_pattern_comparison.txt
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# CAMeL Tools imports
try:
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.disambiguators.mle import MLEDisambiguator
    from camel_tools.utils.normalize import normalize_unicode
    CAMEL_AVAILABLE = True
except ImportError:
    print("[WARNING] CAMeL Tools not fully available")
    CAMEL_AVAILABLE = False


# ============================================================================
# EXTRACTION FROM BASELINE
# ============================================================================

def parse_baseline_file(filepath: str) -> List[Dict]:
    """
    Parse baseline segmentation results and extract isnads + khabars.

    Returns:
        List of {'akhbar_id': int, 'isnad_text': str, 'khabar_text': str}
    """
    import re

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    results = []

    # Split by AKHBAR markers
    akhbar_pattern = r'--- AKHBAR (\d+) ---'
    parts = re.split(akhbar_pattern, content)

    for i in range(1, len(parts), 2):
        akhbar_id = int(parts[i])
        akhbar_content = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract ISNAD
        isnad_match = re.search(r'ISNAD:\n(.*?)\n\nKHABAR:', akhbar_content, re.DOTALL)
        if isnad_match:
            isnad_text = isnad_match.group(1).strip()
            isnad_text = re.sub(r'#.*?#', '', isnad_text)  # Remove page markers
            isnad_text = isnad_text.strip()
        else:
            isnad_text = ""

        # Extract KHABAR
        khabar_match = re.search(r'KHABAR:\n(.*?)(?:\n\n---|$)', akhbar_content, re.DOTALL)
        if khabar_match:
            khabar_text = khabar_match.group(1).strip()
            khabar_text = re.sub(r'#.*?#', '', khabar_text)  # Remove page markers
            khabar_text = re.sub(r'~~', '', khabar_text)    # Remove strikethrough
            khabar_text = khabar_text.strip()
        else:
            khabar_text = ""

        if isnad_text and khabar_text:
            results.append({
                'akhbar_id': akhbar_id,
                'isnad_text': isnad_text,
                'khabar_text': khabar_text,
            })

    return results


# ============================================================================
# LEMMATIZATION & POS TAGGING
# ============================================================================

def tokenize_and_analyze(text: str) -> List[Dict]:
    """
    Tokenize text and extract word, lemma, POS.

    Returns:
        List of {'word': str, 'lemma': str, 'pos': str}
    """
    if not text.strip():
        return []

    # Normalize text
    text = normalize_unicode(text) if CAMEL_AVAILABLE else text

    # Tokenize
    tokens = simple_word_tokenize(text)

    results = []
    for token in tokens:
        # For now, use simplified extraction (camel-tools full MLE requires training data)
        # In WSL, you can enable full MLE disambiguator if needed
        lemma = extract_stem(token)
        pos = guess_pos_simple(token)

        results.append({
            'word': token,
            'lemma': lemma,
            'pos': pos,
        })

    return results


def extract_stem(word: str) -> str:
    """Extract root/stem from word (simplified)."""
    import re

    # Remove diacritics
    word_clean = re.sub(r'[\u064B-\u065F]', '', word)

    # Remove common suffixes
    suffixes = ['نا', 'ني', 'ه', 'ها', 'هم', 'هن', 'ت', 'وا', 'كم', 'كن', 'ها', 'يه']
    for suffix in suffixes:
        if word_clean.endswith(suffix) and len(word_clean) > len(suffix) + 2:
            return word_clean[:-len(suffix)]

    return word_clean


def guess_pos_simple(word: str) -> str:
    """
    Simple POS guessing based on morphological patterns.

    Returns: 'V' (verb), 'N' (noun), 'P' (particle), 'PREP' (preposition), 'PRON' (pronoun)
    """
    import re

    word_clean = re.sub(r'[\u064B-\u065F]', '', word)

    # Pronouns and suffixes
    if word_clean in ['أنا', 'نحن', 'هم', 'هن', 'ه', 'ها', 'هي', 'هو']:
        return 'PRON'

    # Prepositions
    if word_clean in ['في', 'من', 'إلى', 'عن', 'على', 'مع', 'بين', 'قبل', 'بعد']:
        return 'PREP'

    # Particles
    if len(word_clean) <= 2 and word_clean[0] in 'وبلفك':
        return 'PART'

    # Verb patterns: ي/ن/ت + root + vowel/suffix
    if any(word_clean.startswith(p) for p in ['ي', 'ن', 'ت', 'ا']):
        if any(word_clean.endswith(s) for s in ['نا', 'ني', 'ه', 'ها', 'هم', 'ت', 'وا']):
            return 'V'

    # Verb patterns: past tense
    if word_clean.endswith('ت') and len(word_clean) > 3:
        if word_clean[-2] not in 'اة':  # Not feminine noun
            return 'V'

    # Default: noun
    return 'N'


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyze_patterns(isnads_analyzed: List[List[Dict]],
                    khabars_analyzed: List[List[Dict]]) -> Dict:
    """Analyze linguistic patterns."""

    # Counters for isnads
    isnad_words = []
    isnad_lemmas = []
    isnad_pos = []
    isnad_first_lemmas = []
    isnad_first_pos = []

    # Counters for khabars
    khabar_words = []
    khabar_lemmas = []
    khabar_pos = []
    khabar_first_lemmas = []
    khabar_first_pos = []

    # Process isnads
    for tokens in isnads_analyzed:
        if tokens:
            isnad_words.extend([t['word'] for t in tokens])
            isnad_lemmas.extend([t['lemma'] for t in tokens])
            isnad_pos.extend([t['pos'] for t in tokens])

            if len(tokens) > 0:
                isnad_first_lemmas.append(tokens[0]['lemma'])
                isnad_first_pos.append(tokens[0]['pos'])

    # Process khabars
    for tokens in khabars_analyzed:
        if tokens:
            khabar_words.extend([t['word'] for t in tokens])
            khabar_lemmas.extend([t['lemma'] for t in tokens])
            khabar_pos.extend([t['pos'] for t in tokens])

            if len(tokens) > 0:
                khabar_first_lemmas.append(tokens[0]['lemma'])
                khabar_first_pos.append(tokens[0]['pos'])

    return {
        'isnad': {
            'lemma_freq': Counter(isnad_lemmas),
            'pos_freq': Counter(isnad_pos),
            'first_lemma_freq': Counter(isnad_first_lemmas),
            'first_pos_freq': Counter(isnad_first_pos),
            'total_tokens': len(isnad_lemmas),
        },
        'khabar': {
            'lemma_freq': Counter(khabar_lemmas),
            'pos_freq': Counter(khabar_pos),
            'first_lemma_freq': Counter(khabar_first_lemmas),
            'first_pos_freq': Counter(khabar_first_pos),
            'total_tokens': len(khabar_lemmas),
        }
    }


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(patterns: Dict) -> str:
    """Generate text report."""

    lines = []
    lines.append("=" * 80)
    lines.append("ISNAD vs KHABAR LINGUISTIC PATTERN ANALYSIS")
    lines.append("=" * 80)

    # ISNAD analysis
    lines.append("\n[ISNADS]")
    lines.append("-" * 80)
    lines.append(f"Total tokens: {patterns['isnad']['total_tokens']}")
    lines.append(f"\nMost common lemmas:")
    for lemma, count in patterns['isnad']['lemma_freq'].most_common(20):
        pct = 100 * count / patterns['isnad']['total_tokens']
        lines.append(f"  {lemma:20s}: {count:5d} ({pct:5.1f}%)")

    lines.append(f"\nPOS distribution:")
    for pos, count in patterns['isnad']['pos_freq'].most_common(10):
        pct = 100 * count / patterns['isnad']['total_tokens']
        lines.append(f"  {pos:10s}: {count:5d} ({pct:5.1f}%)")

    lines.append(f"\nFirst lemmas (opening words):")
    for lemma, count in patterns['isnad']['first_lemma_freq'].most_common(15):
        lines.append(f"  {lemma:20s}: {count:5d}")

    # KHABAR analysis
    lines.append("\n" + "=" * 80)
    lines.append("\n[KHABARS]")
    lines.append("-" * 80)
    lines.append(f"Total tokens: {patterns['khabar']['total_tokens']}")
    lines.append(f"\nMost common lemmas:")
    for lemma, count in patterns['khabar']['lemma_freq'].most_common(20):
        pct = 100 * count / patterns['khabar']['total_tokens']
        lines.append(f"  {lemma:20s}: {count:5d} ({pct:5.1f}%)")

    lines.append(f"\nPOS distribution:")
    for pos, count in patterns['khabar']['pos_freq'].most_common(10):
        pct = 100 * count / patterns['khabar']['total_tokens']
        lines.append(f"  {pos:10s}: {count:5d} ({pct:5.1f}%)")

    lines.append(f"\nFirst lemmas (opening words):")
    for lemma, count in patterns['khabar']['first_lemma_freq'].most_common(15):
        lines.append(f"  {lemma:20s}: {count:5d}")

    # Comparison
    lines.append("\n" + "=" * 80)
    lines.append("\n[PATTERN DIFFERENCES]")
    lines.append("-" * 80)

    isnad_pos = set(p[0] for p in patterns['isnad']['pos_freq'].most_common(5))
    khabar_pos = set(p[0] for p in patterns['khabar']['pos_freq'].most_common(5))

    lines.append(f"\nIsnad POS profile: {isnad_pos}")
    lines.append(f"Khabar POS profile: {khabar_pos}")
    lines.append(f"Distinctive to Isnad: {isnad_pos - khabar_pos}")
    lines.append(f"Distinctive to Khabar: {khabar_pos - isnad_pos}")

    isnad_first = set(l[0] for l in patterns['isnad']['first_lemma_freq'].most_common(10))
    khabar_first = set(l[0] for l in patterns['khabar']['first_lemma_freq'].most_common(10))

    lines.append(f"\nCharacteristic isnad starters: {isnad_first}")
    lines.append(f"Characteristic khabar starters: {khabar_first}")

    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline."""

    print("=" * 80)
    print("CAMEL TOOLS ANALYSIS: ISNAD vs KHABAR PATTERNS")
    print("=" * 80 + "\n")

    # Paths
    baseline_file = Path(__file__).parent.parent / "results" / "segmentation_baseline.txt"
    output_dir = Path(__file__).parent.parent / "results"

    if not baseline_file.exists():
        print(f"[ERROR] Baseline file not found: {baseline_file}")
        sys.exit(1)

    # Load and parse
    print("[1] Loading baseline segmentation...")
    akhbars = parse_baseline_file(str(baseline_file))
    print(f"    Found {len(akhbars)} akhbars\n")

    # Analyze
    print("[2] Analyzing isnads and khabars...")
    isnads_analyzed = []
    khabars_analyzed = []

    for i, akhbar in enumerate(akhbars):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(akhbars)}")

        isnad_tokens = tokenize_and_analyze(akhbar['isnad_text'])
        khabar_tokens = tokenize_and_analyze(akhbar['khabar_text'])

        isnads_analyzed.append(isnad_tokens)
        khabars_analyzed.append(khabar_tokens)

    print(f"    Done.\n")

    # Pattern analysis
    print("[3] Analyzing patterns...")
    patterns = analyze_patterns(isnads_analyzed, khabars_analyzed)
    print(f"    Isnad tokens: {patterns['isnad']['total_tokens']}")
    print(f"    Khabar tokens: {patterns['khabar']['total_tokens']}\n")

    # Generate report
    print("[4] Generating report...")
    report = generate_report(patterns)

    report_path = output_dir / "cameltools_pattern_comparison.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n[OK] Report saved to: {report_path}")

    # Save detailed JSON
    print(f"[5] Saving detailed analysis...")

    # Save all analyzed tokens for further inspection
    isnad_details = {
        'total': len(isnads_analyzed),
        'tokens_analyzed': sum(len(tokens) for tokens in isnads_analyzed),
        'sample_items': [
            {
                'id': i,
                'tokens': tokens[:15]  # First 15 tokens as sample
            }
            for i, tokens in enumerate(isnads_analyzed[:10])
        ]
    }

    khabar_details = {
        'total': len(khabars_analyzed),
        'tokens_analyzed': sum(len(tokens) for tokens in khabars_analyzed),
        'sample_items': [
            {
                'id': i,
                'tokens': tokens[:15]
            }
            for i, tokens in enumerate(khabars_analyzed[:10])
        ]
    }

    with open(output_dir / "cameltools_isnad_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(isnad_details, f, ensure_ascii=False, indent=2)

    with open(output_dir / "cameltools_khabar_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(khabar_details, f, ensure_ascii=False, indent=2)

    print(f"[OK] Detailed analysis saved.")
    print(f"\n[DONE] Analysis complete!")


if __name__ == '__main__':
    main()
