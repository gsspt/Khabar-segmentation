#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction for XGBoost boundary prediction.

Extract linguistic and statistical features from detected isnads
to predict the correct isnad-to-khabar boundary position.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Transmission verbs for narrative detection
NARRATIVE_VERBS = {
    'قال', 'ذكر', 'أنبأ', 'حكى', 'روى', 'سمع', 'أخبر',
    'قالت', 'قالوا', 'قلت', 'قلنا',
    'فقال', 'فذكر', 'فروى',
}

ISNAD_VERBS = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'حدثت', 'حدثوا',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'أخبرت', 'أخبروا',
    'سمعت', 'سمعنا', 'سمعه', 'سمعها', 'سمعهم',
    'روى', 'رويت', 'روينا', 'رواه', 'روا',
    'أنبأ', 'أنبأني', 'أنبأنا', 'أنبأه', 'أنبأتنا',
    'أنشدني', 'أنشدنا',
}

FIRST_PERSON_PRONOUNS = {'نا', 'ني', 'ه', 'ها', 'هم', 'هن', 'ت'}


def normalize_arabic_text(text: str) -> str:
    """Normaliser le texte arabe."""
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


def extract_features(text: str, isnad_start: int, isnad_end_reference: int,
                     next_isnad_start: int, khabar_text: str = None) -> Dict:
    """
    Extract features for a single isnad segment.

    Args:
        text: Full corpus text
        isnad_start: Position where isnad starts (verb position)
        isnad_end_reference: True position where isnad ends (from reference)
        next_isnad_start: Position of next isnad (or end of text)
        khabar_text: (Optional) text of khabar for context

    Returns:
        Dict of features for XGBoost
    """
    normalized = normalize_arabic_text(text)

    # Window to search for قال and features
    window_start = isnad_start
    window_end = min(isnad_start + 800, next_isnad_start)
    window_text = text[window_start:window_end]
    window_normalized = normalized[window_start:window_end]

    features = {
        'isnad_start': isnad_start,
        'reference_end': isnad_end_reference,
        'target_offset': isnad_end_reference - isnad_start,  # Target: char distance to true boundary
    }

    # ========== FEATURE 1: Distance to قال ==========
    qal_pos = window_normalized.find('قال')
    if qal_pos >= 0:
        features['has_qal'] = 1
        features['qal_distance'] = qal_pos
    else:
        features['has_qal'] = 0
        features['qal_distance'] = 999  # Large value if no قال

    # ========== FEATURE 2: First person pronouns ==========
    pronoun_count = 0
    for pronoun in FIRST_PERSON_PRONOUNS:
        pronoun_count += window_normalized.count(pronoun)
    features['pronoun_count'] = pronoun_count

    # ========== FEATURE 3: Narrative verbs ==========
    narrative_count = 0
    for verb in NARRATIVE_VERBS:
        narrative_count += window_normalized.count(verb)
    features['narrative_verb_count'] = narrative_count

    # ========== FEATURE 4: Text statistics ==========
    word_count = len(window_text.split())
    features['window_word_count'] = word_count
    features['window_char_count'] = len(window_text)
    features['avg_word_length'] = len(window_text) / word_count if word_count > 0 else 0

    # ========== FEATURE 5: Punctuation ==========
    punctuation_marks = window_text.count('،') + window_text.count('۔') + window_text.count('؛')
    features['punctuation_count'] = punctuation_marks

    # ========== FEATURE 6: Special markers ==========
    features['has_faqal'] = 1 if 'فقال' in window_normalized else 0
    features['has_waqal'] = 1 if 'وقال' in window_normalized else 0
    features['has_halath'] = 1 if 'هلاتث' in window_normalized else 0

    # ========== FEATURE 7: Distance to next isnad ==========
    distance_to_next = next_isnad_start - isnad_start
    features['distance_to_next_isnad'] = distance_to_next

    # ========== FEATURE 8: Boundary validation features ==========
    # Is قال within expected isnad range?
    qal_abs_pos = isnad_start + qal_pos if qal_pos >= 0 else -1
    if qal_abs_pos >= 0 and qal_abs_pos < next_isnad_start:
        features['qal_in_valid_range'] = 1
    else:
        features['qal_in_valid_range'] = 0

    # Distance from reference boundary to قال
    if qal_abs_pos >= 0:
        features['qal_to_reference_diff'] = abs(qal_abs_pos - isnad_end_reference)
    else:
        features['qal_to_reference_diff'] = 999

    # ========== FEATURE 9: Character patterns ==========
    # Count of common isnad suffixes (marks where isnads typically end)
    features['has_end_suffix'] = 1 if 'عن' in window_normalized[-50:] else 0

    # ========== FEATURE 10: Reference position features =====
    # Where is the reference boundary relative to window?
    ref_window_pos = isnad_end_reference - isnad_start
    features['reference_in_early_window'] = 1 if ref_window_pos < 250 else 0
    features['reference_in_late_window'] = 1 if ref_window_pos > 500 else 0

    return features


def prepare_training_data(corpus: str, reference_boundaries: List[Dict]) -> List[Dict]:
    """
    Prepare training examples from reference boundaries and corpus.

    Args:
        corpus: Full text
        reference_boundaries: List of reference boundary dicts

    Returns:
        List of training examples with features
    """
    normalized = normalize_arabic_text(corpus)
    training_data = []

    for i, boundary in enumerate(reference_boundaries):
        isnad_start = boundary['start']
        khabar_end = boundary['end']

        # Try to find where isnad ends (look for قال in first 700 chars)
        window_text = corpus[isnad_start:min(isnad_start + 700, khabar_end)]
        window_norm = normalized[isnad_start:min(isnad_start + 700, khabar_end)]

        qal_pos = window_norm.find('قال')
        if qal_pos >= 0:
            isnad_end = isnad_start + qal_pos + 2
        else:
            # Fallback: assume isnad is first 250 chars
            isnad_end = min(isnad_start + 250, khabar_end - 20)

        # Next isnad start (or end of corpus)
        next_isnad_start = khabar_end
        if i + 1 < len(reference_boundaries):
            next_isnad_start = reference_boundaries[i + 1]['start']

        # Extract features
        features = extract_features(
            corpus,
            isnad_start,
            isnad_end,
            next_isnad_start,
            khabar_text=None
        )

        training_data.append(features)

    return training_data


def save_features_csv(training_data: List[Dict], output_path: Path):
    """Save features to CSV for XGBoost training."""
    import csv

    if not training_data:
        print("[ERROR] No training data to save")
        return

    # Get all feature keys except target
    feature_keys = [k for k in training_data[0].keys() if k != 'target_offset']
    feature_keys = sorted([k for k in feature_keys if k not in ['isnad_start', 'reference_end']])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=feature_keys + ['target_offset'])
        writer.writeheader()

        for row in training_data:
            writer.writerow({k: row.get(k, 0) for k in feature_keys + ['target_offset']})

    print(f"[OK] Features saved to {output_path}")
    print(f"    Rows: {len(training_data)}")
    print(f"    Columns: {len(feature_keys)}")


def main():
    """Extract and save features for XGBoost training."""
    print("=" * 80)
    print("XGBOOST FEATURE EXTRACTION")
    print("=" * 80 + "\n")

    # Load data
    print("[1] Loading corpus and reference boundaries...")
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        ref_boundaries = json.load(f)

    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Reference boundaries: {len(ref_boundaries)}\n")

    # Prepare training data
    print("[2] Extracting features from reference boundaries...")
    training_data = prepare_training_data(corpus, ref_boundaries)
    print(f"    Training examples: {len(training_data)}\n")

    # Save to CSV
    print("[3] Saving features...")
    output_path = Path("../data/processed/xgboost_training_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_features_csv(training_data, output_path)

    # Summary statistics
    print("\n[4] Feature statistics:")
    if training_data:
        first_example = training_data[0]
        print(f"    Features per example: {len(first_example)}")
        print(f"    Target variable: 'target_offset' (char distance from isnad start to end)")
        print(f"    Sample features: {list(first_example.keys())[:5]}...\n")

    print("=" * 80)
    print("READY FOR XGBOOST TRAINING")
    print("=" * 80)


if __name__ == "__main__":
    main()
