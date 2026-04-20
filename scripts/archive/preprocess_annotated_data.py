#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess annotated khabar data for CAMeL-BERT training.

Handles:
- Khabars WITH isnad (use transmetteurs field)
- Khabars WITHOUT isnad (mark as such)
- Cases with/without قال marker
- Ambiguous boundaries (log for review)

Outputs clean training data ready for BERT fine-tuning.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ProcessedAkhbar:
    """Processed akhbar entry with clear boundary markers."""
    akhbar_num: int
    full_text: str
    isnad_start: int
    isnad_end: int
    khabar_start: int
    khabar_end: int
    has_isnad: bool
    has_qal: bool
    boundary_type: str  # 'transmitters', 'qal', 'structural', 'fallback', 'ambiguous'
    confidence: float
    notes: str


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for matching."""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # Remove all numerals (Arabic-Indic, Extended, and regular digits)
    text = re.sub(r'[\u0660-\u0669\u06F0-\u06F90-9]', '', text)
    # Normalize alef variants
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # Normalize TA marbuta
    text = text.replace('ة', 'ه')
    return text


def extract_transmitter_text(akhbar: Dict) -> Tuple[str, int, int]:
    """
    Extract transmitter chain text from akhbar.

    Returns:
        (transmitter_text, start_pos, end_pos) or ('', -1, -1) if not found
    """
    transmetteurs = akhbar.get('transmetteurs', [])

    if not transmetteurs:
        return '', -1, -1

    # Get all transmitter names
    transmitter_names = []
    for t in transmetteurs:
        if isinstance(t, dict):
            ar = t.get('ar', '')
            if ar:
                transmitter_names.append(ar)

    if not transmitter_names:
        return '', -1, -1

    # Build text from first to last transmitter
    first_name = transmitter_names[0]
    last_name = transmitter_names[-1]

    content = akhbar.get('content', {})
    segments = content.get('segments', [])
    full_text = ' '.join(seg.get('text', '') for seg in segments)
    normalized_text = normalize_arabic_text(full_text)

    # Find transmitters in text
    first_pos = normalized_text.find(normalize_arabic_text(first_name))
    last_pos = normalized_text.find(normalize_arabic_text(last_name))

    if first_pos < 0 or last_pos < 0:
        return '', -1, -1

    # Transmitter chain likely ends after last transmitter
    # Look for narrative transition markers
    end_pos = last_pos + len(last_name)

    return full_text[first_pos:end_pos], first_pos, end_pos


def find_qal_position(text: str) -> int:
    """Find position of قال marker in text."""
    normalized = normalize_arabic_text(text)
    pos = normalized.find('قال')
    return pos if pos >= 0 else -1


def process_akhbar(akhbar: Dict, index: int) -> ProcessedAkhbar:
    """
    Process a single akhbar entry and extract boundaries.

    Strategy:
    1. Try transmetteurs field (most reliable)
    2. Try قال marker (common but can be ambiguous)
    3. Use structural patterns (fallback)
    4. Mark ambiguous cases
    """

    content = akhbar.get('content', {})
    segments = content.get('segments', [])
    full_text = ' '.join(seg.get('text', '') for seg in segments)

    if not full_text.strip():
        return ProcessedAkhbar(
            akhbar_num=akhbar.get('num', index),
            full_text='',
            isnad_start=-1,
            isnad_end=-1,
            khabar_start=-1,
            khabar_end=-1,
            has_isnad=False,
            has_qal=False,
            boundary_type='empty',
            confidence=0.0,
            notes='Empty akhbar'
        )

    normalized_text = normalize_arabic_text(full_text)
    isnad_start = 0
    isnad_end = -1
    has_isnad = False
    boundary_type = 'none'
    confidence = 0.0
    notes = ''

    # STRATEGY 1: Use transmetteurs field
    transmetteur_text, trans_start, trans_end = extract_transmitter_text(akhbar)

    if trans_start >= 0:
        # Found transmitters - use them
        has_isnad = True
        isnad_start = trans_start

        # Look for قال after transmitters to mark exact end of isnad
        qal_pos = normalized_text.find('قال', trans_end)
        if qal_pos >= 0 and qal_pos < trans_end + 200:  # قال should be near transmitters
            isnad_end = qal_pos + 2
            boundary_type = 'transmitters+qal'
            confidence = 0.95
            notes = 'Isnad boundary via transmitters + qal marker'
        else:
            # قال not found nearby, use end of transmitters
            # But look for narrative transition (verb conjugation, pronouns)
            window_after = normalized_text[trans_end:min(trans_end+150, len(normalized_text))]

            # Simple heuristic: first sequence ending with punctuation or verb marker
            if 'قال' in window_after[:50]:
                qal_offset = window_after.find('قال')
                isnad_end = trans_end + qal_offset + 2
                boundary_type = 'transmitters+qal_nearby'
                confidence = 0.90
                notes = 'Isnad via transmitters, qal found nearby'
            else:
                # Use end of transmitters + small buffer
                isnad_end = trans_end + 10
                boundary_type = 'transmitters'
                confidence = 0.80
                notes = 'Isnad via transmitters, no qal marker'

    # STRATEGY 2: Try قال marker if transmitters failed
    elif isnad_end < 0:
        qal_pos = find_qal_position(full_text)
        if qal_pos >= 0:
            has_isnad = True
            isnad_start = 0
            isnad_end = qal_pos + 2
            boundary_type = 'qal_only'
            confidence = 0.75
            notes = 'Boundary via qal marker only (no transmitters)'
        else:
            # No transmitters and no قال
            has_isnad = False
            isnad_start = -1
            isnad_end = -1
            boundary_type = 'no_markers'
            confidence = 0.0
            notes = 'No isnad markers (transmitters or qal) found'

    # Validate boundaries
    if isnad_end > 0:
        khabar_start = isnad_end
        khabar_end = len(full_text)

        # Check minimum lengths
        isnad_length = isnad_end - isnad_start if isnad_start >= 0 else 0
        khabar_length = khabar_end - khabar_start

        if isnad_length < 5:
            confidence *= 0.5
            notes += ' [WARNING: very short isnad]'

        if khabar_length < 20:
            confidence *= 0.5
            notes += ' [WARNING: very short khabar]'
    else:
        khabar_start = -1
        khabar_end = -1

    return ProcessedAkhbar(
        akhbar_num=akhbar.get('num', index),
        full_text=full_text,
        isnad_start=isnad_start,
        isnad_end=isnad_end,
        khabar_start=khabar_start,
        khabar_end=khabar_end,
        has_isnad=has_isnad,
        has_qal=find_qal_position(full_text) >= 0,
        boundary_type=boundary_type,
        confidence=confidence,
        notes=notes
    )


def main():
    """Main preprocessing pipeline."""
    print("=" * 80)
    print("PREPROCESSING ANNOTATED DATA FOR CAMELBERT TRAINING")
    print("=" * 80 + "\n")

    # Load annotated data
    print("[1] Loading annotated data...")
    # Use absolute path from script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "processed" / "Kitab_Uqala_al_Majanin_annotated.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"    Loaded {len(data['akhbar'])} akhbar\n")

    # Process each akhbar
    print("[2] Processing boundaries...")
    processed = []
    stats = {
        'total': 0,
        'with_isnad': 0,
        'without_isnad': 0,
        'with_qal': 0,
        'by_type': {}
    }

    for i, akhbar in enumerate(data['akhbar']):
        result = process_akhbar(akhbar, i)
        processed.append(result)

        stats['total'] += 1
        if result.has_isnad:
            stats['with_isnad'] += 1
        else:
            stats['without_isnad'] += 1
        if result.has_qal:
            stats['with_qal'] += 1

        boundary_type = result.boundary_type
        stats['by_type'][boundary_type] = stats['by_type'].get(boundary_type, 0) + 1

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(data['akhbar'])} akhbar")

    print(f"\n[3] Statistics:\n")
    print(f"    Total: {stats['total']}")
    print(f"    With isnad: {stats['with_isnad']} ({100*stats['with_isnad']/stats['total']:.1f}%)")
    print(f"    Without isnad: {stats['without_isnad']} ({100*stats['without_isnad']/stats['total']:.1f}%)")
    print(f"    With qal marker: {stats['with_qal']} ({100*stats['with_qal']/stats['total']:.1f}%)\n")

    print(f"    By boundary type:")
    for btype, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        print(f"      {btype:20s}: {count:4d} ({100*count/stats['total']:5.1f}%)")

    # Filter high-confidence examples
    print(f"\n[4] Filtering by confidence...")
    high_conf = [p for p in processed if p.confidence >= 0.75 and p.has_isnad]
    medium_conf = [p for p in processed if 0.60 <= p.confidence < 0.75 and p.has_isnad]
    low_conf = [p for p in processed if p.confidence < 0.60 and p.has_isnad]
    no_isnad = [p for p in processed if not p.has_isnad]

    print(f"    High confidence (>=0.75): {len(high_conf)} ({100*len(high_conf)/stats['total']:.1f}%)")
    print(f"    Medium confidence (0.60-0.75): {len(medium_conf)} ({100*len(medium_conf)/stats['total']:.1f}%)")
    print(f"    Low confidence (<0.60): {len(low_conf)} ({100*len(low_conf)/stats['total']:.1f}%)")
    print(f"    No isnad: {len(no_isnad)} ({100*len(no_isnad)/stats['total']:.1f}%)\n")

    # Save processed data
    print(f"[5] Saving preprocessed data...")
    output_dir = project_root / "data" / "processed" / "camelbert_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all processed data (for reference)
    with open(output_dir / 'all_processed.json', 'w', encoding='utf-8') as f:
        json.dump([
            {
                'akhbar_num': p.akhbar_num,
                'full_text': p.full_text,
                'isnad_start': p.isnad_start,
                'isnad_end': p.isnad_end,
                'khabar_start': p.khabar_start,
                'khabar_end': p.khabar_end,
                'has_isnad': p.has_isnad,
                'has_qal': p.has_qal,
                'boundary_type': p.boundary_type,
                'confidence': p.confidence,
                'notes': p.notes
            }
            for p in processed
        ], f, indent=2, ensure_ascii=False)

    # Save high-confidence training data (recommended for BERT)
    with open(output_dir / 'training_data_high_conf.json', 'w', encoding='utf-8') as f:
        json.dump([
            {
                'akhbar_num': p.akhbar_num,
                'full_text': p.full_text,
                'isnad_start': p.isnad_start,
                'isnad_end': p.isnad_end,
                'khabar_start': p.khabar_start,
                'khabar_end': p.khabar_end,
                'boundary_type': p.boundary_type,
                'confidence': p.confidence
            }
            for p in high_conf
        ], f, indent=2, ensure_ascii=False)

    # Save medium+high confidence (for larger training set)
    with open(output_dir / 'training_data_medium_conf.json', 'w', encoding='utf-8') as f:
        json.dump([
            {
                'akhbar_num': p.akhbar_num,
                'full_text': p.full_text,
                'isnad_start': p.isnad_start,
                'isnad_end': p.isnad_end,
                'khabar_start': p.khabar_start,
                'khabar_end': p.khabar_end,
                'boundary_type': p.boundary_type,
                'confidence': p.confidence
            }
            for p in high_conf + medium_conf
        ], f, indent=2, ensure_ascii=False)

    # Save issue report
    with open(output_dir / 'issue_report.txt', 'w', encoding='utf-8') as f:
        f.write("PREPROCESSING REPORT\n\n")
        f.write(f"Total processed: {stats['total']}\n")
        f.write(f"With high confidence: {len(high_conf)}\n")
        f.write(f"With medium confidence: {len(medium_conf)}\n")
        f.write(f"With low confidence: {len(low_conf)}\n")
        f.write(f"No isnad: {len(no_isnad)}\n\n")

        f.write("ISSUES REQUIRING REVIEW:\n\n")

        if len(no_isnad) > 0:
            f.write(f"Khabars without isnad ({len(no_isnad)}):\n")
            for p in no_isnad[:10]:
                f.write(f"  #{p.akhbar_num}: {p.notes}\n")
            if len(no_isnad) > 10:
                f.write(f"  ... and {len(no_isnad) - 10} more\n")
            f.write("\n")

        if len(low_conf) > 0:
            f.write(f"Low confidence boundaries ({len(low_conf)}):\n")
            for p in low_conf[:10]:
                f.write(f"  #{p.akhbar_num} ({p.boundary_type}, conf={p.confidence:.2f}): {p.notes}\n")
            if len(low_conf) > 10:
                f.write(f"  ... and {len(low_conf) - 10} more\n")
            f.write("\n")

    print(f"    Saved to {output_dir}/:")
    print(f"      - all_processed.json (for reference)")
    print(f"      - training_data_high_conf.json (recommended: {len(high_conf)} examples)")
    print(f"      - training_data_medium_conf.json ({len(high_conf) + len(medium_conf)} examples)")
    print(f"      - issue_report.txt (problems found)\n")

    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nRECOMMENDATION:")
    print(f"  -> Use 'training_data_high_conf.json' for CAMeL-BERT training")
    print(f"  -> Contains {len(high_conf)} high-confidence examples")
    print(f"  -> Review issue_report.txt for {len(no_isnad)} problematic cases")


if __name__ == "__main__":
    main()
