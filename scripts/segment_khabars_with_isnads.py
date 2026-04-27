#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Arabic text into narrative units (khabars) with isnad/matn separation.

Combines CAMeL-BERT boundary detection with clean text to produce JSON output
where each khabar has:
  - isnad: the chain of transmission (usually starts with "قال" or "أخبرنا")
  - matn: the narrative body

Usage:
    python scripts/segment_khabars_with_isnads.py \
        --boundaries results/ibnabdrabbih/camelbert_ibnabdrabbih_char_boundaries.json \
        --text data/processed/ibnabdrabbih_clean.txt \
        --output results/ibnabdrabbih_khabars_segmented.json \
        --isnad-threshold 50 \
        --min-isnad-chars 5
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


def load_boundaries(boundary_file: str) -> List[Dict]:
    """Load CAMeL-BERT boundary detection results."""
    with open(boundary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the list of boundary objects
    boundaries = data.get('khabar_boundaries', [])
    return sorted(boundaries, key=lambda x: x['char_start'])


def load_text(text_file: str) -> str:
    """Load the clean Arabic text."""
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read()


def get_isnad_from_boundary(text: str, boundary: Dict) -> Tuple[int, int]:
    """
    Get exact isnad boundaries from CAMeL-BERT boundary object.

    Args:
        text: Full text
        boundary: Boundary dict with char_start and char_end

    Returns:
        Tuple of (isnad_start, isnad_end) character positions
    """
    return boundary['char_start'], boundary['char_end']


def extract_isnad_and_matn(
    text: str,
    boundary: Dict,
    next_boundary_start: Optional[int] = None
) -> Tuple[str, str]:
    """
    Extract isnad and matn from text using precise boundary positions.

    Args:
        text: Full clean text
        boundary: Boundary dict with char_start (isnad start) and char_end (isnad end)
        next_boundary_start: Start of next khabar's isnad (for matn extraction limit)

    Returns:
        Tuple of (isnad_text, matn_text)
    """

    # Isnad: from char_start to char_end (as defined by CAMeL-BERT)
    isnad_start = boundary['char_start']
    isnad_end = boundary['char_end']
    isnad = text[isnad_start:isnad_end].strip()

    # Matn: from char_end to next boundary start (or end of text)
    matn_start = isnad_end
    if next_boundary_start is not None:
        matn_end = next_boundary_start
    else:
        matn_end = len(text)

    matn = text[matn_start:matn_end].strip()

    return isnad, matn


def segment_khabars(
    text: str,
    boundaries: List[Dict],
    isnad_threshold: int = 50,
    min_isnad_chars: int = 5
) -> List[Dict]:
    """
    Segment text into khabars with isnad/matn separation.

    Each khabar is defined by:
    - isnad: from boundary.char_start to boundary.char_end (as defined by CAMeL-BERT)
    - matn: from boundary.char_end to the next boundary.char_start

    Args:
        text: Full clean Arabic text
        boundaries: List of detected boundaries from CAMeL-BERT
        isnad_threshold: Ignore boundaries closer than this many chars
        min_isnad_chars: Minimum characters to consider as valid isnad

    Returns:
        List of khabar dicts with 'id', 'char_start', 'char_end', 'isnad', 'matn'
    """

    # Filter boundaries by threshold
    filtered = []
    last_char = 0

    for boundary in boundaries:
        char_start = boundary['char_start']
        if char_start - last_char >= isnad_threshold:
            filtered.append(boundary)
            last_char = char_start

    # Build khabars
    khabars = []

    for i, boundary in enumerate(filtered):
        # Next khabar's boundary start (for matn limit)
        if i + 1 < len(filtered):
            next_boundary_start = filtered[i + 1]['char_start']
        else:
            next_boundary_start = None

        # Extract isnad (using precise CAMeL-BERT boundaries) and matn
        isnad, matn = extract_isnad_and_matn(
            text,
            boundary,
            next_boundary_start=next_boundary_start
        )

        # Only include if isnad is substantial enough
        if len(isnad) >= min_isnad_chars:
            khabar_start = boundary['char_start']
            khabar_end = boundary['char_end'] + len(matn)

            khabars.append({
                'id': len(khabars),
                'char_start': khabar_start,
                'char_end': khabar_end,
                'isnad': isnad,
                'matn': matn,
                'isnad_length': len(isnad),
                'matn_length': len(matn),
                'confidence': boundary.get('max_prob', None)
            })

    return khabars


def main():
    parser = argparse.ArgumentParser(
        description='Segment Arabic text into khabars with isnad/matn separation'
    )
    parser.add_argument(
        '--boundaries',
        required=True,
        help='Path to CAMeL-BERT boundaries JSON file'
    )
    parser.add_argument(
        '--text',
        required=True,
        help='Path to clean Arabic text file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--isnad-threshold',
        type=int,
        default=50,
        help='Minimum character distance between boundaries (default: 50)'
    )
    parser.add_argument(
        '--min-isnad-chars',
        type=int,
        default=5,
        help='Minimum isnad length in characters (default: 5)'
    )

    args = parser.parse_args()

    # Load inputs
    print(f"Loading boundaries from {args.boundaries}...")
    boundaries = load_boundaries(args.boundaries)
    print(f"  - Found {len(boundaries)} boundaries")

    print(f"Loading text from {args.text}...")
    text = load_text(args.text)
    print(f"  - Text length: {len(text)} characters")

    # Segment
    print(f"\nSegmenting into khabars...")
    khabars = segment_khabars(
        text,
        boundaries,
        isnad_threshold=args.isnad_threshold,
        min_isnad_chars=args.min_isnad_chars
    )
    print(f"  - Produced {len(khabars)} khabars")

    # Statistics
    total_isnad_chars = sum(k['isnad_length'] for k in khabars)
    total_matn_chars = sum(k['matn_length'] for k in khabars)
    avg_isnad = total_isnad_chars / len(khabars) if khabars else 0
    avg_matn = total_matn_chars / len(khabars) if khabars else 0

    print(f"\nStatistics:")
    print(f"  Isnad chars (total): {total_isnad_chars}")
    print(f"  Matn chars (total): {total_matn_chars}")
    print(f"  Isnad avg length: {avg_isnad:.1f} chars")
    print(f"  Matn avg length: {avg_matn:.1f} chars")

    # Prepare output
    output_data = {
        'metadata': {
            'text_file': args.text,
            'boundary_file': args.boundaries,
            'total_khabars': len(khabars),
            'isnad_threshold': args.isnad_threshold,
            'min_isnad_chars': args.min_isnad_chars,
        },
        'khabars': khabars
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Output saved to {args.output}")


if __name__ == '__main__':
    main()
