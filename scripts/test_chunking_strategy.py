#!/usr/bin/env python3
"""
Test the chunking strategy without making real API calls.

This simulates what the Deepseek gold standard script will do:
1. Load text
2. Create smart chunks
3. Simulate API responses (mock data)
4. Convert positions
5. Merge overlapping units

Usage:
    python scripts/test_chunking_strategy.py \
      --text data/processed/alDarrab_clean.txt \
      --chunk-size 3500 \
      --overlap 500
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def smart_chunk_text(
    text: str,
    chunk_size: int = 3500,
    overlap: int = 500,
    min_chunk_size: int = 500
) -> List[Tuple[int, int, str]]:
    """Split text into overlapping chunks at smart boundaries."""
    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        chunk_start = pos
        chunk_end = min(pos + chunk_size, text_len)

        # Find smart break point
        if chunk_end < text_len:
            search_start = max(chunk_end - 200, chunk_start)
            search_region = text[search_start:chunk_end]

            for delimiter in ['\n', '।', '،', ' ']:
                last_pos = search_region.rfind(delimiter)
                if last_pos != -1:
                    chunk_end = search_start + last_pos + 1
                    break

        chunk_text = text[chunk_start:chunk_end]

        if len(chunk_text) >= min_chunk_size or chunk_end >= text_len:
            chunks.append((chunk_start, chunk_end, chunk_text))

        overlap_chars = min(overlap, len(chunk_text) // 2)
        pos = chunk_end - overlap_chars

        if chunk_end >= text_len:
            break

    return chunks


def simulate_deepseek_response(
    chunk_text: str,
    chunk_start: int
) -> Dict:
    """
    Simulate what Deepseek would return for a chunk.

    In real usage, this would be actual API call results.
    For testing, we generate synthetic units based on text patterns.
    """
    # Find potential isnad markers
    isnad_markers = ['حدثنا', 'أخبرنا', 'قال', 'ذكر', 'أنبأ']

    units = []
    last_end = 0

    for marker in isnad_markers:
        pos = chunk_text.find(marker)
        if pos != -1 and pos > last_end:
            # Found an isnad marker
            unit_start = pos
            # Estimate unit end (simple heuristic: next marker or 300 chars)
            next_pos = len(chunk_text)
            for other_marker in isnad_markers:
                other_pos = chunk_text.find(other_marker, pos + 5)
                if other_pos != -1 and other_pos < next_pos:
                    next_pos = other_pos

            unit_end = min(next_pos, pos + 400)

            if unit_end > unit_start:
                isnad_text = chunk_text[unit_start:min(unit_start + 100, unit_end)]
                units.append({
                    'char_start': unit_start,
                    'char_end': unit_end,
                    'has_isnad': True,
                    'isnad_text': isnad_text.strip(),
                })
                last_end = unit_end

    # Add any remaining text as a unit without isnad
    if last_end < len(chunk_text) - 50:
        units.append({
            'char_start': last_end,
            'char_end': len(chunk_text),
            'has_isnad': False,
            'isnad_text': None,
        })

    return {'units': units}


def convert_chunk_positions(
    chunk_units: List[Dict],
    chunk_start: int
) -> List[Dict]:
    """Convert chunk-relative positions to global positions."""
    converted = []

    for unit in chunk_units:
        char_start = unit.get('char_start', 0)
        char_end = unit.get('char_end', 0)

        if char_start >= char_end:
            continue

        unit['char_start'] = chunk_start + char_start
        unit['char_end'] = chunk_start + char_end
        converted.append(unit)

    return converted


def merge_overlapping_units(
    all_units: List[Dict],
    overlap_threshold: int = 50
) -> List[Dict]:
    """Merge duplicate units from overlapping chunks."""
    if not all_units:
        return []

    sorted_units = sorted(all_units, key=lambda u: u['char_start'])
    merged = []
    current = None

    for unit in sorted_units:
        if current is None:
            current = unit
        else:
            start_diff = abs(unit['char_start'] - current['char_start'])
            end_diff = abs(unit['char_end'] - current['char_end'])

            if start_diff <= overlap_threshold and end_diff <= overlap_threshold:
                # Duplicate - keep the better one
                if len(unit.get('isnad_text', '')) > len(current.get('isnad_text', '')):
                    current = unit
            else:
                merged.append(current)
                current = unit

    if current:
        merged.append(current)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Test chunking strategy for Deepseek gold standard generation'
    )
    parser.add_argument('--text', required=True, help='Path to text file')
    parser.add_argument('--chunk-size', type=int, default=3500, help='Chunk size')
    parser.add_argument('--overlap', type=int, default=500, help='Overlap size')
    parser.add_argument('--output', help='Save test results to JSON')

    args = parser.parse_args()

    # Load text
    logger.info(f"Loading text from {args.text}")
    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()

    text_length = len(text)
    logger.info(f"Text length: {text_length:,} chars")

    # Create chunks
    chunks = smart_chunk_text(text, args.chunk_size, args.overlap)
    logger.info(f"\nCreated {len(chunks)} chunks:")

    for i, (start, end, chunk_text) in enumerate(chunks):
        logger.info(f"  Chunk {i}: chars {start:,}-{end:,} ({end-start} chars)")

    # Simulate API processing
    logger.info(f"\nSimulating Deepseek API responses...")
    all_units = []

    for i, (chunk_start, chunk_end, chunk_text) in enumerate(chunks):
        logger.info(f"\n  Processing chunk {i+1}/{len(chunks)}")

        # Simulate API call
        api_response = simulate_deepseek_response(chunk_text, chunk_start)
        chunk_units = api_response.get('units', [])

        logger.info(f"    Raw units from simulation: {len(chunk_units)}")

        # Convert positions
        converted = convert_chunk_positions(chunk_units, chunk_start)
        logger.info(f"    Converted to global positions: {len(converted)}")

        all_units.extend(converted)

    # Merge
    logger.info(f"\nMerging units from {len(chunks)} chunks...")
    logger.info(f"  Total units before dedup: {len(all_units)}")

    merged = merge_overlapping_units(all_units)
    logger.info(f"  After deduplication: {len(merged)}")

    # Sort and assign IDs
    merged = sorted(merged, key=lambda u: u['char_start'])
    for idx, unit in enumerate(merged):
        unit['unit_id'] = idx

    # Statistics
    units_with_isnad = sum(1 for u in merged if u.get('has_isnad', False))

    logger.info(f"\n{'='*70}")
    logger.info("TEST RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Text length: {text_length:,} chars")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Total units after merging: {len(merged)}")
    logger.info(f"Units with isnad: {units_with_isnad}")
    logger.info(f"Coverage: {sum(u['char_end'] - u['char_start'] for u in merged):,} chars")

    # Sample units
    logger.info(f"\nFirst 3 units:")
    for unit in merged[:3]:
        logger.info(f"  [{unit['unit_id']}] chars {unit['char_start']:,}-{unit['char_end']:,} "
                   f"(isnad: {unit.get('has_isnad', False)})")

    # Validate
    logger.info(f"\nValidation:")
    for unit in merged:
        if unit['char_start'] < 0 or unit['char_end'] > text_length:
            logger.warning(f"  Out of bounds: unit {unit['unit_id']}")
        if unit['char_start'] >= unit['char_end']:
            logger.warning(f"  Invalid range: unit {unit['unit_id']}")

    logger.info("  ✓ All units valid")

    # Save if requested
    if args.output:
        output = {
            'metadata': {
                'test_mode': True,
                'text_length_chars': text_length,
                'total_chunks': len(chunks),
                'chunk_size': args.chunk_size,
                'overlap': args.overlap,
                'total_units': len(merged),
                'units_with_isnad': units_with_isnad,
                'note': 'This is a test with simulated API responses, not real Deepseek results',
            },
            'narrative_units': merged,
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
