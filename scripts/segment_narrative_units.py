#!/usr/bin/env python3
"""
Convert boundary positions into narrative units (isnads + khabars).

Supports both:
1. CAMeL-BERT boundaries (char positions from clustering boundary tokens)
2. Baseline v4 segments (isnad detection via linguistic patterns)

Usage:
    # CAMeL-BERT
    python scripts/segment_narrative_units.py \
      --boundaries results/camelbert_[TEXT]_char_boundaries.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/camelbert_[TEXT]_narrative_units.json

    # Baseline v4
    python scripts/segment_narrative_units.py \
      --segments results/baseline_v4_[TEXT]_segments.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/baseline_v4_[TEXT]_narrative_units.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common isnad verb patterns in Arabic
ISNAD_VERBS = {
    'أخبرنا', 'أخبركم', 'أخبره', 'أخبرها',
    'حدثنا', 'حدثكم', 'حدثه', 'حدثها',
    'أنبأنا', 'أنبأكم', 'أنبأه', 'أنبأها',
    'قال', 'قالت', 'قالوا',
    'ذكر', 'ذكرت', 'ذكروا',
}


def extract_isnad_from_text(text: str, max_length: int = 200) -> Optional[str]:
    """
    Extract isnad text from the beginning of a segment.

    Isnad typically:
    - Starts at beginning of segment
    - Contains transmission verbs (أخبرنا، حدثنا، قال، etc.)
    - Ends before narrative content
    - Max ~200 chars

    Args:
        text: Segment text
        max_length: Maximum isnad length in chars

    Returns:
        Isnad text if found, else None
    """
    if not text:
        return None

    # Limit search to first max_length chars
    search_text = text[:max_length]

    # Find position of isnad verbs
    isnad_verbs = sorted(ISNAD_VERBS, key=len, reverse=True)
    verb_positions = []

    for verb in isnad_verbs:
        matches = re.finditer(re.escape(verb), search_text)
        for match in matches:
            verb_positions.append((match.start(), match.end(), verb))

    if not verb_positions:
        return None

    # Get the last verb position (end of isnad chain)
    verb_positions.sort(key=lambda x: x[0])
    last_verb_start, last_verb_end, verb = verb_positions[-1]

    # Extend to next period, comma, or newline
    end_pos = last_verb_end
    for char in search_text[last_verb_end:]:
        if char in ('۔', '。', '.', '،', ',', '\n'):
            break
        end_pos += 1

    # Return from start to end of isnad
    isnad_text = text[:end_pos].strip()

    # Sanity check: isnad should contain a verb and be reasonably short
    if len(isnad_text) > max_length or not any(v in isnad_text for v in ISNAD_VERBS):
        return None

    return isnad_text if len(isnad_text) > 10 else None


def segment_from_camelbert_boundaries(
    boundaries_file: str,
    corpus_file: str
) -> list[dict]:
    """
    Create narrative units from CAMeL-BERT boundary positions.

    Args:
        boundaries_file: Path to camelbert_*_char_boundaries.json
        corpus_file: Path to cleaned corpus text

    Returns:
        List of narrative unit dicts
    """
    logger.info(f"Loading CAMeL-BERT boundaries from {boundaries_file}")
    with open(boundaries_file, 'r', encoding='utf-8') as f:
        boundaries_data = json.load(f)

    logger.info(f"Loading corpus from {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    boundaries = boundaries_data.get('khabar_boundaries', [])
    logger.info(f"Found {len(boundaries)} boundary clusters")

    narrative_units = []

    for unit_id, boundary in enumerate(boundaries):
        char_start = boundary.get('char_start', 0)
        char_end = boundary.get('char_end', 0)

        # Extract text for this unit
        unit_text = corpus[char_start:char_end]

        # Detect isnad
        isnad_text = extract_isnad_from_text(unit_text)
        has_isnad = isnad_text is not None

        unit = {
            'unit_id': unit_id,
            'type': 'khabar',
            'char_start': char_start,
            'char_end': char_end,
            'has_isnad': has_isnad,
            'isnad_text': isnad_text,
            'text_length': char_end - char_start,
        }

        narrative_units.append(unit)

    logger.info(f"Created {len(narrative_units)} narrative units")
    logger.info(f"Units with isnad: {sum(1 for u in narrative_units if u['has_isnad'])}")

    return narrative_units


def segment_from_baseline_segments(
    segments_file: str,
    corpus_file: str
) -> list[dict]:
    """
    Create narrative units from Baseline v4 segment positions.

    Args:
        segments_file: Path to baseline_v4_*_segments.json
        corpus_file: Path to cleaned corpus text

    Returns:
        List of narrative unit dicts
    """
    logger.info(f"Loading Baseline segments from {segments_file}")
    with open(segments_file, 'r', encoding='utf-8') as f:
        segments_data = json.load(f)

    logger.info(f"Loading corpus from {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    segments = segments_data.get('segments', [])
    logger.info(f"Found {len(segments)} segments")

    narrative_units = []

    for segment in segments:
        unit_id = segment.get('id', 0)
        char_start = segment.get('start', 0)
        char_end = segment.get('end', 0)

        # Extract text for this unit
        unit_text = corpus[char_start:char_end]

        # Detect isnad
        isnad_text = extract_isnad_from_text(unit_text)
        has_isnad = isnad_text is not None

        unit = {
            'unit_id': unit_id,
            'type': 'khabar',
            'char_start': char_start,
            'char_end': char_end,
            'has_isnad': has_isnad,
            'isnad_text': isnad_text,
            'text_length': char_end - char_start,
        }

        narrative_units.append(unit)

    logger.info(f"Created {len(narrative_units)} narrative units")
    logger.info(f"Units with isnad: {sum(1 for u in narrative_units if u['has_isnad'])}")

    return narrative_units


def save_narrative_units(units: list[dict], output_path: str) -> None:
    """Save narrative units to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'total_units': len(units),
            'units_with_isnad': sum(1 for u in units if u['has_isnad']),
        },
        'narrative_units': units
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(units)} narrative units to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert boundary positions to narrative units'
    )

    # Mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--boundaries',
        help='Path to CAMeL-BERT char_boundaries.json'
    )
    input_group.add_argument(
        '--segments',
        help='Path to Baseline v4 segments.json'
    )

    parser.add_argument(
        '--corpus',
        required=True,
        help='Path to cleaned corpus text file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save narrative units JSON'
    )

    args = parser.parse_args()

    try:
        if args.boundaries:
            units = segment_from_camelbert_boundaries(args.boundaries, args.corpus)
        else:
            units = segment_from_baseline_segments(args.segments, args.corpus)

        save_narrative_units(units, args.output)
        logger.info("Segmentation completed successfully")

    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        raise


if __name__ == '__main__':
    main()
