#!/usr/bin/env python3
"""
Generate gold standard segmentation using Deepseek API with optimized prompt.

Strategy:
1. Ask Deepseek to IDENTIFY units and EXTRACT text, NOT to calculate positions
2. Use post-processing to find exact character positions via text search
3. This avoids LLM mistakes in character counting

The key insight: LLMs are bad at precise character positions but excellent at:
- Identifying narrative boundaries
- Extracting relevant text passages
- Recognizing isnads and narrative structure

Usage:
    python scripts/deepseek_segmentation_optimized.py \\
      --text data/processed/alDarrab_clean.txt \\
      --output results/gold_standard_alDarrab_deepseek.json \\
      --chunk-size 3500 \\
      --overlap 500
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai library not installed. Install with: pip install openai")


# ============================================================================
# OPTIMIZED PROMPT: Request TEXT, not positions
# ============================================================================

DEEPSEEK_PROMPT_OPTIMIZED = """Analysez le texte arabe suivant et segmentez-le en unités narratives (khabars).

INSTRUCTIONS:
Pour chaque unité narrative identifiée, fournissez:
1. Le TEXTE EXACT du début de l'unité (premiers 50-100 caractères)
2. Le TEXTE EXACT de la fin de l'unité (derniers 50-100 caractères)
3. Indicateur si un isnad (chaîne de transmission) est présent
4. Si isnad présent: le TEXTE COMPLET de l'isnad
5. Classification: "avec_isnad" ou "sans_isnad"

Définition:
- Un isnad = chaîne de transmission (حدثنا، أخبرنا، قال، etc.)
- Un khabar = unité narrative (isnad + contenu narratif)

IMPORTANT - Répondez UNIQUEMENT en JSON valide:
{{
  "units": [
    {{
      "unit_id": 0,
      "text_start": "Les 50-100 premiers caractères du khabar",
      "text_end": "Les 50-100 derniers caractères du khabar",
      "unit_type": "avec_isnad",
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد بن علي عن أبيه... [isnad complet]"
    }},
    {{
      "unit_id": 1,
      "text_start": "Les premiers caractères...",
      "text_end": "Les derniers caractères...",
      "unit_type": "sans_isnad",
      "has_isnad": false,
      "isnad_text": null
    }}
  ]
}}

TEXTE À ANALYSER:
---
{text}
---
"""


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


def call_deepseek_api(
    text: str,
    api_key: str,
    model: str = 'deepseek-chat'
) -> Optional[dict]:
    """Call Deepseek API to segment a text chunk."""
    if not HAS_OPENAI:
        raise ImportError("openai library required. Install with: pip install openai")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = DEEPSEEK_PROMPT_OPTIMIZED.format(text=text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=8000,
        )

        response_text = response.choices[0].message.content

        # Extract JSON from response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Try markdown code block
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result

            logger.error(f"Could not parse JSON from response: {response_text[:200]}")
            return None

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def find_text_position(
    text: str,
    search_text: str,
    start_from: int = 0,
    tolerance: int = 10
) -> Optional[int]:
    """
    Find position of search_text in text, with tolerance for small variations.

    Args:
        text: Full text to search in
        search_text: Text to find
        start_from: Start searching from this position
        tolerance: Allow fuzzy matching within this char count

    Returns:
        Position if found, None otherwise
    """
    if not search_text or len(search_text) < 3:
        return None

    # Try exact match first
    pos = text.find(search_text, start_from)
    if pos != -1:
        return pos

    # Try without diacritics (normalize)
    normalized_search = re.sub(r'[\u064B-\u065F\u0670]', '', search_text)
    normalized_text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    pos = normalized_text.find(normalized_search, start_from)
    if pos != -1:
        return pos

    # Try with increased search window (fuzzy)
    # Look for partial matches
    search_start = min(20, len(search_text) // 2)
    partial = search_text[:search_start]

    pos = text.find(partial, start_from)
    if pos != -1:
        return pos

    return None


def process_deepseek_units(
    units: List[Dict],
    full_text: str,
    chunk_start: int,
    chunk_end: int
) -> List[Dict]:
    """
    Post-process Deepseek units to find exact character positions.

    Args:
        units: Units from Deepseek API
        full_text: Full corpus text
        chunk_start: Where this chunk starts in corpus
        chunk_end: Where this chunk ends in corpus

    Returns:
        Units with exact char_start/char_end positions found via text search
    """
    processed = []

    for unit in units:
        text_start = unit.get('text_start', '').strip()
        text_end = unit.get('text_end', '').strip()

        if not text_start:
            logger.warning(f"Unit {unit.get('unit_id')} has no text_start, skipping")
            continue

        # Find the start position in full text
        # Search from chunk_start or a bit before to handle edge cases
        search_from = max(0, chunk_start - 100)
        char_start = find_text_position(full_text, text_start, search_from)

        if char_start is None:
            logger.warning(f"Could not find text_start for unit {unit.get('unit_id')}: '{text_start[:50]}'")
            continue

        # Find the end position
        # Search after char_start
        search_from = char_start + len(text_start)
        char_end = find_text_position(full_text, text_end, search_from)

        if char_end is None:
            # Fallback: estimate based on text_start length and typical unit size
            logger.warning(f"Could not find text_end for unit {unit.get('unit_id')}, estimating")
            char_end = min(char_start + 400, len(full_text))
        else:
            # Move to end of matched text
            char_end = char_end + len(text_end)

        # Validate
        if char_start >= char_end:
            logger.warning(f"Invalid unit {unit.get('unit_id')}: start >= end")
            continue

        if char_start < 0 or char_end > len(full_text):
            logger.warning(f"Unit {unit.get('unit_id')} outside corpus bounds")
            continue

        # Build processed unit
        processed_unit = {
            'unit_id': unit.get('unit_id'),
            'char_start': char_start,
            'char_end': char_end,
            'has_isnad': unit.get('has_isnad', False),
            'isnad_text': unit.get('isnad_text'),
            'unit_type': unit.get('unit_type'),
            # Keep original text fragments for reference
            'text_start_from_api': text_start[:50],
            'text_end_from_api': text_end[:50],
        }

        processed.append(processed_unit)

    return processed


def merge_overlapping_units(
    all_units: List[Dict],
    overlap_threshold: int = 100
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

            # Consider duplicate if both start and end are very close
            if start_diff <= overlap_threshold and end_diff <= overlap_threshold:
                # Keep the one with more info (longer isnad_text if present)
                if unit.get('isnad_text') and len(unit.get('isnad_text', '')) > len(current.get('isnad_text', '')):
                    current = unit
            else:
                merged.append(current)
                current = unit

    if current:
        merged.append(current)

    return merged


def generate_gold_standard(
    text_file: str,
    chunk_size: int = 3500,
    overlap: int = 500
) -> dict:
    """Generate gold standard using Deepseek API with optimized prompt."""

    # Load text
    logger.info(f"Loading text from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_length = len(text)
    logger.info(f"Text length: {text_length:,} chars")

    # Get API key from .env
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file or environment")

    # Create chunks
    chunks = smart_chunk_text(text, chunk_size, overlap)
    logger.info(f"Created {len(chunks)} chunks\n")

    # Process chunks
    all_units = []

    for i, (chunk_start, chunk_end, chunk_text) in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} (chars {chunk_start:,}-{chunk_end:,})")

        # Call API
        api_response = call_deepseek_api(chunk_text, api_key)
        if not api_response:
            logger.warning(f"Failed to process chunk {i}, skipping")
            continue

        # Extract and process units
        chunk_units = api_response.get('units', [])
        logger.info(f"  Received {len(chunk_units)} units from API")

        processed_units = process_deepseek_units(chunk_units, text, chunk_start, chunk_end)
        logger.info(f"  Processed {len(processed_units)} units with found positions")

        all_units.extend(processed_units)

    # Merge overlapping units
    logger.info(f"\nMerging units from {len(chunks)} chunks...")
    merged_units = merge_overlapping_units(all_units)
    logger.info(f"After deduplication: {len(merged_units)} unique units")

    # Sort by position and reassign IDs
    merged_units = sorted(merged_units, key=lambda u: u['char_start'])
    for idx, unit in enumerate(merged_units):
        unit['unit_id'] = idx

    # Count units with isnad
    units_with_isnad = sum(1 for u in merged_units if u.get('has_isnad', False))

    result = {
        'metadata': {
            'source': 'deepseek-api-optimized',
            'model': 'deepseek-chat',
            'text_length_chars': text_length,
            'total_chunks': len(chunks),
            'chunk_size': chunk_size,
            'overlap': overlap,
            'total_units': len(merged_units),
            'units_with_isnad': units_with_isnad,
            'method': 'text-extraction-with-post-processing',
            'note': 'Positions determined via text search, not LLM calculation',
        },
        'narrative_units': merged_units
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate gold standard using Deepseek API (optimized)'
    )
    parser.add_argument('--text', required=True, help='Path to cleaned Arabic text')
    parser.add_argument('--output', required=True, help='Path to save gold standard JSON')
    parser.add_argument('--chunk-size', type=int, default=3500, help='Chunk size in chars')
    parser.add_argument('--overlap', type=int, default=500, help='Overlap in chars')

    args = parser.parse_args()

    try:
        logger.info("="*70)
        logger.info("Deepseek Gold Standard Generation (Optimized)")
        logger.info("="*70)

        gold_standard = generate_gold_standard(
            args.text,
            args.chunk_size,
            args.overlap
        )

        # Save
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gold_standard, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*70}")
        logger.info("✓ GENERATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Total units: {gold_standard['metadata']['total_units']}")
        logger.info(f"Units with isnad: {gold_standard['metadata']['units_with_isnad']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
