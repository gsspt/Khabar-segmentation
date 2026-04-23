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
import time
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def calculate_optimal_workers(num_chunks: int, max_workers: int = 8) -> int:
    """
    Calculate optimal number of workers for parallel API calls.

    Deepseek API has rate limits (~60 RPM typical).
    With each chunk taking ~10-30 seconds, we balance:
    - Not too many (avoid rate limiting)
    - Not too few (avoid sequential bottleneck)

    Heuristic: min(num_chunks, max_workers)
    """
    # For small number of chunks, use fewer workers to avoid rate limits
    if num_chunks <= 2:
        return 1
    elif num_chunks <= 6:
        return min(num_chunks, 4)
    elif num_chunks <= 20:
        return min(num_chunks, 6)
    else:
        return min(num_chunks, max_workers)


def call_deepseek_api(
    text: str,
    api_key: str,
    model: str = 'deepseek-chat',
    chunk_index: int = 0,
    total_chunks: int = 1,
    retry_count: int = 0,
    max_retries: int = 3
) -> Optional[dict]:
    """
    Call Deepseek API to segment a text chunk with retry logic.

    Args:
        text: Chunk text to analyze
        api_key: Deepseek API key
        model: Model name
        chunk_index: For logging purposes
        total_chunks: Total number of chunks
        retry_count: Current retry attempt
        max_retries: Maximum retries before giving up
    """
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


def aggressive_dedup(
    units: List[Dict],
    overlap_threshold: int = 50
) -> List[Dict]:
    """
    Remove duplicate units with stricter logic.

    A unit is considered duplicate if:
    - char_start within ±threshold
    - AND char_end within ±threshold

    This is more aggressive than simple merging and handles:
    - Same start, different end positions
    - Close boundaries from overlapping chunks
    """
    if not units:
        return []

    # Sort by start position, then by end position
    sorted_units = sorted(units, key=lambda u: (u['char_start'], u['char_end']))

    merged = []
    skip_until = -1

    for i, unit in enumerate(sorted_units):
        if i <= skip_until:
            continue  # Already merged as duplicate

        current = unit
        j = i + 1

        # Look ahead for duplicates
        while j < len(sorted_units):
            candidate = sorted_units[j]

            # Stop if start position too far away
            if candidate['char_start'] - current['char_start'] > overlap_threshold:
                break

            # Check if duplicate
            start_diff = abs(candidate['char_start'] - current['char_start'])
            end_diff = abs(candidate['char_end'] - current['char_end'])

            if start_diff <= overlap_threshold and end_diff <= overlap_threshold:
                # Keep unit with longer/better isnad_text
                curr_isnad_len = len(current.get('isnad_text', '')) if current.get('isnad_text') else 0
                cand_isnad_len = len(candidate.get('isnad_text', '')) if candidate.get('isnad_text') else 0

                if cand_isnad_len > curr_isnad_len:
                    current = candidate

                skip_until = j  # Mark this duplicate to skip
                j += 1
            else:
                # Not a duplicate, but might find more
                j += 1

        merged.append(current)

    return merged


def merge_overlapping_units(
    units: List[Dict]
) -> List[Dict]:
    """
    Merge overlapping units into single non-overlapping units.

    Problem: When chunks overlap (500 chars), Deepseek extracts units from both
    chunks, creating overlapping/nested results like:
    - Unit A: [223-1855]
    - Unit B: [725-1125] (completely inside A)

    Solution: Merge overlapping units by keeping the larger span.
    For conflicts, prefer the unit with isnad information.
    """
    if not units:
        return []

    # Sort by start position, then by descending end position (longer spans first)
    sorted_units = sorted(
        units,
        key=lambda u: (u['char_start'], -u['char_end'])
    )

    merged = []

    for unit in sorted_units:
        if not merged:
            merged.append(unit)
            continue

        # Check overlap with last merged unit
        last = merged[-1]

        # If no overlap, add as new unit
        if unit['char_start'] >= last['char_end']:
            merged.append(unit)
        else:
            # Overlap detected - merge them
            # Expand the span to cover both units
            new_start = min(last['char_start'], unit['char_start'])
            new_end = max(last['char_end'], unit['char_end'])

            # Choose isnad from whichever unit has one
            new_isnad = (
                unit.get('isnad_text') or last.get('isnad_text')
            )
            new_has_isnad = bool(new_isnad)

            # Update last merged unit
            merged[-1]['char_start'] = new_start
            merged[-1]['char_end'] = new_end
            merged[-1]['isnad_text'] = new_isnad
            merged[-1]['has_isnad'] = new_has_isnad
            merged[-1]['unit_type'] = (
                'avec_isnad' if new_has_isnad else 'sans_isnad'
            )
            # Mark as merged
            if 'merged_from' not in merged[-1]:
                merged[-1]['merged_from'] = [last.get('unit_id')]
            merged[-1]['merged_from'].append(unit.get('unit_id'))

    # Reassign unit IDs
    for idx, unit in enumerate(merged):
        unit['unit_id'] = idx

    return merged


def process_chunk_task(
    chunk_index: int,
    chunk_start: int,
    chunk_end: int,
    chunk_text: str,
    full_text: str,
    api_key: str,
    total_chunks: int
) -> Tuple[int, List[Dict]]:
    """
    Process a single chunk and return (chunk_index, processed_units).

    This is a helper function for parallel processing via ThreadPoolExecutor.
    """
    logger.info(f"[Worker] Processing chunk {chunk_index+1}/{total_chunks} (chars {chunk_start:,}-{chunk_end:,})")

    # Call API with retry
    api_response = call_deepseek_api(
        chunk_text,
        api_key,
        chunk_index=chunk_index,
        total_chunks=total_chunks
    )

    if not api_response:
        logger.warning(f"[Worker] Failed to process chunk {chunk_index}, returning empty")
        return (chunk_index, [])

    # Extract and process units
    chunk_units = api_response.get('units', [])
    logger.info(f"[Worker] Chunk {chunk_index}: Received {len(chunk_units)} units from API")

    processed_units = process_deepseek_units(chunk_units, full_text, chunk_start, chunk_end)
    logger.info(f"[Worker] Chunk {chunk_index}: Processed {len(processed_units)} units with found positions")

    return (chunk_index, processed_units)


def generate_gold_standard(
    text_file: str,
    chunk_size: int = 3500,
    overlap: int = 500
) -> dict:
    """Generate gold standard using Deepseek API with optimized prompt and parallel processing."""

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
    logger.info(f"Created {len(chunks)} chunks")

    # Estimate cost
    estimated_cost = len(chunks) * 0.001  # Rough estimate: $0.001 per chunk
    logger.info(f"Estimated cost: ~${estimated_cost:.2f} (actual may vary by chunk size)")

    # Calculate optimal workers
    optimal_workers = calculate_optimal_workers(len(chunks))
    logger.info(f"Using {optimal_workers} parallel workers for {len(chunks)} chunks\n")

    # Process chunks in parallel
    all_units = [None] * len(chunks)  # Preserve order
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_chunk_task,
                i, chunk_start, chunk_end, chunk_text,
                text, api_key, len(chunks)
            ): i
            for i, (chunk_start, chunk_end, chunk_text) in enumerate(chunks)
        }

        # Process results as they complete
        for future in as_completed(futures):
            chunk_idx, processed_units = future.result()
            all_units[chunk_idx] = processed_units
            completed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(chunks) - completed) / rate if rate > 0 else 0

            logger.info(f"Progress: {completed}/{len(chunks)} chunks ({100*completed//len(chunks)}%) - ETA: {remaining:.0f}s")

    # Flatten results (filter out any None from failed chunks)
    all_units = [u for chunk_units in all_units if chunk_units is not None for u in chunk_units]

    logger.info(f"\nTotal units from all chunks: {len(all_units)}")

    # Aggressive deduplication
    logger.info(f"Deduplicating units from overlapping chunks (±50 char threshold)...")
    deduplicated_units = aggressive_dedup(all_units, overlap_threshold=50)
    removed_count = len(all_units) - len(deduplicated_units)
    logger.info(f"Removed {removed_count} duplicate units")
    logger.info(f"After deduplication: {len(deduplicated_units)} unique units")

    # Merge any remaining overlapping units
    logger.info(f"Merging remaining overlapping units...")
    merged_units = merge_overlapping_units(deduplicated_units)
    merged_count = len(deduplicated_units) - len(merged_units)
    logger.info(f"Merged {merged_count} overlapping pairs")
    logger.info(f"After overlap merging: {len(merged_units)} final units")

    # Sort by position (already done in merge_overlapping_units, but ensure sorted)
    final_units = sorted(merged_units, key=lambda u: u['char_start'])
    for idx, unit in enumerate(final_units):
        unit['unit_id'] = idx

    # Count units with isnad
    units_with_isnad = sum(1 for u in final_units if u.get('has_isnad', False))

    result = {
        'metadata': {
            'source': 'deepseek-api-optimized',
            'model': 'deepseek-chat',
            'text_length_chars': text_length,
            'total_chunks': len(chunks),
            'chunk_size': chunk_size,
            'overlap': overlap,
            'total_units_before_dedup': len(all_units),
            'duplicates_removed': removed_count,
            'units_after_dedup': len(deduplicated_units),
            'units_merged_from_overlap': merged_count,
            'total_units': len(final_units),
            'units_with_isnad': units_with_isnad,
            'method': 'text-extraction-with-post-processing + aggressive-dedup + overlap-merge',
            'dedup_threshold': 50,
            'note': 'Positions determined via text search. Duplicates and overlapping units removed.',
            'parallel_workers': optimal_workers,
        },
        'narrative_units': final_units
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
