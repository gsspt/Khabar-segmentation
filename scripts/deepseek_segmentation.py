#!/usr/bin/env python3
"""
Generate gold standard segmentation using Deepseek API with smart chunking.

Strategy:
1. Split text into manageable chunks (~3000-4000 chars each)
2. Use overlapping boundaries to preserve context
3. Send each chunk to Deepseek API
4. Convert chunk-relative positions to global positions
5. Merge results, deduplicating overlapping regions

Usage:
    python scripts/generate_gold_standard_deepseek_chunked.py \\
      --text data/processed/alDarrab_clean.txt \\
      --output results/gold_standard_alDarrab_deepseek.json \\
      --api-key $DEEPSEEK_API_KEY \\
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


DEEPSEEK_PROMPT_TEMPLATE = """Analysez le texte arabe suivant et segmentez-le en unités narratives (khabars).

Pour chaque khabar identifié:
1. Position de début et fin (numéro de caractère RELATIF AU DÉBUT DE CE TEXTE)
2. Indicateur si un isnad (chaîne de transmission) est présent
3. Texte de l'isnad si présent

Un isnad typique commence par: أخبرنا، حدثنا، قال، ذكر، أنبأنا، etc.

IMPORTANT:
- Répondez UNIQUEMENT en JSON valide, sans texte supplémentaire
- Les positions (char_start, char_end) doivent être RELATIVES AU TEXTE CI-DESSOUS
- Ne segmentez QUE les unités ENTIÈREMENT VISIBLES dans ce texte
- Si un khabar continue avant/après ce texte, incluez-le ENTIÈREMENT même s'il dépasse les limites

Format de réponse:
{{
  "units": [
    {{
      "char_start": 0,
      "char_end": 250,
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد عن علي..."
    }},
    {{
      "char_start": 250,
      "char_end": 500,
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
    """
    Split text into overlapping chunks at smart boundaries.

    Args:
        text: Full text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size to create

    Returns:
        List of (start, end, chunk_text) tuples
    """
    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Calculate chunk boundaries
        chunk_start = pos
        chunk_end = min(pos + chunk_size, text_len)

        # Try to find a smart break point (after newline or punctuation)
        if chunk_end < text_len:
            # Look for sentence/paragraph boundary in the last 200 chars
            search_start = max(chunk_end - 200, chunk_start)
            search_region = text[search_start:chunk_end]

            # Try to break at newline, then period, then space
            for delimiter in ['\n', '।', '،', ' ']:  # Arabic comma, delimiter
                last_pos = search_region.rfind(delimiter)
                if last_pos != -1:
                    chunk_end = search_start + last_pos + 1
                    break

        chunk_text = text[chunk_start:chunk_end]

        # Only add chunk if it meets minimum size or is the last one
        if len(chunk_text) >= min_chunk_size or chunk_end >= text_len:
            chunks.append((chunk_start, chunk_end, chunk_text))

        # Move to next chunk position
        overlap_chars = min(overlap, len(chunk_text) // 2)
        pos = chunk_end - overlap_chars

        # Avoid infinite loop on very small texts
        if chunk_end >= text_len:
            break

    logger.info(f"Split text into {len(chunks)} chunks")
    for i, (start, end, text) in enumerate(chunks):
        logger.info(f"  Chunk {i}: chars {start:,}-{end:,} ({len(text)} chars)")

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
    prompt = DEEPSEEK_PROMPT_TEMPLATE.format(text=text)

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


def convert_chunk_positions(
    chunk_units: List[Dict],
    chunk_start: int,
    chunk_end: int
) -> List[Dict]:
    """
    Convert chunk-relative positions to global positions.

    Args:
        chunk_units: Units with positions relative to chunk
        chunk_start: Global start position of chunk
        chunk_end: Global end position of chunk

    Returns:
        Units with global positions
    """
    converted = []

    for unit in chunk_units:
        char_start = unit.get('char_start', 0)
        char_end = unit.get('char_end', 0)

        # Skip invalid units
        if char_start >= char_end:
            continue

        # Convert to global positions
        global_start = chunk_start + char_start
        global_end = chunk_start + char_end

        # Clamp to chunk boundaries
        global_start = max(global_start, chunk_start)
        global_end = min(global_end, chunk_end)

        # Skip if still invalid
        if global_start >= global_end:
            continue

        unit['char_start'] = global_start
        unit['char_end'] = global_end

        converted.append(unit)

    return converted


def merge_overlapping_units(
    all_units: List[Dict],
    overlap_threshold: int = 50
) -> List[Dict]:
    """
    Merge units from overlapping chunks, removing duplicates.

    Args:
        all_units: All units from all chunks
        overlap_threshold: Max position difference to consider same unit

    Returns:
        Merged units with duplicates removed
    """
    if not all_units:
        return []

    # Sort by start position
    sorted_units = sorted(all_units, key=lambda u: u['char_start'])

    merged = []
    current = None

    for unit in sorted_units:
        if current is None:
            current = unit
        else:
            # Check if this is a duplicate or near-duplicate
            start_diff = abs(unit['char_start'] - current['char_start'])
            end_diff = abs(unit['char_end'] - current['char_end'])

            if start_diff <= overlap_threshold and end_diff <= overlap_threshold:
                # Same unit detected, keep the one with more info
                if len(unit.get('isnad_text', '')) > len(current.get('isnad_text', '')):
                    current = unit
            else:
                # Different unit
                merged.append(current)
                current = unit

    # Add the last unit
    if current:
        merged.append(current)

    return merged


def validate_units(units: List[Dict], text_length: int) -> List[Dict]:
    """Validate and clean unit boundaries."""
    validated = []

    for unit in units:
        char_start = unit.get('char_start', 0)
        char_end = unit.get('char_end', 0)

        # Clamp to text length
        char_start = max(0, min(char_start, text_length))
        char_end = max(char_start, min(char_end, text_length))

        # Skip zero-length units
        if char_start >= char_end:
            continue

        unit['char_start'] = char_start
        unit['char_end'] = char_end
        validated.append(unit)

    return validated


def generate_gold_standard(
    text_file: str,
    api_key: str,
    model: str = 'deepseek-chat',
    chunk_size: int = 3500,
    overlap: int = 500
) -> dict:
    """
    Generate gold standard using Deepseek API with smart chunking.

    Args:
        text_file: Path to cleaned text
        api_key: Deepseek API key
        model: Model name
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        Gold standard dict with narrative units
    """
    # Load text
    logger.info(f"Loading text from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_length = len(text)
    logger.info(f"Text length: {text_length:,} chars")

    # Create chunks
    chunks = smart_chunk_text(text, chunk_size, overlap)

    # Process chunks
    all_units = []

    for i, (chunk_start, chunk_end, chunk_text) in enumerate(chunks):
        logger.info(f"\nProcessing chunk {i+1}/{len(chunks)} (chars {chunk_start:,}-{chunk_end:,})")

        # Call API
        api_response = call_deepseek_api(chunk_text, api_key, model)
        if not api_response:
            logger.warning(f"Failed to process chunk {i}, skipping")
            continue

        # Extract and convert units
        chunk_units = api_response.get('units', [])
        logger.info(f"  Received {len(chunk_units)} units from API")

        converted_units = convert_chunk_positions(chunk_units, chunk_start, chunk_end)
        logger.info(f"  Converted to {len(converted_units)} global units")

        all_units.extend(converted_units)

    # Merge overlapping units
    logger.info(f"\nMerging units from {len(chunks)} chunks...")
    merged_units = merge_overlapping_units(all_units)
    logger.info(f"After deduplication: {len(merged_units)} unique units")

    # Validate boundaries
    validated_units = validate_units(merged_units, text_length)
    logger.info(f"After validation: {len(validated_units)} valid units")

    # Sort by position and assign IDs
    validated_units = sorted(validated_units, key=lambda u: u['char_start'])
    for idx, unit in enumerate(validated_units):
        unit['unit_id'] = idx

    # Count units with isnad
    units_with_isnad = sum(1 for u in validated_units if u.get('has_isnad', False))

    result = {
        'metadata': {
            'source': 'deepseek-api-chunked',
            'model': model,
            'text_length_chars': text_length,
            'total_chunks': len(chunks),
            'chunk_size': chunk_size,
            'overlap': overlap,
            'total_units': len(validated_units),
            'units_with_isnad': units_with_isnad,
        },
        'narrative_units': validated_units
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate gold standard using Deepseek API with smart chunking'
    )
    parser.add_argument('--text', required=True, help='Path to cleaned Arabic text')
    parser.add_argument('--output', required=True, help='Path to save gold standard JSON')
    parser.add_argument('--model', default='deepseek-chat', help='Model name')
    parser.add_argument('--api-key', help='API key (or set DEEPSEEK_API_KEY env var)')
    parser.add_argument('--chunk-size', type=int, default=3500, help='Chunk size in chars')
    parser.add_argument('--overlap', type=int, default=500, help='Overlap in chars')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("API key required (--api-key or DEEPSEEK_API_KEY env var)")

    try:
        gold_standard = generate_gold_standard(
            args.text,
            api_key,
            args.model,
            args.chunk_size,
            args.overlap
        )

        # Save
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gold_standard, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Gold standard saved to {output_file}")
        logger.info(f"  Total units: {gold_standard['metadata']['total_units']}")
        logger.info(f"  Units with isnad: {gold_standard['metadata']['units_with_isnad']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
