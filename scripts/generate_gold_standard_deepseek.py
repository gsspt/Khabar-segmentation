#!/usr/bin/env python3
"""
Generate gold standard segmentation using Deepseek API.

Sends Arabic text to Deepseek for zero-shot khabar segmentation.
Returns narrative unit boundaries with isnad detection.

Usage:
    python scripts/generate_gold_standard_deepseek.py \
      --text data/processed/[TEXT]_clean.txt \
      --model deepseek-chat \
      --output results/gold_standard_[TEXT]_deepseek.json \
      --api-key $DEEPSEEK_API_KEY

Environment:
    DEEPSEEK_API_KEY: Deepseek API key (or pass via --api-key)
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import deepseek client
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai library not installed. Install with: pip install openai")


DEEPSEEK_PROMPT_FR = """Analysez le texte arabe suivant et segmentez-le en unités narratives (khabars).

Pour chaque khabar identifié:
1. Position de début et fin (numéro de caractère dans le texte)
2. Indicateur si un isnad (chaîne de transmission) est présent
3. Texte de l'isnad si présent

Un isnad typique commence par: أخبرنا، حدثنا، قال، ذكر، أنبأنا، etc.

IMPORTANT: Répondez UNIQUEMENT en JSON valide, sans texte supplémentaire.

Format de réponse:
{
  "units": [
    {
      "unit_id": 0,
      "char_start": 0,
      "char_end": 250,
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد عن علي...",
      "text_preview": "أخبرنا... [premiers caractères du khabar]"
    }
  ]
}

TEXTE À ANALYSER:
---
{text}
---
"""

DEEPSEEK_PROMPT_EN = """Analyze the following Arabic text and segment it into narrative units (khabars).

For each khabar identified:
1. Start and end position (character number in the text)
2. Whether an isnad (transmission chain) is present
3. The isnad text if present

An isnad typically starts with: أخبرنا، حدثنا، قال، ذكر، أنبأنا، etc.

IMPORTANT: Respond ONLY with valid JSON, no additional text.

Response format:
{
  "units": [
    {
      "unit_id": 0,
      "char_start": 0,
      "char_end": 250,
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد عن علي...",
      "text_preview": "أخبرنا... [first characters of khabar]"
    }
  ]
}

TEXT TO ANALYZE:
---
{text}
---
"""


def call_deepseek_api(
    text: str,
    api_key: str,
    model: str = 'deepseek-chat',
    language: str = 'fr'
) -> Optional[dict]:
    """
    Call Deepseek API to segment text into narrative units.

    Args:
        text: Arabic text to segment
        api_key: Deepseek API key
        model: Model name (default: deepseek-chat)
        language: Prompt language ('fr' or 'en')

    Returns:
        Parsed JSON response with units, or None if failed
    """
    if not HAS_OPENAI:
        raise ImportError("openai library required. Install with: pip install openai")

    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Select prompt
    prompt_template = DEEPSEEK_PROMPT_FR if language == 'fr' else DEEPSEEK_PROMPT_EN
    prompt = prompt_template.format(text=text)

    logger.info(f"Calling Deepseek API with model: {model}")
    logger.info(f"Text length: {len(text)} chars")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for deterministic output
            max_tokens=8000,
        )

        response_text = response.choices[0].message.content
        logger.info("Received response from Deepseek API")

        # Extract JSON from response
        # Try direct parsing first
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            logger.error("Could not parse JSON from response")
            logger.debug(f"Raw response: {response_text[:500]}")
            return None

    except Exception as e:
        logger.error(f"Error calling Deepseek API: {e}")
        return None


def validate_units(units: list[dict], text_length: int) -> list[dict]:
    """
    Validate and clean unit boundaries.

    Args:
        units: List of unit dicts from API
        text_length: Length of the original text

    Returns:
        Validated units with corrected boundaries
    """
    validated = []

    for unit in units:
        char_start = unit.get('char_start', 0)
        char_end = unit.get('char_end', 0)

        # Clamp boundaries to text length
        char_start = max(0, min(char_start, text_length))
        char_end = max(char_start, min(char_end, text_length))

        # Skip zero-length units
        if char_start >= char_end:
            logger.warning(f"Skipping zero-length unit: {char_start}-{char_end}")
            continue

        unit['char_start'] = char_start
        unit['char_end'] = char_end

        validated.append(unit)

    return validated


def generate_gold_standard(
    text_file: str,
    api_key: str,
    model: str = 'deepseek-chat',
    language: str = 'fr'
) -> dict:
    """
    Generate gold standard segmentation from Deepseek API.

    Args:
        text_file: Path to cleaned Arabic text
        api_key: Deepseek API key
        model: Model to use
        language: Prompt language

    Returns:
        Dict with narrative units and metadata
    """
    logger.info(f"Loading text from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    logger.info(f"Text length: {len(text)} chars, {len(text.split())} words")

    # Call API
    api_response = call_deepseek_api(text, api_key, model, language)
    if not api_response:
        raise RuntimeError("Failed to get response from Deepseek API")

    # Extract units
    units = api_response.get('units', [])
    logger.info(f"Received {len(units)} units from API")

    # Validate boundaries
    units = validate_units(units, len(text))
    logger.info(f"Validated {len(units)} units after boundary checks")

    # Ensure sequential unit_ids
    for idx, unit in enumerate(units):
        unit['unit_id'] = idx

    # Count units with isnad
    units_with_isnad = sum(1 for u in units if u.get('has_isnad', False))

    result = {
        'metadata': {
            'source': 'deepseek-api',
            'model': model,
            'text_length_chars': len(text),
            'total_units': len(units),
            'units_with_isnad': units_with_isnad,
        },
        'narrative_units': units
    }

    return result


def save_gold_standard(gold_standard: dict, output_path: str) -> None:
    """Save gold standard to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved gold standard to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate gold standard using Deepseek API'
    )
    parser.add_argument(
        '--text',
        required=True,
        help='Path to cleaned Arabic text'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save gold standard JSON'
    )
    parser.add_argument(
        '--model',
        default='deepseek-chat',
        help='Deepseek model name (default: deepseek-chat)'
    )
    parser.add_argument(
        '--api-key',
        help='Deepseek API key (or set DEEPSEEK_API_KEY env var)'
    )
    parser.add_argument(
        '--language',
        choices=['fr', 'en'],
        default='fr',
        help='Prompt language (default: fr)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError(
            "Deepseek API key required. "
            "Set via --api-key or DEEPSEEK_API_KEY env var"
        )

    try:
        gold_standard = generate_gold_standard(
            args.text,
            api_key,
            args.model,
            args.language
        )
        save_gold_standard(gold_standard, args.output)
        logger.info("Gold standard generation completed successfully")

    except Exception as e:
        logger.error(f"Error generating gold standard: {e}")
        raise


if __name__ == '__main__':
    main()
