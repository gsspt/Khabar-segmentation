#!/usr/bin/env python3
"""
Script to clean OpenITI texts from metadata while preserving text integrity.

Process:
1. Remove header section (between ### |HEADER-METADATA| and ### |END HEADER|)
2. Remove footer section (after ### |END HEADER|)
3. Remove markup lines (lines starting with ###)
4. Normalize whitespace (multiple newlines → single newline)
5. Preserve text content exactly

Usage:
    python scripts/clean_openiti_text.py \
      --input path/to/0406IbnHabib.raw.txt \
      --output data/processed/0406IbnHabib_clean.txt
"""

import argparse
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_openiti_text(input_path: str, output_path: str) -> dict[str, int]:
    """
    Clean OpenITI text by removing metadata and normalizing whitespace.

    Args:
        input_path: Path to raw OpenITI file
        output_path: Path to save cleaned text

    Returns:
        Dictionary with statistics about the cleaning process
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Reading input file: {input_path}")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_length = len(content)
    logger.info(f"Original file size: {original_length} characters")

    # Step 1: Remove header section (### |HEADER-METADATA| to ### |END HEADER|)
    header_pattern = r'###\s*\|HEADER-METADATA\|.*?###\s*\|END HEADER\|'
    content = re.sub(header_pattern, '', content, flags=re.DOTALL)
    logger.debug("Removed header metadata section")

    # Step 2: Remove footer section (everything after ### |END HEADER| or similar markers)
    footer_pattern = r'###\s*\|.*FOOTER.*\|.*'
    content = re.sub(footer_pattern, '', content, flags=re.DOTALL)
    logger.debug("Removed footer section")

    # Step 3: Remove markup lines (lines starting with ### or #META#)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are purely markup (start with ### or #META#)
        if not stripped.startswith('###') and not stripped.startswith('#META#'):
            cleaned_lines.append(line)

    content = '\n'.join(cleaned_lines)
    logger.debug("Removed markup lines")

    # Step 4: Normalize whitespace
    # Replace multiple consecutive newlines with single newline
    content = re.sub(r'\n\n+', '\n', content)
    logger.debug("Normalized whitespace")

    # Step 5: Strip leading/trailing whitespace from entire document
    content = content.strip()

    final_length = len(content)
    removed_length = original_length - final_length

    logger.info(f"Cleaned file size: {final_length} characters")
    logger.info(f"Removed: {removed_length} characters ({removed_length/original_length*100:.1f}%)")

    # Write cleaned text
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Cleaned text saved to: {output_path}")

    # Count lines and words
    num_lines = len(content.split('\n'))
    num_words = len(content.split())

    return {
        'original_chars': original_length,
        'cleaned_chars': final_length,
        'removed_chars': removed_length,
        'lines': num_lines,
        'words': num_words
    }


def main():
    parser = argparse.ArgumentParser(
        description='Clean OpenITI texts from metadata'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to raw OpenITI file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save cleaned text'
    )

    args = parser.parse_args()

    try:
        stats = clean_openiti_text(args.input, args.output)
        logger.info(f"Statistics:")
        logger.info(f"  Original: {stats['original_chars']:,} chars")
        logger.info(f"  Cleaned: {stats['cleaned_chars']:,} chars")
        logger.info(f"  Lines: {stats['lines']:,}")
        logger.info(f"  Words: {stats['words']:,}")
        logger.info("Cleaning completed successfully")
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
        raise


if __name__ == '__main__':
    main()
