#!/usr/bin/env python3
"""
Optimized cleaning of OpenITI texts for NLP processing.

Process:
1. Remove OpenITI metadata header (#META# ... #META#Header#End#)
2. Remove page/section markers (### |, PageV00P000, etc.)
3. Handle line continuation markers (~~)
4. Remove inline annotations ([قال:], [ص: NN], ms1, etc.)
5. Normalize whitespace while preserving text structure
6. Validate integrity

Usage:
    python scripts/clean_openiti_text.py \
      --input data/alDarrab_Raw.Shamela0027093-ara1 \
      --output data/processed/alDarrab_clean.txt
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


def remove_header(text: str) -> str:
    """Remove OpenITI metadata header section."""
    # Find and remove everything from start up to and including #META#Header#End#
    match = re.search(r"^.*?#META#Header#End#\s*\n", text, re.DOTALL)
    if match:
        text = text[match.end() :]
        logger.debug("Removed #META#Header#End# section")
    # Also try alternate format if needed
    match = re.search(r"^.*?###\s*\|HEADER-METADATA\|.*?###\s*\|END HEADER\|",
                     text, re.DOTALL)
    if match:
        text = text[match.end() :]
        logger.debug("Removed ###|HEADER-METADATA| section")
    return text


def remove_metadata_markers(text: str) -> str:
    """Remove OpenITI metadata markers and page references."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Skip pure metadata lines
        if line.strip().startswith("#META#"):
            continue

        # Skip header start marker
        if line.strip().startswith("######OpenITI#"):
            continue

        # Remove inline page markers (PageV00P000, etc.)
        line = re.sub(r"\s*PageV\d+P\d+\s*$", "", line)

        # Remove section markers (### | 1-, ### | [ص: 23], etc.)
        if re.match(r"^###\s*\|", line):
            continue

        # Remove lines that are just comments/markers
        if re.match(r"^#\s*$", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def handle_line_continuations(text: str) -> str:
    """
    Handle ~~ line continuation markers.

    In OpenITI format, ~~ indicates that a line continues from the previous line.
    We merge them while preserving word boundaries.
    """
    lines = text.split("\n")
    merged_lines = []
    current_line = ""

    for line in lines:
        if line.startswith("~~"):
            # This is a continuation - merge with previous line
            continuation = line[2:].lstrip()  # Remove ~~ and leading spaces
            if current_line:
                current_line += " " + continuation
            else:
                current_line = continuation
        else:
            # New line - save previous if any
            if current_line:
                merged_lines.append(current_line)
            current_line = line

    # Don't forget the last line
    if current_line:
        merged_lines.append(current_line)

    logger.debug("Handled line continuations (~~)")
    return "\n".join(merged_lines)


def remove_inline_annotations(text: str) -> str:
    """Remove inline annotations and editor notes."""
    # Remove footnote markers like ms1, ms10, etc.
    text = re.sub(r"\s*ms\d+\s*", " ", text)

    # Remove inline page references [ص: NN]
    text = re.sub(r"\s*\[ص:\s*\d+\]", "", text)

    # Remove bracketed editor notes at line starts
    text = re.sub(r"^\s*\[.*?:\]\s*", "", text, flags=re.MULTILINE)

    logger.debug("Removed inline annotations")
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.

    - Remove trailing whitespace from lines
    - Collapse multiple blank lines into single blank line
    - Remove leading whitespace from content lines
    """
    lines = text.split("\n")
    normalized = []

    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()

        # Skip empty lines that would create excessive blank lines
        if not line:
            # Only add blank line if previous wasn't blank
            if normalized and normalized[-1] != "":
                normalized.append("")
        else:
            normalized.append(line)

    # Remove trailing blank lines
    while normalized and normalized[-1] == "":
        normalized.pop()

    logger.debug("Normalized whitespace")
    return "\n".join(normalized)


def validate_text(original: str, cleaned: str) -> dict:
    """Validate text integrity after cleaning."""
    orig_lines = original.count("\n")
    clean_lines = cleaned.count("\n")

    # Count actual content characters (rough measure)
    orig_content = len(original.replace(" ", "").replace("\n", ""))
    clean_content = len(cleaned.replace(" ", "").replace("\n", ""))

    # Extract some actual text to verify it's readable
    sample = "\n".join(cleaned.split("\n")[5:10])

    return {
        "original_lines": orig_lines,
        "cleaned_lines": clean_lines,
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "original_content": orig_content,
        "cleaned_content": clean_content,
        "content_retention": clean_content / orig_content if orig_content > 0 else 0,
        "sample_text": sample[:300],
    }


def clean_openiti_text(input_path: str, output_path: str) -> dict:
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
        original_text = f.read()

    original_length = len(original_text)
    logger.info(f"Original file size: {original_length:,} characters")

    # Step 1: Remove metadata header
    text = remove_header(original_text)

    # Step 2: Remove metadata markers
    text = remove_metadata_markers(text)

    # Step 3: Handle line continuations (~~)
    text = handle_line_continuations(text)

    # Step 4: Remove inline annotations
    text = remove_inline_annotations(text)

    # Step 5: Normalize whitespace
    text = normalize_whitespace(text)

    # Validation
    validation = validate_text(original_text, text)
    final_length = len(text)
    removed_length = original_length - final_length

    logger.info(f"Cleaned file size: {final_length:,} characters")
    logger.info(
        f"Removed: {removed_length:,} characters ({removed_length/original_length*100:.1f}%)"
    )
    logger.info(f"Content retention: {validation['content_retention']*100:.1f}%")

    # Write cleaned text
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    logger.info(f"Cleaned text saved to: {output_path}")
    logger.info(f"Lines: {validation['cleaned_lines']:,}")

    return {
        'original_chars': original_length,
        'cleaned_chars': final_length,
        'removed_chars': removed_length,
        'lines': validation['cleaned_lines'],
        'content_retention': validation['content_retention']
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
