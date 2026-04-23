#!/usr/bin/env python3
"""
Diagnose offset accuracy issues from raw inference JSON.

Finds patterns in mismatches to understand what's wrong.

Usage:
    python scripts/diagnose_offset_accuracy.py \
      --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def diagnose_offsets(tokens, offsets, corpus_text, sample_size=200):
    """Diagnose offset accuracy and identify mismatch patterns."""

    print(f"=== OFFSET ACCURACY DIAGNOSIS ===\n")
    print(f"Analyzing {sample_size} random tokens...")
    print(f"Corpus size: {len(corpus_text):,} chars\n")

    matches = 0
    mismatches = []
    error_patterns = defaultdict(int)

    for i in range(min(sample_size, len(tokens))):
        token = tokens[i]
        offset = offsets[i]
        char_start, char_end = offset

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            continue

        # Get text at offset
        if 0 <= char_start < len(corpus_text) and 0 <= char_end <= len(corpus_text):
            corpus_text_at_pos = corpus_text[char_start:char_end]
            token_clean = token.lstrip('#')

            if corpus_text_at_pos == token_clean:
                matches += 1
            else:
                mismatches.append({
                    'index': i,
                    'token': token,
                    'offset': offset,
                    'expected_in_corpus': corpus_text_at_pos,
                    'token_text': token_clean,
                    'match': False
                })

                # Categorize error
                if len(corpus_text_at_pos) != len(token_clean):
                    error_patterns['length_mismatch'] += 1
                elif corpus_text_at_pos.replace(' ', '') == token_clean.replace(' ', ''):
                    error_patterns['whitespace_diff'] += 1
                elif any(ord(c) > 127 for c in corpus_text_at_pos) and any(ord(c) > 127 for c in token_clean):
                    error_patterns['unicode_normalization'] += 1
                else:
                    error_patterns['character_mismatch'] += 1
        else:
            error_patterns['out_of_bounds'] += 1

    accuracy = 100 * matches / min(sample_size, len(tokens))

    print(f"Accuracy: {matches}/{min(sample_size, len(tokens))} = {accuracy:.1f}%\n")

    if accuracy < 95:
        print("⚠️  ACCURACY TOO LOW!")
        print(f"\nError breakdown:")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(mismatches) if mismatches else 0
            print(f"  {pattern:30s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n=== SAMPLE MISMATCHES ===\n")
        for mismatch in mismatches[:10]:
            print(f"Token {mismatch['index']:3d}: '{mismatch['token']}'")
            print(f"  Offset: {mismatch['offset']}")
            print(f"  Expected in corpus: {repr(mismatch['expected_in_corpus'][:50])}")
            print(f"  Token text: {repr(mismatch['token_text'][:50])}")
            print()

        print("DIAGNOSIS:")
        print("- Offsets may be chunk-relative, not absolute")
        print("- Check if chunk overlap handling is correct")
        print("- Verify tokenizer normalization settings")
        print("- Check for Unicode normalization issues (NFC vs NFD)")
        print("\nRECOMMENDATION:")
        print("→ Re-run improved notebook with fixes")
        print("→ If still <95%, investigate tokenizer settings in Colab")

        return False
    else:
        print("✓ Accuracy is acceptable (>95%)")
        print("✓ Offsets are ABSOLUTE corpus positions")
        print("✓ Safe to proceed with convert_boundary_tokens_direct.py")
        return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose offset accuracy")
    parser.add_argument("--raw_inference", required=True, help="Path to raw inference JSON")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of tokens to sample")

    args = parser.parse_args()

    # Load files
    with open(args.raw_inference, encoding="utf-8") as f:
        raw_data = json.load(f)

    corpus_text = Path(args.corpus).read_text(encoding="utf-8")

    results = raw_data["inference_results"]
    tokens = results["tokens"]
    offsets = results["offsets"]

    # Run diagnosis
    success = diagnose_offsets(tokens, offsets, corpus_text, sample_size=args.sample_size)

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
