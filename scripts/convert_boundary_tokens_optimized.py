#!/usr/bin/env python3
"""
Optimized conversion of CAMeL-BERT raw inference to khabar boundaries.

Improvements over direct version:
1. Confidence thresholding (filter low-probability predictions)
2. Adaptive gap clustering (data-driven instead of hardcoded)
3. Merge nearby clusters (resolve fragmented isnads)
4. Boundary validation (check corpus bounds)
5. Configurable parameters for experimentation

Usage:
    python scripts/convert_boundary_tokens_optimized.py \
      --input results/[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/[TEXT]_boundaries_optimized.json \
      --confidence-threshold 0.70 \
      --gap-cluster adaptive
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_inference(input_path: str) -> dict:
    """Load raw inference JSON file."""
    logger.info(f"Loading raw inference from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'inference_results' in data:
        return data['inference_results']
    return data


def load_corpus(corpus_path: str) -> str:
    """Load corpus text."""
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_boundary_tokens(
    tokens: list,
    offsets: list,
    predictions: list,
    probabilities: list,
    confidence_threshold: float = 0.70
) -> dict:
    """
    Extract and deduplicate boundary tokens by char_start.

    Returns:
        Dict mapping char_start → (token, offset, probability)
    """
    logger.info(f"Extracting boundary tokens (confidence >= {confidence_threshold})")

    boundary_by_char = {}
    skipped_low_conf = 0

    for i, (tok, off, pred, prob) in enumerate(zip(tokens, offsets, predictions, probabilities)):
        if pred != 1:
            continue

        # Skip special tokens
        if tok in {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}:
            continue

        # Skip invalid offsets
        if off == [0, 0]:
            continue

        # Skip low-confidence
        if prob < confidence_threshold:
            skipped_low_conf += 1
            continue

        char_start = off[0]
        if char_start not in boundary_by_char or prob > boundary_by_char[char_start][2]:
            boundary_by_char[char_start] = (tok, off, prob)

    logger.info(f"  Boundary positions: {len(boundary_by_char)}")
    logger.info(f"  Skipped (low confidence): {skipped_low_conf}")

    return boundary_by_char


def determine_gap_cluster(
    boundary_positions: dict,
    mode: str = "adaptive"
) -> int:
    """
    Determine optimal GAP_CLUSTER value.

    Modes:
    - "adaptive": Use 75th percentile + 0.5*IQR
    - int: Use fixed value
    """
    char_starts = sorted(boundary_positions.keys())

    if len(char_starts) < 10:
        logger.warning("Too few boundary positions for adaptive clustering")
        return 50

    if mode == "adaptive":
        # Calculate gaps between consecutive boundary tokens
        gaps = [char_starts[i+1] - char_starts[i] for i in range(len(char_starts)-1)]

        # Get quartiles
        try:
            q1, q2, q3 = statistics.quantiles(gaps, n=4)
        except Exception as e:
            logger.warning(f"Could not calculate quantiles: {e}")
            return 50

        iqr = q3 - q1
        gap_cluster = int(q3 + 0.5 * iqr)

        logger.info(f"Adaptive GAP_CLUSTER calculation:")
        logger.info(f"  Min gap: {min(gaps)} chars")
        logger.info(f"  Q1 (25%): {q1:.0f} chars")
        logger.info(f"  Q3 (75%): {q3:.0f} chars")
        logger.info(f"  IQR: {iqr:.0f} chars")
        logger.info(f"  GAP_CLUSTER = Q3 + 0.5*IQR = {gap_cluster}")

        return gap_cluster
    else:
        return mode


def cluster_boundary_tokens(
    boundary_positions: dict,
    gap_cluster: int
) -> list:
    """
    Cluster boundary tokens into isnad groups.
    """
    logger.info(f"Clustering with GAP_CLUSTER={gap_cluster}")

    sorted_chars = sorted(boundary_positions.keys())
    clusters = []

    if not sorted_chars:
        return clusters

    cur_start = sorted_chars[0]
    cur_end = boundary_positions[sorted_chars[0]][1][1]
    cur_tokens = [sorted_chars[0]]

    for char_start in sorted_chars[1:]:
        off = boundary_positions[char_start][1]
        gap = char_start - cur_end

        if gap <= gap_cluster:
            # Extend cluster
            cur_end = max(cur_end, off[1])
            cur_tokens.append(char_start)
        else:
            # Close cluster, start new one
            clusters.append({
                'char_start': cur_start,
                'char_end': cur_end,
                'n_tokens': len(cur_tokens),
                'token_positions': cur_tokens
            })
            cur_start = char_start
            cur_end = off[1]
            cur_tokens = [char_start]

    # Add last cluster
    clusters.append({
        'char_start': cur_start,
        'char_end': cur_end,
        'n_tokens': len(cur_tokens),
        'token_positions': cur_tokens
    })

    logger.info(f"  Created {len(clusters)} clusters")

    # Statistics
    sizes = [c['n_tokens'] for c in clusters]
    logger.info(f"  Cluster size stats: min={min(sizes)}, max={max(sizes)}, "
               f"avg={sum(sizes)/len(sizes):.1f}")

    return clusters


def merge_nearby_clusters(
    clusters: list,
    min_gap: int = 10
) -> list:
    """
    Merge clusters that are too close together.

    Args:
        clusters: List of cluster dicts
        min_gap: Minimum gap to keep clusters separate (chars)
    """
    logger.info(f"Merging clusters with gap < {min_gap} chars")

    if not clusters:
        return clusters

    merged = []
    i = 0

    while i < len(clusters):
        current = clusters[i].copy()

        # Try to merge with next cluster(s)
        while i + 1 < len(clusters):
            next_cluster = clusters[i + 1]
            gap = next_cluster['char_start'] - current['char_end']

            if gap < min_gap:
                # Merge
                current['char_end'] = next_cluster['char_end']
                current['n_tokens'] += next_cluster['n_tokens']
                current['token_positions'].extend(next_cluster['token_positions'])
                i += 1
            else:
                break

        merged.append(current)
        i += 1

    logger.info(f"  Merged: {len(clusters)} → {len(merged)} clusters")

    return merged


def validate_boundaries(
    clusters: list,
    corpus_len: int
) -> list:
    """
    Validate and fix boundary positions.
    """
    logger.info("Validating boundaries")

    validated = []
    fixed = 0

    for cluster in clusters:
        # Clamp to corpus bounds
        char_start = max(0, min(cluster['char_start'], corpus_len - 1))
        char_end = max(char_start + 1, min(cluster['char_end'], corpus_len))

        if char_start != cluster['char_start'] or char_end != cluster['char_end']:
            fixed += 1

        validated.append({
            'char_start': char_start,
            'char_end': char_end,
            'n_tokens': cluster['n_tokens'],
            'token_positions': cluster.get('token_positions', [])
        })

    if fixed > 0:
        logger.info(f"  Fixed {fixed} boundary positions")

    return validated


def convert_optimized(
    input_path: str,
    corpus_path: str,
    output_path: str,
    confidence_threshold: float = 0.70,
    gap_cluster: str = "adaptive",
    merge_close_clusters: bool = True,
    min_gap_merge: int = 10,
    verbose: bool = True
) -> None:
    """
    Optimized conversion pipeline.
    """
    logger.info("=" * 80)
    logger.info("CAMeL-BERT Optimized Conversion")
    logger.info("=" * 80)
    logger.info(f"Parameters:")
    logger.info(f"  confidence_threshold: {confidence_threshold}")
    logger.info(f"  gap_cluster: {gap_cluster}")
    logger.info(f"  merge_close_clusters: {merge_close_clusters}")
    logger.info(f"  min_gap_merge: {min_gap_merge}")

    # Load data
    raw = load_raw_inference(input_path)
    corpus = load_corpus(corpus_path)

    corpus_len = len(corpus)
    logger.info(f"\nCorpus: {corpus_len:,} characters")

    # Step 1: Extract boundary tokens
    boundary_by_char = extract_boundary_tokens(
        raw['tokens'],
        raw['offsets'],
        raw['predictions'],
        raw['probabilities'],
        confidence_threshold
    )

    # Step 2: Determine gap cluster
    if isinstance(gap_cluster, str) and gap_cluster == "adaptive":
        gap_cluster_value = determine_gap_cluster(boundary_by_char, mode="adaptive")
    else:
        gap_cluster_value = int(gap_cluster)

    # Step 3: Cluster
    clusters = cluster_boundary_tokens(boundary_by_char, gap_cluster_value)

    # Step 4: Merge (optional)
    if merge_close_clusters:
        clusters = merge_nearby_clusters(clusters, min_gap_merge)

    # Step 5: Validate
    clusters = validate_boundaries(clusters, corpus_len)

    # Step 6: Extract boundaries
    khabar_boundaries = [c['char_start'] for c in clusters]

    logger.info(f"\nOutput:")
    logger.info(f"  Khabar boundaries: {len(khabar_boundaries)}")

    # Save output
    output_data = {
        'metadata': {
            'method': 'optimized_with_parameters',
            'corpus': corpus_path,
            'corpus_length': corpus_len,
            'confidence_threshold': confidence_threshold,
            'gap_cluster_mode': gap_cluster,
            'gap_cluster_value': gap_cluster_value,
            'merge_close_clusters': merge_close_clusters,
            'min_gap_merge': min_gap_merge,
            'unique_boundary_positions': len(boundary_by_char),
            'n_clusters': len(clusters),
            'n_khabar_boundaries': len(khabar_boundaries)
        },
        'khabar_boundaries': khabar_boundaries,
        'clusters': [
            {
                'cluster_id': i,
                'char_start': c['char_start'],
                'char_end': c['char_end'],
                'length': c['char_end'] - c['char_start'],
                'n_tokens': c['n_tokens']
            }
            for i, c in enumerate(clusters)
        ]
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSaved to: {output_path}")
    logger.info("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimized CAMeL-BERT raw inference conversion'
    )
    parser.add_argument('--input', required=True, help='Raw inference JSON')
    parser.add_argument('--corpus', required=True, help='Cleaned corpus text')
    parser.add_argument('--output', required=True, help='Output JSON')
    parser.add_argument('--confidence-threshold', type=float, default=0.70,
                       help='Minimum confidence to keep boundary (default: 0.70)')
    parser.add_argument('--gap-cluster', default='adaptive',
                       help='Gap clustering mode: "adaptive" or integer (default: adaptive)')
    parser.add_argument('--merge-close-clusters', type=bool, default=True,
                       help='Merge clusters with gap < min-gap-merge')
    parser.add_argument('--min-gap-merge', type=int, default=10,
                       help='Min gap to keep clusters separate (default: 10)')

    args = parser.parse_args()

    convert_optimized(
        args.input,
        args.corpus,
        args.output,
        confidence_threshold=args.confidence_threshold,
        gap_cluster=args.gap_cluster,
        merge_close_clusters=args.merge_close_clusters,
        min_gap_merge=args.min_gap_merge
    )


if __name__ == '__main__':
    main()
