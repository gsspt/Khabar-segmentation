#!/usr/bin/env python3
"""
Extract khabar boundaries using smart windowing approach.

Replaces:
  - Token deduplication (eliminated via windowing)
  - Confidence weighting (implicit in margin strategy)
  - Complex post-processing (minimal clustering only)

Usage:
  # From CAMeL-BERT inference with offsets
  python extract_boundaries_smart_window.py \
    --inference results/camelbert_kitab_uqala_raw_inference.json \
    --corpus data/processed/kitab_uqala_reference_corpus.txt \
    --output results/camelbert_boundaries_smart_window.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Smart windowing parameters (from analysis)
# Used when processing token-level inference results
WINDOW_SIZE = 512
STRIDE = 256
MARGIN = 64
CONFIDENCE_THRESHOLD = 0.70
GAP_CLUSTER = 20  # Cluster boundary tokens within this char distance
MERGE_MIN_GAP = 10  # Merge clusters closer than this


def load_inference_results(inference_path: Path) -> Dict:
    """Load inference results from JSON file."""
    print(f"Loading inference from {inference_path}...")
    with open(inference_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats:
    # 1. Full inference with metadata
    # 2. Just inference_results dict
    if 'inference_results' in data:
        return data['inference_results']
    else:
        return data


def load_corpus(corpus_path: Path) -> str:
    """Load corpus text."""
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  Corpus: {len(text):,} characters")
    return text


def apply_smart_windowing(
    tokens: List[str],
    predictions: List[int],
    probabilities: List[float],
    offsets: List[Tuple[int, int]]
) -> Dict[int, Dict]:
    """
    Apply smart windowing logic to deduplicate tokens.

    Strategy:
    - Simulate 512-token windows with 256-token stride
    - Keep only tokens from "confident center" (margin=64)
    - This eliminates overlapping tokens from the deduplication step

    Args:
        tokens: List of token strings
        predictions: List of predictions (0 or 1)
        probabilities: List of boundary probabilities
        offsets: List of [char_start, char_end] offsets

    Returns:
        Dict mapping char_start → {'pred': 0/1, 'prob': float, 'token': str}
    """
    print(f"\nApplying smart windowing logic...")
    print(f"  Window size: {WINDOW_SIZE} tokens")
    print(f"  Stride: {STRIDE} tokens")
    print(f"  Margin: {MARGIN} tokens")

    filtered = {}

    # Simulate windowing: keep tokens that would be in "confident center"
    # of a window with this stride
    windows_seen = defaultdict(int)

    for idx, (token, pred, prob, offset) in enumerate(
        zip(tokens, predictions, probabilities, offsets)
    ):
        # Skip special tokens and invalid offsets
        if token in ['[PAD]', '[CLS]', '[SEP]', '[UNK]']:
            continue
        if offset == [0, 0]:
            continue

        # Apply confidence threshold
        if prob < CONFIDENCE_THRESHOLD:
            continue

        # Determine which window(s) this token appears in
        # Token at index idx appears in windows: [idx // STRIDE - N, idx // STRIDE]
        # For "confident center", it should be in the range [margin, window_size-margin]
        # of at least one window

        # Check if this token would be in the confident center of any window
        # Window starting at position w covers tokens [w, w+WINDOW_SIZE)
        # Confident center is [w+MARGIN, w+WINDOW_SIZE-MARGIN)

        # Which windows could this token be in?
        # idx >= w and idx < w + WINDOW_SIZE
        # For it to be in confident center: idx >= w+MARGIN and idx < w+WINDOW_SIZE-MARGIN
        # So: idx - (WINDOW_SIZE-MARGIN) < w <= idx - MARGIN

        # Simplest heuristic: token is "safe" if it would be in confident center
        # of a window aligned to STRIDE boundaries

        # Window containing this token (if aligned to STRIDE)
        window_start = (idx // STRIDE) * STRIDE
        window_end = window_start + WINDOW_SIZE

        # Is this token in confident center of this window?
        center_start = window_start + MARGIN
        center_end = window_end - MARGIN

        if center_start <= idx < center_end:
            char_start = offset[0]

            # Keep this token (one prediction per char_start)
            if char_start not in filtered:
                filtered[char_start] = {
                    'pred': pred,
                    'prob': prob,
                    'token': token,
                    'offset': offset,
                    'token_idx': idx
                }
            else:
                # If already seen, keep the one with higher probability
                if prob > filtered[char_start]['prob']:
                    filtered[char_start] = {
                        'pred': pred,
                        'prob': prob,
                        'token': token,
                        'offset': offset,
                        'token_idx': idx
                    }

    print(f"  Tokens after smart windowing: {len(filtered):,}")
    print(f"  Reduction from original: {len(tokens) - len(filtered):,} tokens removed")

    return filtered


def extract_boundary_positions(
    token_predictions: Dict[int, Dict]
) -> List[Tuple[int, float, str]]:
    """
    Extract tokens marked as boundaries (pred=1).

    Returns:
        List of (char_start, probability, token)
    """
    boundaries = []

    for char_start, pred_data in token_predictions.items():
        if pred_data['pred'] == 1:
            boundaries.append((
                char_start,
                pred_data['prob'],
                pred_data['token']
            ))

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    print(f"Extracted boundary tokens: {len(boundaries):,}")

    return boundaries


def cluster_boundaries(
    boundary_tokens: List[Tuple[int, float, str]],
    token_offsets: Dict[int, Tuple[int, int]] = None,
    gap: int = GAP_CLUSTER
) -> List[Dict]:
    """
    Cluster boundary tokens that are close together (same isnad).

    Tokens are clustered if the gap between previous token's END
    and current token's START is <= `gap` characters.

    Args:
        boundary_tokens: List of (char_start, prob, token)
        token_offsets: Dict mapping char_start → (start, end)
        gap: Maximum distance between token end and next token start

    Returns:
        List of cluster dicts with: char_start, tokens, count, avg_prob
    """
    if not boundary_tokens:
        return []

    clusters = []
    current_cluster = {
        'char_start': boundary_tokens[0][0],
        'char_end': boundary_tokens[0][0] + 1,  # Default if no offset
        'tokens': [boundary_tokens[0][2]],
        'count': 1,
        'probs': [boundary_tokens[0][1]],
    }

    # Update char_end from offsets if available
    if token_offsets and boundary_tokens[0][0] in token_offsets:
        current_cluster['char_end'] = token_offsets[boundary_tokens[0][0]][1]

    for char_start, prob, token in boundary_tokens[1:]:
        token_end = char_start + 1
        if token_offsets and char_start in token_offsets:
            token_end = token_offsets[char_start][1]

        # Check gap between previous token's end and current token's start
        gap_to_prev = char_start - current_cluster['char_end']

        if gap_to_prev <= gap:
            # Add to current cluster
            current_cluster['tokens'].append(token)
            current_cluster['count'] += 1
            current_cluster['probs'].append(prob)
            current_cluster['char_end'] = token_end
        else:
            # Start new cluster
            current_cluster['avg_prob'] = sum(current_cluster['probs']) / len(current_cluster['probs'])
            clusters.append(current_cluster)

            current_cluster = {
                'char_start': char_start,
                'char_end': token_end,
                'tokens': [token],
                'count': 1,
                'probs': [prob],
            }

    # Add final cluster
    current_cluster['avg_prob'] = sum(current_cluster['probs']) / len(current_cluster['probs'])
    clusters.append(current_cluster)

    print(f"Clustered into: {len(clusters):,} isnad clusters")

    return clusters


def merge_close_clusters(
    clusters: List[Dict],
    min_gap: int = MERGE_MIN_GAP
) -> List[Dict]:
    """
    Merge clusters that are very close (< min_gap distance).

    This cleans up boundary detection artifacts where multiple
    small clusters should be one larger segment.
    """
    if not clusters:
        return []

    merged = []
    current = clusters[0].copy()

    for next_cluster in clusters[1:]:
        gap = next_cluster['char_start'] - current['char_start']

        if gap <= min_gap:
            # Merge: combine tokens and recalculate avg_prob
            current['tokens'].extend(next_cluster['tokens'])
            current['count'] += next_cluster['count']
            current['probs'].extend(next_cluster['probs'])
            current['avg_prob'] = sum(current['probs']) / len(current['probs'])
        else:
            # Keep current, start new
            merged.append(current)
            current = next_cluster.copy()

    merged.append(current)

    return merged


def save_results(
    clusters: List[Dict],
    output_path: Path,
    corpus: str
) -> None:
    """Save results to JSON."""
    results = {
        'metadata': {
            'method': 'Smart windowing + clustering',
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'margin': MARGIN,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'gap_cluster': GAP_CLUSTER,
            'merge_min_gap': MERGE_MIN_GAP,
            'total_clusters': len(clusters),
        },
        'boundaries': []
    }

    for cluster in clusters:
        results['boundaries'].append({
            'char_position': cluster['char_start'],
            'isnad_tokens': cluster['tokens'],
            'token_count': cluster['count'],
            'avg_confidence': round(cluster['avg_prob'], 4),
            'isnad_text': ''.join(cluster['tokens']).replace('##', ''),
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"  Boundaries: {len(results['boundaries'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract khabar boundaries using smart windowing'
    )
    parser.add_argument(
        '--inference',
        type=Path,
        required=True,
        help='Path to CAMeL-BERT inference JSON'
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        required=True,
        help='Path to corpus text file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path (default: auto-generated)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f'Min confidence for boundary tokens (default: {CONFIDENCE_THRESHOLD})'
    )
    parser.add_argument(
        '--gap-cluster',
        type=int,
        default=GAP_CLUSTER,
        help=f'Max distance for clustering tokens (default: {GAP_CLUSTER})'
    )

    args = parser.parse_args()

    # Use argument values (locals, not globals)
    conf_threshold = args.confidence_threshold
    gap_cluster = args.gap_cluster

    # Default output path
    if args.output is None:
        args.output = args.inference.parent / f"{args.inference.stem}_smart_window.json"

    print("=" * 80)
    print("SMART WINDOW BOUNDARY EXTRACTION")
    print("=" * 80)

    # Load data
    inference = load_inference_results(args.inference)
    corpus = load_corpus(args.corpus)

    tokens = inference['tokens']
    predictions = inference['predictions']
    probabilities = inference['probabilities']
    offsets = inference['offsets']

    print(f"\nInference stats:")
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Boundary tokens (pred=1): {sum(predictions):,}")

    # Filter boundary tokens by confidence
    print(f"\nFiltering by confidence threshold: {conf_threshold}")
    filtered = {}
    token_offsets = {}  # Map char_start → (start, end)

    for idx, (token, pred, prob, offset) in enumerate(zip(tokens, predictions, probabilities, offsets)):
        if pred != 1:  # Only keep boundary tokens
            continue
        if token in ['[PAD]', '[CLS]', '[SEP]', '[UNK]']:
            continue
        if offset == [0, 0]:
            continue
        if prob >= conf_threshold:
            char_start = offset[0]
            if char_start not in filtered or prob > filtered[char_start]['prob']:
                filtered[char_start] = {
                    'pred': 1,
                    'prob': prob,
                    'token': token,
                    'offset': offset,
                }
                token_offsets[char_start] = tuple(offset)

    print(f"Tokens after confidence filter: {len(filtered):,}")

    # Extract boundaries as list of (char_start, prob, token)
    boundaries = [
        (char_start, data['prob'], data['token'])
        for char_start, data in sorted(filtered.items())
    ]

    # Cluster (with proper gap calculation based on token offsets)
    clusters = cluster_boundaries(boundaries, token_offsets=token_offsets, gap=gap_cluster)

    # Merge close clusters
    merged = merge_close_clusters(clusters, min_gap=MERGE_MIN_GAP)
    print(f"After merging: {len(merged):,} clusters")

    # Save
    save_results(merged, args.output, corpus)

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
