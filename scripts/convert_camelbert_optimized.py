#!/usr/bin/env python3
"""
CAMeL-BERT post-processing - Optimized for generalization across texts.

Combines two techniques:
1. Chunk position estimation (handles chunk-relative offsets)
2. Very high confidence filtering (extracts only strongest boundary signals)

This approach works for both well-studied texts (al-Darrab) and challenging texts (Ibn Jawzi).

Usage:
    python scripts/convert_camelbert_optimized.py \
      --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/[TEXT]/camelbert_[TEXT]_char_boundaries.json \
      --confidence_threshold 0.98 \
      --gap_cluster 20
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_chunk_positions(tokens: List[str], offsets: List[tuple], corpus_len: int) -> Dict[int, int]:
    """Estimate absolute chunk positions from offset patterns."""
    chunk_positions = {0: 0}
    sep_indices = [i for i, t in enumerate(tokens) if t == "[SEP]"]
    SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

    for chunk_id in range(1, len(sep_indices)):
        # Get offset range for previous chunk
        start_idx = sep_indices[chunk_id - 2] + 1 if chunk_id > 1 else 0
        prev_end_idx = sep_indices[chunk_id - 1]

        prev_offsets = []
        for i in range(start_idx, prev_end_idx):
            if tokens[i] not in SPECIAL_TOKENS and offsets[i] != [0, 0]:
                prev_offsets.append(offsets[i][0])

        # Get offset range for current chunk
        curr_start_idx = sep_indices[chunk_id - 1] + 1
        curr_end_idx = sep_indices[chunk_id]

        curr_offsets = []
        for i in range(curr_start_idx, curr_end_idx):
            if tokens[i] not in SPECIAL_TOKENS and offsets[i] != [0, 0]:
                curr_offsets.append(offsets[i][0])

        if prev_offsets and curr_offsets:
            # Estimate from overlap
            prev_max = max(prev_offsets)
            curr_min = min(curr_offsets)
            overlap = prev_max - curr_min

            chunk_positions[chunk_id] = chunk_positions[chunk_id - 1] + overlap
        else:
            # Fallback estimate
            chunk_positions[chunk_id] = chunk_positions[chunk_id - 1] + 500

        chunk_positions[chunk_id] = max(0, min(chunk_positions[chunk_id], corpus_len))

    return chunk_positions, sep_indices


def extract_boundaries(
    tokens: List[str],
    offsets: List[tuple],
    predictions: List[int],
    probabilities: List[float],
    corpus_text: str,
    chunk_positions: Dict[int, int],
    sep_indices: List[int],
    confidence_threshold: float = 0.98,
) -> Dict[int, Dict]:
    """Extract very high-confidence boundary positions."""
    SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}
    char_to_info = {}
    filtered_count = 0

    for tok_idx, (token, offset, pred, prob) in enumerate(
        zip(tokens, offsets, predictions, probabilities)
    ):
        # Only process boundary tokens
        if pred != 1:
            continue

        # Filter by confidence threshold
        if prob < confidence_threshold:
            filtered_count += 1
            continue

        # Skip special tokens
        if token in SPECIAL_TOKENS or offset == [0, 0]:
            continue

        # Find chunk
        chunk_id = 0
        for sep_idx in sep_indices:
            if tok_idx > sep_idx:
                chunk_id += 1

        if chunk_id not in chunk_positions:
            continue

        # Convert to absolute position
        offset_within_chunk = offset[0]
        chunk_start = chunk_positions[chunk_id]
        absolute_pos = chunk_start + offset_within_chunk

        # Validate
        if absolute_pos < 0 or absolute_pos > len(corpus_text):
            continue

        # Keep highest probability
        if absolute_pos not in char_to_info or prob > char_to_info[absolute_pos]["prob"]:
            char_to_info[absolute_pos] = {"prob": prob, "token": token}

    logger.info(f"Filtered {filtered_count} low-confidence tokens (threshold: {confidence_threshold})")
    logger.info(f"Extracted {len(char_to_info)} unique high-confidence boundary positions")
    return char_to_info


def cluster_boundaries(
    char_to_info: Dict[int, Dict], corpus_text: str, gap_cluster: int = 20
) -> List[Dict]:
    """Cluster adjacent boundary positions."""
    if not char_to_info:
        return []

    positions = sorted(char_to_info.keys())
    clusters = []
    current_cluster = [positions[0]]

    for i in range(1, len(positions)):
        pos = positions[i]
        prev_pos = positions[i - 1]
        gap = pos - prev_pos

        if gap <= gap_cluster:
            current_cluster.append(pos)
        else:
            # Save cluster
            cluster_start = current_cluster[0]
            cluster_end = current_cluster[-1]
            context_start = max(0, cluster_start - 40)
            context_end = min(len(corpus_text), cluster_end + 80)
            text_context = corpus_text[context_start:context_end]

            clusters.append(
                {
                    "boundary_id": len(clusters),
                    "char_start": cluster_start,
                    "char_end": cluster_end,
                    "n_tokens": len(current_cluster),
                    "tokens": [char_to_info[p]["token"] for p in current_cluster[:10]],
                    "max_prob": max(char_to_info[p]["prob"] for p in current_cluster),
                    "text_context": text_context,
                }
            )

            current_cluster = [pos]

    # Last cluster
    if current_cluster:
        cluster_start = current_cluster[0]
        cluster_end = current_cluster[-1]
        context_start = max(0, cluster_start - 40)
        context_end = min(len(corpus_text), cluster_end + 80)
        text_context = corpus_text[context_start:context_end]

        clusters.append(
            {
                "boundary_id": len(clusters),
                "char_start": cluster_start,
                "char_end": cluster_end,
                "n_tokens": len(current_cluster),
                "tokens": [char_to_info[p]["token"] for p in current_cluster[:10]],
                "max_prob": max(char_to_info[p]["prob"] for p in current_cluster),
                "text_context": text_context,
            }
        )

    return clusters


def validate_boundaries(clusters: List[Dict], corpus_text: str) -> Dict:
    """Validate boundary quality."""
    report = {
        "total_boundaries": len(clusters),
        "text_coverage": 0,
        "gap_stats": {},
        "mid_word_boundaries": 0,
    }

    if not clusters:
        return report

    total_span = sum(c["char_end"] - c["char_start"] for c in clusters)
    report["text_coverage"] = total_span / len(corpus_text) * 100

    gaps = [clusters[i + 1]["char_start"] - clusters[i]["char_end"] for i in range(len(clusters) - 1)]

    if gaps:
        gaps_sorted = sorted(gaps)
        report["gap_stats"] = {
            "min": min(gaps),
            "max": max(gaps),
            "median": gaps_sorted[len(gaps_sorted) // 2],
            "mean": sum(gaps) / len(gaps),
        }

    mid_word_count = 0
    for cluster in clusters:
        pos = cluster["char_start"]
        if pos > 0 and pos < len(corpus_text):
            before_char = corpus_text[pos - 1]
            after_char = corpus_text[pos]
            if before_char not in " \n" and after_char not in " \n":
                mid_word_count += 1

    report["mid_word_boundaries"] = mid_word_count

    return report


def main():
    parser = argparse.ArgumentParser(
        description="CAMeL-BERT post-processing: Optimized for generalization"
    )
    parser.add_argument("--raw_inference", required=True, help="Path to raw inference JSON")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--output", required=True, help="Path to save output JSON")
    parser.add_argument("--confidence_threshold", type=float, default=0.98, help="Confidence threshold (0-1)")
    parser.add_argument("--gap_cluster", type=int, default=20, help="Clustering gap threshold (chars)")

    args = parser.parse_args()

    # Load files
    logger.info("Loading files...")
    with open(args.raw_inference, encoding="utf-8") as f:
        raw_data = json.load(f)

    results = raw_data["inference_results"]
    tokens = results["tokens"]
    offsets = results["offsets"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    corpus_text = Path(args.corpus).read_text(encoding="utf-8")

    logger.info(f"  Corpus: {len(corpus_text):,} chars")
    logger.info(f"  Tokens: {len(tokens):,}")
    logger.info(f"  Boundary predictions (pred=1): {sum(predictions):,}")

    # Estimate chunk positions
    logger.info("\nEstimating chunk positions from offset patterns...")
    chunk_positions, sep_indices = estimate_chunk_positions(tokens, offsets, len(corpus_text))
    logger.info(f"  Estimated positions for {len(chunk_positions)} chunks")

    # Extract boundaries with high confidence filtering
    logger.info(f"\nExtracting very high-confidence boundaries (threshold >= {args.confidence_threshold})...")
    char_to_info = extract_boundaries(
        tokens, offsets, predictions, probabilities, corpus_text, chunk_positions, sep_indices,
        confidence_threshold=args.confidence_threshold
    )

    # Cluster boundaries
    logger.info(f"\nClustering boundaries (gap_cluster={args.gap_cluster})...")
    clusters = cluster_boundaries(char_to_info, corpus_text, gap_cluster=args.gap_cluster)

    logger.info(f"  Clusters found: {len(clusters)}")

    # Validate
    logger.info("\nValidating boundaries...")
    validation = validate_boundaries(clusters, corpus_text)

    logger.info(f"  Total boundaries: {validation['total_boundaries']}")
    logger.info(f"  Text coverage: {validation['text_coverage']:.1f}%")
    if validation["gap_stats"]:
        logger.info(
            f"  Gap stats: min={validation['gap_stats']['min']}, "
            f"max={validation['gap_stats']['max']}, "
            f"median={validation['gap_stats']['median']}"
        )
    logger.info(f"  Mid-word boundaries: {validation['mid_word_boundaries']}")

    # Save output
    output = {
        "metadata": {
            "method": "chunk_estimation_with_confidence_filtering",
            "description": (
                "Optimized approach combining chunk position estimation (handles chunk-relative offsets) "
                f"with very high confidence filtering (>={args.confidence_threshold}). "
                "Extracted strongest boundary signals and clustered them."
            ),
            "source": args.raw_inference,
            "corpus": args.corpus,
            "corpus_chars": len(corpus_text),
            "total_tokens": len(tokens),
            "boundary_tokens_raw": sum(predictions),
            "confidence_threshold": args.confidence_threshold,
            "boundary_tokens_filtered": len(char_to_info),
            "unique_positions": len(char_to_info),
            "gap_cluster": args.gap_cluster,
            "clusters_found": len(clusters),
            "validation": validation,
        },
        "khabar_boundaries": clusters,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n[OK] Output saved to {out_path}")
    logger.info(f"  {len(clusters)} boundaries extracted and clustered")


if __name__ == "__main__":
    main()
