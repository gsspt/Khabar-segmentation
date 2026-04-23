#!/usr/bin/env python3
"""
Direct CAMeL-BERT post-processing: Use offset_mapping directly.

The raw inference file contains HuggingFace offset_mapping outputs:
each token's offsets are GLOBAL corpus character positions, not chunk-relative.

Strategy:
1. Extract all boundary tokens (pred=1) with their global offsets
2. Deduplicate by position (keep highest confidence)
3. Cluster adjacent boundaries (gap_cluster threshold)
4. Output segments with char_start/char_end positions

This is the simplest and most robust approach.

Usage:
    python scripts/convert_boundary_tokens_direct.py \
      --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/[TEXT]/camelbert_[TEXT]_char_boundaries.json \
      --gap_cluster 20
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_boundaries(
    tokens: List[str],
    offsets: List[tuple],
    predictions: List[int],
    probabilities: List[float],
    corpus_text: str,
) -> Dict[int, Dict]:
    """
    Extract boundary positions using global offsets directly.

    Args:
        tokens: Token list from raw inference
        offsets: Global corpus character positions [char_start, char_end]
        predictions: Prediction array (1 = boundary, 0 = non-boundary)
        probabilities: Confidence scores
        corpus_text: Full corpus text for validation

    Returns:
        Dict mapping char_start -> {prob, token}
    """
    SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}
    char_to_info = {}

    for tok_idx, (token, offset, pred, prob) in enumerate(
        zip(tokens, offsets, predictions, probabilities)
    ):
        # Only process boundary tokens
        if pred != 1:
            continue

        # Skip special tokens
        if token in SPECIAL_TOKENS:
            continue

        # Skip tokens with null offset
        if offset == [0, 0]:
            continue

        cs = offset[0]  # char_start (position in corpus)

        # Validate position is within corpus
        if cs < 0 or cs > len(corpus_text):
            continue

        # Keep highest probability for each position
        if cs not in char_to_info or prob > char_to_info[cs]["prob"]:
            char_to_info[cs] = {
                "prob": prob,
                "token": token,
            }

    logger.info(f"Extracted {len(char_to_info)} unique boundary positions")
    return char_to_info


def cluster_boundaries(
    char_to_info: Dict[int, Dict], corpus_text: str, gap_cluster: int = 20
) -> List[Dict]:
    """Cluster adjacent boundary positions into narrative units."""
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
            # Continue same cluster
            current_cluster.append(pos)
        else:
            # Save cluster and start new one
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

    # Text coverage
    total_span = sum(c["char_end"] - c["char_start"] for c in clusters)
    report["text_coverage"] = total_span / len(corpus_text) * 100

    # Gap analysis
    gaps = [clusters[i + 1]["char_start"] - clusters[i]["char_end"] for i in range(len(clusters) - 1)]

    if gaps:
        gaps_sorted = sorted(gaps)
        report["gap_stats"] = {
            "min": min(gaps),
            "max": max(gaps),
            "median": gaps_sorted[len(gaps_sorted) // 2],
            "mean": sum(gaps) / len(gaps),
            "very_short": sum(1 for g in gaps if g < 50),
            "very_long": sum(1 for g in gaps if g > 5000),
        }

    # Mid-word boundaries
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
        description="Direct CAMeL-BERT post-processing using global offsets"
    )
    parser.add_argument("--raw_inference", required=True, help="Path to raw inference JSON")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--output", required=True, help="Path to save output JSON")
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

    # Extract boundaries using global offsets
    logger.info("\nExtracting boundaries from global offsets...")
    char_to_info = extract_boundaries(tokens, offsets, predictions, probabilities, corpus_text)

    logger.info(f"  Unique boundary positions: {len(char_to_info):,}")

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
            "method": "direct_global_offsets",
            "description": (
                "Direct extraction using global HuggingFace offset_mapping. "
                "Each token offset is already a global corpus character position. "
                "No chunk localization needed."
            ),
            "source": args.raw_inference,
            "corpus": args.corpus,
            "corpus_chars": len(corpus_text),
            "total_tokens": len(tokens),
            "boundary_tokens": sum(predictions),
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
