#!/usr/bin/env python3
"""
CAMeL-BERT post-processing with chunk position validation.

Strategy:
1. Use offset pattern analysis to estimate chunk positions
2. Validate by checking if anchor token offsets match corpus
3. Adjust chunk positions if validation fails
4. Extract and cluster boundaries with validated positions

This is more robust than pure offset pattern analysis.

Usage:
    python scripts/convert_camelbert_validated.py \
      --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/[TEXT]/camelbert_[TEXT]_char_boundaries.json \
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


class ValidatedChunkLocator:
    """Estimate and validate chunk positions."""

    def __init__(self, tokens: List[str], offsets: List[tuple], corpus_text: str):
        self.tokens = tokens
        self.offsets = offsets
        self.corpus_text = corpus_text
        self.SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

        # Find chunk boundaries
        self.sep_indices = [i for i, t in enumerate(tokens) if t == "[SEP]"]
        logger.info(f"Found {len(self.sep_indices)} chunks")

    def _extract_chunk_offsets(self, chunk_id: int) -> List[int]:
        """Extract all non-zero offsets for a chunk."""
        start_idx = 0 if chunk_id == 0 else self.sep_indices[chunk_id - 1] + 1
        end_idx = self.sep_indices[chunk_id]

        offsets = []
        for i in range(start_idx, end_idx):
            token = self.tokens[i]
            offset = self.offsets[i]

            if token not in self.SPECIAL_TOKENS and offset != [0, 0]:
                offsets.append(offset[0])

        return sorted(set(offsets))

    def estimate_positions(self) -> Dict[int, int]:
        """Estimate chunk positions from offset patterns."""
        chunk_positions = {0: 0}  # First chunk starts at 0

        for chunk_id in range(1, len(self.sep_indices)):
            prev_offsets = self._extract_chunk_offsets(chunk_id - 1)
            curr_offsets = self._extract_chunk_offsets(chunk_id)

            if not prev_offsets or not curr_offsets:
                # Estimate from previous
                chunk_positions[chunk_id] = chunk_positions[chunk_id - 1] + 500
                continue

            # Estimate: if prev max offset was 500, and curr min offset is 20,
            # then there's a 480-char overlap, so current chunk starts at prev_start + (500-480)
            prev_max = max(prev_offsets)
            curr_min = min(curr_offsets)

            # Simple estimation: assume chunks are ~500 chars apart
            estimated = chunk_positions[chunk_id - 1] + (prev_max - curr_min)

            chunk_positions[chunk_id] = max(0, min(estimated, len(self.corpus_text)))

        logger.info(f"Estimated positions for {len(chunk_positions)} chunks")
        return chunk_positions

    def validate_positions(self, chunk_positions: Dict[int, int]) -> Dict[int, int]:
        """Validate positions by checking if tokens match corpus text."""
        corrected = chunk_positions.copy()

        for chunk_id in range(len(self.sep_indices)):
            start_idx = 0 if chunk_id == 0 else self.sep_indices[chunk_id - 1] + 1
            end_idx = self.sep_indices[chunk_id]

            # Find first real token in chunk
            anchor_token = None
            anchor_offset = None
            for idx in range(start_idx, end_idx):
                tok = self.tokens[idx]
                off = self.offsets[idx]
                if tok not in self.SPECIAL_TOKENS and off != [0, 0]:
                    anchor_token = tok
                    anchor_offset = off[0]
                    break

            if not anchor_token:
                continue

            # Check if anchor token appears at the estimated position
            chunk_start = corrected[chunk_id]
            expected_corpus_pos = chunk_start + anchor_offset

            if 0 <= expected_corpus_pos < len(self.corpus_text):
                # Check if token matches
                tok_clean = anchor_token.lstrip('#')
                if expected_corpus_pos + len(tok_clean) <= len(self.corpus_text):
                    corpus_text_at_pos = self.corpus_text[expected_corpus_pos:expected_corpus_pos + len(tok_clean)]
                    if corpus_text_at_pos == tok_clean:
                        # Validation passed
                        pass
                    else:
                        # Validation failed - try to find token nearby
                        search_start = max(0, expected_corpus_pos - 100)
                        search_end = min(len(self.corpus_text), expected_corpus_pos + 100)
                        search_range = self.corpus_text[search_start:search_end]

                        found_pos = search_range.find(tok_clean)
                        if found_pos >= 0:
                            # Found token in nearby range - adjust chunk position
                            actual_pos = search_start + found_pos
                            corrected[chunk_id] = max(0, actual_pos - anchor_offset)
                            logger.debug(f"Chunk {chunk_id}: corrected from {chunk_start} to {corrected[chunk_id]}")

        return corrected


def extract_boundaries(
    tokens: List[str],
    offsets: List[tuple],
    predictions: List[int],
    probabilities: List[float],
    chunk_positions: Dict[int, int],
    sep_indices: List[int],
    corpus_text: str,
) -> Dict[int, Dict]:
    """Extract boundary positions using validated chunk positions."""
    SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}
    char_to_info = {}

    for tok_idx, (token, offset, pred, prob) in enumerate(
        zip(tokens, offsets, predictions, probabilities)
    ):
        if pred != 1:
            continue

        if token in SPECIAL_TOKENS or offset == [0, 0]:
            continue

        # Find which chunk this token belongs to
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

        # Validate position
        if absolute_pos < 0 or absolute_pos > len(corpus_text):
            continue

        # Keep highest probability
        if absolute_pos not in char_to_info or prob > char_to_info[absolute_pos]["prob"]:
            char_to_info[absolute_pos] = {"prob": prob, "token": token}

    logger.info(f"Extracted {len(char_to_info)} unique boundary positions")
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
        description="CAMeL-BERT post-processing with chunk position validation"
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

    # Estimate and validate chunk positions
    logger.info("\nEstimating chunk positions...")
    locator = ValidatedChunkLocator(tokens, offsets, corpus_text)
    chunk_positions = locator.estimate_positions()

    logger.info("Validating chunk positions...")
    chunk_positions = locator.validate_positions(chunk_positions)

    # Extract boundaries
    logger.info("\nExtracting boundaries...")
    sep_indices = [i for i, t in enumerate(tokens) if t == "[SEP]"]
    char_to_info = extract_boundaries(tokens, offsets, predictions, probabilities, chunk_positions, sep_indices, corpus_text)

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
        logger.info(f"  Gap stats: min={validation['gap_stats']['min']}, max={validation['gap_stats']['max']}, median={validation['gap_stats']['median']}")
    logger.info(f"  Mid-word boundaries: {validation['mid_word_boundaries']}")

    # Save output
    output = {
        "metadata": {
            "method": "validated_chunk_position_estimation",
            "description": (
                "Offset pattern analysis with validation. "
                "Estimates chunk positions from offset ranges, validates by checking "
                "if anchor tokens match corpus text, and adjusts if needed."
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
