#!/usr/bin/env python3
"""
Corrected CAMeL-BERT post-processing: Text-based chunk localization.

CRITICAL FIX: The previous approach estimated chunk positions from offset patterns,
which was fundamentally flawed. This version locates chunks by searching for
actual token text in the corpus.

Strategy:
1. For each chunk, find the first real token (not [PAD]/[CLS]/[SEP])
2. Search for this token text in the corpus
3. Use actual position to determine chunk start (position = token_pos - token_offset)
4. All subsequent tokens in chunk use: absolute = chunk_start + chunk_relative_offset

This is robust and doesn't require any assumptions about chunk structure.

Usage:
    python scripts/convert_camelbert_corrected.py \
      --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/[TEXT]/camelbert_[TEXT]_char_boundaries.json
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


class TextBasedChunkLocator:
    """Locate chunk positions by searching for token text in corpus."""

    def __init__(self, tokens: List[str], offsets: List[tuple], corpus_text: str):
        """
        Initialize locator.

        Args:
            tokens: Token list from raw inference
            offsets: Offset list (chunk-relative positions)
            corpus_text: Full corpus text
        """
        self.tokens = tokens
        self.offsets = offsets
        self.corpus_text = corpus_text
        self.SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

        # Find chunk boundaries
        self.sep_indices = [i for i, t in enumerate(tokens) if t == "[SEP]"]
        logger.info(f"Found {len(self.sep_indices)} chunks (SEP tokens)")

    def _find_token_in_corpus(
        self, token_text: str, search_start: int = 0, search_end: Optional[int] = None
    ) -> Optional[int]:
        """
        Search for token text in corpus (case-sensitive, exact match).

        Args:
            token_text: Text to search for (without ## prefix)
            search_start: Start position in corpus
            search_end: End position in corpus

        Returns:
            Position in corpus where token starts, or None if not found
        """
        if search_end is None:
            search_end = len(self.corpus_text)

        search_start = max(0, search_start)
        search_end = min(len(self.corpus_text), search_end)

        search_substr = self.corpus_text[search_start:search_end]
        pos = search_substr.find(token_text)

        return search_start + pos if pos >= 0 else None

    def locate_chunk_positions(self) -> Dict[int, int]:
        """
        Locate absolute corpus position for each chunk by finding token text.

        Returns:
            Dict mapping chunk_id -> absolute_corpus_position
        """
        chunk_positions = {}

        for chunk_id in range(len(self.sep_indices)):
            # Get token range for this chunk
            start_idx = 0 if chunk_id == 0 else self.sep_indices[chunk_id - 1] + 1
            end_idx = self.sep_indices[chunk_id]

            # Find first real token in this chunk
            anchor_token = None
            anchor_offset = None
            anchor_idx = None

            for idx in range(start_idx, end_idx):
                tok = self.tokens[idx]
                off = self.offsets[idx]

                if tok not in self.SPECIAL_TOKENS and off != [0, 0]:
                    anchor_token = tok
                    anchor_offset = off[0]  # char_start relative to chunk
                    anchor_idx = idx
                    break

            if not anchor_token:
                logger.warning(f"Chunk {chunk_id}: No anchor token found")
                # Estimate from previous chunk
                if chunk_positions:
                    chunk_positions[chunk_id] = list(chunk_positions.values())[-1] + 500
                else:
                    chunk_positions[chunk_id] = chunk_id * 500
                continue

            # Search for anchor token in corpus
            anchor_token_clean = anchor_token.lstrip("##")

            # Determine search window
            if chunk_id == 0:
                search_start = 0
                search_end = len(self.corpus_text)
            else:
                # Estimate: previous chunk should be before current
                prev_chunk_pos = chunk_positions.get(chunk_id - 1, 0)
                search_start = max(0, prev_chunk_pos)
                search_end = min(len(self.corpus_text), prev_chunk_pos + 2000)

            # Find token in corpus
            token_pos_in_corpus = self._find_token_in_corpus(
                anchor_token_clean, search_start, search_end
            )

            if token_pos_in_corpus is not None:
                # Token found at position X in corpus
                # Token has offset O relative to chunk start
                # Therefore: chunk_start = X - O
                chunk_start = token_pos_in_corpus - anchor_offset
                chunk_start = max(0, min(chunk_start, len(self.corpus_text)))
                chunk_positions[chunk_id] = chunk_start

                logger.debug(
                    f"Chunk {chunk_id}: Found anchor '{anchor_token}' at corpus pos "
                    f"{token_pos_in_corpus}, chunk starts at {chunk_start}"
                )
            else:
                logger.warning(
                    f"Chunk {chunk_id}: Could not find anchor token '{anchor_token_clean}' "
                    f"in search range [{search_start}, {search_end}]"
                )
                # Estimate
                if chunk_positions:
                    chunk_positions[chunk_id] = list(chunk_positions.values())[-1] + 500
                else:
                    chunk_positions[chunk_id] = 0

        logger.info(f"Located {len(chunk_positions)} chunks")
        logger.info(f"Chunk positions: {list(chunk_positions.values())[:10]}...")

        return chunk_positions

    def extract_absolute_boundaries(
        self, predictions: List[int], probabilities: List[float], chunk_positions: Dict[int, int]
    ) -> Dict[int, Dict]:
        """
        Extract boundary positions with absolute corpus coordinates.

        Args:
            predictions: Prediction array (1 = boundary, 0 = non-boundary)
            probabilities: Confidence scores
            chunk_positions: Dict mapping chunk_id -> absolute position

        Returns:
            Dict mapping char_start -> {prob, token, chunk_id}
        """
        char_to_info = {}
        found_count = 0
        missing_chunk_count = 0

        for tok_idx, (token, offset, pred, prob) in enumerate(
            zip(self.tokens, self.offsets, predictions, probabilities)
        ):
            if pred != 1:  # Only boundary tokens
                continue

            if token in self.SPECIAL_TOKENS or offset == [0, 0]:
                continue

            # Find which chunk this token belongs to
            chunk_id = 0
            for sep_idx in self.sep_indices:
                if tok_idx > sep_idx:
                    chunk_id += 1

            if chunk_id not in chunk_positions:
                missing_chunk_count += 1
                continue

            # Convert to absolute position
            cs_rel = offset[0]
            chunk_start_abs = chunk_positions[chunk_id]
            cs_abs = chunk_start_abs + cs_rel

            # Validate: position must be within corpus
            if cs_abs < 0 or cs_abs > len(self.corpus_text):
                continue

            # Keep highest probability for each position
            if cs_abs not in char_to_info or prob > char_to_info[cs_abs]["prob"]:
                char_to_info[cs_abs] = {
                    "prob": prob,
                    "token": token,
                    "chunk_id": chunk_id,
                }
                found_count += 1

        logger.info(f"Extracted {found_count} boundary tokens -> {len(char_to_info)} unique positions")
        if missing_chunk_count > 0:
            logger.warning(f"Skipped {missing_chunk_count} tokens in unlocated chunks")

        return char_to_info


def cluster_boundaries(
    char_to_info: Dict[int, Dict], corpus_text: str, gap_cluster: int = 20
) -> List[Dict]:
    """Cluster adjacent boundary tokens into narrative units."""
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
        "quality_issues": [],
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
        description="Corrected CAMeL-BERT post-processing with text-based chunk localization"
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

    # Locate chunks using token text
    logger.info("\nLocating chunk positions via token search...")
    locator = TextBasedChunkLocator(tokens, offsets, corpus_text)
    chunk_positions = locator.locate_chunk_positions()

    logger.info("Extracting boundary positions...")
    char_to_info = locator.extract_absolute_boundaries(predictions, probabilities, chunk_positions)

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

    if validation["quality_issues"]:
        logger.warning("Quality issues detected:")
        for issue in validation["quality_issues"]:
            logger.warning(f"    - {issue}")

    # Save output
    output = {
        "metadata": {
            "method": "text_based_chunk_localization",
            "description": (
                "Corrected extraction using text-based chunk location detection. "
                "Finds anchor tokens in corpus to determine chunk positions, then maps "
                "offsets to absolute coordinates. No assumptions about chunk structure."
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
