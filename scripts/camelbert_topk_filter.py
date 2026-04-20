#!/usr/bin/env python3
"""
Alternative approach: Keep only top-K boundaries by confidence.

Instead of a fixed threshold, we select the N highest-confidence boundaries
and assume those are the true segment boundaries.

This directly targets the expected segment count (~600 khabars).
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class TopKBoundaryConverter:
    """Convert CAMeL-BERT predictions by selecting top-K high-confidence boundaries."""

    def __init__(self, raw_inference_json: str):
        """Load raw inference results from Colab."""
        with open(raw_inference_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.predictions = np.array(self.data['inference_results']['predictions'], dtype=int)
        self.probabilities = np.array(self.data['inference_results']['probabilities'])
        self.offsets = np.array(self.data['inference_results']['offsets'])
        self.tokens = self.data['inference_results']['tokens']

        print(f"[OK] Loaded inference data")
        print(f"  Total tokens: {len(self.predictions):,}")
        print(f"  Boundary tokens: {np.sum(self.predictions):,}")

    def filter_by_top_k(self, k: int) -> Tuple[np.ndarray, int]:
        """
        Filter boundary tokens to keep only top-K by confidence.

        Logic:
        - Find all predicted boundaries
        - Sort by confidence (probability)
        - Keep only top-K
        - Set all others to 0

        Args:
            k: Number of boundaries to keep

        Returns:
            - filtered_predictions: Predictions with only top-K boundaries set to 1
            - min_confidence: Minimum confidence of kept boundaries
        """
        filtered_predictions = np.zeros_like(self.predictions)

        # Find all boundary indices and their probabilities
        boundary_indices = np.where(self.predictions == 1)[0]
        boundary_confidences = [(idx, self.probabilities[idx]) for idx in boundary_indices]

        # Sort by confidence descending
        boundary_confidences.sort(key=lambda x: x[1], reverse=True)

        # Keep top-K
        min_confidence = 0
        if len(boundary_confidences) >= k:
            for i, (idx, conf) in enumerate(boundary_confidences[:k]):
                filtered_predictions[idx] = 1
            min_confidence = boundary_confidences[k-1][1]
        else:
            # If fewer boundaries than K, keep all
            for idx, conf in boundary_confidences:
                filtered_predictions[idx] = 1
            min_confidence = boundary_confidences[-1][1] if boundary_confidences else 0

        print(f"\n[INFO] Top-K filtering (keeping top {k} boundaries):")
        print(f"  Original boundary tokens: {np.sum(self.predictions):,}")
        print(f"  Kept (top-K): {np.sum(filtered_predictions):,}")
        print(f"  Removed: {np.sum(self.predictions) - np.sum(filtered_predictions):,}")
        print(f"  Min confidence of kept: {min_confidence:.4f}")
        print(f"  Reduction: {(np.sum(self.predictions) - np.sum(filtered_predictions))/np.sum(self.predictions)*100:.1f}%")

        return filtered_predictions, min_confidence

    def find_transitions(self, predictions: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find boundary transitions in predictions."""
        preds_with_prefix = np.concatenate([[0], predictions])
        diffs = np.diff(preds_with_prefix)

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        print(f"\n[INFO] Found boundary transitions:")
        print(f"  Boundary starts (0->1): {len(starts)}")
        print(f"  Boundary ends (1->0): {len(ends)}")

        return starts.tolist(), ends.tolist()

    def pair_boundaries(self, boundary_starts: List[int], boundary_ends: List[int]) -> List[Tuple[int, int]]:
        """Pair up boundary starts and ends."""
        boundary_pairs = []

        for start in boundary_starts:
            ends_after = [e for e in boundary_ends if e > start]
            if ends_after:
                end = ends_after[0]
                boundary_pairs.append((start, end))

        print(f"  Boundary pairs (isnad spans): {len(boundary_pairs)}")
        return boundary_pairs

    def extract_segments(
        self,
        text: str,
        boundary_pairs: List[Tuple[int, int]]
    ) -> List[Dict]:
        """Extract segments from text based on boundary transitions."""
        segments = []

        if not boundary_pairs:
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'type': 'unknown',
                'length': len(text)
            }]

        current_pos = 0

        for start_idx, end_idx in boundary_pairs:
            char_start = self.offsets[start_idx][0] if start_idx < len(self.offsets) else 0
            char_end = self.offsets[end_idx][1] if end_idx < len(self.offsets) else len(text)

            # Prose before this isnad
            if current_pos < char_start:
                prose_text = text[current_pos:char_start].strip()
                if prose_text:
                    segments.append({
                        'text': prose_text,
                        'start': int(current_pos),
                        'end': int(char_start),
                        'type': 'prose',
                        'length': len(prose_text)
                    })

            # Isnad segment
            isnad_text = text[char_start:char_end].strip()
            if isnad_text:
                segments.append({
                    'text': isnad_text,
                    'start': int(char_start),
                    'end': int(char_end),
                    'type': 'isnad',
                    'length': len(isnad_text)
                })

            current_pos = char_end

        # Remaining text
        if current_pos < len(text):
            prose_text = text[current_pos:].strip()
            if prose_text:
                segments.append({
                    'text': prose_text,
                    'start': int(current_pos),
                    'end': int(len(text)),
                    'type': 'prose',
                    'length': len(prose_text)
                })

        return segments

    def process(self, text: str, k: int) -> Dict:
        """Full conversion pipeline with top-K filtering."""
        print(f"\n[INFO] Processing text ({len(text):,} chars)...")

        # Step 1: Filter by top-K
        filtered_predictions, min_confidence = self.filter_by_top_k(k)

        # Step 2: Find boundary transitions
        starts, ends = self.find_transitions(filtered_predictions)

        # Step 3: Pair boundaries
        boundary_pairs = self.pair_boundaries(starts, ends)

        # Step 4: Extract segments
        segments = self.extract_segments(text, boundary_pairs)

        print(f"  Extracted {len(segments)} segments")

        # Type breakdown
        type_counts = {}
        for seg in segments:
            t = seg['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"  Type breakdown:")
        for t, count in sorted(type_counts.items()):
            pct = 100 * count / len(segments)
            print(f"    {t}: {count} ({pct:.1f}%)")

        return {
            'segments': segments,
            'total_segments': len(segments),
            'type_breakdown': type_counts,
            'boundary_starts': len(starts),
            'boundary_ends': len(ends),
            'statistics': {
                'avg_segment_length': int(np.mean([s['length'] for s in segments])) if segments else 0,
                'min_segment_length': int(np.min([s['length'] for s in segments])) if segments else 0,
                'max_segment_length': int(np.max([s['length'] for s in segments])) if segments else 0,
            },
            'topk_stats': {
                'original_boundary_tokens': int(np.sum(self.predictions)),
                'top_k_value': k,
                'kept_boundaries': int(np.sum(filtered_predictions)),
                'min_confidence_kept': float(min_confidence)
            }
        }


def evaluate_against_gold_standard(
    camelbert_segments: int,
    gold_standard: int = 613
) -> Dict:
    """Compare CAMeL-BERT segments against gold standard."""
    recall = camelbert_segments / gold_standard if gold_standard > 0 else 0
    ratio = camelbert_segments / gold_standard if gold_standard > 0 else 0

    print(f"\n[EVALUATION vs Gold Standard ({gold_standard} khabars)]")
    print(f"  CAMeL-BERT segments: {camelbert_segments}")
    print(f"  Gold standard: {gold_standard}")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  Ratio: {ratio:.2f}x")

    # Determine assessment
    if 0.85 <= ratio <= 1.15:
        assessment = "WELL-ALIGNED"
    elif ratio > 1.2:
        assessment = "OVER-segments"
    elif ratio < 0.8:
        assessment = "UNDER-segments"
    else:
        assessment = "SLIGHTLY-OFF"

    print(f"  Assessment: {assessment}")

    return {
        'gold_standard': gold_standard,
        'camelbert_segments': camelbert_segments,
        'recall': f'{recall*100:.1f}%',
        'ratio': f'{ratio:.2f}x',
        'assessment': assessment
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert token-level CAMeL-BERT predictions using top-K confidence filtering'
    )
    parser.add_argument('--input', required=True, help='Input JSON with raw inference results')
    parser.add_argument('--output', required=True, help='Output JSON with segments')
    parser.add_argument('--text', default='data/processed/kitab_uqala_reference_corpus.txt',
                        help='Original text file')
    parser.add_argument('--gold-standard', type=int, default=613,
                        help='Gold standard segment count for evaluation')
    parser.add_argument('--top-k', type=int, default=600,
                        help='Keep only top-K boundaries by confidence')

    args = parser.parse_args()

    # Load inference results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load original text
    text_path = Path(args.text)
    if not text_path.exists():
        print(f"ERROR: Text file not found: {text_path}")
        sys.exit(1)

    print(f"[INFO] Loading text from {text_path}...")
    with open(text_path, encoding='utf-8') as f:
        text = f.read()

    print(f"  Size: {len(text):,} chars, {len(text.split()):,} words")

    # Convert tokens to segments
    converter = TopKBoundaryConverter(str(input_path))
    result = converter.process(text, args.top_k)

    # Evaluate
    evaluation = evaluate_against_gold_standard(result['total_segments'], args.gold_standard)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert segments
    serializable_segments = []
    for seg in result['segments']:
        serializable_segments.append({
            'text': seg['text'],
            'start': int(seg['start']),
            'end': int(seg['end']),
            'type': seg['type'],
            'length': int(seg['length'])
        })

    output_data = {
        'metadata': {
            'source': str(input_path),
            'text': str(text_path),
            'gold_standard': args.gold_standard,
            'method': 'topk_confidence_filtering',
            'top_k': args.top_k
        },
        'segmentation': {
            'segments': serializable_segments,
            'total_segments': result['total_segments'],
            'type_breakdown': result['type_breakdown'],
            'boundary_starts': result['boundary_starts'],
            'boundary_ends': result['boundary_ends'],
            'statistics': result['statistics']
        },
        'topk_filtering': {
            'topk_stats': result['topk_stats']
        },
        'evaluation': evaluation
    }

    print(f"\n[INFO] Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Complete!")

    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY - TOP-K CONFIDENCE FILTERING")
    print(f"="*80)
    print(f"  Top-K value: {args.top_k}")
    print(f"  Original boundary tokens: {result['topk_stats']['original_boundary_tokens']:,}")
    print(f"  Kept (top-K): {result['topk_stats']['kept_boundaries']:,}")
    print(f"  Min confidence: {result['topk_stats']['min_confidence_kept']:.4f}")
    print(f"\n  Final Results:")
    print(f"    Total segments: {result['total_segments']}")
    print(f"    Type breakdown: {result['type_breakdown']}")
    print(f"    Recall vs gold: {evaluation['recall']}")
    print(f"    Assessment: {evaluation['assessment']}")
    print(f"="*80)


if __name__ == '__main__':
    main()
