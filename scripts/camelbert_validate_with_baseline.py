#!/usr/bin/env python3
"""
Option 3: Validate CAMeL-BERT boundaries against Baseline v4.

Strategy:
1. Load CAMeL-BERT predictions (token-level)
2. Load Baseline v4 results (segment-level boundaries)
3. Filter CAMeL-BERT boundaries to keep only those near Baseline v4 boundaries
4. Extract validated segments

This approach assumes Baseline v4 is conservative but precise, and uses it
as an anchor to filter CAMeL-BERT's more aggressive boundary detection.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np


class ValidatedBoundaryConverter:
    """Convert CAMeL-BERT predictions using Baseline v4 as validation anchor."""

    def __init__(self, raw_inference_json: str, baseline_segments_json: str = None):
        """
        Load inference results and optionally baseline segments for validation.

        If baseline_segments_json is None, we'll use the Baseline v4 results
        stored in the repository.
        """
        with open(raw_inference_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.predictions = np.array(self.data['inference_results']['predictions'], dtype=int)
        self.probabilities = np.array(self.data['inference_results']['probabilities'])
        self.offsets = np.array(self.data['inference_results']['offsets'])
        self.tokens = self.data['inference_results']['tokens']

        print(f"[OK] Loaded CAMeL-BERT inference data")
        print(f"  Total tokens: {len(self.predictions):,}")
        print(f"  Boundary tokens: {np.sum(self.predictions):,}")

        # Load baseline segments if provided
        self.baseline_segments = None
        if baseline_segments_json and Path(baseline_segments_json).exists():
            with open(baseline_segments_json, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
                if 'segmentation' in baseline_data:
                    self.baseline_segments = baseline_data['segmentation']['segments']
                elif 'segments' in baseline_data:
                    self.baseline_segments = baseline_data['segments']
                else:
                    self.baseline_segments = []

            print(f"[OK] Loaded Baseline v4 with {len(self.baseline_segments)} segments")

    def find_baseline_boundary_positions(self, text: str, tolerance: int = 50) -> Set[int]:
        """
        Extract character positions of Baseline v4 segment boundaries.

        Args:
            text: Original text
            tolerance: How far a CAMeL-BERT boundary can be from baseline boundary (chars)

        Returns:
            Set of character positions where baseline identifies boundaries
        """
        if not self.baseline_segments:
            return set()

        baseline_boundaries = set()

        for seg in self.baseline_segments:
            # Boundaries are at segment ends
            if 'end' in seg:
                baseline_boundaries.add(seg['end'])

        print(f"\n[INFO] Baseline v4 boundary positions:")
        print(f"  Total boundaries identified: {len(baseline_boundaries)}")

        return baseline_boundaries

    def filter_by_baseline_proximity(
        self,
        baseline_boundaries: Set[int],
        tolerance: int = 50
    ) -> Tuple[np.ndarray, int]:
        """
        Filter CAMeL-BERT boundary tokens to keep only those near Baseline v4 boundaries.

        Logic:
        - For each predicted boundary token, check if any Baseline v4 boundary is within tolerance
        - Keep the prediction if it aligns with baseline
        - This filters out CAMeL-BERT's spurious boundaries

        Args:
            baseline_boundaries: Set of character positions from Baseline v4
            tolerance: Distance threshold for alignment (default 50 chars)

        Returns:
            - filtered_predictions: Predictions with non-aligned boundaries set to 0
            - kept_count: Number of boundaries kept
        """
        filtered_predictions = self.predictions.copy()
        kept_count = 0
        removed_count = 0

        boundary_indices = np.where(self.predictions == 1)[0]

        for idx in boundary_indices:
            # Get character position of this token
            token_char_start, token_char_end = self.offsets[idx]
            token_char_mid = (token_char_start + token_char_end) // 2

            # Check if any baseline boundary is within tolerance
            aligned = False
            for baseline_pos in baseline_boundaries:
                if abs(token_char_mid - baseline_pos) <= tolerance:
                    aligned = True
                    break

            if aligned:
                kept_count += 1
            else:
                filtered_predictions[idx] = 0
                removed_count += 1

        print(f"\n[INFO] Baseline alignment filtering (tolerance={tolerance} chars):")
        print(f"  Original boundary tokens: {np.sum(self.predictions):,}")
        print(f"  Aligned with baseline: {kept_count:,}")
        print(f"  Removed (not aligned): {removed_count:,}")
        print(f"  Reduction: {removed_count/np.sum(self.predictions)*100:.1f}%")

        return filtered_predictions, kept_count

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

    def process(self, text: str, baseline_json: str = None, tolerance: int = 50) -> Dict:
        """Full conversion pipeline with baseline validation."""
        print(f"\n[INFO] Processing text ({len(text):,} chars)...")

        # Step 1: Find baseline boundaries (if available)
        baseline_boundaries = self.find_baseline_boundary_positions(text)

        if baseline_boundaries:
            # Step 2: Filter by baseline proximity
            filtered_predictions, kept_count = self.filter_by_baseline_proximity(
                baseline_boundaries, tolerance
            )
        else:
            print(f"\n[WARNING] No baseline boundaries found - using all CAMeL-BERT predictions")
            filtered_predictions = self.predictions.copy()
            kept_count = np.sum(filtered_predictions)

        # Step 3: Find boundary transitions
        starts, ends = self.find_transitions(filtered_predictions)

        # Step 4: Pair boundaries
        boundary_pairs = self.pair_boundaries(starts, ends)

        # Step 5: Extract segments
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
            'validation_stats': {
                'original_boundary_tokens': int(np.sum(self.predictions)),
                'baseline_boundaries_identified': len(baseline_boundaries),
                'aligned_with_baseline': kept_count,
                'filtered_by_baseline': int(np.sum(self.predictions) - kept_count) if baseline_boundaries else 0,
                'tolerance_chars': tolerance
            }
        }


def evaluate_against_gold_standard(
    camelbert_segments: int,
    baseline_segments: int = 575,
    gold_standard: int = 613
) -> Dict:
    """Compare CAMeL-BERT segments against gold standard and baseline."""
    recall = camelbert_segments / gold_standard if gold_standard > 0 else 0
    ratio = camelbert_segments / gold_standard if gold_standard > 0 else 0
    baseline_ratio = baseline_segments / gold_standard if gold_standard > 0 else 0

    print(f"\n[EVALUATION vs Gold Standard ({gold_standard} khabars)]")
    print(f"  Baseline v4:         {baseline_segments:3d} ({baseline_ratio*100:.1f}% recall)")
    print(f"  CAMeL-BERT (validated): {camelbert_segments:3d} ({recall*100:.1f}% recall)")
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
        'baseline_segments': baseline_segments,
        'camelbert_segments': camelbert_segments,
        'recall': f'{recall*100:.1f}%',
        'ratio': f'{ratio:.2f}x',
        'assessment': assessment
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert token-level CAMeL-BERT predictions using Baseline v4 validation'
    )
    parser.add_argument('--input', required=True, help='Input JSON with raw CAMeL-BERT inference results')
    parser.add_argument('--output', required=True, help='Output JSON with validated segments')
    parser.add_argument('--text', default='data/processed/kitab_uqala_reference_corpus.txt',
                        help='Original text file')
    parser.add_argument('--baseline', default='results/baseline_v4_kitab_uqala.json',
                        help='Baseline v4 segments (for validation reference)')
    parser.add_argument('--gold-standard', type=int, default=613,
                        help='Gold standard segment count')
    parser.add_argument('--tolerance', type=int, default=50,
                        help='Distance tolerance for baseline boundary alignment (chars)')

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
    converter = ValidatedBoundaryConverter(str(input_path), args.baseline)
    result = converter.process(text, args.baseline, args.tolerance)

    # Evaluate
    baseline_count = 575  # Baseline v4 on Kitab Uqala
    evaluation = evaluate_against_gold_standard(
        result['total_segments'],
        baseline_count,
        args.gold_standard
    )

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
            'baseline_reference': args.baseline,
            'gold_standard': args.gold_standard,
            'method': 'baseline_validated_boundaries',
            'tolerance_chars': args.tolerance
        },
        'segmentation': {
            'segments': serializable_segments,
            'total_segments': result['total_segments'],
            'type_breakdown': result['type_breakdown'],
            'boundary_starts': result['boundary_starts'],
            'boundary_ends': result['boundary_ends'],
            'statistics': result['statistics']
        },
        'validation': {
            'validation_stats': result['validation_stats']
        },
        'evaluation': evaluation
    }

    print(f"\n[INFO] Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Complete!")

    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY - BASELINE-VALIDATED BOUNDARY METHOD")
    print(f"="*80)
    print(f"  Tolerance: {args.tolerance} chars")
    print(f"\n  Baseline v4 Analysis:")
    print(f"    Baseline boundaries identified: {result['validation_stats']['baseline_boundaries_identified']}")
    print(f"    CAMeL-BERT boundaries aligned: {result['validation_stats']['aligned_with_baseline']}")
    print(f"    CAMeL-BERT boundaries filtered: {result['validation_stats']['filtered_by_baseline']}")
    print(f"\n  Final Results:")
    print(f"    Total segments: {result['total_segments']}")
    print(f"    Type breakdown: {result['type_breakdown']}")
    print(f"    Recall vs gold: {evaluation['recall']}")
    print(f"    Ratio to baseline: {float(evaluation['ratio'].rstrip('x')):.2f}x")
    print(f"    Assessment: {evaluation['assessment']}")
    print(f"="*80)


if __name__ == '__main__':
    main()
