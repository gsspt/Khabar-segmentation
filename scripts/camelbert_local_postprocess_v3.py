#!/usr/bin/env python3
"""
HYBRID APPROACH: Convert token-level CAMeL-BERT predictions to khabar-level segments.

Key improvements:
1. Confidence threshold filtering (remove low-confidence boundary predictions)
2. Adjacent isnad merging (merge consecutive isnads separated by short prose)
3. Statistical tracking of filtering/merging effects

Logic:
1. Filter boundary tokens by confidence threshold (default 0.90)
2. Find boundary transitions from filtered predictions
3. Extract segments between transitions
4. Merge adjacent isnads if prose between them < threshold (default 50 chars)
5. Compare with gold standard
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class HybridBoundaryConverter:
    """Convert token-level predictions to segments using confidence filtering and merging."""

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
        print(f"  Boundary ratio: {np.sum(self.predictions)/len(self.predictions)*100:.1f}%")

    def filter_by_confidence(self, threshold: float = 0.90) -> Tuple[np.ndarray, List[int]]:
        """
        Filter boundary tokens by confidence threshold.

        Args:
            threshold: Minimum probability to keep a boundary prediction

        Returns:
            - filtered_predictions: New predictions with low-confidence boundaries set to 0
            - filtered_indices: Indices of tokens that were filtered out
        """
        filtered_predictions = self.predictions.copy()
        filtered_indices = []

        # For each position where model predicted boundary (1),
        # check if confidence is below threshold
        boundary_indices = np.where(self.predictions == 1)[0]

        for idx in boundary_indices:
            if self.probabilities[idx] < threshold:
                filtered_predictions[idx] = 0  # Downgrade to non-boundary
                filtered_indices.append(idx)

        print(f"\n[INFO] Confidence filtering (threshold={threshold}):")
        print(f"  Original boundary tokens: {np.sum(self.predictions):,}")
        print(f"  Filtered out (low confidence): {len(filtered_indices):,}")
        print(f"  Remaining boundary tokens: {np.sum(filtered_predictions):,}")
        print(f"  Reduction: {len(filtered_indices)/np.sum(self.predictions)*100:.1f}%")

        return filtered_predictions, filtered_indices

    def find_transitions(self, predictions: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find boundary transitions in token predictions.

        Returns:
            - starts: Token indices where 0->1 transition occurs
            - ends: Token indices where 1->0 transition occurs
        """
        # Prepend 0 to detect boundary at token 0
        preds_with_prefix = np.concatenate([[0], predictions])

        # Find differences (0->1 = +1, 1->0 = -1)
        diffs = np.diff(preds_with_prefix)

        # Transitions
        starts = np.where(diffs == 1)[0]  # 0->1 transitions
        ends = np.where(diffs == -1)[0]   # 1->0 transitions

        print(f"\n[INFO] Found boundary transitions:")
        print(f"  Boundary starts (0->1): {len(starts)}")
        print(f"  Boundary ends (1->0): {len(ends)}")

        return starts.tolist(), ends.tolist()

    def pair_boundaries(self, boundary_starts: List[int], boundary_ends: List[int]) -> List[Tuple[int, int]]:
        """
        Pair up boundary starts and ends to identify isnad spans.

        Returns:
            - boundary_pairs: List of (start_idx, end_idx) tuples
        """
        boundary_pairs = []

        for start in boundary_starts:
            # Find the next end after this start
            ends_after = [e for e in boundary_ends if e > start]
            if ends_after:
                end = ends_after[0]
                boundary_pairs.append((start, end))

        print(f"  Boundary pairs (isnad spans): {len(boundary_pairs)}")
        return boundary_pairs

    def extract_segments_with_boundaries(
        self,
        text: str,
        boundary_pairs: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Extract segments from text based on boundary transitions.
        Does NOT merge yet.
        """
        segments = []

        if not boundary_pairs:
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'type': 'unknown',
                'length': len(text),
                'token_span': (0, len(self.offsets)-1)
            }]

        current_pos = 0

        for start_idx, end_idx in boundary_pairs:
            # Get character positions from offsets
            char_start = self.offsets[start_idx][0] if start_idx < len(self.offsets) else 0
            char_end = self.offsets[end_idx][1] if end_idx < len(self.offsets) else len(text)

            # Prose before this isnad (if any)
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

        # Remaining text after last isnad
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

    def merge_adjacent_isnads(self, segments: List[Dict], text: str, min_prose_length: int = 50) -> Tuple[List[Dict], int]:
        """
        Merge isnads separated by short prose blocks.

        Logic:
        - If an isnad is followed by prose < min_prose_length chars, followed by another isnad,
          merge all three into a single isnad segment.

        Args:
            segments: List of segment dicts
            text: Original text (needed to extract merged segment text)
            min_prose_length: Maximum prose length to trigger merge (default 50 chars)

        Returns:
            - merged_segments: Segments with adjacent isnads merged
            - merge_count: Number of merge operations performed
        """
        if len(segments) < 3:
            return segments, 0

        merged = []
        i = 0
        merge_count = 0

        while i < len(segments):
            current_seg = segments[i]

            # Check if we can merge: current=isnad, next=short prose, next-next=isnad
            if (current_seg['type'] == 'isnad' and
                i + 2 < len(segments) and
                segments[i + 1]['type'] == 'prose' and
                segments[i + 2]['type'] == 'isnad' and
                segments[i + 1]['length'] < min_prose_length):

                # Merge current isnad + prose + next isnad
                merged_start = current_seg['start']
                merged_end = segments[i + 2]['end']
                merged_text = text[merged_start:merged_end].strip()

                merged.append({
                    'text': merged_text,
                    'start': int(merged_start),
                    'end': int(merged_end),
                    'type': 'isnad',
                    'length': len(merged_text)
                })

                merge_count += 1
                i += 3  # Skip the prose and next isnad

            else:
                merged.append(current_seg)
                i += 1

        print(f"\n[INFO] Adjacent isnad merging (prose < {min_prose_length} chars):")
        print(f"  Merged groups: {merge_count}")
        print(f"  Segments before merge: {len(segments)}")
        print(f"  Segments after merge: {len(merged)}")

        return merged, merge_count

    def process(self, text: str, confidence_threshold: float = 0.90, min_prose_length: int = 50) -> Dict:
        """Full conversion pipeline with confidence filtering and merging."""
        print(f"\n[INFO] Processing text ({len(text):,} chars)...")

        # Step 1: Filter by confidence
        filtered_predictions, filtered_indices = self.filter_by_confidence(confidence_threshold)

        # Step 2: Find boundary transitions in filtered predictions
        starts, ends = self.find_transitions(filtered_predictions)

        # Step 3: Pair boundaries
        boundary_pairs = self.pair_boundaries(starts, ends)

        # Step 4: Extract segments
        segments = self.extract_segments_with_boundaries(text, boundary_pairs)

        print(f"  Extracted {len(segments)} segments (before merging)")

        # Step 5: Merge adjacent isnads
        merged_segments, merge_count = self.merge_adjacent_isnads(segments, text, min_prose_length)

        print(f"  Extracted {len(merged_segments)} segments (after merging)")

        # Type breakdown
        type_counts = {}
        for seg in merged_segments:
            t = seg['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"  Type breakdown:")
        for t, count in sorted(type_counts.items()):
            pct = 100 * count / len(merged_segments)
            print(f"    {t}: {count} ({pct:.1f}%)")

        return {
            'segments': merged_segments,
            'total_segments': len(merged_segments),
            'type_breakdown': type_counts,
            'boundary_starts': len(starts),
            'boundary_ends': len(ends),
            'statistics': {
                'avg_segment_length': int(np.mean([s['length'] for s in merged_segments])) if merged_segments else 0,
                'min_segment_length': int(np.min([s['length'] for s in merged_segments])) if merged_segments else 0,
                'max_segment_length': int(np.max([s['length'] for s in merged_segments])) if merged_segments else 0,
            },
            'filtering_stats': {
                'original_boundary_tokens': int(np.sum(self.predictions)),
                'filtered_low_confidence': len(filtered_indices),
                'remaining_boundary_tokens': int(np.sum(filtered_predictions)),
            },
            'merging_stats': {
                'merge_count': merge_count,
                'segments_before_merge': len(segments),
                'segments_after_merge': len(merged_segments),
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
        description='Convert token-level CAMeL-BERT predictions to khabar segments (v3: hybrid approach)'
    )
    parser.add_argument('--input', required=True, help='Input JSON with raw inference results')
    parser.add_argument('--output', required=True, help='Output JSON with segments')
    parser.add_argument('--text', default='data/processed/kitab_uqala_reference_corpus.txt',
                        help='Original text file')
    parser.add_argument('--gold-standard', type=int, default=613,
                        help='Gold standard segment count for evaluation')
    parser.add_argument('--confidence-threshold', type=float, default=0.90,
                        help='Minimum confidence to keep boundary prediction (0.0-1.0)')
    parser.add_argument('--min-prose-length', type=int, default=50,
                        help='Maximum prose length between isnads to trigger merge')

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
    converter = HybridBoundaryConverter(str(input_path))
    result = converter.process(text, args.confidence_threshold, args.min_prose_length)

    # Evaluate
    evaluation = evaluate_against_gold_standard(result['total_segments'], args.gold_standard)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert segments to JSON-serializable format
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
            'method': 'hybrid_boundary_transitions_v3',
            'confidence_threshold': args.confidence_threshold,
            'min_prose_length': args.min_prose_length
        },
        'segmentation': {
            'segments': serializable_segments,
            'total_segments': result['total_segments'],
            'type_breakdown': result['type_breakdown'],
            'boundary_starts': result['boundary_starts'],
            'boundary_ends': result['boundary_ends'],
            'statistics': result['statistics']
        },
        'filtering_and_merging': {
            'filtering_stats': result['filtering_stats'],
            'merging_stats': result['merging_stats']
        },
        'evaluation': evaluation
    }

    print(f"\n[INFO] Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Complete!")
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY - HYBRID BOUNDARY TRANSITION METHOD (V3)")
    print(f"="*80)
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Min prose length for merging: {args.min_prose_length} chars")
    print(f"\n  Filtering results:")
    print(f"    Original boundary tokens: {result['filtering_stats']['original_boundary_tokens']:,}")
    print(f"    Filtered (low confidence): {result['filtering_stats']['filtered_low_confidence']:,}")
    print(f"    Remaining boundary tokens: {result['filtering_stats']['remaining_boundary_tokens']:,}")
    print(f"\n  Merging results:")
    print(f"    Merged groups: {result['merging_stats']['merge_count']}")
    print(f"    Segments before merge: {result['merging_stats']['segments_before_merge']}")
    print(f"    Segments after merge: {result['merging_stats']['segments_after_merge']}")
    print(f"\n  Final results:")
    print(f"    Total segments: {result['total_segments']}")
    print(f"    Type breakdown: {result['type_breakdown']}")
    print(f"    Recall vs gold: {evaluation['recall']}")
    print(f"    Assessment: {evaluation['assessment']}")
    print(f"="*80)


if __name__ == '__main__':
    main()
