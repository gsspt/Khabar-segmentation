#!/usr/bin/env python3
"""
REDESIGNED: Convert token-level CAMeL-BERT predictions to khabar-level segments.

Key change: Identify actual BOUNDARY TRANSITIONS (0→1, 1→0) instead of clustering tokens.

Logic:
1. Find where predictions transition from non-boundary to boundary (0→1)
   = Start of an isnad segment
2. Find where predictions transition from boundary to non-boundary (1→0)
   = End of an isnad segment
3. Extract segments between these transitions
4. Compare with gold standard
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class BoundaryTransitionConverter:
    """Convert token-level predictions to segments using boundary transitions."""

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

    def find_transitions(self) -> Tuple[List[int], List[int]]:
        """
        Find boundary transitions in token predictions.

        Returns:
            - starts: Token indices where 0→1 transition occurs (boundary starts)
            - ends: Token indices where 1→0 transition occurs (boundary ends)
        """
        # Prepend 0 to detect boundary at token 0
        preds_with_prefix = np.concatenate([[0], self.predictions])

        # Find differences (0→1 = +1, 1→0 = -1)
        diffs = np.diff(preds_with_prefix)

        # Transitions
        starts = np.where(diffs == 1)[0]  # 0→1 transitions
        ends = np.where(diffs == -1)[0]   # 1→0 transitions

        print(f"\n[INFO] Found boundary transitions:")
        print(f"  Boundary starts (0->1): {len(starts)}")
        print(f"  Boundary ends (1->0): {len(ends)}")

        return starts.tolist(), ends.tolist()

    def extract_segments(
        self,
        text: str,
        boundary_starts: List[int],
        boundary_ends: List[int]
    ) -> List[Dict]:
        """
        Extract segments from text based on boundary transitions.

        Logic:
        - Isnad segment: From boundary_start to boundary_end
        - Prose segment: Between boundary_end and next boundary_start
        """
        segments = []

        # Pair up starts and ends
        # Each start should have a corresponding end
        boundary_pairs = []

        for start in boundary_starts:
            # Find the next end after this start
            ends_after = [e for e in boundary_ends if e > start]
            if ends_after:
                end = ends_after[0]
                boundary_pairs.append((start, end))

        print(f"  Boundary pairs (isnad spans): {len(boundary_pairs)}")

        if not boundary_pairs:
            # No boundaries found, return whole text as one segment
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'type': 'unknown',
                'length': len(text),
                'token_span': (0, len(self.offsets)-1)
            }]

        # Extract segments
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

    def process(self, text: str) -> Dict:
        """Full conversion pipeline: text → segments."""
        print(f"\n[INFO] Processing text ({len(text):,} chars)...")

        # Find boundary transitions
        starts, ends = self.find_transitions()

        # Extract segments
        segments = self.extract_segments(text, starts, ends)

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
        description='Convert token-level CAMeL-BERT predictions to khabar segments (v2: boundary transitions)'
    )
    parser.add_argument('--input', required=True, help='Input JSON with raw inference results')
    parser.add_argument('--output', required=True, help='Output JSON with segments')
    parser.add_argument('--text', default='data/processed/kitab_uqala_reference_corpus.txt',
                        help='Original text file')
    parser.add_argument('--gold-standard', type=int, default=613,
                        help='Gold standard segment count for evaluation')

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
    converter = BoundaryTransitionConverter(str(input_path))
    result = converter.process(text)

    # Evaluate
    evaluation = evaluate_against_gold_standard(result['total_segments'], args.gold_standard)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert segments to JSON-serializable format (ensure all ints)
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
            'method': 'boundary_transitions_v2'
        },
        'segmentation': {
            'segments': serializable_segments,
            'total_segments': result['total_segments'],
            'type_breakdown': result['type_breakdown'],
            'boundary_starts': result['boundary_starts'],
            'boundary_ends': result['boundary_ends'],
            'statistics': result['statistics']
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
    print(f"SUMMARY - BOUNDARY TRANSITION METHOD")
    print(f"="*80)
    print(f"  Total segments: {result['total_segments']}")
    print(f"  Type breakdown: {result['type_breakdown']}")
    print(f"  Recall vs gold: {evaluation['recall']}")
    print(f"  Assessment: {evaluation['assessment']}")
    print(f"  Method: Identified {result['boundary_starts']} boundary starts / {result['boundary_ends']} boundary ends")
    print(f"="*80)


if __name__ == '__main__':
    main()
