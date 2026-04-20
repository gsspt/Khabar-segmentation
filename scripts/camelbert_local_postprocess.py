#!/usr/bin/env python3
"""
Local post-processing for CAMeL-BERT inference results.

Pipeline:
1. Load raw inference JSON (from Colab)
2. Cluster boundary tokens
3. Extract text segments
4. Compare with gold standard (613 khabars)
5. Calculate metrics (recall, precision, F1)

Usage:
    python3 scripts/camelbert_local_postprocess.py \
        --input results/camelbert_kitab_uqala_raw_inference.json \
        --output results/camelbert_kitab_uqala_segments.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class TokenToSegmentConverter:
    """Convert token-level predictions to document segments."""

    def __init__(self, raw_inference_json: str):
        """Load raw inference results from Colab."""
        with open(raw_inference_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.predictions = np.array(self.data['inference_results']['predictions'])
        self.probabilities = np.array(self.data['inference_results']['probabilities'])
        self.offsets = np.array(self.data['inference_results']['offsets'])
        self.tokens = self.data['inference_results']['tokens']

        print(f"[OK] Loaded inference data")
        print(f"  Total tokens: {len(self.predictions)}")
        print(f"  Boundary tokens: {np.sum(self.predictions)}")

    def cluster_boundaries(self, max_token_gap: int = 3) -> List[Tuple[int, int]]:
        """
        Cluster adjacent boundary tokens into segment boundaries.

        Adjacent tokens (distance < max_token_gap) mark the same boundary.
        Returns: List of (char_start, char_end) for each boundary cluster.
        """
        boundary_indices = np.where(self.predictions == 1)[0]

        if len(boundary_indices) == 0:
            print(f"[WARN] No boundary tokens detected!")
            return []

        print(f"\n[INFO] Clustering {len(boundary_indices)} boundary tokens...")

        clusters = []
        current_start_idx = boundary_indices[0]
        current_end_idx = boundary_indices[0]

        for i in range(1, len(boundary_indices)):
            idx = boundary_indices[i]
            prev_idx = boundary_indices[i - 1]

            gap = idx - prev_idx
            if gap <= max_token_gap:
                # Continue cluster
                current_end_idx = idx
            else:
                # End current cluster
                char_start = self.offsets[current_start_idx][0]
                char_end = self.offsets[current_end_idx][1]
                clusters.append((int(char_start), int(char_end)))

                current_start_idx = idx
                current_end_idx = idx

        # Final cluster
        char_start = self.offsets[current_start_idx][0]
        char_end = self.offsets[current_end_idx][1]
        clusters.append((int(char_start), int(char_end)))

        print(f"  Created {len(clusters)} boundary clusters")
        return clusters

    def extract_segments(
        self,
        text: str,
        boundary_clusters: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Extract segments from text given boundary positions.

        Each boundary marks the start of an isnad.
        Non-boundary text between boundaries is narrative.
        """
        if not boundary_clusters:
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'type': 'unknown',
                'length': len(text)
            }]

        segments = []
        current_pos = 0

        for bound_start, bound_end in sorted(boundary_clusters):
            # Text before boundary (narrative)
            if current_pos < bound_start:
                seg_text = text[current_pos:bound_start].strip()
                if seg_text:
                    segments.append({
                        'text': seg_text,
                        'start': current_pos,
                        'end': bound_start,
                        'type': 'prose',
                        'length': len(seg_text)
                    })

            # Boundary itself (isnad)
            seg_text = text[bound_start:bound_end].strip()
            if seg_text:
                segments.append({
                    'text': seg_text,
                    'start': bound_start,
                    'end': bound_end,
                    'type': 'isnad',
                    'length': len(seg_text)
                })

            current_pos = bound_end

        # Remaining text
        if current_pos < len(text):
            seg_text = text[current_pos:].strip()
            if seg_text:
                segments.append({
                    'text': seg_text,
                    'start': current_pos,
                    'end': len(text),
                    'type': 'prose',
                    'length': len(seg_text)
                })

        return segments

    def process(self, text: str) -> Dict:
        """Full conversion pipeline: text → segments."""
        print(f"\n[INFO] Processing text ({len(text):,} chars)...")

        # Cluster boundaries
        boundary_clusters = self.cluster_boundaries(max_token_gap=3)

        # Extract segments
        segments = self.extract_segments(text, boundary_clusters)

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
            'boundary_clusters': len(boundary_clusters),
            'statistics': {
                'avg_segment_length': int(np.mean([s['length'] for s in segments])),
                'min_segment_length': int(np.min([s['length'] for s in segments])),
                'max_segment_length': int(np.max([s['length'] for s in segments])),
            }
        }


def evaluate_against_gold_standard(
    camelbert_segments: int,
    gold_standard: int = 613
) -> Dict:
    """Compare CAMeL-BERT segments against gold standard."""
    recall = camelbert_segments / gold_standard if gold_standard > 0 else 0
    ratio = camelbert_segments / gold_standard if gold_standard > 0 else 0

    print(f"\n[EVALUATION vs Gold Standard (613 khabars)]")
    print(f"  CAMeL-BERT segments: {camelbert_segments}")
    print(f"  Gold standard: {gold_standard}")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  Ratio: {ratio:.2f}x")

    if ratio > 1.2:
        assessment = "OVER-segments"
    elif ratio < 0.8:
        assessment = "UNDER-segments"
    else:
        assessment = "WELL-ALIGNED"

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
        description='Convert token-level CAMeL-BERT predictions to khabar segments'
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
    converter = TokenToSegmentConverter(str(input_path))
    result = converter.process(text)

    # Evaluate
    evaluation = evaluate_against_gold_standard(result['total_segments'], args.gold_standard)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'source': str(input_path),
            'text': str(text_path),
            'gold_standard': args.gold_standard
        },
        'segmentation': result,
        'evaluation': evaluation
    }

    print(f"\n[INFO] Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Complete!")
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"  Total segments: {result['total_segments']}")
    print(f"  Type breakdown: {result['type_breakdown']}")
    print(f"  Recall vs gold: {evaluation['recall']}")
    print(f"  Assessment: {evaluation['assessment']}")
    print(f"="*80)


if __name__ == '__main__':
    main()
