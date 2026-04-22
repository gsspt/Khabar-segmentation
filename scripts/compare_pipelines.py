#!/usr/bin/env python3
"""
Compare three pipelines (CAMeL-BERT, Baseline v4, Deepseek) with standard metrics.

Compares:
1. CAMeL-BERT vs Gold Standard
2. Baseline v4 vs Gold Standard
3. CAMeL-BERT vs Baseline v4

Metrics:
- Precision: (# correct detections) / (# predicted)
- Recall: (# correct detections) / (# gold standard)
- F1: 2 * (Precision * Recall) / (Precision + Recall)
- Median distance: median char distance between predicted and gold positions
- Mean overlap: mean overlap % between segments

Usage:
    python scripts/compare_pipelines.py \
      --camelbert results/camelbert_[TEXT]_narrative_units.json \
      --baseline results/baseline_v4_[TEXT]_narrative_units.json \
      --gold results/gold_standard_[TEXT]_deepseek.json \
      --output results/comparison_[TEXT].json
"""

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_boundaries(data_file: str) -> list[tuple[int, int]]:
    """
    Load narrative units and extract (char_start, char_end) boundaries.

    Args:
        data_file: Path to narrative_units.json or gold_standard.json

    Returns:
        List of (start, end) tuples sorted by start position
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    units = data.get('narrative_units', data.get('units', []))

    boundaries = []
    for unit in units:
        start = unit.get('char_start')
        end = unit.get('char_end')
        if start is not None and end is not None:
            boundaries.append((start, end))

    # Sort by start position
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def calculate_overlap(seg1: tuple[int, int], seg2: tuple[int, int]) -> float:
    """
    Calculate overlap percentage between two segments.

    Args:
        seg1: (start, end) tuple
        seg2: (start, end) tuple

    Returns:
        Overlap as percentage (0-100)
    """
    s1_start, s1_end = seg1
    s2_start, s2_end = seg2

    overlap_start = max(s1_start, s2_start)
    overlap_end = min(s1_end, s2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start
    seg1_length = s1_end - s1_start
    seg2_length = s2_end - s2_start

    # Overlap as % of smaller segment
    min_length = min(seg1_length, seg2_length)
    return (overlap_length / min_length * 100) if min_length > 0 else 0.0


def find_closest_match(
    predicted: tuple[int, int],
    gold_boundaries: list[tuple[int, int]],
    tolerance: int = 80
) -> Optional[tuple[int, int]]:
    """
    Find gold boundary closest to predicted boundary.

    Args:
        predicted: (start, end) tuple
        gold_boundaries: List of gold (start, end) tuples
        tolerance: Maximum char distance to consider a match

    Returns:
        Closest gold boundary if within tolerance, else None
    """
    pred_start, pred_end = predicted

    # Find gold with start closest to predicted start
    min_distance = tolerance + 1
    closest = None

    for gold_start, gold_end in gold_boundaries:
        distance = abs(pred_start - gold_start)
        if distance < min_distance:
            min_distance = distance
            closest = (gold_start, gold_end)

    return closest if min_distance <= tolerance else None


def compare_boundaries(
    predicted: list[tuple[int, int]],
    gold: list[tuple[int, int]],
    tolerance: int = 80
) -> dict:
    """
    Compare predicted boundaries against gold standard.

    Args:
        predicted: List of (start, end) tuples from pipeline
        gold: List of (start, end) tuples from gold standard
        tolerance: Char distance tolerance for matching

    Returns:
        Dict with metrics (precision, recall, f1, distances, overlaps)
    """
    logger.info(f"Comparing {len(predicted)} predicted vs {len(gold)} gold boundaries")
    logger.info(f"Tolerance: ±{tolerance} chars")

    # Find matches
    matches = []
    distances = []
    overlaps = []
    unmatched_predicted = []

    for pred_boundary in predicted:
        match = find_closest_match(pred_boundary, gold, tolerance)
        if match:
            matches.append((pred_boundary, match))
            distance = abs(pred_boundary[0] - match[0])
            distances.append(distance)
            overlap = calculate_overlap(pred_boundary, match)
            overlaps.append(overlap)
        else:
            unmatched_predicted.append(pred_boundary)

    # Calculate metrics
    tp = len(matches)
    fp = len(unmatched_predicted)
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Statistics on distances and overlaps
    median_distance = statistics.median(distances) if distances else None
    mean_distance = statistics.mean(distances) if distances else None
    mean_overlap = statistics.mean(overlaps) if overlaps else None

    result = {
        'predicted_count': len(predicted),
        'gold_count': len(gold),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'median_distance_chars': median_distance,
        'mean_distance_chars': round(mean_distance, 1) if mean_distance else None,
        'mean_overlap_percent': round(mean_overlap, 1) if mean_overlap else None,
        'tolerance_chars': tolerance,
    }

    return result


def compare_all_pipelines(
    camelbert_file: str,
    baseline_file: str,
    gold_file: str,
    tolerance: int = 80
) -> dict:
    """
    Compare all three pipelines against each other and gold standard.

    Args:
        camelbert_file: Path to CAMeL-BERT narrative_units.json
        baseline_file: Path to Baseline v4 narrative_units.json
        gold_file: Path to gold_standard_deepseek.json
        tolerance: Char distance tolerance for matching

    Returns:
        Dict with comparisons: camelbert_vs_gold, baseline_vs_gold, camelbert_vs_baseline
    """
    logger.info("Loading boundaries from all pipelines...")

    camelbert = load_boundaries(camelbert_file)
    baseline = load_boundaries(baseline_file)
    gold = load_boundaries(gold_file)

    logger.info(f"Loaded {len(camelbert)} CAMeL-BERT, "
                f"{len(baseline)} Baseline, {len(gold)} Gold boundaries")

    comparisons = {
        'camelbert_vs_gold': compare_boundaries(camelbert, gold, tolerance),
        'baseline_vs_gold': compare_boundaries(baseline, gold, tolerance),
        'camelbert_vs_baseline': compare_boundaries(camelbert, baseline, tolerance),
    }

    return comparisons


def save_comparison_results(comparisons: dict, output_path: str) -> None:
    """Save comparison results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'tolerance_chars': 80,
        },
        'comparison': comparisons
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Comparison results saved to {output_path}")

    # Print summary
    logger.info("\n=== COMPARISON SUMMARY ===")
    for comparison_name, metrics in comparisons.items():
        logger.info(f"\n{comparison_name}:")
        logger.info(f"  Precision: {metrics['precision']}")
        logger.info(f"  Recall: {metrics['recall']}")
        logger.info(f"  F1: {metrics['f1']}")
        logger.info(f"  TP: {metrics['true_positives']}, "
                   f"FP: {metrics['false_positives']}, "
                   f"FN: {metrics['false_negatives']}")
        if metrics['median_distance_chars'] is not None:
            logger.info(f"  Median distance: {metrics['median_distance_chars']} chars")


def main():
    parser = argparse.ArgumentParser(
        description='Compare pipeline results against gold standard'
    )
    parser.add_argument(
        '--camelbert',
        required=True,
        help='Path to CAMeL-BERT narrative_units.json'
    )
    parser.add_argument(
        '--baseline',
        required=True,
        help='Path to Baseline v4 narrative_units.json'
    )
    parser.add_argument(
        '--gold',
        required=True,
        help='Path to gold_standard_deepseek.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save comparison results'
    )
    parser.add_argument(
        '--tolerance',
        type=int,
        default=80,
        help='Char distance tolerance for matching (default: 80)'
    )

    args = parser.parse_args()

    try:
        comparisons = compare_all_pipelines(
            args.camelbert,
            args.baseline,
            args.gold,
            args.tolerance
        )
        save_comparison_results(comparisons, args.output)
        logger.info("Comparison completed successfully")

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise


if __name__ == '__main__':
    main()
