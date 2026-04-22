#!/usr/bin/env python3
"""
Compare two segmentation pipelines directly.

Metrics:
- Boundary position distances
- Overlap between segments
- Coverage differences

Usage:
    python scripts/compare_two_pipelines.py \\
      --pipeline1 results/alDarrab_narrative_units_camelbert.json \\
      --pipeline2 results/alDarrab_narrative_units_baseline.json \\
      --name1 camelbert \\
      --name2 baseline \\
      --output results/alDarrab_comparison.json
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Tuple


def load_units(data_file: str) -> List[Dict]:
    """Load narrative units from JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    units = data.get('narrative_units', data.get('units', []))
    return sorted(units, key=lambda u: u.get('char_start', 0))


def get_boundaries(units: List[Dict]) -> List[int]:
    """Extract boundary start positions from units."""
    return sorted([u.get('char_start', 0) for u in units])


def find_closest(pos: int, positions: List[int], max_distance: int = 500) -> Tuple[int, int]:
    """Find closest position and return (position, distance)."""
    if not positions:
        return None, float('inf')

    closest = min(positions, key=lambda p: abs(p - pos))
    distance = abs(pos - closest)

    if distance <= max_distance:
        return closest, distance
    else:
        return closest, distance


def calculate_metrics(p1_units: List[Dict], p2_units: List[Dict], name1: str, name2: str) -> Dict:
    """Calculate comparison metrics."""

    p1_boundaries = get_boundaries(p1_units)
    p2_boundaries = get_boundaries(p2_units)

    # Distance analysis
    p1_to_p2_distances = []
    for pos in p1_boundaries:
        _, dist = find_closest(pos, p2_boundaries)
        p1_to_p2_distances.append(dist)

    p2_to_p1_distances = []
    for pos in p2_boundaries:
        _, dist = find_closest(pos, p1_boundaries)
        p2_to_p1_distances.append(dist)

    # Calculate percentiles
    def calc_percentiles(distances):
        if not distances:
            return None
        sorted_dist = sorted(distances)
        return {
            'min': min(distances),
            'q25': sorted_dist[len(sorted_dist)//4],
            'median': sorted_dist[len(sorted_dist)//2],
            'q75': sorted_dist[3*len(sorted_dist)//4],
            'max': max(distances),
            'mean': statistics.mean(distances),
        }

    # Alignment quality
    def count_within_tolerance(distances, tolerance):
        return sum(1 for d in distances if d <= tolerance)

    return {
        'summary': {
            f'{name1}_boundaries': len(p1_boundaries),
            f'{name2}_boundaries': len(p2_boundaries),
            'difference': len(p2_boundaries) - len(p1_boundaries),
            'difference_pct': (len(p2_boundaries) - len(p1_boundaries)) / len(p1_boundaries) * 100 if p1_boundaries else 0,
        },
        f'{name1}_to_{name2}': {
            'distances': calc_percentiles(p1_to_p2_distances),
            'within_20': count_within_tolerance(p1_to_p2_distances, 20),
            'within_50': count_within_tolerance(p1_to_p2_distances, 50),
            'within_100': count_within_tolerance(p1_to_p2_distances, 100),
            'within_500': count_within_tolerance(p1_to_p2_distances, 500),
        },
        f'{name2}_to_{name1}': {
            'distances': calc_percentiles(p2_to_p1_distances),
            'within_20': count_within_tolerance(p2_to_p1_distances, 20),
            'within_50': count_within_tolerance(p2_to_p1_distances, 50),
            'within_100': count_within_tolerance(p2_to_p1_distances, 100),
            'within_500': count_within_tolerance(p2_to_p1_distances, 500),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two segmentation pipelines"
    )
    parser.add_argument("--pipeline1", required=True, help="First pipeline JSON")
    parser.add_argument("--pipeline2", required=True, help="Second pipeline JSON")
    parser.add_argument("--name1", default="pipeline1", help="Name for pipeline 1")
    parser.add_argument("--name2", default="pipeline2", help="Name for pipeline 2")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.name1} from {args.pipeline1}...")
    p1_units = load_units(args.pipeline1)
    print(f"  Units: {len(p1_units)}")

    print(f"Loading {args.name2} from {args.pipeline2}...")
    p2_units = load_units(args.pipeline2)
    print(f"  Units: {len(p2_units)}")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(p1_units, p2_units, args.name1, args.name2)

    # Prepare output
    output = {
        "metadata": {
            "comparison": f"{args.name1} vs {args.name2}",
            "pipeline1": args.pipeline1,
            "pipeline2": args.pipeline2,
        },
        "metrics": metrics,
    }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Comparison saved to {out_path}")

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{args.name1}: {metrics['summary'][f'{args.name1}_boundaries']} boundaries")
    print(f"{args.name2}: {metrics['summary'][f'{args.name2}_boundaries']} boundaries")
    print(f"Difference: {metrics['summary']['difference']} ({metrics['summary']['difference_pct']:+.1f}%)")

    print(f"\n{args.name1} alignment to {args.name2}:")
    dist_data = metrics[f'{args.name1}_to_{args.name2}']['distances']
    print(f"  Median distance: {dist_data['median']} chars")
    print(f"  Within +-100 chars: {metrics[f'{args.name1}_to_{args.name2}']['within_100']}/{len(p1_units)}")

    print(f"\n{args.name2} alignment to {args.name1}:")
    dist_data = metrics[f'{args.name2}_to_{args.name1}']['distances']
    print(f"  Median distance: {dist_data['median']} chars")
    print(f"  Within +-100 chars: {metrics[f'{args.name2}_to_{args.name1}']['within_100']}/{len(p2_units)}")


if __name__ == "__main__":
    main()
