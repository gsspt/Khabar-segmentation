#!/usr/bin/env python3
"""
Compare three pipelines (Deepseek, CAMeL-BERT, Baseline) for Arabic text segmentation.

Supports any text by specifying --text argument.

Usage:
    python scripts/compare_pipelines.py --text ibjawzi
    python scripts/compare_pipelines.py --text aldarrab
    python scripts/compare_pipelines.py --text 0406IbnHabib
"""

import json
import argparse
import sys
from pathlib import Path

def find_results_directory(text_name: str) -> Path:
    """
    Find the results directory for the given text.

    Handles common name variations and common abbreviations.
    """
    results_base = Path("results")
    if not results_base.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base}")

    # Common name mappings for known texts
    name_mappings = {
        'ibjawzi': 'IbnJawzi',
        'ibn jawzi': 'IbnJawzi',
        'hamqa': 'IbnJawzi',
        'aldarrab': 'alDarrab',
        'al darrab': 'alDarrab',
        'kitab uqala': 'Kitab_Uqala_al_Majanin',
        'uqala': 'Kitab_Uqala_al_Majanin',
    }

    # Check if there's a known mapping
    text_lower = text_name.lower()
    if text_lower in name_mappings:
        mapped_name = name_mappings[text_lower]
        result_path = results_base / mapped_name
        if result_path.exists() and result_path.is_dir():
            return result_path

    # Normalize text_name for comparison (remove underscores, lowercase)
    text_normalized = text_lower.replace("_", "").replace("-", "")

    # First pass: exact case-insensitive match
    for subdir in results_base.iterdir():
        if not subdir.is_dir():
            continue

        subdir_normalized = subdir.name.lower().replace("_", "").replace("-", "")

        if subdir_normalized == text_normalized:
            return subdir

    # If not found, list available options
    available = [d.name for d in results_base.iterdir() if d.is_dir()]
    raise FileNotFoundError(
        f"Could not find results directory for text: {text_name}.\n"
        f"Available directories: {available}\n"
        f"Try: --text ibnjawzi, --text aldarrab, or --text uqala"
    )

def build_file_paths(text_name: str, results_dir: Path) -> dict:
    """
    Build file paths for the three pipelines.

    Tries multiple naming conventions to be flexible.
    """
    # Normalize text name for file matching
    text_lower = text_name.lower()

    # Try to find Deepseek file
    deepseek_patterns = [
        f"gold_standard_{text_lower}_hamqa_deepseek.json",
        f"gold_standard_{text_lower}_deepseek.json",
        f"deepseek_{text_lower}_narrative_units.json",
        f"gold_standard_{text_name}_deepseek.json",
    ]
    deepseek_file = None
    for pattern in deepseek_patterns:
        candidate = results_dir / pattern
        if candidate.exists():
            deepseek_file = candidate
            break

    # Try to find CAMeL-BERT file
    camelbert_patterns = [
        f"camelbert_{text_lower}_narrative_units.json",
        f"camelbert_{text_name}_narrative_units.json",
        f"{text_lower}_camelbert_narrative_units.json",
    ]
    camelbert_file = None
    for pattern in camelbert_patterns:
        candidate = results_dir / pattern
        if candidate.exists():
            camelbert_file = candidate
            break

    # Try to find Baseline file
    baseline_patterns = [
        f"baseline_v4_{text_lower}_narrative_units.json",
        f"baseline_v4_{text_lower}_hamqa_segments.json",
        f"baseline_{text_lower}_narrative_units.json",
        f"{text_lower}_baseline_narrative_units.json",
    ]
    baseline_file = None
    for pattern in baseline_patterns:
        candidate = results_dir / pattern
        if candidate.exists():
            baseline_file = candidate
            break

    return {
        'deepseek': deepseek_file,
        'camelbert': camelbert_file,
        'baseline': baseline_file
    }

def load_pipeline_results(text_name: str):
    """Load results for the given text."""
    # Find results directory
    try:
        results_dir = find_results_directory(text_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using results directory: {results_dir}")

    # Build file paths
    file_paths = build_file_paths(text_name, results_dir)

    # Check that all files exist
    missing = [name for name, path in file_paths.items() if path is None]
    if missing:
        print(f"[ERROR] Missing files for: {', '.join(missing)}", file=sys.stderr)
        print(f"[INFO] Looked in: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load files
    print("\nLoading pipelines...")
    with open(file_paths['deepseek'], encoding='utf-8') as f:
        deepseek_data = json.load(f)

    with open(file_paths['camelbert'], encoding='utf-8') as f:
        camelbert_data = json.load(f)

    with open(file_paths['baseline'], encoding='utf-8') as f:
        baseline_data = json.load(f)

    deepseek_units = deepseek_data.get('narrative_units', [])
    camelbert_units = camelbert_data.get('narrative_units', [])
    baseline_units = baseline_data.get('narrative_units', [])

    return deepseek_units, camelbert_units, baseline_units

def main():
    parser = argparse.ArgumentParser(
        description='Compare three segmentation pipelines for Arabic text'
    )
    parser.add_argument(
        '--text',
        default='ibjawzi',
        help='Text name (e.g., ibjawzi, aldarrab, 0406IbnHabib). Default: ibjawzi'
    )
    parser.add_argument(
        '--tolerance',
        type=int,
        default=80,
        help='Boundary alignment tolerance in chars. Default: 80'
    )

    args = parser.parse_args()

    # Load results
    deepseek_units, camelbert_units, baseline_units = load_pipeline_results(args.text)

    print(f"  Deepseek:   {len(deepseek_units):3d} units")
    print(f"  CAMeL-BERT: {len(camelbert_units):3d} units")
    print(f"  Baseline:   {len(baseline_units):3d} units")

    # Get boundaries
    deepseek_boundaries = sorted([u.get('char_start', 0) for u in deepseek_units])
    camelbert_boundaries = sorted([u.get('char_start', 0) for u in camelbert_units])
    baseline_boundaries = sorted([u.get('char_start', 0) for u in baseline_units])

    print("\n" + "=" * 70)
    print(f"THREE-PIPELINE COMPARISON: {args.text.upper()}")
    print("=" * 70)
    print(f"\nBoundary Alignment Tolerance: +/- {args.tolerance} chars")
    print("=" * 70)

    def compare_boundaries(b1, b2, name1, name2, tolerance=80):
        """Compare two boundary sets"""
        matches = 0
        for b in b1:
            closest = min(abs(b - b2_val) for b2_val in b2) if b2 else float('inf')
            if closest <= tolerance:
                matches += 1

        precision = matches / len(b1) if b1 else 0
        recall = matches / len(b2) if b2 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{name1} vs {name2}:")
        print(f"  Matches: {matches}/{len(b1)} (precision: {precision:.1%}, recall: {recall:.1%})")
        print(f"  F1 Score: {f1:.3f}")
        return matches, precision, recall, f1

    # Comparisons
    m1, p1, r1, f1_1 = compare_boundaries(deepseek_boundaries, camelbert_boundaries, "Deepseek", "CAMeL-BERT", args.tolerance)
    m2, p2, r2, f1_2 = compare_boundaries(deepseek_boundaries, baseline_boundaries, "Deepseek", "Baseline", args.tolerance)
    m3, p3, r3, f1_3 = compare_boundaries(camelbert_boundaries, baseline_boundaries, "CAMeL-BERT", "Baseline", args.tolerance)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nUnit counts:")
    print(f"  Deepseek:   {len(deepseek_units):3d}")
    print(f"  CAMeL-BERT: {len(camelbert_units):3d}")
    print(f"  Baseline:   {len(baseline_units):3d}")

    print("\nF1 Scores:")
    print(f"  Deepseek vs CAMeL-BERT: {f1_1:.3f}")
    print(f"  Deepseek vs Baseline:   {f1_2:.3f}")
    print(f"  CAMeL-BERT vs Baseline: {f1_3:.3f}")

    # Key finding
    print("\n" + "=" * 70)
    if abs(len(deepseek_units) - len(camelbert_units)) <= 10:
        print("KEY: Deepseek and CAMeL-BERT have similar unit counts!")
        print(f"      Difference: {abs(len(deepseek_units) - len(camelbert_units))} units")
    else:
        print(f"NOTE: Significant difference between pipelines")
        print(f"      Deepseek:   {len(deepseek_units)}")
        print(f"      CAMeL-BERT: {len(camelbert_units)}")
        print(f"      Baseline:   {len(baseline_units)}")

    print("=" * 70)
    print("\n[OK] Comparison complete!")

if __name__ == '__main__':
    main()
