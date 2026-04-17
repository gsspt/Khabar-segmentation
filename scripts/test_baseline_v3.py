#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Baseline V2 vs V3

Tests both versions against the reference corpus and compares:
- Detection count
- Boundary accuracy
- Coverage statistics
"""

import json
import sys
from pathlib import Path

# Load data
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

print("=" * 80)
print("BASELINE V2 vs V3 COMPARISON TEST")
print("=" * 80 + "\n")

print(f"Reference: {len(boundaries)} akhbars\n")

# Run both versions
sys.path.insert(0, '.')

print("[1] Running Baseline V2...")
from baseline_v2_isnad_first import segment_akhbars_v2
detected_v2 = segment_akhbars_v2(corpus)
print(f"    Detected: {len(detected_v2)}\n")

print("[2] Running Baseline V3...")
from baseline_v3_refined import segment_akhbars_v3
detected_v3 = segment_akhbars_v3(corpus)
print(f"    Detected: {len(detected_v3)}\n")

# Analyze improvement
print("[3] Comparing results...\n")

diff = len(detected_v3) - len(detected_v2)
pct_improvement = (diff / len(detected_v2) * 100) if detected_v2 else 0

print("DETECTION COUNT:")
print(f"  V2: {len(detected_v2)}/613 ({100*len(detected_v2)/613:.1f}%)")
print(f"  V3: {len(detected_v3)}/613 ({100*len(detected_v3)/613:.1f}%)")
print(f"  Improvement: {diff:+d} ({pct_improvement:+.1f}%)\n")

# Boundary accuracy
print("BOUNDARY ACCURACY:")

def compute_boundary_accuracy(detected, boundaries):
    """Compute distance from detected to nearest reference boundary."""
    ref_positions = {b['start'] for b in boundaries}
    det_positions = {d['start'] for d in detected}

    matches = []
    for ref_pos in ref_positions:
        closest_det = None
        min_dist = float('inf')

        for det_pos in det_positions:
            distance = abs(ref_pos - det_pos)
            if distance < min_dist:
                min_dist = distance
                closest_det = det_pos

        if closest_det is not None:
            matches.append(min_dist)

    if not matches:
        return {'perfect': 0, 'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'very_poor': 0, 'failed': 0}

    perfect = sum(1 for d in matches if d < 5)
    excellent = sum(1 for d in matches if 5 <= d < 20)
    good = sum(1 for d in matches if 20 <= d < 50)
    acceptable = sum(1 for d in matches if 50 <= d < 100)
    poor = sum(1 for d in matches if 100 <= d < 200)
    very_poor = sum(1 for d in matches if 200 <= d < 500)
    failed = sum(1 for d in matches if d >= 500)

    return {
        'perfect': perfect,
        'excellent': excellent,
        'good': good,
        'acceptable': acceptable,
        'poor': poor,
        'very_poor': very_poor,
        'failed': failed,
        'total': len(matches),
    }

acc_v2 = compute_boundary_accuracy(detected_v2, boundaries)
acc_v3 = compute_boundary_accuracy(detected_v3, boundaries)

print("Accuracy Tiers (% of matched boundaries):\n")
print("Tier              V2              V3          Change")
print("-" * 60)

tiers = ['perfect', 'excellent', 'good', 'acceptable', 'poor', 'very_poor', 'failed']
tier_names = {
    'perfect': 'Perfect (0-5)',
    'excellent': 'Excellent (5-20)',
    'good': 'Good (20-50)',
    'acceptable': 'Acceptable (50-100)',
    'poor': 'Poor (100-200)',
    'very_poor': 'Very Poor (200-500)',
    'failed': 'Failed (500+)',
}

for tier in tiers:
    if acc_v2['total'] > 0 and acc_v3['total'] > 0:
        v2_pct = acc_v2[tier] / acc_v2['total'] * 100
        v3_pct = acc_v3[tier] / acc_v3['total'] * 100
        change = v3_pct - v2_pct
        print(f"{tier_names[tier]:16} {v2_pct:5.1f}%  {v3_pct:5.1f}%  {change:+5.1f}%")

# Usable boundaries (< 50 chars accuracy)
if acc_v2['total'] > 0 and acc_v3['total'] > 0:
    usable_v2 = acc_v2['perfect'] + acc_v2['excellent'] + acc_v2['good']
    usable_v3 = acc_v3['perfect'] + acc_v3['excellent'] + acc_v3['good']
    usable_pct_v2 = usable_v2 / acc_v2['total'] * 100
    usable_pct_v3 = usable_v3 / acc_v3['total'] * 100

    print("-" * 60)
    print(f"{'Usable (<50)':16} {usable_pct_v2:5.1f}%  {usable_pct_v3:5.1f}%  {usable_pct_v3-usable_pct_v2:+5.1f}%\n")

# Coverage statistics
print("COVERAGE STATISTICS:\n")

cov_v2 = sum(d['end'] - d['start'] for d in detected_v2) / len(corpus) * 100
cov_v3 = sum(d['end'] - d['start'] for d in detected_v3) / len(corpus) * 100

print(f"  V2 Coverage: {cov_v2:.1f}%")
print(f"  V3 Coverage: {cov_v3:.1f}%\n")

# Segment length statistics
print("SEGMENT LENGTH STATISTICS:\n")

def compute_stats(detected):
    isnad_lens = [d['isnad_end'] - d['start'] for d in detected]
    khabar_lens = [d['end'] - d['isnad_end'] for d in detected]

    return {
        'avg_isnad': sum(isnad_lens) / len(isnad_lens) if isnad_lens else 0,
        'avg_khabar': sum(khabar_lens) / len(khabar_lens) if khabar_lens else 0,
    }

stats_v2 = compute_stats(detected_v2)
stats_v3 = compute_stats(detected_v3)

print(f"Average isnad length:")
print(f"  V2: {stats_v2['avg_isnad']:.0f} chars")
print(f"  V3: {stats_v3['avg_isnad']:.0f} chars\n")

print(f"Average khabar length:")
print(f"  V2: {stats_v2['avg_khabar']:.0f} chars")
print(f"  V3: {stats_v3['avg_khabar']:.0f} chars\n")

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80 + "\n")

print(f"Baseline V3 improves detection by {diff:+d} akhbars ({pct_improvement:+.1f}%)")
print(f"Detection: {len(detected_v2)} -> {len(detected_v3)}/613")

if acc_v2['total'] > 0 and acc_v3['total'] > 0:
    print(f"Boundary accuracy (< 50 chars): {usable_pct_v2:.1f}% -> {usable_pct_v3:.1f}%")

print(f"Coverage: {cov_v2:.1f}% -> {cov_v3:.1f}%\n")

# Save results
report_path = Path("../results/baseline_v3_comparison.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Baseline V2 vs V3 Comparison\n\n")

    f.write("## Detection Results\n\n")
    f.write(f"| Metric | V2 | V3 | Change |\n")
    f.write(f"|--------|----|----|--------|\n")
    f.write(f"| Detected | {len(detected_v2)}/613 | {len(detected_v3)}/613 | {diff:+d} |\n")
    f.write(f"| Detection Rate | {100*len(detected_v2)/613:.1f}% | {100*len(detected_v3)/613:.1f}% | {pct_improvement:+.1f}% |\n")
    f.write(f"| Coverage | {cov_v2:.1f}% | {cov_v3:.1f}% | {cov_v3-cov_v2:+.1f}% |\n\n")

    f.write("## Boundary Accuracy\n\n")
    f.write("| Tier | V2 | V3 | Change |\n")
    f.write("|------|----|----|--------|\n")
    for tier in tiers:
        if acc_v2['total'] > 0 and acc_v3['total'] > 0:
            v2_pct = acc_v2[tier] / acc_v2['total'] * 100
            v3_pct = acc_v3[tier] / acc_v3['total'] * 100
            change = v3_pct - v2_pct
            f.write(f"| {tier_names[tier]} | {v2_pct:.1f}% | {v3_pct:.1f}% | {change:+.1f}% |\n")

    f.write("\n## Key Improvements\n\n")
    f.write(f"1. **Detection**: +{diff} akhbars ({pct_improvement:+.1f}%)\n")
    if acc_v2['total'] > 0 and acc_v3['total'] > 0:
        f.write(f"2. **Boundary Accuracy**: {usable_pct_v2:.1f}% → {usable_pct_v3:.1f}% with < 50 chars error\n")
    f.write(f"3. **Coverage**: {cov_v2:.1f}% → {cov_v3:.1f}%\n")

print(f"Report saved to: {report_path}\n")
