#!/usr/bin/env python3
"""
Detailed comparison and analysis of clustering strategies.
"""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).parent.parent

print("=" * 80)
print("CLUSTERING STRATEGIES - DETAILED COMPARISON")
print("=" * 80)

# Load results
results_path = ROOT / "results/clustering_strategies_comparison.json"
with open(results_path, encoding='utf-8') as f:
    results = json.load(f)

strategies = results['strategies']

# ============================================================================
# TABLE 1: COMPREHENSIVE METRICS
# ============================================================================

print("\n" + "-" * 80)
print("TABLE 1: ALL METRICS COMPARISON")
print("-" * 80)

print(f"\n{'Strategy':<25} {'Clusters':<10} {'F1':<10} {'Prec':<10} {'Recall':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
print("-" * 100)

for strategy_name in ['baseline_gap50', 'optimal_gap20', 'confidence_weighted', 'linguistic', 'hierarchical', 'ensemble']:
    if strategy_name not in strategies:
        continue

    data = strategies[strategy_name]['evaluation']
    s = strategies[strategy_name]

    print(f"{strategy_name:<25} {s['clusters']:<10} {data['f1']:<10.4f} {data['precision']:<10.4f} {data['recall']:<10.4f} {data['tp']:<8} {data['fp']:<8} {data['fn']:<8}")

# ============================================================================
# TABLE 2: RANKING BY DIFFERENT CRITERIA
# ============================================================================

print("\n" + "-" * 80)
print("TABLE 2: RANKING BY DIFFERENT CRITERIA")
print("-" * 80)

# F1 ranking
print("\nBy F1 Score (highest first):")
ranked_f1 = sorted(strategies.items(), key=lambda x: x[1]['evaluation']['f1'], reverse=True)
for i, (name, data) in enumerate(ranked_f1, 1):
    print(f"{i}. {name:<25} {data['evaluation']['f1']:.4f}")

# Precision ranking
print("\nBy Precision (highest first):")
ranked_prec = sorted(strategies.items(), key=lambda x: x[1]['evaluation']['precision'], reverse=True)
for i, (name, data) in enumerate(ranked_prec, 1):
    print(f"{i}. {name:<25} {data['evaluation']['precision']:.4f}")

# Recall ranking
print("\nBy Recall (highest first):")
ranked_recall = sorted(strategies.items(), key=lambda x: x[1]['evaluation']['recall'], reverse=True)
for i, (name, data) in enumerate(ranked_recall, 1):
    print(f"{i}. {name:<25} {data['evaluation']['recall']:.4f}")

# ============================================================================
# TABLE 3: IMPROVEMENT OVER BASELINE
# ============================================================================

print("\n" + "-" * 80)
print("TABLE 3: IMPROVEMENT OVER BASELINE (gap=50)")
print("-" * 80)

baseline_f1 = strategies['baseline_gap50']['evaluation']['f1']
baseline_tp = strategies['baseline_gap50']['evaluation']['tp']
baseline_fp = strategies['baseline_gap50']['evaluation']['fp']
baseline_fn = strategies['baseline_gap50']['evaluation']['fn']

print(f"\nBaseline (gap=50): F1={baseline_f1:.4f}, TP={baseline_tp}, FP={baseline_fp}, FN={baseline_fn}")
print(f"\n{'Strategy':<25} {'F1 Change':<15} {'TP Change':<15} {'FP Change':<15} {'FN Change':<15}")
print("-" * 80)

for strategy_name in ['optimal_gap20', 'confidence_weighted', 'linguistic', 'hierarchical', 'ensemble']:
    if strategy_name not in strategies:
        continue

    data = strategies[strategy_name]['evaluation']
    f1_change = data['f1'] - baseline_f1
    tp_change = data['tp'] - baseline_tp
    fp_change = data['fp'] - baseline_fp
    fn_change = data['fn'] - baseline_fn

    print(f"{strategy_name:<25} {f1_change:+.4f} ({f1_change*100:+.2f}%)  {tp_change:+d}          {fp_change:+d}          {fn_change:+d}")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS PER STRATEGY")
print("=" * 80)

strategies_info = {
    'baseline_gap50': {
        'description': 'Original approach: gap=50 chars',
        'pros': ['Established baseline', 'Moderate precision'],
        'cons': ['Sub-optimal gap value', 'Misses 127 boundaries'],
    },
    'optimal_gap20': {
        'description': 'Data-driven: gap=20 chars',
        'pros': ['Highest F1 score (0.8646)', 'Detects 12 more boundaries', 'Based on gap distribution analysis'],
        'cons': ['Slightly lower precision (-1.07%)', 'More false positives (+7)'],
    },
    'confidence_weighted': {
        'description': 'Confidence filtering (conf >= 0.70)',
        'pros': ['Highest precision (95.59%)', 'Fewest false positives (22)', 'Filters low-quality tokens'],
        'cons': ['Lowest recall (77.81%)', 'Misses 136 boundaries', 'F1 same as baseline'],
    },
    'linguistic': {
        'description': 'Boosts isnad verbs (hadathna, akhbarna)',
        'pros': ['Uses linguistic knowledge', 'Simple to implement'],
        'cons': ['Same F1 as baseline', 'Linguistic boost too small', 'No improvement in detection'],
    },
    'hierarchical': {
        'description': 'Two-pass: gap=30 then merge if < 10 chars',
        'pros': ['Good balance (F1=0.8629)', 'Better recall than baseline', 'Reduces over-segmentation'],
        'cons': ['Slightly complex logic', 'Still lower than optimal_gap20'],
    },
    'ensemble': {
        'description': 'Multi-criteria: distance + confidence + linguistic',
        'pros': ['Balanced metrics', 'F1=0.8629', 'Adaptive gap based on confidence'],
        'cons': ['Complexity', 'Same F1 as hierarchical', 'Not better than simple gap=20'],
    }
}

for strategy in ['baseline_gap50', 'optimal_gap20', 'confidence_weighted', 'linguistic', 'hierarchical', 'ensemble']:
    if strategy not in strategies_info:
        continue

    info = strategies_info[strategy]
    data = strategies[strategy]['evaluation']

    print(f"\n{strategy.upper()}")
    print(f"Description: {info['description']}")
    print(f"Metrics: F1={data['f1']:.4f}, P={data['precision']:.4f}, R={data['recall']:.4f}")
    print(f"Detections: TP={data['tp']}, FP={data['fp']}, FN={data['fn']}")
    print(f"Pros:")
    for pro in info['pros']:
        print(f"  - {pro}")
    print(f"Cons:")
    for con in info['cons']:
        print(f"  - {con}")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("""
Based on the comprehensive evaluation:

WINNER: optimal_gap20
- Highest F1 score: 0.8646 (+0.67% vs baseline)
- Detects 12 additional boundaries (TP: 498 vs 486)
- Data-driven: optimized from gap distribution analysis
- Simple: single parameter change (gap=50 -> gap=20)
- Generalizable: likely works on other texts

RUNNER-UP: hierarchical (F1=0.8629)
- Nearly equal performance (-0.17% vs optimal_gap20)
- Better precision than optimal_gap20
- Two-pass approach adds slight complexity
- Good alternative if cluster merging needed

WHY NOT OTHERS:

confidence_weighted (F1=0.8579):
- Highest precision BUT lowest recall
- Loses 136 boundaries (more than baseline)
- Not worth the trade-off for this task

linguistic (F1=0.8579):
- No improvement despite linguistic boost
- Isnad verb boost too small (15%)
- Same results as baseline

ensemble (F1=0.8629):
- Same F1 as hierarchical but more complex
- Not justified by the marginal benefit
- Harder to understand/maintain

IMPLEMENTATION:

1. IMMEDIATE: Change gap from 50 to 20
   Location: scripts/convert_boundary_tokens_direct.py, line 87
   Change: GAP_CLUSTER = 20

2. VALIDATION: Test on 2-3 other OpenITI texts
   Confirm F1 improvement generalizes

3. DEPLOYMENT: Use as new standard for all texts

EXPECTED IMPACT:
- F1 improvement: +0.67% (0.8579 -> 0.8646)
- Better recall: +1.96% (79.28% -> 81.24%)
- Acceptable precision: 92.39% (vs 93.46%)
- 12 more boundaries correctly detected
""")

print("\n[OK] Analysis complete!")
