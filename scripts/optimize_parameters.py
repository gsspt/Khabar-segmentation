#!/usr/bin/env python3
"""
Parameter optimization for CAMeL-BERT boundary conversion.

Tests different parameter combinations to find optimal F1 score.
"""

import json
import statistics
from pathlib import Path
from typing import Callable

# Load data once
raw_path = Path('results/Kitab_Uqala_al_Majanin/camelbert_kitab_uqala_raw_inference.json')
corpus_path = Path('data/processed/kitab_uqala_reference_corpus.txt')
gold_path = Path('data/processed/kitab_uqala_boundaries.json')

print("Loading data...")
with open(raw_path, 'r', encoding='utf-8') as f:
    raw = json.load(f)['inference_results']

with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

with open(gold_path, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

gold_bounds = sorted([int(b['start']) for b in gold_data])
print(f"  Gold standard: {len(gold_bounds)} boundaries\n")


def extract_boundaries(
    confidence_threshold: float = 0.70,
    gap_cluster: int = 50,
    merge_min_gap: int = 10
) -> tuple[list[int], dict]:
    """Extract boundaries with given parameters."""

    # Extract boundary tokens
    boundary_by_char = {}
    for i, (tok, off, pred, prob) in enumerate(
        zip(raw['tokens'], raw['offsets'], raw['predictions'], raw['probabilities'])
    ):
        if pred != 1 or off == [0, 0]:
            continue
        if tok in {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}:
            continue
        if prob < confidence_threshold:
            continue

        char_start = off[0]
        if char_start not in boundary_by_char or prob > boundary_by_char[char_start][2]:
            boundary_by_char[char_start] = (tok, off, prob)

    # Cluster
    sorted_chars = sorted(boundary_by_char.keys())
    clusters = []
    if sorted_chars:
        cur_start = sorted_chars[0]
        cur_end = boundary_by_char[sorted_chars[0]][1][1]
        cur_tokens = [sorted_chars[0]]

        for char_start in sorted_chars[1:]:
            off = boundary_by_char[char_start][1]
            gap = char_start - cur_end

            if gap <= gap_cluster:
                cur_end = max(cur_end, off[1])
                cur_tokens.append(char_start)
            else:
                clusters.append({
                    'char_start': cur_start,
                    'char_end': cur_end,
                    'n_tokens': len(cur_tokens)
                })
                cur_start = char_start
                cur_end = off[1]
                cur_tokens = [char_start]

        clusters.append({
            'char_start': cur_start,
            'char_end': cur_end,
            'n_tokens': len(cur_tokens)
        })

    # Merge
    if merge_min_gap > 0:
        merged = []
        i = 0
        while i < len(clusters):
            current = clusters[i].copy()
            while i + 1 < len(clusters):
                gap = clusters[i + 1]['char_start'] - current['char_end']
                if gap < merge_min_gap:
                    current['char_end'] = clusters[i + 1]['char_end']
                    current['n_tokens'] += clusters[i + 1]['n_tokens']
                    i += 1
                else:
                    break
            merged.append(current)
            i += 1
        clusters = merged

    boundaries = [c['char_start'] for c in clusters]

    return boundaries, {
        'unique_positions': len(boundary_by_char),
        'clusters': len(clusters),
        'boundaries': len(boundaries)
    }


def evaluate(predictions: list[int], gold: list[int], tolerance: int = 80) -> dict:
    """Evaluate predictions against gold standard."""
    matched_gold = set()
    matched_pred = set()

    for i, p in enumerate(predictions):
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            if abs(p - g) <= tolerance:
                matched_gold.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_gold)
    fp = len(predictions) - len(matched_pred)
    fn = len(gold) - tp

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': prec, 'recall': rec, 'f1': f1
    }


# Test different parameters
print("=" * 100)
print("PARAMETER OPTIMIZATION")
print("=" * 100)

results = []

print("\n1. Testing CONFIDENCE_THRESHOLD (with gap=50, merge=true)")
print("-" * 100)
print(f"{'Conf':<8} {'Unique':<8} {'Clusters':<10} {'Bounds':<8} {'P':<8} {'R':<8} {'F1':<8} {'TP/FP/FN':<20}")
print("-" * 100)

for conf_thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    bounds, stats = extract_boundaries(
        confidence_threshold=conf_thresh,
        gap_cluster=50,
        merge_min_gap=10
    )
    evals = evaluate(bounds, gold_bounds, 80)
    results.append({
        'confidence': conf_thresh,
        'gap_cluster': 50,
        'merge_min_gap': 10,
        'bounds': len(bounds),
        'f1': evals['f1'],
        'precision': evals['precision'],
        'recall': evals['recall'],
        'tp': evals['tp'],
        'fp': evals['fp'],
        'fn': evals['fn']
    })
    print(f"{conf_thresh:<8.2f} {stats['unique_positions']:<8} {stats['clusters']:<10} "
          f"{stats['boundaries']:<8} {evals['precision']:<8.4f} {evals['recall']:<8.4f} "
          f"{evals['f1']:<8.4f} {evals['tp']}/{evals['fp']}/{evals['fn']:<20}")

print("\n2. Testing GAP_CLUSTER (with conf=0.70, merge=true)")
print("-" * 100)
print(f"{'Gap':<8} {'Unique':<8} {'Clusters':<10} {'Bounds':<8} {'P':<8} {'R':<8} {'F1':<8} {'TP/FP/FN':<20}")
print("-" * 100)

for gap in [5, 10, 15, 20, 30, 40, 50, 75, 100, 150]:
    bounds, stats = extract_boundaries(
        confidence_threshold=0.70,
        gap_cluster=gap,
        merge_min_gap=10
    )
    evals = evaluate(bounds, gold_bounds, 80)
    results.append({
        'confidence': 0.70,
        'gap_cluster': gap,
        'merge_min_gap': 10,
        'bounds': len(bounds),
        'f1': evals['f1'],
        'precision': evals['precision'],
        'recall': evals['recall'],
        'tp': evals['tp'],
        'fp': evals['fp'],
        'fn': evals['fn']
    })
    print(f"{gap:<8} {stats['unique_positions']:<8} {stats['clusters']:<10} "
          f"{stats['boundaries']:<8} {evals['precision']:<8.4f} {evals['recall']:<8.4f} "
          f"{evals['f1']:<8.4f} {evals['tp']}/{evals['fp']}/{evals['fn']:<20}")

print("\n3. Testing MERGE_MIN_GAP (with conf=0.70, gap=50)")
print("-" * 100)
print(f"{'Merge':<8} {'Unique':<8} {'Clusters':<10} {'Bounds':<8} {'P':<8} {'R':<8} {'F1':<8} {'TP/FP/FN':<20}")
print("-" * 100)

for merge_gap in [0, 5, 10, 15, 20, 30, 50]:
    bounds, stats = extract_boundaries(
        confidence_threshold=0.70,
        gap_cluster=50,
        merge_min_gap=merge_gap
    )
    evals = evaluate(bounds, gold_bounds, 80)
    results.append({
        'confidence': 0.70,
        'gap_cluster': 50,
        'merge_min_gap': merge_gap,
        'bounds': len(bounds),
        'f1': evals['f1'],
        'precision': evals['precision'],
        'recall': evals['recall'],
        'tp': evals['tp'],
        'fp': evals['fp'],
        'fn': evals['fn']
    })
    print(f"{merge_gap:<8} {stats['unique_positions']:<8} {stats['clusters']:<10} "
          f"{stats['boundaries']:<8} {evals['precision']:<8.4f} {evals['recall']:<8.4f} "
          f"{evals['f1']:<8.4f} {evals['tp']}/{evals['fp']}/{evals['fn']:<20}")

# Find best
print("\n" + "=" * 100)
print("TOP 5 CONFIGURATIONS")
print("=" * 100)

best = sorted(results, key=lambda x: x['f1'], reverse=True)[:5]
for rank, config in enumerate(best, 1):
    print(f"\n{rank}. F1={config['f1']:.4f} (P={config['precision']:.4f}, R={config['recall']:.4f})")
    print(f"   Conf={config['confidence']:.2f}, Gap={config['gap_cluster']}, "
          f"Merge={config['merge_min_gap']}, Bounds={config['bounds']}")
    print(f"   TP={config['tp']}, FP={config['fp']}, FN={config['fn']}")

# Save results
with open('results/parameter_optimization.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: results/parameter_optimization.json")
