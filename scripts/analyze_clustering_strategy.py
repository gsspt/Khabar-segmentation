#!/usr/bin/env python3
"""
Analyze current clustering strategy and propose improvements.

Current: GAP_CLUSTER=50 (hardcoded)
Goal: Find optimal gap value and alternative strategies.
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(__file__).parent.parent

print("=" * 80)
print("CLUSTERING STRATEGY ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
with open(ROOT / "data/processed/kitab_uqala_boundaries.json", encoding='utf-8') as f:
    gold_data = json.load(f)
gold_boundaries = sorted(int(b["start"]) for b in gold_data)

with open(ROOT / "results/Kitab_Uqala_al_Majanin/camelbert_kitab_uqala_raw_inference.json", encoding='utf-8') as f:
    raw = json.load(f)['inference_results']

corpus_text = (ROOT / "data/processed/kitab_uqala_reference_corpus.txt").read_text(encoding='utf-8')

# ============================================================================
# EXTRACT BOUNDARY TOKENS (same as convert_boundary_tokens_direct.py)
# ============================================================================

SPECIAL = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

tokens = raw["tokens"]
offsets = raw["offsets"]
preds = raw["predictions"]
probs = raw["probabilities"]

char_to_prob = {}
char_to_pred = {}
char_to_tok = {}
char_to_end = {}

for tok, off, pred, prob in zip(tokens, offsets, preds, probs):
    if tok in SPECIAL or off == [0, 0]:
        continue
    cs, ce = off
    if cs not in char_to_prob or prob > char_to_prob[cs]:
        char_to_prob[cs] = prob
        char_to_pred[cs] = pred
        char_to_tok[cs] = tok
        char_to_end[cs] = ce

boundary_chars_all = sorted([cs for cs, pred in char_to_pred.items() if pred == 1])

print(f"Boundary tokens: {len(boundary_chars_all):,}")
print(f"Gold boundaries: {len(gold_boundaries)}")

# ============================================================================
# ANALYZE GAP DISTRIBUTION
# ============================================================================

print("\n" + "-" * 80)
print("CURRENT STRATEGY: GAP_CLUSTER = 50 CHARS")
print("-" * 80)

# Compute gaps between consecutive boundary tokens
gaps = []
for i in range(1, len(boundary_chars_all)):
    cs_curr = boundary_chars_all[i]
    ce_prev = char_to_end[boundary_chars_all[i-1]]
    gap = cs_curr - ce_prev
    gaps.append(gap)

print(f"\nGap distribution (between boundary token end and next token start):")
print(f"  Count: {len(gaps)}")
print(f"  Min: {min(gaps):,} chars")
print(f"  Q25: {sorted(gaps)[len(gaps)//4]:,} chars")
print(f"  Median: {statistics.median(gaps):,} chars")
print(f"  Q75: {sorted(gaps)[3*len(gaps)//4]:,} chars")
print(f"  Mean: {statistics.mean(gaps):,.0f} chars")
print(f"  Max: {max(gaps):,} chars")
print(f"  Stdev: {statistics.stdev(gaps):,.0f} chars")

# Histogram
print(f"\nGap histogram:")
gap_bins = [0, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, float('inf')]
for i in range(len(gap_bins)-1):
    count = sum(1 for g in gaps if gap_bins[i] <= g < gap_bins[i+1])
    pct = 100 * count / len(gaps)
    label = f"<{gap_bins[i+1]}" if gap_bins[i+1] < float('inf') else f">={gap_bins[i]}"
    print(f"  [{gap_bins[i]:4d}-{label:6s}]: {count:5d} ({pct:5.1f}%)")

# ============================================================================
# EVALUATE DIFFERENT GAP THRESHOLDS
# ============================================================================

def cluster_with_gap(boundary_chars, char_to_end, gap_threshold):
    """Cluster boundaries with given gap threshold."""
    clusters = []
    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_count = 1

    for cs in boundary_chars[1:]:
        gap = cs - cur_end
        if gap <= gap_threshold:
            cur_end = max(cur_end, char_to_end[cs])
            cur_count += 1
        else:
            clusters.append({'start': cur_start, 'end': cur_end, 'count': cur_count})
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_count = 1

    clusters.append({'start': cur_start, 'end': cur_end, 'count': cur_count})
    return clusters


def evaluate(predictions, gold, tol=80):
    """Evaluate predictions against gold."""
    matched_gold = set()
    matched_pred = set()

    for i, p in enumerate(predictions):
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            if abs(p - g) <= tol:
                matched_gold.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(gold) - len(matched_gold)

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1}


print("\n" + "-" * 80)
print("EVALUATING DIFFERENT GAP THRESHOLDS")
print("-" * 80)

gap_thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500]
results_by_gap = {}

print(f"\n{'Gap':<10} {'Clusters':<12} {'F1':<10} {'Prec':<10} {'Recall':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
print("-" * 90)

for gap in gap_thresholds:
    clusters = cluster_with_gap(boundary_chars_all, char_to_end, gap)
    boundaries = [c['start'] for c in clusters]
    eval_result = evaluate(boundaries, gold_boundaries, tol=80)
    results_by_gap[gap] = {
        'clusters': len(clusters),
        'boundaries': len(boundaries),
        'eval': eval_result
    }

    f1 = eval_result['f1']
    p = eval_result['p']
    r = eval_result['r']
    print(f"{gap:<10} {len(clusters):<12} {f1:<10.4f} {p:<10.4f} {r:<10.4f} {eval_result['tp']:<8} {eval_result['fp']:<8} {eval_result['fn']:<8}")

# Find best gap
best_gap = max(results_by_gap.keys(), key=lambda g: results_by_gap[g]['eval']['f1'])
best_f1 = results_by_gap[best_gap]['eval']['f1']
print(f"\nBest gap: {best_gap} chars -> F1={best_f1:.4f}")

# ============================================================================
# ANALYZE WHAT'S BEING MISSED/OVER-DETECTED
# ============================================================================

print("\n" + "-" * 80)
print("ERROR ANALYSIS (current gap=50)")
print("-" * 80)

current_clusters = cluster_with_gap(boundary_chars_all, char_to_end, 50)
current_boundaries = [c['start'] for c in current_clusters]
current_eval = evaluate(current_boundaries, gold_boundaries, tol=80)

print(f"\nCurrent performance (gap=50):")
print(f"  Detected: {len(current_boundaries)}")
print(f"  Gold: {len(gold_boundaries)}")
print(f"  TP: {current_eval['tp']}, FP: {current_eval['fp']}, FN: {current_eval['fn']}")
print(f"  F1: {current_eval['f1']:.4f}")

# Find unmatched gold boundaries
unmatched_gold = []
matched_set = set()

for i, p in enumerate(current_boundaries):
    for j, g in enumerate(gold_boundaries):
        if j in matched_set:
            continue
        if abs(p - g) <= 80:
            matched_set.add(j)
            break

for j, g in enumerate(gold_boundaries):
    if j not in matched_set:
        # Find closest predicted boundary
        closest = min(current_boundaries, key=lambda b: abs(b - g)) if current_boundaries else -1
        dist = abs(closest - g) if closest != -1 else 999999
        unmatched_gold.append({'gold': g, 'closest_pred': closest, 'dist': dist})

print(f"\nUnmatched gold boundaries (FN): {len(unmatched_gold)}")
print(f"  First 10:")
for item in unmatched_gold[:10]:
    print(f"    gold={item['gold']:6d}  closest={item['closest_pred']:6d}  dist={item['dist']:4d}")

# ============================================================================
# ANALYZE CLUSTERING QUALITY METRICS
# ============================================================================

print("\n" + "-" * 80)
print("CLUSTER QUALITY ANALYSIS (gap=50)")
print("-" * 80)

cluster_sizes = [c['count'] for c in current_clusters]
cluster_lengths = [c['end'] - c['start'] for c in current_clusters]

print(f"\nCluster size distribution:")
print(f"  Mean tokens per cluster: {statistics.mean(cluster_sizes):.1f}")
print(f"  Median: {statistics.median(cluster_sizes):.0f}")
print(f"  Max: {max(cluster_sizes)}")

print(f"\nCluster length distribution:")
print(f"  Mean length: {statistics.mean(cluster_lengths):,.0f} chars")
print(f"  Median: {statistics.median(cluster_lengths):,.0f} chars")
print(f"  Max: {max(cluster_lengths):,} chars")

# Analyze inter-cluster gaps
inter_cluster_gaps = []
for i in range(1, len(current_clusters)):
    gap = current_clusters[i]['start'] - current_clusters[i-1]['end']
    inter_cluster_gaps.append(gap)

print(f"\nInter-cluster gaps (khabar content between isnads):")
print(f"  Mean: {statistics.mean(inter_cluster_gaps):,.0f} chars")
print(f"  Median: {statistics.median(inter_cluster_gaps):,.0f} chars")
print(f"  Min: {min(inter_cluster_gaps):,} chars")
print(f"  Max: {max(inter_cluster_gaps):,} chars")

# ============================================================================
# SUGGESTIONS FOR IMPROVEMENT
# ============================================================================

print("\n" + "=" * 80)
print("IMPROVEMENT SUGGESTIONS")
print("=" * 80)

print(f"""
1. ADAPTIVE GAP THRESHOLD (Data-driven)
   Current: GAP_CLUSTER = 50 (hardcoded)
   Observation: Gap distribution has median ~250 chars

   Suggested approach:
   - Use multiple thresholds: 20, 30, 50, 100
   - Or compute adaptive gap based on gap distribution
   - Gap should separate "isnad tokens" from "prose tokens"

2. CONFIDENCE-WEIGHTED CLUSTERING
   Current: All boundary tokens treated equally
   Issue: Some tokens have low confidence (0.5001)

   Suggested approach:
   - Filter tokens by confidence > 0.70
   - Weight clustering decision by average confidence in cluster
   - Discard low-confidence clusters

3. LINGUISTIC FEATURES
   Current: Pure distance-based clustering
   Issue: Doesn't use Arabic linguistic knowledge

   Suggested approach:
   - Detect isnad verbs: hadathna, akhbarna, qala
   - Boost confidence for clusters with isnad verbs
   - Penalize clusters that are pure verbs/names (likely not khaba start)

4. HIERARCHICAL/SPECTRAL CLUSTERING
   Current: Simple greedy clustering (gap-based)
   Issue: Doesn't capture complex patterns

   Suggested approach:
   - Use sklearn.cluster.AgglomerativeClustering
   - Or compute distance matrix between boundary tokens
   - Use linkage strategies: complete, average, ward

5. MULTI-FEATURE CLUSTERING
   Combine:
   - Distance (current)
   - Confidence score
   - Token type (isnad verb, name, pronoun, etc.)
   - Context (preceding/following tokens)

6. POST-PROCESSING MERGE/SPLIT
   Current: Single clustering pass
   Issue: May miss small clusters or over-segment

   Suggested approach:
   - Merge very close clusters (< 10 chars apart)
   - Split large clusters (> 1000 chars) if confidence drops
   - Validate boundaries against text structure (paragraphs)
""")

print("\nRECOMMENDATION:")
print(f"Best immediate improvement: Test gap={best_gap} instead of 50")
print(f"  Expected improvement: F1 = {best_f1:.4f} (current: {current_eval['f1']:.4f})")
print(f"  Difference: {(best_f1 - current_eval['f1'])*100:+.2f}%")

print("\n[OK] Analysis complete!")
