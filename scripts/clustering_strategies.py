#!/usr/bin/env python3
"""
Multiple clustering strategies for CAMeL-BERT boundary detection.

Strategies:
1. Baseline (gap=50) - current approach
2. Optimal gap (gap=20) - data-driven
3. Confidence-weighted - filters low-confidence tokens
4. Linguistic features - boosts isnad verbs
5. Hierarchical - sklearn-based clustering
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from collections import defaultdict

ROOT = Path(__file__).parent.parent

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_boundary_data():
    """Load boundary tokens and metadata."""
    with open(ROOT / "results/Kitab_Uqala_al_Majanin/camelbert_kitab_uqala_raw_inference.json", encoding='utf-8') as f:
        raw = json.load(f)['inference_results']

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

    boundary_chars = sorted([cs for cs, pred in char_to_pred.items() if pred == 1])
    return boundary_chars, char_to_prob, char_to_pred, char_to_tok, char_to_end


def load_gold_standard():
    """Load gold boundaries."""
    with open(ROOT / "data/processed/kitab_uqala_boundaries.json", encoding='utf-8') as f:
        gold_data = json.load(f)
    return sorted(int(b["start"]) for b in gold_data)


def evaluate(predictions: List[int], gold: List[int], tol: int = 80) -> Dict:
    """Evaluate predictions against gold standard."""
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

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1}


# ============================================================================
# STRATEGY 1: BASELINE (GAP=50)
# ============================================================================

def strategy_baseline(boundary_chars, char_to_end, char_to_prob):
    """Original approach: gap=50."""
    GAP_CLUSTER = 50
    clusters = []

    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_tokens = [boundary_chars[0]]
    cur_probs = [char_to_prob[boundary_chars[0]]]

    for cs in boundary_chars[1:]:
        gap = cs - cur_end
        if gap <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(char_to_prob[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'strategy': 'baseline_gap50'
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [char_to_prob[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'strategy': 'baseline_gap50'
    })

    return clusters


# ============================================================================
# STRATEGY 2: OPTIMAL GAP (GAP=20)
# ============================================================================

def strategy_optimal_gap(boundary_chars, char_to_end, char_to_prob):
    """Data-driven approach: gap=20."""
    GAP_CLUSTER = 20
    clusters = []

    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_tokens = [boundary_chars[0]]
    cur_probs = [char_to_prob[boundary_chars[0]]]

    for cs in boundary_chars[1:]:
        gap = cs - cur_end
        if gap <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(char_to_prob[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'strategy': 'optimal_gap20'
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [char_to_prob[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'strategy': 'optimal_gap20'
    })

    return clusters


# ============================================================================
# STRATEGY 3: CONFIDENCE-WEIGHTED
# ============================================================================

def strategy_confidence_weighted(boundary_chars, char_to_end, char_to_prob):
    """Filter by confidence + weighted clustering."""
    CONFIDENCE_THRESHOLD = 0.70
    GAP_CLUSTER = 50

    # Filter by confidence
    filtered_chars = [cs for cs in boundary_chars if char_to_prob[cs] >= CONFIDENCE_THRESHOLD]

    clusters = []
    if not filtered_chars:
        return clusters

    cur_start = filtered_chars[0]
    cur_end = char_to_end[filtered_chars[0]]
    cur_tokens = [filtered_chars[0]]
    cur_probs = [char_to_prob[filtered_chars[0]]]

    for cs in filtered_chars[1:]:
        gap = cs - cur_end
        if gap <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(char_to_prob[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'strategy': 'confidence_weighted'
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [char_to_prob[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'strategy': 'confidence_weighted'
    })

    return clusters


# ============================================================================
# STRATEGY 4: LINGUISTIC FEATURES
# ============================================================================

def strategy_linguistic(boundary_chars, char_to_end, char_to_prob, char_to_tok):
    """Boost isnad verbs: hadathna, akhbarna, qala."""
    ISNAD_VERBS = {'حدثنا', 'أخبرنا', 'قال', 'أخبرني', 'قالا', 'قالوا', 'أخبرنيه'}
    CONFIDENCE_BOOST = 0.15
    GAP_CLUSTER = 50

    # Boost confidence for isnad verbs
    char_to_prob_boosted = {}
    for cs in boundary_chars:
        tok = char_to_tok[cs]
        prob = char_to_prob[cs]

        # Check if token is isnad verb (or contains it)
        is_isnad_verb = any(verb in tok for verb in ISNAD_VERBS)
        if is_isnad_verb:
            prob = min(1.0, prob + CONFIDENCE_BOOST)

        char_to_prob_boosted[cs] = prob

    # Cluster with boosted probabilities
    clusters = []
    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_tokens = [boundary_chars[0]]
    cur_probs = [char_to_prob_boosted[boundary_chars[0]]]

    for cs in boundary_chars[1:]:
        gap = cs - cur_end
        if gap <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(char_to_prob_boosted[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'strategy': 'linguistic'
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [char_to_prob_boosted[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'strategy': 'linguistic'
    })

    return clusters


# ============================================================================
# STRATEGY 5: HIERARCHICAL CLUSTERING
# ============================================================================

def strategy_hierarchical(boundary_chars, char_to_end, char_to_prob):
    """Hierarchical clustering based on distance and confidence."""
    # Simple hierarchical: start with gap-based, then merge/split
    GAP_CLUSTER = 30  # Start with moderate gap
    MIN_MERGE_GAP = 10  # Merge clusters < 10 chars apart

    # Initial clustering
    clusters = []
    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_tokens = [boundary_chars[0]]
    cur_probs = [char_to_prob[boundary_chars[0]]]

    for cs in boundary_chars[1:]:
        gap = cs - cur_end
        if gap <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(char_to_prob[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'tokens': cur_tokens
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [char_to_prob[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'tokens': cur_tokens
    })

    # Merge very close clusters
    merged = []
    i = 0
    while i < len(clusters):
        cur = clusters[i].copy()
        j = i + 1
        while j < len(clusters):
            gap_to_next = clusters[j]['start'] - cur['end']
            if gap_to_next <= MIN_MERGE_GAP:
                cur['end'] = max(cur['end'], clusters[j]['end'])
                cur['count'] += clusters[j]['count']
                cur['avg_prob'] = (cur['avg_prob'] + clusters[j]['avg_prob']) / 2
                j += 1
            else:
                break

        cur['strategy'] = 'hierarchical'
        merged.append(cur)
        i = j

    return merged


# ============================================================================
# STRATEGY 6: ENSEMBLE (MULTI-CRITERIA)
# ============================================================================

def strategy_ensemble(boundary_chars, char_to_end, char_to_prob, char_to_tok):
    """Multi-criteria approach: distance + confidence + linguistic."""
    ISNAD_VERBS = {'حدثنا', 'أخبرنا', 'قال', 'أخبرني', 'قالا', 'قالوا', 'أخبرنيه'}

    # Compute scoring for each boundary token
    scores = {}
    for cs in boundary_chars:
        prob_score = char_to_prob[cs]
        tok = char_to_tok[cs]
        is_isnad = any(verb in tok for verb in ISNAD_VERBS)
        linguistic_score = 1.15 if is_isnad else 1.0
        scores[cs] = prob_score * linguistic_score

    # Adaptive gap based on score
    clusters = []
    if not boundary_chars:
        return clusters

    cur_start = boundary_chars[0]
    cur_end = char_to_end[boundary_chars[0]]
    cur_tokens = [boundary_chars[0]]
    cur_probs = [scores[boundary_chars[0]]]

    for i, cs in enumerate(boundary_chars[1:], 1):
        gap = cs - cur_end
        # Adaptive threshold: lower for high-confidence tokens
        avg_score = statistics.mean(cur_probs)
        adaptive_gap = 40 if avg_score > 0.8 else 60

        if gap <= adaptive_gap:
            cur_end = max(cur_end, char_to_end[cs])
            cur_tokens.append(cs)
            cur_probs.append(scores[cs])
        else:
            avg_prob = statistics.mean(cur_probs)
            clusters.append({
                'start': cur_start,
                'end': cur_end,
                'count': len(cur_tokens),
                'avg_prob': avg_prob,
                'strategy': 'ensemble'
            })
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_tokens = [cs]
            cur_probs = [scores[cs]]

    avg_prob = statistics.mean(cur_probs)
    clusters.append({
        'start': cur_start,
        'end': cur_end,
        'count': len(cur_tokens),
        'avg_prob': avg_prob,
        'strategy': 'ensemble'
    })

    return clusters


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all strategies and save results."""
    print("=" * 80)
    print("CLUSTERING STRATEGIES - IMPLEMENTATION")
    print("=" * 80)

    # Load data
    boundary_chars, char_to_prob, char_to_pred, char_to_tok, char_to_end = load_boundary_data()
    gold = load_gold_standard()

    print(f"\nData loaded:")
    print(f"  Boundary tokens: {len(boundary_chars):,}")
    print(f"  Gold boundaries: {len(gold)}")

    # Run all strategies
    strategies = {
        'baseline_gap50': lambda: strategy_baseline(boundary_chars, char_to_end, char_to_prob),
        'optimal_gap20': lambda: strategy_optimal_gap(boundary_chars, char_to_end, char_to_prob),
        'confidence_weighted': lambda: strategy_confidence_weighted(boundary_chars, char_to_end, char_to_prob),
        'linguistic': lambda: strategy_linguistic(boundary_chars, char_to_end, char_to_prob, char_to_tok),
        'hierarchical': lambda: strategy_hierarchical(boundary_chars, char_to_end, char_to_prob),
        'ensemble': lambda: strategy_ensemble(boundary_chars, char_to_end, char_to_prob, char_to_tok),
    }

    results = {}

    print("\n" + "-" * 80)
    print("RUNNING STRATEGIES")
    print("-" * 80)

    for strategy_name, strategy_func in strategies.items():
        clusters = strategy_func()
        boundaries = [c['start'] for c in clusters]
        eval_result = evaluate(boundaries, gold, tol=80)

        results[strategy_name] = {
            'clusters': len(clusters),
            'boundaries': len(boundaries),
            'eval': eval_result,
            'clusters_data': clusters
        }

        print(f"\n{strategy_name}:")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Boundaries: {len(boundaries)}")
        print(f"  F1: {eval_result['f1']:.4f}  P: {eval_result['p']:.4f}  R: {eval_result['r']:.4f}")
        print(f"  TP: {eval_result['tp']}  FP: {eval_result['fp']}  FN: {eval_result['fn']}")

    # Save results
    output = {
        'gold_boundaries': len(gold),
        'boundary_tokens': len(boundary_chars),
        'strategies': {}
    }

    for strategy_name, data in results.items():
        output['strategies'][strategy_name] = {
            'clusters': data['clusters'],
            'boundaries': data['boundaries'],
            'evaluation': {
                'f1': float(data['eval']['f1']),
                'precision': float(data['eval']['p']),
                'recall': float(data['eval']['r']),
                'tp': int(data['eval']['tp']),
                'fp': int(data['eval']['fp']),
                'fn': int(data['eval']['fn']),
            }
        }

    output_path = ROOT / "results/clustering_strategies_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Results saved to {output_path}")

    # Ranking
    print("\n" + "=" * 80)
    print("RANKING BY F1 SCORE")
    print("=" * 80)

    ranked = sorted(results.items(), key=lambda x: x[1]['eval']['f1'], reverse=True)
    for i, (name, data) in enumerate(ranked, 1):
        print(f"{i}. {name:<25} F1={data['eval']['f1']:.4f}  TP={data['eval']['tp']}  FP={data['eval']['fp']}  FN={data['eval']['fn']}")

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
