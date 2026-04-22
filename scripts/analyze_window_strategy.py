#!/usr/bin/env python3
"""
Analyze current windowing strategy and propose optimal approach.

Compare:
1. Current: overlap + deduplication by max probability
2. Smart Windows: overlap + keep only confident center
3. No Overlap: no overlap, accept edge effects

This script analyzes the current raw_inference to understand what windowing was used.
"""

import json
from collections import defaultdict
from pathlib import Path

# Load current raw inference
raw_path = Path('results/Kitab_Uqala_al_Majanin/camelbert_kitab_uqala_raw_inference.json')
print(f"Loading: {raw_path}\n")

with open(raw_path, 'r', encoding='utf-8') as f:
    raw = json.load(f)

inf = raw['inference_results']
tokens = inf['tokens']
offsets = inf['offsets']
preds = inf['predictions']
probs = inf['probabilities']

print("=" * 100)
print("CURRENT WINDOWING ANALYSIS")
print("=" * 100)

# Count tokens with offsets
valid_offsets = [off for off in offsets if off != [0, 0]]
print(f"\nTotal tokens: {len(tokens):,}")
print(f"Valid offsets: {len(valid_offsets):,}")
print(f"Special/padding tokens: {len(tokens) - len(valid_offsets):,}")

# Analyze char_start distribution
char_starts = [off[0] for off in valid_offsets]
char_start_freq = defaultdict(list)

for i, off in enumerate(offsets):
    if off != [0, 0]:
        cs = off[0]
        char_start_freq[cs].append({
            'idx': i,
            'token': tokens[i],
            'prob': probs[i],
            'pred': preds[i]
        })

print(f"\nUnique char_start positions: {len(char_start_freq):,}")

# Find char_starts with multiple tokens
multiple_char_starts = {cs: v for cs, v in char_start_freq.items() if len(v) > 1}
print(f"Char_starts appearing multiple times: {len(multiple_char_starts):,}")

print(f"\nDuplicate char_start statistics:")
dup_counts = [len(v) for v in multiple_char_starts.values()]
print(f"  Min duplicates: {min(dup_counts)}")
print(f"  Max duplicates: {max(dup_counts)}")
print(f"  Mean duplicates: {sum(dup_counts)/len(dup_counts):.2f}")
print(f"  Total duplicate tokens: {sum(dup_counts):,}")

# Example of duplication
print(f"\nExample of duplicated char_start (showing first 5):")
for cs, occurrences in list(multiple_char_starts.items())[:5]:
    print(f"\n  char_start={cs}:")
    for occ in occurrences:
        print(f"    idx={occ['idx']:6d} pred={occ['pred']} prob={occ['prob']:.4f}")

# Analyze conflicting predictions
conflicts = {}
for cs, occurrences in multiple_char_starts.items():
    preds_set = set(occ['pred'] for occ in occurrences)
    if len(preds_set) > 1:
        conflicts[cs] = occurrences

print(f"\n" + "=" * 100)
print("PREDICTION CONFLICTS")
print("=" * 100)

print(f"\nChar_starts with conflicting predictions: {len(conflicts):,}")

# Analyze conflict types
conflict_types = defaultdict(int)
for cs, occurrences in conflicts.items():
    preds_list = [occ['pred'] for occ in occurrences]
    conflict_key = tuple(sorted(set(preds_list)))
    conflict_types[conflict_key] += 1

print(f"\nConflict types:")
for conflict_key, count in sorted(conflict_types.items()):
    print(f"  {conflict_key}: {count} positions")

# Show some conflict examples
print(f"\nExamples of conflicts (pred mismatch):")
conflict_examples = []
for cs, occurrences in conflicts.items():
    preds_list = [occ['pred'] for occ in occurrences]
    probs_list = [occ['prob'] for occ in occurrences]
    if len(set(preds_list)) == 2:  # 0 vs 1 conflict
        conflict_examples.append({
            'char_start': cs,
            'occurrences': occurrences,
            'prob_range': (min(probs_list), max(probs_list))
        })

for example in conflict_examples[:3]:
    cs = example['char_start']
    occs = example['occurrences']
    prob_min, prob_max = example['prob_range']

    print(f"\n  char_start={cs}:")
    preds_str = [str(occ['pred']) for occ in occs]
    probs_str = [f"{occ['prob']:.4f}" for occ in occs]
    print(f"    Predictions: {preds_str}")
    print(f"    Probabilities: {probs_str}")
    print(f"    Probability range: {prob_min:.4f} - {prob_max:.4f}")
    print(f"    Decision impact: 0 to 1 or 1 to 0 depending on max prob strategy")

# Estimate windowing strategy
print(f"\n" + "=" * 100)
print("INFERRED WINDOWING STRATEGY")
print("=" * 100)

# If 50% overlap: stride=256
# Expected: each token appears in ~2 windows (middle of window)
# Some tokens in overlap appear in 2 windows

# Calculate: if we had 50% overlap (256 stride)
# Then for 297,984 tokens: 297,984 / 256 ≈ 1164 windows
# Some tokens appear 1×, some 2×

tokens_appearing_once = len([v for v in char_start_freq.values() if len(v) == 1])
tokens_appearing_twice = len([v for v in char_start_freq.values() if len(v) == 2])
tokens_appearing_thrice = len([v for v in char_start_freq.values() if len(v) == 3])

print(f"\nToken appearance frequency:")
print(f"  Appearing 1× (center of window): {tokens_appearing_once:,}")
print(f"  Appearing 2× (in overlap zone): {tokens_appearing_twice:,}")
print(f"  Appearing 3+× (edge cases): {tokens_appearing_thrice:,}")

ratio_2x = tokens_appearing_twice / (tokens_appearing_once + tokens_appearing_twice)
print(f"\nRatio of 2× appearances: {ratio_2x:.1%}")

if ratio_2x > 0.20:
    print(f"\n✓ CONCLUSION: 50% Overlap WAS USED")
    print(f"  Expected: ~33% tokens in overlap (256 stride)")
    print(f"  Observed: {ratio_2x:.1%}")
    print(f"  This matches 50% overlap strategy")
else:
    print(f"\n✗ CONCLUSION: NO Overlap (or different strategy)")

# Analyze how deduplication affects boundaries
print(f"\n" + "=" * 100)
print("DEDUPLICATION IMPACT")
print("=" * 100)

# Current approach: max probability
current_decisions = {}
for cs, occurrences in multiple_char_starts.items():
    best = max(occurrences, key=lambda x: x['prob'])
    current_decisions[cs] = {
        'pred': best['pred'],
        'prob': best['prob'],
        'n_candidates': len(occurrences),
        'prob_range': (min(occ['prob'] for occ in occurrences),
                      max(occ['prob'] for occ in occurrences))
    }

# Analyze decision confidence
high_confidence = sum(1 for v in current_decisions.values() if v['prob'] > 0.9)
medium_confidence = sum(1 for v in current_decisions.values() if 0.7 < v['prob'] <= 0.9)
low_confidence = sum(1 for v in current_decisions.values() if v['prob'] <= 0.7)

print(f"\nDeduplication decisions (max probability):")
print(f"  High confidence (prob > 0.90): {high_confidence:,}")
print(f"  Medium confidence (0.70-0.90): {medium_confidence:,}")
print(f"  Low confidence (prob ≤ 0.70): {low_confidence:,}")

# Show decisions for conflicts (pred mismatch)
conflict_decisions = {
    cs: current_decisions[cs]
    for cs in conflicts.keys()
    if cs in current_decisions
}

conflict_high = sum(1 for v in conflict_decisions.values() if v['prob'] > 0.9)
conflict_medium = sum(1 for v in conflict_decisions.values() if 0.7 < v['prob'] <= 0.9)
conflict_low = sum(1 for v in conflict_decisions.values() if v['prob'] <= 0.7)

print(f"\nOf the {len(conflict_decisions)} conflicting char_starts:")
print(f"  High confidence winner: {conflict_high}")
print(f"  Medium confidence winner: {conflict_medium}")
print(f"  Low confidence winner: {conflict_low}")

# Conclusion about duplication impact
print(f"\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)

print(f"""
Based on analysis:

1. CURRENT STRATEGY (overlap + max prob):
   ✓ Works: F1=0.8579 is excellent
   ✓ Handles conflicts: Takes highest probability
   ✗ Loses information: Can't tell if decision was unanimous
   ✗ Complexity: Deduplication adds post-processing step

2. SMART WINDOWS STRATEGY:
   ✓ Keep only center of each window (margin=64)
   ✓ Eliminates deduplication (each token seen once)
   ✓ Keeps confident center context (512 token window)
   ✗ Need to handle final chunk separately
   ✗ Slightly more complex inference code

3. NO OVERLAP STRATEGY:
   ✓ Simplest: no conflicts to resolve
   ✓ 50% faster: no overlapping windows
   ✗ Edge effects: tokens at window boundaries have limited context
   ✗ Quality risk: boundary-spanning isnads might split prediction

NEXT STEP: Test smart windows on Kitab Uqala
- If smart windows achieves F1 ≥ 0.86: recommend for future
- If current is better: stick with current (it's good enough)
""")
