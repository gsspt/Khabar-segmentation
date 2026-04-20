#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test different verb combinations to find optimal set
"""

import json
import sys
from baseline_v3_refined import normalize_arabic_text, find_all_isnad_starts, find_isnad_end, ISNAD_MIN_LENGTH, ISNAD_MAX_LENGTH, ISNAD_ABSOLUTE_MAX

# Load corpus
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

# Define verb combinations to test
VERBS_BASE = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'حدثت', 'حدثوا',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'أخبرت', 'أخبروا',
    'سمعت', 'سمعنا', 'سمعه', 'سمعها', 'سمعهم',
    'روى', 'رويت', 'روينا', 'رواه', 'روا',
    'أنبأ', 'أنبأني', 'أنبأنا', 'أنبأه', 'أنبأتنا',
    'أنشدني', 'أنشدنا',
}

VERBS_PREFIXED = {
    'وأخبرنا', 'وأخبرني', 'وأخبره', 'وأخبرها', 'وأخبرهم',
    'وحدثنا', 'وحدثني', 'وحدثه', 'وحدثها', 'وحدثهم',
    'وسمعت', 'وسمعنا', 'وسمعه', 'وسمعها', 'وسمعهم',
    'وروى', 'ورويت', 'وروينا', 'ورواه',
    'وأنبأ', 'وأنبأني', 'وأنبأنا', 'وأنبأه',
    'وأنشدني', 'وأنشدنا',
}

VERBS_MISSING_HIGH_CONF = {
    'وقال',       # 20 misses
    'وبهذا',      # 9 misses
    'ومنهم',      # 8 misses
    'ومنها',      # 5 misses
    'ولبعضهم',   # 2 misses
    'وحكى',       # 1 miss
    'وبلغني',     # 1 miss
}

VERBS_MISSING_RISKY = {
    'قال',        # 17 misses - but very generic
    'ومن',        # 3 misses - but very generic
    'وله',        # 3 misses - but generic
}

# Combinations to test
combinations = [
    ('Base + Prefixed (Current V3)', VERBS_BASE | VERBS_PREFIXED | VERBS_MISSING_HIGH_CONF),
    ('Base + Prefixed + قال', VERBS_BASE | VERBS_PREFIXED | VERBS_MISSING_HIGH_CONF | {'قال'}),
    ('Base + Prefixed + ومن', VERBS_BASE | VERBS_PREFIXED | VERBS_MISSING_HIGH_CONF | {'ومن'}),
    ('Base + Prefixed + وله', VERBS_BASE | VERBS_PREFIXED | VERBS_MISSING_HIGH_CONF | {'وله'}),
    ('Base + Prefixed + all missing', VERBS_BASE | VERBS_PREFIXED | VERBS_MISSING_HIGH_CONF | VERBS_MISSING_RISKY),
]

print("=" * 80)
print("TESTING VERB COMBINATIONS")
print("=" * 80 + "\n")

results = []

for name, verb_set in combinations:
    # Temporarily replace ISNAD_START_VERBS
    import baseline_v3_refined as baseline
    original_verbs = baseline.ISNAD_START_VERBS
    baseline.ISNAD_START_VERBS = verb_set

    # Segment
    detected = baseline.segment_akhbars_v3(corpus)
    count = len(detected)
    pct = 100 * count / len(boundaries)

    results.append((name, count, pct, len(verb_set)))

    # Restore
    baseline.ISNAD_START_VERBS = original_verbs

    print(f"Config {len(results)}:")
    print(f"  Verbs: {len(verb_set)}")
    print(f"  Detected: {count}/613 ({pct:.1f}%)")
    print(f"  vs Reference: {count - len(boundaries):+d}\n")

# Summary table
print("=" * 80)
print("SUMMARY")
print("=" * 80 + "\n")

print("Config                                    Verbs  Detected    %")
print("-" * 65)

for i, (name, count, pct, verb_count) in enumerate(results, 1):
    print(f"Config {i:<33} {verb_count:3d}    {count:3d}   {pct:5.1f}%")

print("\nRecommendation:")
best = max(results, key=lambda x: x[1])
best_idx = next(i for i, r in enumerate(results) if r == best)
print(f"  Config {best_idx + 1}: {best[1]} detected ({best[2]:.1f}%)")

# Save detailed results
from pathlib import Path

report_path = Path("../results/verb_combination_test.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Verb Combination Testing Results\n\n")
    f.write("Testing different verb sets to optimize detection.\n\n")

    f.write("| Config | Verb Count | Detected | % | Improvement |\n")
    f.write("|--------|------------|----------|---|-------------|\n")

    baseline_count = 396  # v2
    for name, count, pct, verb_count in results:
        improvement = count - baseline_count
        f.write(f"| {name} | {verb_count} | {count}/613 | {pct:.1f}% | {improvement:+d} |\n")

    f.write("\n## Findings\n\n")
    f.write(f"- Baseline V2: 396 detected\n")
    f.write(f"- Best: {best[0]} = {best[1]} detected\n")
    f.write(f"- Improvement: {best[1] - 396:+d} akhbars\n")

print(f"\nReport saved to: {report_path}\n")
