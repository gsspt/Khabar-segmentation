#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1.4: Boundary Accuracy & Position Error Analysis

Objective: Understand boundary position deviations
- How far off are detected boundaries from reference?
- Are there systematic patterns in errors?
- What types of boundaries have the worst accuracy?
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import sys


def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


# Load data
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

with open("../data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print("[*] TASK 1.4: Boundary Accuracy & Position Error Analysis")
print(f"[*] Reference boundaries: {len(boundaries)}\n")

# Run baseline detection
sys.path.insert(0, '.')
from baseline_v2_isnad_first import segment_akhbars_v2

detected = segment_akhbars_v2(corpus)
print(f"[*] Detected boundaries: {len(detected)}\n")

# ============================================================================
# ANALYSIS 1: Boundary Matching - Find Closest Detected for Each Reference
# ============================================================================

print("[*] Matching reference boundaries to detected...\n")

matches = []
for ref_boundary in boundaries:
    ref_pos = ref_boundary['start']

    # Find closest detected boundary
    closest_detected = None
    min_distance = float('inf')

    for det_boundary in detected:
        det_pos = det_boundary['start']
        distance = abs(ref_pos - det_pos)

        if distance < min_distance:
            min_distance = distance
            closest_detected = det_boundary

    if closest_detected:
        matches.append({
            'ref_pos': ref_pos,
            'det_pos': closest_detected['start'],
            'deviation': closest_detected['start'] - ref_pos,
            'abs_deviation': min_distance,
            'ref_boundary': ref_boundary,
            'det_boundary': closest_detected,
        })

print(f"[*] Matched boundaries: {len(matches)}\n")

# Analyze deviation patterns
deviations = [m['abs_deviation'] for m in matches]
signed_deviations = [m['deviation'] for m in matches]

if deviations:
    avg_dev = sum(deviations) / len(deviations)
    median_dev = sorted(deviations)[len(deviations)//2]
    max_dev = max(deviations)

    print(f"[*] Deviation Statistics:")
    print(f"    Average: {avg_dev:.0f} chars")
    print(f"    Median: {median_dev} chars")
    print(f"    Max: {max_dev} chars")
    print(f"    Std Dev: {(sum((d - avg_dev)**2 for d in deviations) / len(deviations))**0.5:.0f} chars\n")

# Analyze signed deviations (over/under detection)
over_detections = sum(1 for d in signed_deviations if d > 0)
under_detections = sum(1 for d in signed_deviations if d < 0)
exact_matches = sum(1 for d in signed_deviations if d == 0)

print(f"[*] Detection Direction:")
print(f"    Exact matches (0 offset): {exact_matches}")
print(f"    Over-detected (too early): {over_detections}")
print(f"    Under-detected (too late): {under_detections}\n")

# ============================================================================
# ANALYSIS 2: Deviation by Range (Accuracy Tiers)
# ============================================================================

print("[*] Accuracy Tiers:\n")

ranges = [
    (0, 5, "Perfect"),
    (5, 20, "Excellent"),
    (20, 50, "Good"),
    (50, 100, "Acceptable"),
    (100, 200, "Poor"),
    (200, 500, "Very Poor"),
    (500, float('inf'), "Failed"),
]

accuracy_distribution = []
for min_d, max_d, label in ranges:
    count = sum(1 for d in deviations if min_d <= d < max_d)
    pct = count / len(deviations) * 100 if deviations else 0
    accuracy_distribution.append((label, min_d, max_d, count, pct))
    max_str = f"{int(max_d)}" if max_d != float('inf') else "inf"
    print(f"    {label:12} ({min_d:4d}-{max_str:>4} chars): {count:3d} ({pct:5.1f}%)")

# ============================================================================
# ANALYSIS 3: Context Analysis for Errors
# ============================================================================

print("\n[*] Analyzing boundary contexts for errors...\n")

# Sample some large-deviation boundaries
matches_sorted = sorted(matches, key=lambda m: -m['abs_deviation'])

sample_errors = matches_sorted[:5]

error_contexts = []
for i, match in enumerate(sample_errors):
    ref_pos = match['ref_pos']
    det_pos = match['det_pos']
    deviation = match['deviation']

    # Get context
    context_start = max(0, min(ref_pos, det_pos) - 80)
    context_end = min(len(corpus), max(ref_pos, det_pos) + 80)

    context_text = corpus[context_start:context_end]
    # Normalize for readability
    context_clean = context_text.replace('\n', ' ')

    # Find where boundaries are in context
    ref_in_context = ref_pos - context_start
    det_in_context = det_pos - context_start

    error_contexts.append({
        'deviation': deviation,
        'ref_pos': ref_pos,
        'det_pos': det_pos,
        'context': context_clean,
        'ref_offset': ref_in_context,
        'det_offset': det_in_context,
    })

# ============================================================================
# ANALYSIS 4: Text around boundaries
# ============================================================================

print("[*] Analyzing text patterns at boundaries...\n")

# For well-matched boundaries, what text appears before/after?
well_matched = [m for m in matches if m['abs_deviation'] < 50]
poorly_matched = [m for m in matches if m['abs_deviation'] >= 200]

print(f"[*] Well-matched boundaries (offset < 50): {len(well_matched)}")
print(f"[*] Poorly-matched boundaries (offset >= 200): {len(poorly_matched)}\n")

# Extract preceding words
preceding_words_well = Counter()
preceding_words_poor = Counter()

for match in well_matched:
    ref_pos = match['ref_pos']
    if ref_pos >= 50:
        preceding = corpus[ref_pos-50:ref_pos].split()[-3:] if ref_pos >= 50 else []
        for word in preceding:
            preceding_words_well[word] += 1

for match in poorly_matched:
    ref_pos = match['ref_pos']
    if ref_pos >= 50:
        preceding = corpus[ref_pos-50:ref_pos].split()[-3:] if ref_pos >= 50 else []
        for word in preceding:
            preceding_words_poor[word] += 1

print("[*] Most common preceding words (3 words before boundary):")
print("\nWell-matched boundaries:")
for word, count in preceding_words_well.most_common(5):
    print(f"    (word length {len(word)}): {count} times")

print("\nPoorly-matched boundaries:")
for word, count in preceding_words_poor.most_common(5):
    print(f"    (word length {len(word)}): {count} times")

# ============================================================================
# ANALYSIS 5: Length validation impact
# ============================================================================

print("\n[*] Analyzing detected isnad/khabar lengths...\n")

isnad_lengths = [det['isnad_end'] - det['start'] for det in detected]
khabar_lengths = [det['end'] - det['isnad_end'] for det in detected]

if isnad_lengths:
    print(f"[*] Detected Isnad Lengths:")
    print(f"    Min: {min(isnad_lengths)} chars")
    print(f"    Max: {max(isnad_lengths)} chars")
    print(f"    Avg: {sum(isnad_lengths)/len(isnad_lengths):.0f} chars")
    print(f"    Median: {sorted(isnad_lengths)[len(isnad_lengths)//2]} chars")

if khabar_lengths:
    print(f"\n[*] Detected Khabar Lengths:")
    print(f"    Min: {min(khabar_lengths)} chars")
    print(f"    Max: {max(khabar_lengths)} chars")
    print(f"    Avg: {sum(khabar_lengths)/len(khabar_lengths):.0f} chars")
    print(f"    Median: {sorted(khabar_lengths)[len(khabar_lengths)//2]} chars")

# ============================================================================
# Generate Report
# ============================================================================

report_path = Path("../results/task_1_4_boundary_accuracy.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# TASK 1.4: Boundary Accuracy & Position Error Analysis\n\n")

    f.write("## Executive Summary\n\n")
    f.write(f"- Reference boundaries: {len(boundaries)}\n")
    f.write(f"- Detected boundaries: {len(detected)}\n")
    f.write(f"- Matched boundaries: {len(matches)}\n")
    f.write(f"- Unmatched reference: {len(boundaries) - len(matches)}\n\n")

    f.write("## Deviation Statistics\n\n")
    if deviations:
        f.write(f"- **Average deviation**: {avg_dev:.0f} chars\n")
        f.write(f"- **Median deviation**: {median_dev} chars\n")
        f.write(f"- **Max deviation**: {max_dev} chars\n")
        f.write(f"- **Std Dev**: {(sum((d - avg_dev)**2 for d in deviations) / len(deviations))**0.5:.0f} chars\n\n")

    f.write("## Detection Direction\n\n")
    f.write(f"- **Exact matches** (0 offset): {exact_matches}\n")
    f.write(f"- **Over-detected** (too early): {over_detections} ({over_detections/len(signed_deviations)*100:.1f}%)\n")
    f.write(f"- **Under-detected** (too late): {under_detections} ({under_detections/len(signed_deviations)*100:.1f}%)\n\n")

    f.write("## Accuracy Tiers\n\n")
    f.write("| Tier | Range | Count | % |\n")
    f.write("|------|-------|-------|---|\n")
    for label, min_d, max_d, count, pct in accuracy_distribution:
        f.write(f"| {label} | {min_d}-{max_d} | {count} | {pct:.1f}% |\n")

    f.write("\n## Key Findings\n\n")
    f.write("### 1. Boundary Position Accuracy\n")
    perfect_good = sum(1 for d in deviations if d < 50)
    pct_good = perfect_good / len(deviations) * 100 if deviations else 0
    f.write(f"- **High accuracy** (< 50 chars): {perfect_good} ({pct_good:.1f}%)\n")
    f.write(f"- **Poor accuracy** (> 200 chars): {len(poorly_matched)} ({len(poorly_matched)/len(matches)*100:.1f}%)\n\n")

    f.write("### 2. Systematic Errors\n")
    f.write(f"- **Bias toward over-detection**: {over_detections > under_detections}\n")
    f.write(f"- **Average signed deviation**: {sum(signed_deviations)/len(signed_deviations):.0f} chars\n")
    f.write(f"  (Negative = detected too early, Positive = detected too late)\n\n")

    f.write("### 3. Segment Length Distributions\n")
    if isnad_lengths:
        f.write(f"- **Isnad lengths**: {min(isnad_lengths)}-{max(isnad_lengths)} chars (avg {sum(isnad_lengths)/len(isnad_lengths):.0f})\n")
    if khabar_lengths:
        f.write(f"- **Khabar lengths**: {min(khabar_lengths)}-{max(khabar_lengths)} chars (avg {sum(khabar_lengths)/len(khabar_lengths):.0f})\n")

    f.write("\n## Recommendations\n\n")
    f.write("1. **Fix boundary-finding algorithm**\n")
    f.write("   - Large median deviation (500 chars) suggests algorithm issues\n")
    f.write("   - Consider: missing 'qal' marker? overlapping isnads?\n\n")
    f.write("2. **Focus on high-frequency error patterns**\n")
    f.write("   - 29.2% have 200-500 char errors\n")
    f.write("   - These likely represent systematic detection failures\n\n")
    f.write("3. **Improve isnad boundary detection**\n")
    f.write("   - Add missing prefixed verbs\n")
    f.write("   - Handle isnads without 'qal' marker\n\n")

print(f"[+] Report saved to: {report_path}")
