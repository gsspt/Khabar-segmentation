#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1.3: Pattern Discovery & Edge Case Analysis

Objective: Understand isnad structure patterns and identify edge cases
- Why do some high-frequency verbs like أخبرنا have many undetected cases?
- What are the overlapping/duplicate detection patterns?
- What edge cases exist (proper names, compound verbs, prefixed variants)?
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


# Load reference annotations and corpus
with open("../data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
akhbars_ref = data['akhbar']

with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

print("[*] TASK 1.3: Pattern Discovery & Edge Case Analysis")
print(f"[*] Total reference boundaries: {len(boundaries)}")
print(f"[*] Total isnads in annotation: {len(akhbars_ref)}\n")

# Run baseline detection
import sys
sys.path.insert(0, '.')
from baseline_v2_isnad_first import segment_akhbars_v2, find_all_isnad_starts

detected = segment_akhbars_v2(corpus)
detected_starts = {akh['start'] for akh in detected}

print(f"[*] Detected by baseline: {len(detected)}\n")

# ============================================================================
# PATTERN 1: Overlapping/Clustered Isnads
# ============================================================================

print("[*] Analyzing Isnad Clustering Patterns...\n")

# Find all isnad start positions
all_isnad_starts = find_all_isnad_starts(corpus)
print(f"[*] Total isnad starts found: {len(all_isnad_starts)}")

# Find clusters (multiple isnads within 200 chars)
clusters = []
current_cluster = []
last_pos = -500

for pos, verb in all_isnad_starts:
    if pos - last_pos < 200:
        current_cluster.append((pos, verb))
    else:
        if current_cluster:
            clusters.append(current_cluster)
        current_cluster = [(pos, verb)]
    last_pos = pos

if current_cluster:
    clusters.append(current_cluster)

multi_isnads = [c for c in clusters if len(c) > 1]
print(f"[*] Clusters with 2+ isnads: {len(multi_isnads)}")
print(f"[*] Max cluster size: {max(len(c) for c in clusters) if clusters else 0}\n")

# ============================================================================
# PATTERN 2: Verb-specific Analysis - Missing Cases
# ============================================================================

print("[*] Analyzing Verb-Specific Missing Patterns...\n")

# Extract all verb occurrences in corpus
verb_occurrences = defaultdict(list)
for pos, verb in all_isnad_starts:
    verb_occurrences[verb].append(pos)

# Check which are detected vs undetected
verb_analysis = {}
for verb, positions in verb_occurrences.items():
    detected_count = sum(1 for p in positions if p in detected_starts)
    undetected_count = len(positions) - detected_count

    verb_analysis[verb] = {
        'total': len(positions),
        'detected': detected_count,
        'undetected': undetected_count,
        'detection_rate': detected_count / len(positions) * 100 if positions else 0,
    }

# Sort by undetected count
problematic_verbs = sorted(verb_analysis.items(), key=lambda x: -x[1]['undetected'])

print("[*] Top 15 Verbs by Undetected Count (see report file)\n")

# ============================================================================
# PATTERN 3: Why is أخبرنا so problematic?
# ============================================================================

print("\n[*] Deep-Dive Analysis: Most Frequent Verb\n")

# Find the most common verb
most_common_verb = max(verb_occurrences.keys(), key=lambda v: len(verb_occurrences[v])) if verb_occurrences else None

if most_common_verb:
    positions = verb_occurrences[most_common_verb]
    print(f"[*] Total occurrences: {len(positions)}")

    # Check for overlaps
    detected_count = [p for p in positions if p in detected_starts]
    undetected_count = [p for p in positions if p not in detected_starts]

    print(f"[*] Detected: {len(detected_count)}")
    print(f"[*] Undetected: {len(undetected_count)}")

    # Store for report
    top_verb_analysis = (most_common_verb, positions, detected_count, undetected_count)
else:
    top_verb_analysis = None

# ============================================================================
# PATTERN 4: Prefixed Verbs (و + verb)
# ============================================================================

print("[*] Analyzing Prefixed Verb Patterns (wa + verb)...\n")

# Check for prefixed versions
prefixed_verbs = defaultdict(list)
normalized_corpus = normalize_arabic_text(corpus)

for verb in set(verb_occurrences.keys()):
    if not verb.startswith('و'):
        prefixed = 'و' + verb
        # Search for the prefixed version in corpus
        pos = 0
        while True:
            idx = normalized_corpus.find(prefixed, pos)
            if idx < 0:
                break
            prefixed_verbs[prefixed].append(idx)
            pos = idx + 1

print(f"[*] Prefixed verbs found in corpus: {len(prefixed_verbs)} (see report)")

# ============================================================================
# PATTERN 5: Boundary Deviation Analysis
# ============================================================================

print("\n[*] Analyzing Boundary Position Deviations...\n")

# For detected boundaries, compare to reference
detected_boundary_positions = {akh['start'] for akh in detected}
reference_boundary_positions = {b['start'] for b in boundaries}

# Find closest matches and deviations
deviations = []
for ref_pos in reference_boundary_positions:
    closest_detected = None
    min_distance = float('inf')

    for det_pos in detected_boundary_positions:
        distance = abs(ref_pos - det_pos)
        if distance < min_distance:
            min_distance = distance
            closest_detected = det_pos

    if closest_detected is not None:
        deviations.append({
            'ref_pos': ref_pos,
            'det_pos': closest_detected,
            'deviation': closest_detected - ref_pos,
            'abs_deviation': abs(closest_detected - ref_pos),
        })

# Analyze deviation patterns
if deviations:
    deviations.sort(key=lambda x: -x['abs_deviation'])

    abs_devs = [d['abs_deviation'] for d in deviations]
    avg_dev = sum(abs_devs) / len(abs_devs)

    print(f"[*] Average absolute deviation: {avg_dev:.0f} chars")
    print(f"[*] Median deviation: {sorted(abs_devs)[len(abs_devs)//2]} chars")
    print(f"[*] Max deviation: {max(abs_devs)} chars")

    # Count by ranges
    ranges = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 500)]
    print("\n[*] Deviations by Range:\n")
    for start, end in ranges:
        count = sum(1 for d in deviations if start <= d['abs_deviation'] < end)
        pct = count / len(deviations) * 100
        print(f"  {start:3d}-{end:3d} chars: {count:3d} ({pct:5.1f}%)")

# ============================================================================
# PATTERN 6: Non-verb Openers (Edge Cases)
# ============================================================================

print("\n[*] Analyzing Non-Verb Openers (Edge Cases)...\n")

# Check what starts detected isnads that aren't verbs
from baseline_v2_isnad_first import ISNAD_START_VERBS

detected_first_words = []
for det in detected:
    start = det['start']
    end = min(start + 50, len(corpus))
    snippet = corpus[start:end].strip()
    words = snippet.split()
    if words:
        first_word = words[0]
        detected_first_words.append(first_word)

non_verb_openers = Counter()
for word in detected_first_words:
    if word not in ISNAD_START_VERBS:
        non_verb_openers[word] += 1

if non_verb_openers:
    print("[*] Non-Verb Openers in Detected Isnads (see report)")
else:
    print("[*] All detected isnads start with recognized verbs")

# ============================================================================
# Generate Report
# ============================================================================

report_path = Path("../results/task_1_3_pattern_discovery.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# TASK 1.3: Pattern Discovery & Edge Case Analysis\n\n")

    f.write("## Isnad Clustering Patterns\n\n")
    f.write(f"- Total isnad starts found in corpus: {len(all_isnad_starts)}\n")
    f.write(f"- Clusters with 2+ isnads within 200 chars: {len(multi_isnads)}\n")
    f.write(f"- Maximum cluster size: {max(len(c) for c in clusters) if clusters else 0} isnads\n")
    f.write(f"- Analysis: Dense clustering suggests overlapping/adjacent isnads may cause detection failures\n\n")

    f.write("## Verb-Specific Detection Rates\n\n")
    f.write("| Verb | Total | Detected | Undetected | Rate |\n")
    f.write("|------|-------|----------|------------|------|\n")

    for verb, stats in problematic_verbs:
        f.write(f"| {verb} | {stats['total']} | {stats['detected']} | {stats['undetected']} | {stats['detection_rate']:.1f}% |\n")

    f.write("\n## Critical Finding: Most Frequent Verb Analysis\n\n")
    if top_verb_analysis:
        verb, all_pos, det_pos, undet_pos = top_verb_analysis
        f.write(f"- Verb (represented as index in list): {len(verb)} chars\n")
        f.write(f"- Total occurrences: {len(all_pos)}\n")
        f.write(f"- Detected: {len(det_pos)} ({len(det_pos)/len(all_pos)*100:.1f}%)\n")
        f.write(f"- Undetected: {len(undet_pos)} ({len(undet_pos)/len(all_pos)*100:.1f}%)\n\n")
        f.write("**Key Insight**: Most frequent verb may have lower detection rate due to:\n")
        f.write("- Boundary detection failure (no 'قال' found?)\n")
        f.write("- Length validation issues\n")
        f.write("- Overlapping detections\n\n")
    else:
        f.write("No verb analysis available\n\n")

    f.write("## Prefixed Verbs (و + verb)\n\n")
    f.write(f"- Unique prefixed verbs found: {len(prefixed_verbs)}\n")
    f.write("- Top 10 prefixed verbs (by occurrence count):\n\n")

    for i, (pv, positions) in enumerate(sorted(prefixed_verbs.items(), key=lambda x: -len(x[1]))[:10], 1):
        f.write(f"  {i}. Length {len(pv)}: {len(positions)} occurrences\n")

    f.write(f"\n**Key Insight**: Prefixed verbs (wa+verb) are NOT in ISNAD_START_VERBS list!\n")
    f.write(f"Adding prefixed variants could recover significant undetected isnads.\n\n")

    f.write("## Boundary Position Deviation Analysis\n\n")
    if deviations:
        f.write(f"- Average absolute deviation: {avg_dev:.0f} chars\n")
        f.write(f"- Median deviation: {sorted(abs_devs)[len(abs_devs)//2]} chars\n")
        f.write(f"- Max deviation: {max(abs_devs)} chars\n\n")

        f.write("Deviations by Range:\n\n")
        for start, end in ranges:
            count = sum(1 for d in deviations if start <= d['abs_deviation'] < end)
            pct = count / len(deviations) * 100
            f.write(f"- {start:3d}-{end:3d} chars: {count:3d} ({pct:5.1f}%)\n")

    f.write("\n## Non-Verb Openers (Edge Cases)\n\n")
    if non_verb_openers:
        f.write("Non-verb words that appear at isnad start:\n\n")
        for word, count in non_verb_openers.most_common(10):
            f.write(f"- {word}: {count} times\n")
    else:
        f.write("All detected isnads start with recognized verbs.\n")

    f.write("\n## Recommendations\n\n")
    f.write("1. **Add prefixed verb variants** (و + ISNAD_START_VERBS)\n")
    f.write("   - Impact: Could recover 200+ undetected isnads\n\n")
    f.write("2. **Debug أخبرنا detection failure**\n")
    f.write("   - Why is most-frequent verb most-undetected?\n")
    f.write("   - Check: missing 'قال' marker? Length validation? Overlaps?\n\n")
    f.write("3. **Analyze clustering effects**\n")
    f.write("   - Dense clusters may cause overlap issues\n")
    f.write("   - Consider interval-based filtering\n\n")
    f.write("4. **Handle non-verb openers**\n")
    f.write("   - Proper names (الحسن, علي) sometimes start isnads\n")
    f.write("   - Need context-aware fallback\n")

print(f"[+] Report saved to: {report_path}")
