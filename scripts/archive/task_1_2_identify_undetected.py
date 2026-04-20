#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1.2: Analyze UNDETECTED Isnads

Objective: Find the 217 isnads we DON'T detect (613 - 396 detected)
Categorize by: starting verb, length, structure
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter


def normalize_arabic_text(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


# Load reference annotations
with open("../data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

akhbars_ref = data['akhbar']

# Load reference corpus and boundaries
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

print("[*] TASK 1.2: Identifying Undetected Isnads")
print(f"[*] Total reference boundaries: {len(boundaries)}")

# Run baseline detection on reference corpus
sys.path.insert(0, '.')
from baseline_v2_isnad_first import segment_akhbars_v2

detected = segment_akhbars_v2(corpus)
print(f"[*] Detected by baseline: {len(detected)}")
print(f"[*] Undetected: {len(boundaries) - len(detected)}\n")

# Extract detected boundaries (from detected akhbars)
detected_positions = set()
for akh in detected:
    detected_positions.add(akh['start'])

# Find undetected boundaries
undetected = []
for ref in boundaries:
    # Check if this boundary was detected
    # We do fuzzy matching: if any detected akhbar starts within 100 chars, consider it detected
    is_detected = False
    for det_start in detected_positions:
        if abs(ref['start'] - det_start) <= 100:
            is_detected = True
            break

    if not is_detected:
        undetected.append(ref)

print(f"[*] Undetected boundaries (allowing 100 char tolerance): {len(undetected)}\n")

# Analyze undetected by verb
undetected_by_verb = defaultdict(list)
for ref_boundary in undetected:
    start = ref_boundary['start']
    end = ref_boundary['end']
    isnad_text = corpus[start:end].strip()

    # Extract first word
    first_word = isnad_text.split()[0] if isnad_text.split() else ""

    undetected_by_verb[first_word].append({
        'boundary': ref_boundary,
        'text': isnad_text,
        'length': end - start,
    })

# Generate report
report_path = Path("../results/task_1_2_undetected_isnads.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# TASK 1.2: Undetected Isnads Analysis\n\n")

    f.write("## Summary\n\n")
    f.write(f"- Total reference boundaries: {len(boundaries)}\n")
    f.write(f"- Detected by baseline: {len(detected)}\n")
    f.write(f"- Undetected: {len(undetected)}\n")
    f.write(f"- Detection rate: {len(detected)/len(boundaries)*100:.1f}%\n\n")

    f.write("## Undetected Isnads by Starting Verb\n\n")

    sorted_verbs = sorted(undetected_by_verb.items(), key=lambda x: -len(x[1]))

    f.write("| Verb | Count | Avg Length | Min | Max |\n")
    f.write("|------|-------|-----------|-----|-----|\n")

    for verb, items in sorted_verbs:
        count = len(items)
        lengths = [i['length'] for i in items]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        f.write(f"| {verb} | {count} | {avg_len:.0f} | {min_len} | {max_len} |\n")

    # Most problematic verbs
    f.write("\n## Most Problematic Verbs (>3 undetected)\n\n")
    for verb, items in sorted_verbs[:10]:
        if len(items) >= 3:
            count = len(items)
            avg_len = sum(i['length'] for i in items) / len(items)
            f.write(f"- **{verb}**: {count} missed (avg length: {avg_len:.0f})\n")

    # Length analysis of undetected
    f.write("\n## Undetected Isnads Length Distribution\n\n")

    all_undetected_lengths = [i['length'] for items in undetected_by_verb.values() for i in items]
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]

    f.write("| Range | Count | % |\n")
    f.write("|-------|-------|---|\n")
    for start, end in ranges:
        count = sum(1 for l in all_undetected_lengths if start <= l < end)
        pct = count / len(all_undetected_lengths) * 100
        f.write(f"| {start}-{end} | {count} | {pct:.1f}% |\n")

    # Analysis
    f.write("\n## Analysis\n\n")

    f.write("### Hypothesis: Why are these isnads undetected?\n\n")

    # Check if verbs are in our verb list
    from baseline_v2_isnad_first import ISNAD_START_VERBS

    missing_verbs = defaultdict(int)
    for verb in undetected_by_verb.keys():
        if verb not in ISNAD_START_VERBS:
            missing_verbs[verb] = len(undetected_by_verb[verb])

    if missing_verbs:
        f.write("**1. Missing Verbs from ISNAD_START_VERBS:**\n\n")
        for verb, count in sorted(missing_verbs.items(), key=lambda x: -x[1])[:15]:
            f.write(f"- {verb}: {count} undetected\n")
        f.write(f"\nTotal undetected due to missing verbs: {sum(missing_verbs.values())}\n\n")

    # Check if they have no qal
    no_qal = 0
    for items in undetected_by_verb.values():
        for item in items:
            if 'قال' not in item['text']:
                no_qal += 1

    if no_qal > 0:
        f.write(f"\n**2. Missing 'قال' Marker:**\n\n")
        f.write(f"- Isnads without 'قال': {no_qal} ({no_qal/len(undetected)*100:.1f}%)\n")
        f.write(f"- These likely fail at khabar boundary detection\n\n")

    # Check if they're very short
    very_short = sum(1 for l in all_undetected_lengths if l < 10)
    if very_short > 0:
        f.write(f"\n**3. Very Short Isnads (<10 chars):**\n\n")
        f.write(f"- Count: {very_short} ({very_short/len(undetected)*100:.1f}%)\n")
        f.write(f"- These may be filtered by ISNAD_MIN_LENGTH\n\n")

    f.write("## Recommendations\n\n")
    f.write("1. Add missing high-frequency verbs from undetected list\n")
    f.write("2. For isnads without 'قال': use next verb as fallback boundary\n")
    f.write("3. Review length bounds - may be too strict\n")

print(f"[+] Report saved to: {report_path}")

# Summary stats
import sys
sys.exit(0)
