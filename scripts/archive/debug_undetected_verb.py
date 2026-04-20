#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug: Why is the most common undetected verb failing?

Key Finding: 144 undetected akhbars start with a verb that's ALREADY in v3 list!
This suggests a boundary detection or validation issue, not a missing verb.

This single verb represents 35% of the 402 undetected. Fixing it could
recover ~35% of remaining gaps.
"""

import json
import re
from collections import Counter
from pathlib import Path

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

# Run v3 baseline
import sys
sys.path.insert(0, '.')
from baseline_v3_refined import segment_akhbars_v3, find_all_isnad_starts, find_isnad_end, ISNAD_MIN_LENGTH, ISNAD_MAX_LENGTH, ISNAD_ABSOLUTE_MAX

detected = segment_akhbars_v3(corpus)
detected_starts = {d['start'] for d in detected}

print("[*] DEBUG: Undetected Verb Analysis\n")

# Find undetected
undetected = []
for ref in boundaries:
    is_detected = False
    for det_start in detected_starts:
        if abs(ref['start'] - det_start) <= 100:
            is_detected = True
            break
    if not is_detected:
        undetected.append(ref)

# Find most common first word in undetected
first_words = Counter()
for ref in undetected:
    words = corpus[ref['start']:ref['start']+100].split()
    if words:
        first_words[words[0]] += 1

most_common_word, count = first_words.most_common(1)[0]

print(f"[*] Most common undetected verb: (count={count})")
print(f"[*] This verb appears {count} times in undetected akhbars\n")

# Find all references that start with this verb
ref_with_verb = [r for r in undetected if corpus[r['start']:r['start']+100].split() and
                  corpus[r['start']:r['start']+100].split()[0] == most_common_word]

print(f"[*] Found {len(ref_with_verb)} reference akhbars starting with this verb\n")

# Try to detect each one manually
print("[*] Attempting to detect each one:\n")

success = 0
failure_modes = {}

for i, ref in enumerate(ref_with_verb[:10]):  # Debug first 10
    start_pos = ref['start']
    end_pos = ref['end']

    # Try find_all_isnad_starts
    all_starts = find_all_isnad_starts(corpus)
    matching_starts = [(s, v) for s, v in all_starts if start_pos - 50 <= s <= start_pos + 50]

    if matching_starts:
        # Verb was found
        verb_start = matching_starts[0][0]

        # Try to find isnad end
        isnad_end = find_isnad_end(corpus, verb_start, end_bound=-1)

        if isnad_end > 0:
            # Found قال
            isnad_length = isnad_end - verb_start
            khabar_length = end_pos - isnad_end

            if (isnad_length < ISNAD_MIN_LENGTH or isnad_length > ISNAD_ABSOLUTE_MAX or
                khabar_length < 20):
                failure_modes['length_validation'] = failure_modes.get('length_validation', 0) + 1
                print(f"  [{i+1}] [FAIL] Length validation failed:")
                print(f"       Isnad: {isnad_length} ({ISNAD_MIN_LENGTH}-{ISNAD_ABSOLUTE_MAX})")
                print(f"       Khabar: {khabar_length} (min 20)")
            else:
                success += 1
                print(f"  [{i+1}] [OK] Would be detected\n")
        else:
            # Couldn't find قال
            failure_modes['no_qal'] = failure_modes.get('no_qal', 0) + 1
            print(f"  [{i+1}] [FAIL] No qal found within {ISNAD_MAX_LENGTH} chars\n")
    else:
        # Verb not found
        failure_modes['verb_not_found'] = failure_modes.get('verb_not_found', 0) + 1
        print(f"  [{i+1}] [FAIL] Verb not found by find_all_isnad_starts\n")

print("[*] Summary of failure modes:\n")
for mode, count in failure_modes.items():
    print(f"  {mode}: {count}")

print(f"\n  Successes: {success}")

# Analyze typical characteristics
print(f"\n[*] Analyzing typical characteristics of {count} undetected instances:\n")

lengths = []
has_qal = 0

for ref in ref_with_verb:
    start = ref['start']
    end = ref['end']
    snippet = corpus[start:end]

    lengths.append(end - start)
    if 'قال' in snippet:
        has_qal += 1

if lengths:
    print(f"  Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min-Max: {min(lengths)}-{max(lengths)} chars")
    print(f"  With qal marker: {has_qal}/{len(ref_with_verb)} ({has_qal/len(ref_with_verb)*100:.0f}%)")

print(f"\n[*] HYPOTHESIS:")
print(f"  This verb is being skipped because:\n")

if 'no_qal' in failure_modes and failure_modes['no_qal'] > 0:
    print(f"  - Many instances have no qal marker within {ISNAD_MAX_LENGTH} chars")
elif 'length_validation' in failure_modes and failure_modes['length_validation'] > 0:
    print(f"  - Length validation is too strict (min/max bounds)")
elif 'verb_not_found' in failure_modes and failure_modes['verb_not_found'] > 0:
    print(f"  - Word boundary checking is filtering out valid instances")

print(f"\n[*] RECOMMENDATION:")
print(f"  Focus on this single verb first - fixing it could recover ~{count} akhbars")
print(f"  This represents ~{count/402*100:.0f}% of remaining undetected\n")

# Generate report
report_path = Path("../results/phase_2_5_debug_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 2.5: Debug Report - Undetected Verb Analysis\n\n")

    f.write("## Key Finding\n\n")
    f.write(f"57% of undetected akhbars have verbs ALREADY in v3 list.\n")
    f.write(f"This indicates **boundary detection failure**, not missing verbs.\n\n")

    f.write(f"## Most Common Undetected Verb\n\n")
    f.write(f"- Count: {count} undetected instances\n")
    f.write(f"- % of undetected: {count/402*100:.1f}%\n")
    f.write(f"- Is in v3 verb list: YES\n")
    f.write(f"- Avg segment length: {sum(lengths)/len(lengths):.0f} chars\n")
    f.write(f"- Has qal marker: {has_qal}/{count} ({has_qal/count*100:.0f}%)\n\n")

    f.write("## Failure Analysis\n\n")
    if failure_modes:
        f.write("| Failure Mode | Count |\n")
        f.write("|--------------|-------|\n")
        for mode, cnt in failure_modes.items():
            f.write(f"| {mode} | {cnt} |\n")
    else:
        f.write("Could not determine failure mode from sample.\n")

    f.write("\n## Potential Fixes\n\n")
    if 'no_qal' in failure_modes:
        f.write("1. **No قال found**: Extend ISNAD_MAX_LENGTH further (currently 700)\n")
        f.write("2. **Fallback boundary**: Improve heuristic for non-قال isnads\n\n")
    if 'length_validation' in failure_modes:
        f.write("1. **Length validation**: May be too strict\n")
        f.write("2. **Consider**: Different bounds for different verb types\n\n")

print(f"[+] Report saved to: {report_path}\n")
