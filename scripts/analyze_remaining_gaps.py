#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2.5: Analyze Remaining Undetected Akhbars

Goal: Find the 161 undetected akhbars and identify patterns:
- Which verbs start them?
- What are common characteristics?
- Can we add more verbs to recover them?
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
from baseline_v3_refined import segment_akhbars_v3

detected = segment_akhbars_v3(corpus)
detected_starts = {d['start'] for d in detected}

print("[*] PHASE 2.5: Analyzing Remaining Undetected Akhbars\n")
print(f"[*] Reference boundaries: {len(boundaries)}")
print(f"[*] Detected by v3: {len(detected)}")

# Find undetected (allowing 100 char tolerance like task 1.2)
undetected = []
for ref in boundaries:
    is_detected = False
    for det_start in detected_starts:
        if abs(ref['start'] - det_start) <= 100:
            is_detected = True
            break
    if not is_detected:
        undetected.append(ref)

print(f"[*] Undetected (100 char tolerance): {len(undetected)}\n")

# Extract first word from each undetected boundary
print("[*] Analyzing first words (potential verbs)...\n")

first_words = Counter()
undetected_by_verb = {}

for ref in undetected:
    start = ref['start']
    end = min(start + 100, len(corpus))  # Look at first 100 chars
    snippet = corpus[start:end].strip()
    words = snippet.split()

    if words:
        first_word = words[0]
        first_words[first_word] += 1

        if first_word not in undetected_by_verb:
            undetected_by_verb[first_word] = []
        undetected_by_verb[first_word].append(ref)

print("[*] Top 20 First Words in Undetected Akhbars:\n")

for word, count in first_words.most_common(20):
    # Check if it's already in v3 verb list
    from baseline_v3_refined import ISNAD_START_VERBS
    in_v3 = "IN" if word in ISNAD_START_VERBS else "NOT"
    print(f"  [{in_v3}] Count: {count:3d}  Word (len={len(word):2d})")

print("\n[*] Analysis of undetected by category:\n")

# Categorize undetected
in_verb_list = 0
not_in_verb_list = 0
no_clear_verb = 0

for ref in undetected:
    start = ref['start']
    snippet = corpus[start:start+100].strip()
    words = snippet.split()

    if words:
        first_word = words[0]
        if first_word in ISNAD_START_VERBS:
            in_verb_list += 1
        else:
            not_in_verb_list += 1
    else:
        no_clear_verb += 1

print(f"Category 1: First word IN v3 verb list: {in_verb_list} ({in_verb_list/len(undetected)*100:.1f}%)")
print(f"Category 2: First word NOT in v3 list: {not_in_verb_list} ({not_in_verb_list/len(undetected)*100:.1f}%)")
print(f"Category 3: No clear first word: {no_clear_verb} ({no_clear_verb/len(undetected)*100:.1f}%)\n")

# Sample undetected from each category
print("[*] Sampling Undetected Akhbars by Type:\n")

# Category 1 samples (verb in list - why is it not detected?)
in_list_samples = [r for r in undetected if corpus[r['start']:r['start']+100].split() and
                   corpus[r['start']:r['start']+100].split()[0] in ISNAD_START_VERBS][:3]

if in_list_samples:
    print("Category 1 - Verb IN list (boundary detection failure?):")
    for i, ref in enumerate(in_list_samples, 1):
        first_word = corpus[ref['start']:ref['start']+100].split()[0]
        count = first_words[first_word]
        print(f"  [{i}] Position {ref['start']}, First word (count={count}, len={len(first_word)})\n")

# Category 2 samples (new verbs to add)
not_in_list_samples = [r for r in undetected if corpus[r['start']:r['start']+100].split() and
                       corpus[r['start']:r['start']+100].split()[0] not in ISNAD_START_VERBS][:5]

if not_in_list_samples:
    print("Category 2 - NEW Verbs to Consider:")
    for i, ref in enumerate(not_in_list_samples, 1):
        first_word = corpus[ref['start']:ref['start']+100].split()[0]
        count = first_words[first_word]
        print(f"  [{i}] Position {ref['start']}, First word (count={count}, len={len(first_word)})\n")

# Generate report
report_path = Path("../results/phase_2_5_gap_analysis.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 2.5: Remaining Undetected Akhbars Analysis\n\n")

    f.write("## Summary\n\n")
    f.write(f"- Reference boundaries: {len(boundaries)}\n")
    f.write(f"- Detected by v3: {len(detected)} ({100*len(detected)/len(boundaries):.1f}%)\n")
    f.write(f"- Undetected (100 char tolerance): {len(undetected)} ({100*len(undetected)/len(boundaries):.1f}%)\n\n")

    f.write("## Categorization of Undetected\n\n")
    f.write(f"| Category | Count | % | Interpretation |\n")
    f.write(f"|----------|-------|---|----------------|\n")
    f.write(f"| Verb in v3 list | {in_verb_list} | {in_verb_list/len(undetected)*100:.1f}% | Boundary detection failure |\n")
    f.write(f"| Verb NOT in list | {not_in_verb_list} | {not_in_verb_list/len(undetected)*100:.1f}% | Missing verbs to add |\n")
    f.write(f"| No clear verb | {no_clear_verb} | {no_clear_verb/len(undetected)*100:.1f}% | Edge cases |\n\n")

    f.write("## Top 20 First Words in Undetected\n\n")
    f.write("| Rank | Count | In V3? | Notes |\n")
    f.write("|------|-------|--------|-------|\n")

    for rank, (word, count) in enumerate(first_words.most_common(20), 1):
        in_v3 = "✓" if word in ISNAD_START_VERBS else "✗"
        f.write(f"| {rank} | {count} | {in_v3} | Length: {len(word)} |\n")

    f.write("\n## High-Priority Verbs to Add\n\n")
    f.write("Top candidates with count >= 3 and not in v3:\n\n")

    candidates = [(w, c) for w, c in first_words.most_common(20)
                  if c >= 3 and w not in ISNAD_START_VERBS]
    for word, count in candidates:
        f.write(f"- Word (count={count}): {count} undetected akhbars\n")

    f.write("\n## Recommendations\n\n")
    f.write(f"1. **Recover boundary detection failures** ({in_verb_list} akhbars)\n")
    f.write(f"   - These have verbs in v3 list but aren't detected\n")
    f.write(f"   - Likely issues: missing قال marker, length validation, clustering\n")
    f.write(f"   - Action: Debug a sample to understand failure mode\n\n")

    f.write(f"2. **Add missing verbs** ({not_in_verb_list} akhbars)\n")
    f.write(f"   - High-priority: verbs with count >= 3\n")
    f.write(f"   - Estimated recovery: {sum(c for w, c in candidates)} akhbars\n\n")

    f.write(f"3. **Handle edge cases** ({no_clear_verb} akhbars)\n")
    f.write(f"   - Non-verb openers (proper names, titles)\n")
    f.write(f"   - Contextual patterns\n")

print(f"\n[+] Report saved to: {report_path}")
print(f"[+] Analysis complete\n")
