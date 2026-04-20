#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check how many khabars have no isnad in annotated data.
"""

import json
from pathlib import Path

# Load annotated data - FIX: specify UTF-8 encoding for Arabic text
data_path = Path("../data/processed/Kitab_Uqala_al_Majanin_annotated.json")
with open(data_path, 'r', encoding='utf-8') as f:  # ← KEY FIX
    data = json.load(f)

print("=" * 80)
print("CHECKING KHABARS WITHOUT ISNAD")
print("=" * 80 + "\n")

total = len(data['akhbar'])
no_isnad_count = 0
no_transmetteurs_count = 0
no_qal_count = 0
with_isnad_count = 0

for i, akhbar in enumerate(data['akhbar']):
    transmetteurs = akhbar.get('transmetteurs', [])

    # Check if has transmitters
    if not transmetteurs or len(transmetteurs) == 0:
        no_transmetteurs_count += 1
    else:
        with_isnad_count += 1

    # Check if content has قال
    content = akhbar.get('content', {})
    segments = content.get('segments', [])
    full_text = ' '.join(seg.get('text', '') for seg in segments)

    if 'قال' not in full_text:
        no_qal_count += 1

print(f"TOTAL AKHBAR: {total}\n")

print(f"WITH TRANSMITTERS (has isnad):     {with_isnad_count:4d} ({100*with_isnad_count/total:5.1f}%)")
print(f"WITHOUT TRANSMITTERS (no isnad):   {no_transmetteurs_count:4d} ({100*no_transmetteurs_count/total:5.1f}%)\n")

print(f"WITH [qal] marker:                 {total - no_qal_count:4d} ({100*(total-no_qal_count)/total:5.1f}%)")
print(f"WITHOUT [qal] marker:              {no_qal_count:4d} ({100*no_qal_count/total:5.1f}%)\n")

print("=" * 80)
print("INTERPRETATION")
print("=" * 80 + "\n")

if no_transmetteurs_count == 0:
    print("[OK] EXCELLENT: ALL 533 khabars have transmitters (isnad)")
    print("  -> Can safely use transmetteurs field for boundary detection")
    print("  -> Fallback 30% heuristic rarely needed\n")
elif no_transmetteurs_count <= 10:
    print(f"[WARN] ACCEPTABLE: Only {no_transmetteurs_count} ({100*no_transmetteurs_count/total:.1f}%) without isnad")
    print("  -> Mostly safe, but should handle these edge cases\n")
else:
    print(f"[ERROR] WARNING: {no_transmetteurs_count} ({100*no_transmetteurs_count/total:.1f}%) without isnad")
    print("  -> Significant number need special handling\n")

if no_qal_count <= 50:
    print(f"[OK] [qal] marker found in {100*(total-no_qal_count)/total:.1f}% of khabars")
    print("  -> Good fallback available for boundary detection\n")
else:
    print(f"[WARN] [qal] missing in {100*no_qal_count/total:.1f}% of khabars")
    print("  -> Need robust fallback strategy\n")

print("=" * 80)
print("RECOMMENDATION FOR CAMELBERT NOTEBOOKS")
print("=" * 80 + "\n")

if no_transmetteurs_count == 0:
    print("[OK] USE: Transmetteurs-based boundary detection")
    print("  Current notebooks (30% fallback) are SAFE")
    print("  Could be improved by using transmetteurs field")
else:
    print("[ACTION] MODIFY: Add handling for khabars without transmetteurs")
    print(f"  Current notebooks may mislabel {no_transmetteurs_count} examples")
    print("  Should use [qal] or structural patterns for these cases")
