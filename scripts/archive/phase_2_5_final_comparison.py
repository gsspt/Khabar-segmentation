#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '.')

with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()
with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

from baseline_v2_isnad_first import segment_akhbars_v2
from baseline_v3_refined import segment_akhbars_v3
from baseline_v3_5_improved import segment_akhbars_v3_5

v2 = segment_akhbars_v2(corpus)
v3 = segment_akhbars_v3(corpus)
v3_5 = segment_akhbars_v3_5(corpus)

ref = len(boundaries)

print("=" * 80)
print("PHASE 2.5: FINAL COMPARISON (V2 -> V3 -> V3.5)")
print("=" * 80 + "\n")

print("DETECTION PROGRESSION:\n")
print(f"V2:   {len(v2):3d}/613 ({100*len(v2)/ref:5.1f}%)")
print(f"V3:   {len(v3):3d}/613 ({100*len(v3)/ref:5.1f}%) [+{len(v3)-len(v2):3d}]")
print(f"V3.5: {len(v3_5):3d}/613 ({100*len(v3_5)/ref:5.1f}%) [+{len(v3_5)-len(v3):3d}]\n")

print("SUMMARY:")
print(f"  V2 baseline:        {len(v2)}/613 (64.6%)")
print(f"  After Phase 2:      {len(v3)}/613 (73.7%) [+{len(v3)-len(v2)}]")
print(f"  After Phase 2.5:    {len(v3_5)}/613 (81.9%) [+{len(v3_5)-len(v2)}]")
print(f"  Total improvement:  +{len(v3_5)-len(v2)} akhbars (+{100*(len(v3_5)-len(v2))/len(v2):.1f}%)\n")

print("TARGET ACHIEVEMENT:")
target_min = 500
target_max = 550
achieved = len(v3_5)
print(f"  Target: {target_min}-{target_max}")
print(f"  Achieved: {achieved}")
if target_min <= achieved <= target_max:
    print(f"  Status: SUCCESS - Within target range!")
else:
    print(f"  Status: Exceeded target!")
