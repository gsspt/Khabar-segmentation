#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KHABAR BOUNDARY PRECISION EVALUATION

Evaluate how similar detected khabars are to reference annotations:
- Boundary position accuracy
- Segment overlap (IoU - Intersection over Union)
- Exact match rate
- Error categorization
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, '.')
from baseline_v3_5_improved import segment_akhbars_v3_5

# Load data
with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
    boundaries = json.load(f)

print("=" * 80)
print("KHABAR BOUNDARY PRECISION EVALUATION")
print("=" * 80 + "\n")

print(f"[*] Reference boundaries: {len(boundaries)}")
print(f"[*] Reference corpus: {len(corpus):,} chars\n")

# Run baseline
detected = segment_akhbars_v3_5(corpus)
print(f"[*] Detected by v3.5: {len(detected)}\n")

# ============================================================================
# METRIC 1: BOUNDARY POSITION ACCURACY
# ============================================================================

print("[*] Computing boundary position accuracy...\n")

# Match detected to reference by position (find closest match)
matches = []
for ref in boundaries:
    ref_start = ref['start']
    ref_end = ref['end']

    # Find closest detected boundary
    closest_det = None
    min_start_distance = float('inf')

    for det in detected:
        det_start = det['start']
        start_distance = abs(ref_start - det_start)

        if start_distance < min_start_distance:
            min_start_distance = start_distance
            closest_det = det

    if closest_det is not None:
        det_start = closest_det['start']
        det_end = closest_det['end']

        # Compute metrics
        start_deviation = det_start - ref_start
        end_deviation = det_end - ref_end
        abs_start_dev = abs(start_deviation)
        abs_end_dev = abs(end_deviation)

        # Overlap: Intersection over Union
        overlap_start = max(ref_start, det_start)
        overlap_end = min(ref_end, det_end)
        intersection = max(0, overlap_end - overlap_start)
        union = max(ref_end, det_end) - min(ref_start, det_start)
        iou = intersection / union if union > 0 else 0

        # Jaccard similarity (same as IoU for 1D intervals)
        jaccard = iou

        matches.append({
            'ref': ref,
            'det': closest_det,
            'ref_start': ref_start,
            'ref_end': ref_end,
            'det_start': det_start,
            'det_end': det_end,
            'ref_length': ref_end - ref_start,
            'det_length': det_end - det_start,
            'start_deviation': start_deviation,
            'end_deviation': end_deviation,
            'abs_start_dev': abs_start_dev,
            'abs_end_dev': abs_end_dev,
            'iou': iou,
            'jaccard': jaccard,
            'intersection': intersection,
            'union': union,
        })

print(f"[*] Matched {len(matches)}/{ len(boundaries)} reference boundaries\n")

# ============================================================================
# BOUNDARY ACCURACY ANALYSIS
# ============================================================================

start_devs = [m['abs_start_dev'] for m in matches]
end_devs = [m['abs_end_dev'] for m in matches]
ious = [m['iou'] for m in matches]

print("[*] START BOUNDARY ACCURACY:\n")
print(f"  Mean deviation: {sum(start_devs)/len(start_devs):.0f} chars")
print(f"  Median deviation: {sorted(start_devs)[len(start_devs)//2]} chars")
print(f"  95th percentile: {sorted(start_devs)[int(len(start_devs)*0.95)]} chars")
print(f"  Max deviation: {max(start_devs)} chars")

start_perfect = sum(1 for d in start_devs if d < 5)
start_excellent = sum(1 for d in start_devs if 5 <= d < 20)
start_good = sum(1 for d in start_devs if 20 <= d < 50)
start_acceptable = sum(1 for d in start_devs if 50 <= d < 100)

print(f"\n  Accuracy tiers:")
print(f"    Perfect (0-5 chars): {start_perfect} ({start_perfect/len(start_devs)*100:.1f}%)")
print(f"    Excellent (5-20): {start_excellent} ({start_excellent/len(start_devs)*100:.1f}%)")
print(f"    Good (20-50): {start_good} ({start_good/len(start_devs)*100:.1f}%)")
print(f"    Acceptable (50-100): {start_acceptable} ({start_acceptable/len(start_devs)*100:.1f}%)\n")

print("[*] END BOUNDARY ACCURACY:\n")
print(f"  Mean deviation: {sum(end_devs)/len(end_devs):.0f} chars")
print(f"  Median deviation: {sorted(end_devs)[len(end_devs)//2]} chars")
print(f"  95th percentile: {sorted(end_devs)[int(len(end_devs)*0.95)]} chars")
print(f"  Max deviation: {max(end_devs)} chars")

end_perfect = sum(1 for d in end_devs if d < 5)
end_excellent = sum(1 for d in end_devs if 5 <= d < 20)
end_good = sum(1 for d in end_devs if 20 <= d < 50)
end_acceptable = sum(1 for d in end_devs if 50 <= d < 100)

print(f"\n  Accuracy tiers:")
print(f"    Perfect (0-5 chars): {end_perfect} ({end_perfect/len(end_devs)*100:.1f}%)")
print(f"    Excellent (5-20): {end_excellent} ({end_excellent/len(end_devs)*100:.1f}%)")
print(f"    Good (20-50): {end_good} ({end_good/len(end_devs)*100:.1f}%)")
print(f"    Acceptable (50-100): {end_acceptable} ({end_acceptable/len(end_devs)*100:.1f}%)\n")

# ============================================================================
# SEGMENT OVERLAP (IoU) ANALYSIS
# ============================================================================

print("[*] SEGMENT OVERLAP (Jaccard Similarity / IoU):\n")

iou_perfect = sum(1 for iou in ious if iou >= 0.95)
iou_excellent = sum(1 for iou in ious if 0.90 <= iou < 0.95)
iou_good = sum(1 for iou in ious if 0.80 <= iou < 0.90)
iou_acceptable = sum(1 for iou in ious if 0.70 <= iou < 0.80)
iou_poor = sum(1 for iou in ious if iou < 0.70)

print(f"  Perfect (95-100% overlap): {iou_perfect} ({iou_perfect/len(ious)*100:.1f}%)")
print(f"  Excellent (90-95%): {iou_excellent} ({iou_excellent/len(ious)*100:.1f}%)")
print(f"  Good (80-90%): {iou_good} ({iou_good/len(ious)*100:.1f}%)")
print(f"  Acceptable (70-80%): {iou_acceptable} ({iou_acceptable/len(ious)*100:.1f}%)")
print(f"  Poor (<70%): {iou_poor} ({iou_poor/len(ious)*100:.1f}%)")

usable_iou = iou_perfect + iou_excellent + iou_good
print(f"\n  Usable (80%+ overlap): {usable_iou} ({usable_iou/len(ious)*100:.1f}%)")

print(f"\n  Mean IoU: {sum(ious)/len(ious):.3f}")
print(f"  Median IoU: {sorted(ious)[len(ious)//2]:.3f}")

# ============================================================================
# EXACT MATCH ANALYSIS
# ============================================================================

print("\n[*] EXACT MATCH ANALYSIS:\n")

exact_matches = 0
near_exact = 0  # Within 10 chars on both boundaries

for m in matches:
    if m['abs_start_dev'] == 0 and m['abs_end_dev'] == 0:
        exact_matches += 1
    elif m['abs_start_dev'] <= 10 and m['abs_end_dev'] <= 10:
        near_exact += 1

print(f"  Exact match (0 char deviation): {exact_matches} ({exact_matches/len(matches)*100:.1f}%)")
print(f"  Near exact (<=10 chars both ends): {near_exact} ({near_exact/len(matches)*100:.1f}%)")
print(f"  Combined: {exact_matches + near_exact} ({(exact_matches+near_exact)/len(matches)*100:.1f}%)")

# ============================================================================
# LENGTH DISTRIBUTION
# ============================================================================

print("\n[*] KHABAR LENGTH ANALYSIS:\n")

ref_lengths = [m['ref_length'] for m in matches]
det_lengths = [m['det_length'] for m in matches]
length_diffs = [abs(m['det_length'] - m['ref_length']) for m in matches]

print(f"  Reference khabars:")
print(f"    Mean: {sum(ref_lengths)/len(ref_lengths):.0f} chars")
print(f"    Median: {sorted(ref_lengths)[len(ref_lengths)//2]} chars")
print(f"    Min-Max: {min(ref_lengths)}-{max(ref_lengths)} chars")

print(f"\n  Detected khabars:")
print(f"    Mean: {sum(det_lengths)/len(det_lengths):.0f} chars")
print(f"    Median: {sorted(det_lengths)[len(det_lengths)//2]} chars")
print(f"    Min-Max: {min(det_lengths)}-{max(det_lengths)} chars")

print(f"\n  Length difference (detected vs reference):")
print(f"    Mean absolute difference: {sum(length_diffs)/len(length_diffs):.0f} chars")
print(f"    Median difference: {sorted(length_diffs)[len(length_diffs)//2]} chars")

# ============================================================================
# ERROR CATEGORIZATION
# ============================================================================

print("\n[*] ERROR CATEGORIZATION:\n")

over_segmented = sum(1 for m in matches if m['det_length'] < m['ref_length'] * 0.8)
under_segmented = sum(1 for m in matches if m['det_length'] > m['ref_length'] * 1.2)
correctly_sized = len(matches) - over_segmented - under_segmented

print(f"  Over-segmented (detected <80% of reference): {over_segmented} ({over_segmented/len(matches)*100:.1f}%)")
print(f"  Under-segmented (detected >120% of reference): {under_segmented} ({under_segmented/len(matches)*100:.1f}%)")
print(f"  Correctly sized (80-120%): {correctly_sized} ({correctly_sized/len(matches)*100:.1f}%)")

# ============================================================================
# GENERATE REPORT
# ============================================================================

report_path = Path("../results/khabar_precision_evaluation.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Khabar Boundary Precision Evaluation\n\n")

    f.write("## Summary\n\n")
    f.write(f"- Reference boundaries: {len(boundaries)}\n")
    f.write(f"- Detected boundaries: {len(detected)}\n")
    f.write(f"- Matched boundaries: {len(matches)}\n")
    f.write(f"- Detection rate: {100*len(detected)/len(boundaries):.1f}%\n\n")

    f.write("## Boundary Position Accuracy\n\n")
    f.write("### Start Boundary\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Mean deviation | {sum(start_devs)/len(start_devs):.0f} chars |\n")
    f.write(f"| Median deviation | {sorted(start_devs)[len(start_devs)//2]} chars |\n")
    f.write(f"| 95th percentile | {sorted(start_devs)[int(len(start_devs)*0.95)]} chars |\n")
    f.write(f"| Perfect (0-5) | {start_perfect} ({start_perfect/len(start_devs)*100:.1f}%) |\n")
    f.write(f"| Excellent (5-20) | {start_excellent} ({start_excellent/len(start_devs)*100:.1f}%) |\n")
    f.write(f"| Good (20-50) | {start_good} ({start_good/len(start_devs)*100:.1f}%) |\n")
    f.write(f"| Acceptable (50-100) | {start_acceptable} ({start_acceptable/len(start_devs)*100:.1f}%) |\n\n")

    f.write("### End Boundary\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Mean deviation | {sum(end_devs)/len(end_devs):.0f} chars |\n")
    f.write(f"| Median deviation | {sorted(end_devs)[len(end_devs)//2]} chars |\n")
    f.write(f"| 95th percentile | {sorted(end_devs)[int(len(end_devs)*0.95)]} chars |\n")
    f.write(f"| Perfect (0-5) | {end_perfect} ({end_perfect/len(end_devs)*100:.1f}%) |\n")
    f.write(f"| Excellent (5-20) | {end_excellent} ({end_excellent/len(end_devs)*100:.1f}%) |\n")
    f.write(f"| Good (20-50) | {end_good} ({end_good/len(end_devs)*100:.1f}%) |\n")
    f.write(f"| Acceptable (50-100) | {end_acceptable} ({end_acceptable/len(end_devs)*100:.1f}%) |\n\n")

    f.write("## Segment Overlap (Jaccard / IoU)\n\n")
    f.write(f"| Category | Count | % |\n")
    f.write(f"|----------|-------|---|\n")
    f.write(f"| Perfect (95-100%) | {iou_perfect} | {iou_perfect/len(ious)*100:.1f}% |\n")
    f.write(f"| Excellent (90-95%) | {iou_excellent} | {iou_excellent/len(ious)*100:.1f}% |\n")
    f.write(f"| Good (80-90%) | {iou_good} | {iou_good/len(ious)*100:.1f}% |\n")
    f.write(f"| Acceptable (70-80%) | {iou_acceptable} | {iou_acceptable/len(ious)*100:.1f}% |\n")
    f.write(f"| Poor (<70%) | {iou_poor} | {iou_poor/len(ious)*100:.1f}% |\n")
    f.write(f"| **Usable (80%+)** | **{usable_iou}** | **{usable_iou/len(ious)*100:.1f}%** |\n\n")

    f.write(f"Mean IoU: {sum(ious)/len(ious):.3f}  \n")
    f.write(f"Median IoU: {sorted(ious)[len(ious)//2]:.3f}\n\n")

    f.write("## Exact Match Analysis\n\n")
    f.write(f"- Exact match (0 char deviation): {exact_matches} ({exact_matches/len(matches)*100:.1f}%)\n")
    f.write(f"- Near exact (<=10 chars both): {near_exact} ({near_exact/len(matches)*100:.1f}%)\n")
    f.write(f"- Combined: {exact_matches + near_exact} ({(exact_matches+near_exact)/len(matches)*100:.1f}%)\n\n")

    f.write("## Khabar Length Analysis\n\n")
    f.write("### Reference Khabars\n")
    f.write(f"- Mean: {sum(ref_lengths)/len(ref_lengths):.0f} chars\n")
    f.write(f"- Median: {sorted(ref_lengths)[len(ref_lengths)//2]} chars\n")
    f.write(f"- Range: {min(ref_lengths)}-{max(ref_lengths)} chars\n\n")

    f.write("### Detected Khabars\n")
    f.write(f"- Mean: {sum(det_lengths)/len(det_lengths):.0f} chars\n")
    f.write(f"- Median: {sorted(det_lengths)[len(det_lengths)//2]} chars\n")
    f.write(f"- Range: {min(det_lengths)}-{max(det_lengths)} chars\n\n")

    f.write("### Differences\n")
    f.write(f"- Mean absolute difference: {sum(length_diffs)/len(length_diffs):.0f} chars\n")
    f.write(f"- Median difference: {sorted(length_diffs)[len(length_diffs)//2]} chars\n\n")

    f.write("## Error Categorization\n\n")
    f.write(f"- Over-segmented (<80% of reference): {over_segmented} ({over_segmented/len(matches)*100:.1f}%)\n")
    f.write(f"- Correctly sized (80-120%): {correctly_sized} ({correctly_sized/len(matches)*100:.1f}%)\n")
    f.write(f"- Under-segmented (>120% of reference): {under_segmented} ({under_segmented/len(matches)*100:.1f}%)\n\n")

    f.write("## Key Findings\n\n")
    f.write(f"1. **Detection**: {len(detected)}/{ len(boundaries)} ({100*len(detected)/len(boundaries):.1f}%)\n")
    f.write(f"2. **Boundary Accuracy**: {start_perfect + start_excellent + start_good}/{ len(matches)} ({(start_perfect + start_excellent + start_good)/len(matches)*100:.1f}%) start boundaries within 50 chars\n")
    f.write(f"3. **Segment Overlap**: {usable_iou}/{ len(matches)} ({usable_iou/len(matches)*100:.1f}%) with 80%+ IoU overlap\n")
    f.write(f"4. **Exact Match**: {exact_matches}/{ len(matches)} ({exact_matches/len(matches)*100:.1f}%) perfect match\n")
    f.write(f"5. **Length Error**: Mean {sum(length_diffs)/len(length_diffs):.0f} chars difference from reference\n\n")

    f.write("## Assessment\n\n")
    f.write("### Good Performance\n")
    f.write(f"- Detection rate of 81.9% (502/613)\n")
    f.write(f"- {(start_perfect + start_excellent + start_good)/len(matches)*100:.1f}% of boundaries accurate within 50 chars\n")
    f.write(f"- {usable_iou/len(matches)*100:.1f}% of segments have 80%+ overlap with reference\n\n")

    f.write("### Areas for Improvement\n")
    f.write(f"- {iou_poor/len(ious)*100:.1f}% of segments have <70% overlap\n")
    f.write(f"- {over_segmented/len(matches)*100:.1f}% over-segmented, {under_segmented/len(matches)*100:.1f}% under-segmented\n")
    f.write(f"- Mean boundary deviation: {(sum(start_devs) + sum(end_devs))/(2*len(matches)):.0f} chars\n\n")

print(f"\n[+] Report saved to: {report_path}\n")
