#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1.1: Extract & Categorize ALL Isnads from Reference Annotations

Objective: Understand the actual transmission verbs and their patterns in the reference.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def normalize_arabic_text(text: str) -> str:
    """Normaliser le texte arabe."""
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


def extract_all_isnads(akhbars):
    """Extraire TOUS les isnads avec métadonnées complètes."""
    isnads = []

    for akh in akhbars:
        if akh.get('subtype') == 'preambule':
            continue

        akh_num = akh.get('num')
        content = akh.get('content', {})
        segments = content.get('segments', [])

        # Chercher le segment isnad
        for seg in segments:
            if seg.get('type') == 'isnad':
                isnad_text = seg.get('text', '').strip()
                if isnad_text:
                    # Extraire le premier mot (verbe de transmission)
                    first_word = isnad_text.split()[0] if isnad_text.split() else ""
                    normalized_first = normalize_arabic_text(first_word)

                    # Compter les séparateurs de transmetteurs
                    xan_count = isnad_text.count('عن')

                    # Vérifier présence de 'قال'
                    has_qal = 'قال' in isnad_text

                    isnads.append({
                        'akh_num': akh_num,
                        'text': isnad_text,
                        'length': len(isnad_text),
                        'word_count': len(isnad_text.split()),
                        'first_word': first_word,
                        'normalized_first': normalized_first,
                        'xan_count': xan_count,
                        'transmitter_count': xan_count + 1,
                        'has_qal': has_qal,
                    })
                break

    return isnads


def main():
    """Analyser les isnads du corpus de référence."""
    print("[*] TASK 1.1: Extracting and Categorizing ALL Isnads")
    print("[*] Loading reference annotations...\n")

    # Load annotations
    with open("../data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    akhbars = data['akhbar']
    isnads = extract_all_isnads(akhbars)

    print(f"[+] Extracted {len(isnads)} isnads\n")

    # Analyze by starting verb
    verbs_analysis = defaultdict(list)
    for isnad in isnads:
        verb = isnad['first_word']
        verbs_analysis[verb].append(isnad)

    # Generate report
    report_path = Path("../results/task_1_1_reference_isnads_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TASK 1.1: Reference Isnads Analysis\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total isnads extracted: {len(isnads)}\n")
        f.write(f"- Unique starting verbs: {len(verbs_analysis)}\n\n")

        # Detailed verb analysis
        f.write("## Starting Verbs (Transmission Verbs) - Ranked by Frequency\n\n")
        f.write("| Verb | Count | Avg Length | Min | Max | Has Qal | Transmitters |\n")
        f.write("|------|-------|-----------|-----|-----|---------|---------------|\n")

        sorted_verbs = sorted(verbs_analysis.items(), key=lambda x: -len(x[1]))

        for verb, isnad_list in sorted_verbs:
            count = len(isnad_list)
            lengths = [i['length'] for i in isnad_list]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)

            # Count how many have qal
            qal_count = sum(1 for i in isnad_list if i['has_qal'])
            qal_pct = qal_count / count * 100

            # Transmitter count distribution
            trans_counts = Counter(i['transmitter_count'] for i in isnad_list)
            trans_dist = ", ".join([f"{k}T:{v}" for k, v in sorted(trans_counts.items())])

            f.write(f"| {verb} | {count} | {avg_len:.0f} | {min_len} | {max_len} | {qal_pct:.0f}% | {trans_dist} |\n")

        # Distribution analysis
        f.write("\n## Isnad Length Distribution\n\n")

        lengths = [i['length'] for i in isnads]
        ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]

        f.write("| Range | Count | % |\n")
        f.write("|-------|-------|---|\n")
        for start, end in ranges:
            count = sum(1 for l in lengths if start <= l < end)
            pct = count / len(isnads) * 100
            f.write(f"| {start}-{end} | {count} | {pct:.1f}% |\n")

        # Transmitter distribution
        f.write("\n## Transmitter Count Distribution\n\n")

        trans_dist = Counter(i['transmitter_count'] for i in isnads)

        f.write("| Transmitters | Count | % |\n")
        f.write("|--------------|-------|---|\n")
        for count_val in sorted(trans_dist.keys()):
            freq = trans_dist[count_val]
            pct = freq / len(isnads) * 100
            f.write(f"| {count_val} | {freq} | {pct:.1f}% |\n")

        # Qal analysis
        f.write("\n## Presence of 'قال' (End Marker)\n\n")
        qal_count = sum(1 for i in isnads if i['has_qal'])
        qal_pct = qal_count / len(isnads) * 100
        f.write(f"- Isnads WITH 'قال': {qal_count} ({qal_pct:.1f}%)\n")
        f.write(f"- Isnads WITHOUT 'قال': {len(isnads) - qal_count} ({100-qal_pct:.1f}%)\n\n")

        # Top 5 verbs detail
        f.write("## Top 5 Starting Verbs - Detailed Breakdown\n\n")

        for i, (verb, isnad_list) in enumerate(sorted_verbs[:5]):
            f.write(f"### {i+1}. {verb} ({len(isnad_list)} occurrences)\n\n")

            lengths = [akh['length'] for akh in isnad_list]
            f.write(f"- Average length: {sum(lengths)/len(lengths):.0f} chars\n")
            f.write(f"- Median length: {sorted(lengths)[len(lengths)//2]} chars\n")
            f.write(f"- With 'قال': {sum(1 for akh in isnad_list if akh['has_qal'])} / {len(isnad_list)}\n")

            trans_counts = Counter(akh['transmitter_count'] for akh in isnad_list)
            f.write(f"- Transmitter distribution: ")
            f.write(", ".join([f"{k}T: {v}" for k, v in sorted(trans_counts.items())]))
            f.write("\n\n")

        # Recommendations
        f.write("## Recommendations for Baseline Verb List\n\n")
        f.write("### High Priority (>50 occurrences):\n")
        for verb, isnad_list in sorted_verbs:
            if len(isnad_list) >= 50:
                f.write(f"- {verb} ({len(isnad_list)})\n")

        f.write("\n### Medium Priority (10-50 occurrences):\n")
        for verb, isnad_list in sorted_verbs:
            if 10 <= len(isnad_list) < 50:
                f.write(f"- {verb} ({len(isnad_list)})\n")

        f.write("\n### Low Priority (<10 occurrences):\n")
        for verb, isnad_list in sorted_verbs:
            if len(isnad_list) < 10:
                f.write(f"- {verb} ({len(isnad_list)})\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **Qal Presence**: ")
        if qal_pct > 90:
            f.write("Strong - almost all isnads end with قال\n")
        elif qal_pct > 70:
            f.write("Good - majority have قال marker\n")
        else:
            f.write(f"Moderate - {qal_pct:.0f}% have قال marker\n")

        f.write("2. **Transmitter Patterns**: ")
        single_trans = sum(1 for i in isnads if i['transmitter_count'] == 1)
        multi_trans = len(isnads) - single_trans
        f.write(f"{single_trans} single-transmitter, {multi_trans} multi-transmitter\n")

        f.write("3. **Length Stability**: ")
        avg_len = sum(i['length'] for i in isnads) / len(isnads)
        f.write(f"Average {avg_len:.0f} chars - good for boundary detection\n\n")

    print(f"[+] Report saved to: {report_path}\n")
    print(f"[+] Total isnads analyzed: {len(isnads)}")
    print(f"[+] Unique verbs found: {len(verbs_analysis)}")
    print(f"[+] Isnads with قال: {sum(1 for i in isnads if i['has_qal'])}/{len(isnads)}")


if __name__ == "__main__":
    main()
