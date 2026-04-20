#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tester la baseline améliorée (avec filtrage des verbes ambigus).
Compare avec la baseline originale.
"""

import json
from pathlib import Path
from baseline_isnad_segmentation import segment_akhbars


def load_reference_data():
    """Charger le corpus et les boundaries."""
    with open("data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        boundaries = json.load(f)

    return corpus, boundaries


def compare_boundaries(ref_boundaries: list, detected_boundaries: list, tolerance: int = 50) -> dict:
    """Comparer les boundaries avec tolérance."""
    ref_starts = [b['start'] for b in ref_boundaries]
    ref_ends = [b['end'] for b in ref_boundaries]

    det_starts = [b[0] for b in detected_boundaries]
    det_ends = [b[1] for b in detected_boundaries]

    # Matching avec tolérance
    matched_starts = 0
    for det_start in det_starts:
        for ref_start in ref_starts:
            if abs(det_start - ref_start) <= tolerance:
                matched_starts += 1
                break

    matched_ends = 0
    for det_end in det_ends:
        for ref_end in ref_ends:
            if abs(det_end - ref_end) <= tolerance:
                matched_ends += 1
                break

    # Calcul des métriques
    precision_start = matched_starts / len(det_starts) if det_starts else 0
    recall_start = matched_starts / len(ref_starts) if ref_starts else 0
    f1_start = 2 * (precision_start * recall_start) / (precision_start + recall_start) if (precision_start + recall_start) > 0 else 0

    precision_end = matched_ends / len(det_ends) if det_ends else 0
    recall_end = matched_ends / len(ref_ends) if ref_ends else 0
    f1_end = 2 * (precision_end * recall_end) / (precision_end + recall_end) if (precision_end + recall_end) > 0 else 0

    return {
        "f1_start": f1_start,
        "f1_end": f1_end,
        "f1_avg": (f1_start + f1_end) / 2,
    }


def main():
    """Tester la baseline améliorée."""
    print("=" * 70)
    print("TEST DE LA BASELINE AMELIOREE")
    print("=" * 70 + "\n")

    # Charger données
    print("[1] Chargement du corpus de reference...")
    corpus, boundaries = load_reference_data()
    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Boundaries reference: {len(boundaries)}\n")

    # Exécuter la baseline
    print("[2] Execution de la baseline amelioree...")
    akhbars_detected = segment_akhbars(corpus)
    print(f"    Akhbars detectes: {len(akhbars_detected)}\n")

    # Comparaison
    print("[3] Comparaison des resultats\n")

    diff = len(akhbars_detected) - len(boundaries)
    pct = 100 * diff / len(boundaries) if boundaries else 0

    print(f"    Reference: {len(boundaries)} akhbars")
    print(f"    Baseline:  {len(akhbars_detected)} akhbars")
    print(f"    Ecart:     {diff:+d} ({pct:+.1f}%)\n")

    # Extraire les boundaries détectés
    detected_boundaries = []
    current_pos = 0
    for i, akh in enumerate(akhbars_detected, 1):
        isnad = akh.get('isnad', '')
        khabar = akh.get('khabar', '')
        full_text = (isnad + ' ' + khabar).strip() if (isnad or khabar) else ""

        idx = corpus.find(full_text, current_pos)
        if idx != -1:
            detected_boundaries.append((idx, idx + len(full_text), i))
            current_pos = idx + len(full_text)

    # Calculer les métriques
    print("[4] Metriques de frontieres (tolerance: 50 chars)\n")
    metrics = compare_boundaries(boundaries, detected_boundaries, tolerance=50)

    print(f"    F1 debut:   {metrics['f1_start']:.4f}")
    print(f"    F1 fin:     {metrics['f1_end']:.4f}")
    print(f"    F1 MOYENNE: {metrics['f1_avg']:.4f}\n")

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70 + "\n")

    if metrics['f1_avg'] > 0.9:
        verdict = "[EXCELLENT]"
    elif metrics['f1_avg'] > 0.8:
        verdict = "[BON]"
    elif metrics['f1_avg'] > 0.7:
        verdict = "[ACCEPTABLE]"
    elif metrics['f1_avg'] > 0.5:
        verdict = "[MOYEN]"
    else:
        verdict = "[FAIBLE]"

    print(f"Verdict: {verdict}\n")
    print(f"Akhbars detectes: {len(akhbars_detected)}/{ len(boundaries)} ({100*len(akhbars_detected)/len(boundaries):.1f}%)")
    print(f"Ecart: {diff:+d} ({pct:+.1f}%)")
    print(f"F1 frontieres: {metrics['f1_avg']:.1%}\n")

    # Sauvegarder le rapport
    report_path = Path("results/baseline_improved_test.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Test de la Baseline Amelioree\n\n")
        f.write("## Changements\n")
        f.write("- Verbes ambigus (qal, dhakara, etc.) ENLEVES\n")
        f.write("- Garde seulement les verbes fiables (hadithna, akhbarna, etc.)\n\n")

        f.write("## Resultats\n")
        f.write(f"- Akhbars detectes: {len(akhbars_detected)}\n")
        f.write(f"- Akhbars attendus: {len(boundaries)}\n")
        f.write(f"- Ecart: {diff:+d} ({pct:+.1f}%)\n\n")

        f.write("## Metriques\n")
        f.write(f"- F1 debut: {metrics['f1_start']:.4f}\n")
        f.write(f"- F1 fin: {metrics['f1_end']:.4f}\n")
        f.write(f"- **F1 MOYENNE: {metrics['f1_avg']:.4f}**\n\n")

        f.write("## Verdict\n")
        f.write(f"**{verdict}**\n")

    print(f"Rapport: {report_path}\n")


if __name__ == "__main__":
    main()
