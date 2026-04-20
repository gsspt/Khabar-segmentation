#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valider la baseline contre le corpus de référence.

1. Lancer la baseline sur le corpus de référence
2. Extraire les positions des akhbars détectés
3. Comparer avec les boundaries de référence
4. Calculer les métriques (Pk, WindowDiff, F1)
"""

import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Importer la baseline
sys.path.insert(0, str(Path(__file__).parent))
from baseline_isnad_segmentation import (
    segment_akhbars, ISNAD_VERBS, TRANSMISSION_MARKERS
)

try:
    from segeval import pk, window_diff
    HAS_SEGEVAL = True
except ImportError:
    HAS_SEGEVAL = False


def load_reference_boundaries() -> list:
    """Charger les boundaries de référence."""
    with open("data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def load_reference_corpus() -> str:
    """Charger le corpus de référence."""
    with open("data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        return f.read()


def run_baseline_on_text(text: str) -> list:
    """
    Lancer la baseline et retourner les boundaries détectés.
    Chaque boundary = (start_char, end_char, akhbar_num)
    """
    boundaries = []

    # Utiliser la fonction segment_text de baseline_isnad_segmentation
    akhbars = segment_text(text)

    # Convertir les résultats en boundaries
    current_pos = 0
    for akhbar_num, akhbar in enumerate(akhbars, 1):
        isnad = akhbar.get('isnad', '')
        khabar = akhbar.get('khabar', '')

        # Chercher cet akhbar dans le texte
        full_akhbar = isnad + ' ' + khabar if isnad and khabar else (isnad or khabar)

        # Trouver dans le texte
        idx = text.find(full_akhbar, current_pos)
        if idx != -1:
            start_pos = idx
            end_pos = idx + len(full_akhbar)
            boundaries.append((start_pos, end_pos, akhbar_num))
            current_pos = end_pos
        else:
            # Si exact match ne fonctionne pas, essayer fuzzy
            # Pour l'instant, juste noter
            pass

    return boundaries


def compare_boundaries(ref_boundaries: list, detected_boundaries: list, tolerance: int = 50) -> dict:
    """
    Comparer les boundaries de référence avec ceux détectés.
    tolerance = nombre de caractères d'erreur acceptable
    """
    # Extraire juste les positions
    ref_starts = [b['start'] for b in ref_boundaries]
    ref_ends = [b['end'] for b in ref_boundaries]

    det_starts = [b[0] for b in detected_boundaries]
    det_ends = [b[1] for b in detected_boundaries]

    # Matching starts avec tolérance
    matched_starts = 0
    for det_start in det_starts:
        for ref_start in ref_starts:
            if abs(det_start - ref_start) <= tolerance:
                matched_starts += 1
                break

    # Matching ends avec tolérance
    matched_ends = 0
    for det_end in det_ends:
        for ref_end in ref_ends:
            if abs(det_end - ref_end) <= tolerance:
                matched_ends += 1
                break

    # Métriques
    precision_start = matched_starts / len(det_starts) if det_starts else 0
    recall_start = matched_starts / len(ref_starts) if ref_starts else 0
    f1_start = 2 * (precision_start * recall_start) / (precision_start + recall_start) if (precision_start + recall_start) > 0 else 0

    precision_end = matched_ends / len(det_ends) if det_ends else 0
    recall_end = matched_ends / len(ref_ends) if ref_ends else 0
    f1_end = 2 * (precision_end * recall_end) / (precision_end + recall_end) if (precision_end + recall_end) > 0 else 0

    f1_avg = (f1_start + f1_end) / 2

    return {
        "f1_start": f1_start,
        "f1_end": f1_end,
        "f1_avg": f1_avg,
        "precision_start": precision_start,
        "recall_start": recall_start,
        "precision_end": precision_end,
        "recall_end": recall_end,
        "matched_starts": matched_starts,
        "total_starts": len(ref_starts),
        "matched_ends": matched_ends,
        "total_ends": len(ref_ends),
    }


def main():
    """Workflow principal de validation."""
    print("=" * 70)
    print("VALIDATION BASELINE CONTRE CORPUS DE RÉFÉRENCE")
    print("=" * 70 + "\n")

    # 1. Charger corpus et boundaries
    print("[1] Chargement du corpus de référence...")
    ref_corpus = load_reference_corpus()
    ref_boundaries = load_reference_boundaries()
    print(f"    Longueur: {len(ref_corpus):,} caractères")
    print(f"    Akhbars: {len(ref_boundaries)}\n")

    # 2. Lancer la baseline
    print("[2] Exécution de la baseline...")
    try:
        akhbars_detected = segment_akhbars(ref_corpus)
        print(f"    Akhbars détectés: {len(akhbars_detected)}\n")
    except Exception as e:
        print(f"    ERREUR: {e}\n")
        return

    # 3. Comparer les nombres d'akhbars
    print("[3] Comparaison des résultats")
    print(f"    Référence: {len(ref_boundaries)} akhbars")
    print(f"    Baseline:  {len(akhbars_detected)} akhbars")
    diff = len(akhbars_detected) - len(ref_boundaries)
    pct = 100 * diff / len(ref_boundaries) if ref_boundaries else 0
    print(f"    Écart:     {diff:+d} ({pct:+.1f}%)\n")

    # 4. Calculer les métriques de boundaries
    print("[4] Calcul des métriques de frontières...")
    print(f"    Tolérance: 50 caractères\n")

    # Extraire les boundaries détectés (approximativement)
    detected_boundaries = []
    current_pos = 0
    for i, akh in enumerate(akhbars_detected, 1):
        isnad = akh.get('isnad', '')
        khabar = akh.get('khabar', '')
        full_text = (isnad + ' ' + khabar).strip() if (isnad or khabar) else ""

        # Chercher dans le corpus
        idx = ref_corpus.find(full_text, current_pos)
        if idx != -1:
            detected_boundaries.append((idx, idx + len(full_text), i))
            current_pos = idx + len(full_text)

    metrics = compare_boundaries(ref_boundaries, detected_boundaries, tolerance=50)

    print(f"    F1 début de segment: {metrics['f1_start']:.4f}")
    print(f"      Précision: {metrics['precision_start']:.4f} ({metrics['matched_starts']}/{metrics['total_starts']})")
    print(f"      Rappel: {metrics['recall_start']:.4f}\n")

    print(f"    F1 fin de segment: {metrics['f1_end']:.4f}")
    print(f"      Précision: {metrics['precision_end']:.4f} ({metrics['matched_ends']}/{metrics['total_ends']})")
    print(f"      Rappel: {metrics['recall_end']:.4f}\n")

    print(f"    F1 MOYENNE: {metrics['f1_avg']:.4f}\n")

    # 5. Rapport final
    print("=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70 + "\n")

    if metrics['f1_avg'] > 0.9:
        verdict = "[EXCELLENT]"
    elif metrics['f1_avg'] > 0.8:
        verdict = "[BON]"
    elif metrics['f1_avg'] > 0.7:
        verdict = "[ACCEPTABLE]"
    else:
        verdict = "[FAIBLE]"

    print(f"Verdict: {verdict}")
    print(f"\nRésumé:")
    print(f"  - Akhbars détectés: {len(akhbars_detected)}/{ len(ref_boundaries)} ({100*len(akhbars_detected)/len(ref_boundaries):.1f}%)")
    print(f"  - Écart: {diff:+d} ({pct:+.1f}%)")
    print(f"  - F1 frontières: {metrics['f1_avg']:.1%}")

    # Sauvegarder le rapport
    report_path = Path("results/baseline_validation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Rapport de Validation : Baseline vs Corpus de Référence\n\n")
        f.write("## Configuration\n")
        f.write(f"- Corpus: kitab_uqala_reference_corpus.txt ({len(ref_corpus):,} chars)\n")
        f.write(f"- Akhbars référence: {len(ref_boundaries)}\n")
        f.write(f"- Tolérance: 50 caractères\n\n")

        f.write("## Résultats\n")
        f.write(f"- Akhbars détectés: {len(akhbars_detected)}\n")
        f.write(f"- Écart: {diff:+d} ({pct:+.1f}%)\n\n")

        f.write("## Métriques de Frontières\n")
        f.write(f"- F1 début: {metrics['f1_start']:.4f}\n")
        f.write(f"- F1 fin: {metrics['f1_end']:.4f}\n")
        f.write(f"- **F1 MOYENNE: {metrics['f1_avg']:.4f}**\n\n")

        f.write("## Verdict\n")
        f.write(f"**{verdict}**\n")

    print(f"\nRapport sauvegardé: {report_path}\n")


if __name__ == "__main__":
    main()
