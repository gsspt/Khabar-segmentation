#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation de la baseline contre les annotations manuelles du Kitab Uqala.

Workflow :
1. Charger 612 annotations manuelles
2. Extraire le texte de chaque akhbar (concaténer segments)
3. Charger le texte brut OpenITI
4. Chercher chaque akhbar avec fuzzy matching
5. Déterminer les positions exactes (char indices)
6. Comparer avec la baseline_isnad_segmentation
7. Calculer Pk, WindowDiff, F1
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict

# Tenter d'importer segeval pour les métriques
try:
    from segeval import pk, window_diff
    HAS_SEGEVAL = True
except ImportError:
    HAS_SEGEVAL = False
    print("[!] segeval non trouvé. Installez : pip install segeval")


def load_annotations(annot_path: Path) -> Dict:
    """Charger les annotations manuelles."""
    with open(annot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_akhbar_text(akhbar: Dict) -> str:
    """Extraire le texte complet d'un akhbar en concaténant les segments."""
    segments = akhbar.get("content", {}).get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def normalize_text(text: str) -> str:
    """Normaliser minimalement le texte - juste espacements."""
    # Normaliser les espacements multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fuzzy_find(needle: str, haystack: str, threshold: float = 0.75) -> Optional[Tuple[int, int]]:
    """Trouver l'index de `needle` dans `haystack` avec fuzzy matching.

    Retourne (start_idx, end_idx) ou None si confiance < threshold.
    """
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)

    if not needle_norm or not haystack_norm:
        return None

    # Essayer d'abord exact matching
    exact_idx = haystack_norm.find(needle_norm)
    if exact_idx != -1:
        return (exact_idx, exact_idx + len(needle_norm))

    # Sinon, sliding window avec SequenceMatcher
    best_score = 0
    best_pos = None
    window_size = len(needle_norm)

    # Chercher partout dans le haystack
    for i in range(len(haystack_norm) - window_size + 1):
        window = haystack_norm[i:i + window_size]
        matcher = SequenceMatcher(None, needle_norm, window)
        ratio = matcher.ratio()

        if ratio > best_score:
            best_score = ratio
            best_pos = (i, i + window_size)

    if best_score >= threshold:
        return best_pos

    return None


def build_annotation_boundaries(annot_data: Dict, raw_text: str) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Construire les frontières de segmentation à partir des annotations.

    Retourne :
    - boundaries: liste de (start_idx, end_idx) pour chaque akhbar
    - frontier_indices: indices de début de chaque akhbar dans le texte normalisé
    """
    boundaries = []
    frontier_indices = []

    raw_text_norm = normalize_text(raw_text)
    current_pos = 0

    akhbars = annot_data.get("akhbar", [])
    print(f"*** Chargé {len(akhbars)} akhbars annotés")

    not_found = []

    for idx, akhbar in enumerate(akhbars):
        akhbar_text = extract_akhbar_text(akhbar)
        akhbar_num = akhbar.get("num", idx + 1)

        if not akhbar_text.strip():
            print(f"  [!]  Akhbar {akhbar_num}: texte vide")
            continue

        # Chercher avec fuzzy matching
        match = fuzzy_find(akhbar_text, raw_text_norm, threshold=0.82)

        if match:
            start_idx, end_idx = match
            boundaries.append((start_idx, end_idx))
            frontier_indices.append(start_idx)

            if idx < 5:  # Log les premiers
                print(f"  [OK] Akhbar {akhbar_num}: trouvé à {start_idx}-{end_idx} "
                      f"({end_idx - start_idx} chars)")
        else:
            not_found.append((akhbar_num, len(akhbar_text)))
            print(f"  [KO] Akhbar {akhbar_num}: non trouvé (len={len(akhbar_text)})")

    if not_found:
        print(f"\n[!]  {len(not_found)} akhbars non trouvés :")
        for num, length in not_found[:10]:
            print(f"    - Akhbar {num} ({length} chars)")

    print(f"\n[OK] Trouvé {len(boundaries)}/{len(akhbars)} akhbars")

    return boundaries, sorted(frontier_indices)


def segments_to_boundaries(text_len: int, frontier_indices: List[int]) -> List[Tuple[int, int]]:
    """Convertir les indices de frontières en liste de segments (start, end)."""
    # Ajouter 0 au début et text_len à la fin si absent
    indices = sorted(set([0] + frontier_indices + [text_len]))

    boundaries = []
    for i in range(len(indices) - 1):
        boundaries.append((indices[i], indices[i + 1]))

    return boundaries


def run_baseline(text_path: Path) -> List[Tuple[int, int]]:
    """Exécuter la baseline et retourner les segmentations (positions dans le texte)."""
    import subprocess

    # Exécuter le script baseline
    result = subprocess.run(
        [sys.executable, "scripts/baseline_isnad_segmentation.py",
         "--input", str(text_path)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    if result.returncode != 0:
        print(f"[ERR] Erreur baseline: {result.stderr}")
        return []

    # Parser le résultat (format : "AKHBAR X: start-end")
    boundaries = []
    for line in result.stdout.split('\n'):
        if 'AKHBAR' in line and ':' in line:
            match = re.search(r'(\d+)-(\d+)', line)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                boundaries.append((start, end))

    return boundaries


def boundaries_to_segmentation(text_len: int, boundaries: List[Tuple[int, int]]) -> str:
    """Convertir des boundaries en chaîne de segmentation pour segeval.

    Format : "===0===texte===x===texte===x===" où ===X=== sont des séparateurs.
    """
    # Créer array de positions de rupture
    breaks = set([0, text_len])
    for start, end in boundaries:
        breaks.add(start)
        breaks.add(end)

    breaks = sorted(breaks)

    # Créer segmentation pour segeval
    segments = []
    for i in range(len(breaks) - 1):
        segments.append(breaks[i + 1] - breaks[i])

    return ",".join(map(str, segments))


def calculate_metrics(annotation_seg: str, baseline_seg: str) -> Dict:
    """Calculer Pk et WindowDiff entre deux segmentations."""
    metrics = {}

    if not HAS_SEGEVAL:
        return {"error": "segeval not installed"}

    try:
        annotation_boundaries = [int(x) for x in annotation_seg.split(",")]
        baseline_boundaries = [int(x) for x in baseline_seg.split(",")]

        # Normaliser les longueurs
        min_len = min(len(annotation_boundaries), len(baseline_boundaries))
        annotation_boundaries = annotation_boundaries[:min_len]
        baseline_boundaries = baseline_boundaries[:min_len]

        # Pk
        pk_score = pk(annotation_boundaries, baseline_boundaries)
        metrics["pk"] = pk_score

        # WindowDiff
        wd_score = window_diff(annotation_boundaries, baseline_boundaries)
        metrics["windowdiff"] = wd_score

    except Exception as e:
        metrics["error"] = str(e)

    return metrics


def compute_f1_boundaries(ref_boundaries: List[Tuple[int, int]],
                          hyp_boundaries: List[Tuple[int, int]],
                          tolerance: int = 50) -> Dict:
    """Calculer F1 sur les frontières avec tolérance (tolerance chars).

    Une frontière est considérée correcte si elle est à tolerance chars de la référence.
    """
    ref_starts = {b[0] for b in ref_boundaries}
    ref_ends = {b[1] for b in ref_boundaries}

    hyp_starts = {b[0] for b in hyp_boundaries}
    hyp_ends = {b[1] for b in hyp_boundaries}

    # Matching starts avec tolérance
    matched_starts = 0
    for hyp_start in hyp_starts:
        for ref_start in ref_starts:
            if abs(hyp_start - ref_start) <= tolerance:
                matched_starts += 1
                break

    # Matching ends avec tolérance
    matched_ends = 0
    for hyp_end in hyp_ends:
        for ref_end in ref_ends:
            if abs(hyp_end - ref_end) <= tolerance:
                matched_ends += 1
                break

    # Précision et rappel
    precision_start = matched_starts / len(hyp_starts) if hyp_starts else 0
    recall_start = matched_starts / len(ref_starts) if ref_starts else 0

    precision_end = matched_ends / len(hyp_ends) if hyp_ends else 0
    recall_end = matched_ends / len(ref_ends) if ref_ends else 0

    # F1
    f1_start = 2 * (precision_start * recall_start) / (precision_start + recall_start) if (precision_start + recall_start) > 0 else 0
    f1_end = 2 * (precision_end * recall_end) / (precision_end + recall_end) if (precision_end + recall_end) > 0 else 0

    return {
        "f1_boundaries_start": f1_start,
        "f1_boundaries_end": f1_end,
        "f1_boundaries_avg": (f1_start + f1_end) / 2,
        "precision_start": precision_start,
        "recall_start": recall_start,
        "precision_end": precision_end,
        "recall_end": recall_end,
        "matched_starts": matched_starts,
        "total_ref_starts": len(ref_starts),
        "matched_ends": matched_ends,
        "total_ref_ends": len(ref_ends),
    }


def main():
    """Workflow principal de validation."""
    # Chemins
    annot_path = Path("data/processed/Kitab_Uqala_al_Majanin_annotated.json")
    text_path = Path("data/processed/0406IbnHabib_cleaned.txt")
    output_path = Path("results/validation_report.md")

    print("=" * 70)
    print("VALIDATION DE LA BASELINE CONTRE ANNOTATIONS MANUELLES")
    print("=" * 70)

    # 1. Charger annotations
    print(f"\n[1]  Chargement des annotations...")
    annot_data = load_annotations(annot_path)
    print(f"   [OK] Chargé {annot_data['total']} akhbars")

    # 2. Charger texte brut (déjà nettoyé)
    print(f"\n[2]  Chargement du texte brut nettoyé...")
    with open(text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print(f"   [OK] Texte: {len(raw_text)} caractères")

    # 3. Construire boundaries d'annotation
    print(f"\n[3]  Matching annotations avec fuzzy matching...")
    annot_boundaries, frontier_indices = build_annotation_boundaries(annot_data, raw_text)

    # 4. Convertir en segmentation
    print(f"\n[4]  Conversion en format de segmentation...")
    annot_seg = boundaries_to_segmentation(len(raw_text), annot_boundaries)
    print(f"   [OK] Segmentation annotation: {len(frontier_indices)} frontières")

    # 5. Exécuter baseline
    print(f"\n[5]  Exécution de la baseline...")
    print(f"   (ce peut prendre ~30 secondes...)")
    baseline_boundaries = run_baseline(text_path)
    baseline_seg = boundaries_to_segmentation(len(raw_text), baseline_boundaries)
    print(f"   [OK] Baseline: {len(baseline_boundaries)} segments")

    # 6. Calculer métriques
    print(f"\n[6]  Calcul des métriques...")

    # Métriques segeval
    segeval_metrics = calculate_metrics(annot_seg, baseline_seg)

    # F1 boundaries
    f1_metrics = compute_f1_boundaries(annot_boundaries, baseline_boundaries, tolerance=50)

    # 7. Générer rapport
    print(f"\n[7]  Génération du rapport...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Rapport de Validation : Baseline vs Annotations Manuelles\n\n")

        f.write("## Configuration\n")
        f.write(f"- **Annotations** : {annot_data['total']} akhbars du Kitab Uqala al-Majanin\n")
        f.write(f"- **Texte brut** : {len(raw_text):,} caractères (406IbnHabibNaysaburi)\n")
        f.write(f"- **Méthode** : Fuzzy matching (threshold=0.82) + segeval metrics\n\n")

        f.write("## Résultats de Matching\n")
        f.write(f"- **Akhbars trouvés** : {len(annot_boundaries)} / {annot_data['total']}\n")
        f.write(f"- **Couverture** : {100 * len(annot_boundaries) / annot_data['total']:.1f}%\n\n")

        f.write("## Métriques de Segmentation\n\n")

        if "error" in segeval_metrics:
            f.write(f"[!]  Erreur segeval: {segeval_metrics['error']}\n\n")
        else:
            f.write(f"### Pk (Probabilité d'erreur)\n")
            f.write(f"- **Valeur** : {segeval_metrics.get('pk', 'N/A'):.4f}\n")
            f.write(f"- **Interprétation** : Plus bas = mieux (0 = parfait)\n\n")

            f.write(f"### WindowDiff\n")
            f.write(f"- **Valeur** : {segeval_metrics.get('windowdiff', 'N/A'):.4f}\n")
            f.write(f"- **Interprétation** : Plus bas = mieux\n\n")

        f.write("### F1 sur Frontières (tolérance ±50 chars)\n")
        f1 = f1_metrics.get("f1_boundaries_avg", 0)
        f.write(f"- **F1 moyenne** : {f1:.4f}\n")
        f.write(f"- **F1 début de segment** : {f1_metrics.get('f1_boundaries_start', 0):.4f}\n")
        f.write(f"  - Précision : {f1_metrics.get('precision_start', 0):.4f} ({f1_metrics.get('matched_starts', 0)}/{f1_metrics.get('total_ref_starts', 0)})\n")
        f.write(f"  - Rappel : {f1_metrics.get('recall_start', 0):.4f}\n")
        f.write(f"- **F1 fin de segment** : {f1_metrics.get('f1_boundaries_end', 0):.4f}\n")
        f.write(f"  - Précision : {f1_metrics.get('precision_end', 0):.4f} ({f1_metrics.get('matched_ends', 0)}/{f1_metrics.get('total_ref_ends', 0)})\n")
        f.write(f"  - Rappel : {f1_metrics.get('recall_end', 0):.4f}\n\n")

        f.write("## Résumé\n")

        total_akhbars_ref = len(annot_boundaries)
        total_akhbars_hyp = len(baseline_boundaries)

        f.write(f"- **Référence (annotations)** : {total_akhbars_ref} akhbars\n")
        f.write(f"- **Hypothèse (baseline)** : {total_akhbars_hyp} akhbars\n")
        if total_akhbars_ref > 0:
            pct = 100 * (total_akhbars_hyp - total_akhbars_ref) / total_akhbars_ref
            f.write(f"- **Différence** : {total_akhbars_hyp - total_akhbars_ref} ({pct:+.1f}%)\n\n")
        else:
            f.write(f"- **Différence** : N/A (aucun akhbar trouv­ dans annotations)\n\n")

        if f1 > 0.9:
            verdict = "[OK] EXCELLENT"
        elif f1 > 0.8:
            verdict = "[OK] BON"
        elif f1 > 0.7:
            verdict = "[!]  ACCEPTABLE"
        else:
            verdict = "[ERR] FAIBLE"

        f.write(f"**Verdict** : {verdict}\n\n")

        f.write("## Notes\n")
        f.write("- Le fuzzy matching a une tolérance de 0.82 (82% de similarité)\n")
        f.write("- La F1 sur frontières accepte une tolérance de ±50 caractères\n")
        f.write("- Les akhbars très courts ou mal trouvés peuvent affecter les métriques\n")

    print(f"\n[OK] Rapport généré: {output_path}\n")

    # Afficher résumé
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Akhbars référence  : {total_akhbars_ref}")
    print(f"Akhbars baseline   : {total_akhbars_hyp}")
    print(f"Écart              : {total_akhbars_hyp - total_akhbars_ref} ({100 * (total_akhbars_hyp - total_akhbars_ref) / total_akhbars_ref:+.1f}%)")
    print(f"\nF1 frontières      : {f1:.4f}")
    if "pk" in segeval_metrics:
        print(f"Pk                 : {segeval_metrics['pk']:.4f}")
        print(f"WindowDiff         : {segeval_metrics['windowdiff']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
