#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tester la baseline améliorée sur le fichier OpenITI nettoyé.
Celui-ci a plus de structure que le corpus de référence.
"""

import json
from pathlib import Path
from baseline_isnad_segmentation import segment_akhbars


def main():
    """Tester sur le fichier nettoyé."""
    print("=" * 70)
    print("TEST BASELINE AMELIOREE SUR FICHIER OPENITI NETTOYÉ")
    print("=" * 70 + "\n")

    # 1. Charger le fichier nettoyé
    print("[1] Chargement du fichier nettoyé...")
    cleaned_path = Path("data/processed/0406IbnHabib_cleaned.txt")
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        cleaned_text = f.read()
    print(f"    Longueur: {len(cleaned_text):,} caractères\n")

    # 2. Charger les boundaries de référence
    print("[2] Chargement des boundaries de référence...")
    boundaries_path = Path("data/processed/kitab_uqala_boundaries.json")
    with open(boundaries_path, 'r', encoding='utf-8') as f:
        ref_boundaries = json.load(f)
    print(f"    Boundaries: {len(ref_boundaries)} akhbars\n")

    # 3. Exécuter la baseline
    print("[3] Execution de la baseline amelioree...")
    akhbars_detected = segment_akhbars(cleaned_text)
    print(f"    Akhbars detectes: {len(akhbars_detected)}\n")

    # 4. Comparaison simple des nombres
    print("[4] Comparaison\n")
    diff = len(akhbars_detected) - len(ref_boundaries)
    pct = 100 * diff / len(ref_boundaries)

    print(f"    Reference: {len(ref_boundaries)} akhbars")
    print(f"    Baseline:  {len(akhbars_detected)} akhbars")
    print(f"    Écart:     {diff:+d} ({pct:+.1f}%)\n")

    # 5. Analyser les akhbars détectés
    print("[5] Analyse des akhbars detectes\n")

    # Statistiques
    akhbars_with_isnad = sum(1 for a in akhbars_detected if a.get('isnad', '').strip())
    akhbars_with_khabar = sum(1 for a in akhbars_detected if a.get('khabar', '').strip())
    avg_isnad_len = sum(len(a.get('isnad', '')) for a in akhbars_detected) / len(akhbars_detected) if akhbars_detected else 0
    avg_khabar_len = sum(len(a.get('khabar', '')) for a in akhbars_detected) / len(akhbars_detected) if akhbars_detected else 0

    print(f"    Akhbars avec isnad: {akhbars_with_isnad}/{len(akhbars_detected)}")
    print(f"    Akhbars avec khabar: {akhbars_with_khabar}/{len(akhbars_detected)}")
    print(f"    Longueur moyenne isnad: {avg_isnad_len:.1f} caractères")
    print(f"    Longueur moyenne khabar: {avg_khabar_len:.1f} caractères\n")

    # 6. Comparer avec les boundaries de référence
    print("[6] Analyse de la couverture\n")

    # Chercher combien d'akhbars de référence sont "couverts" par la baseline
    ref_text_covered = 0
    for ref_b in ref_boundaries:
        ref_text = cleaned_text[ref_b['start']:ref_b['end']]

        # Chercher si cet akhbar est trouvé dans les akhbars détectés
        for det_a in akhbars_detected:
            det_text = (det_a.get('isnad', '') + ' ' + det_a.get('khabar', '')).strip()
            if len(ref_text) > 50 and ref_text[:50] in cleaned_text:
                if det_text and ref_text[:100] in det_text:
                    ref_text_covered += 1
                    break

    coverage_pct = 100 * ref_text_covered / len(ref_boundaries) if ref_boundaries else 0
    print(f"    Akhbars de référence couverts: {ref_text_covered}/{len(ref_boundaries)} ({coverage_pct:.1f}%)\n")

    # 7. Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70 + "\n")

    if abs(pct) < 10:
        verdict = "[BON]"
        explanation = "L'écart est acceptable. La baseline fonctionne bien sur ce fichier."
    elif abs(pct) < 20:
        verdict = "[ACCEPTABLE]"
        explanation = "L'écart est modéré. À améliorer."
    else:
        verdict = "[FAIBLE]"
        explanation = "L'écart est trop élevé. Besoin d'amélioration."

    print(f"Verdict: {verdict}")
    print(f"Explication: {explanation}\n")

    print(f"Akhbars detectes: {len(akhbars_detected)}/{len(ref_boundaries)} ({100*len(akhbars_detected)/len(ref_boundaries):.1f}%)")
    print(f"Écart: {diff:+d} ({pct:+.1f}%)")
    print(f"Couverture: {coverage_pct:.1f}%\n")

    # Sauvegarder le rapport
    report_path = Path("results/baseline_on_cleaned_test.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Test sur Fichier OpenITI Nettoyé\n\n")
        f.write("## Configuration\n")
        f.write(f"- Fichier: 0406IbnHabib_cleaned.txt\n")
        f.write(f"- Taille: {len(cleaned_text):,} caractères\n")
        f.write(f"- Baseline: version améliorée (verbes fiables uniquement)\n\n")

        f.write("## Résultats\n")
        f.write(f"- Akhbars détectés: {len(akhbars_detected)}\n")
        f.write(f"- Akhbars attendus: {len(ref_boundaries)}\n")
        f.write(f"- Écart: {diff:+d} ({pct:+.1f}%)\n\n")

        f.write("## Statistiques\n")
        f.write(f"- Avec isnad: {akhbars_with_isnad}/{len(akhbars_detected)}\n")
        f.write(f"- Avec khabar: {akhbars_with_khabar}/{len(akhbars_detected)}\n")
        f.write(f"- Longueur moyenne isnad: {avg_isnad_len:.1f} chars\n")
        f.write(f"- Longueur moyenne khabar: {avg_khabar_len:.1f} chars\n\n")

        f.write("## Verdict\n")
        f.write(f"**{verdict}** - {explanation}\n")

    print(f"Rapport: {report_path}\n")


if __name__ == "__main__":
    main()
