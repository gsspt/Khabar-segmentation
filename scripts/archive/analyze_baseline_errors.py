#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyser pourquoi la baseline sur-segmente sur le corpus de référence.

1. Exécuter la baseline
2. Identifier les verbes de transmission détectés
3. Comparer avec les ruptures attendues
4. Identifier les fausses ruptures
"""

import json
import re
from pathlib import Path
from baseline_isnad_segmentation import segment_akhbars, ISNAD_VERBS


def load_reference_data():
    """Charger le corpus et les boundaries."""
    with open("data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        boundaries = json.load(f)

    return corpus, boundaries


def analyze_detected_akhbars(corpus: str, akhbars: list, boundaries: list) -> dict:
    """
    Analyser les akhbars détectés pour comprendre les erreurs.
    """
    print("[*] Analyse des akhbars détectés...\n")

    # Statistiques
    stats = {
        "total_detected": len(akhbars),
        "total_expected": len(boundaries),
        "isnad_verbs_found": 0,
        "false_breaks": 0,
        "correct_matches": 0,
        "examples_oversegment": [],
        "most_common_verbs": {},
    }

    # Analyser chaque akhbar détecté
    for i, akh in enumerate(akhbars[:50]):  # Analyser les 50 premiers
        isnad = akh.get('isnad', '').strip()
        khabar = akh.get('khabar', '').strip()[:100]  # Premiers 100 chars du khabar

        if isnad:
            stats["isnad_verbs_found"] += 1

            # Extraire le verbe de transmission
            verb_match = None
            for verb in ISNAD_VERBS:
                if verb in isnad:
                    verb_match = verb
                    if verb not in stats["most_common_verbs"]:
                        stats["most_common_verbs"][verb] = 0
                    stats["most_common_verbs"][verb] += 1
                    break

            # Vérifier si c'est une vraie rupture ou une fausse
            # Chercher le texte complet dans le corpus
            full_akh = (isnad + " " + khabar).strip()
            is_real = False
            for boundary in boundaries:
                boundary_text = corpus[boundary['start']:boundary['end']]
                if isnad in boundary_text:
                    is_real = True
                    break

            if not is_real and len(stats["examples_oversegment"]) < 20:
                stats["examples_oversegment"].append({
                    "num": i + 1,
                    "isnad": isnad[:80],
                    "khabar": khabar[:80],
                    "verb": verb_match
                })
                stats["false_breaks"] += 1

    # Trier les verbes par fréquence
    stats["most_common_verbs"] = dict(sorted(
        stats["most_common_verbs"].items(),
        key=lambda x: x[1],
        reverse=True
    ))

    return stats


def main():
    """Analyser les erreurs de segmentation."""
    print("=" * 70)
    print("ANALYSE DES ERREURS DE BASELINE")
    print("=" * 70 + "\n")

    # 1. Charger les données
    print("[1] Chargement du corpus de référence...")
    corpus, boundaries = load_reference_data()
    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Boundaries référence: {len(boundaries)}\n")

    # 2. Exécuter la baseline
    print("[2] Exécution de la baseline...")
    akhbars_detected = segment_akhbars(corpus)
    print(f"    Akhbars détectés: {len(akhbars_detected)}\n")

    # 3. Analyser les erreurs
    print("[3] Analyse des erreurs...\n")
    stats = analyze_detected_akhbars(corpus, akhbars_detected, boundaries)

    # 4. Afficher les résultats
    print("=" * 70)
    print("RÉSULTATS DE L'ANALYSE")
    print("=" * 70 + "\n")

    print(f"Statistiques globales:")
    print(f"  Akhbars détectés: {stats['total_detected']}")
    print(f"  Akhbars attendus: {stats['total_expected']}")
    print(f"  Verbes de transmission trouvés: {stats['isnad_verbs_found']}")
    print(f"  Ruptures fausses (en sample): {stats['false_breaks']}\n")

    print(f"Verbes les plus frequents:")
    for verb, count in list(stats["most_common_verbs"].items())[:10]:
        print(f"  Verbe: {count} fois")

    print(f"\nExemples de fausses ruptures (non validees par les boundaries):")
    for ex in stats["examples_oversegment"][:10]:
        print(f"\n  Akhbar {ex['num']}:")
        print(f"    Has Isnad: yes")
        print(f"    Verb found: {bool(ex['verb'])}")
        print(f"    Khabar length: {len(ex['khabar'])} chars")

    # 5. Diagnostiquer le problème
    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC")
    print("=" * 70 + "\n")

    ratio = stats['total_detected'] / stats['total_expected']
    print(f"La baseline détecte {ratio:.2f}x plus de ruptures que prévu.\n")

    if stats['isnad_verbs_found'] > stats['total_expected']:
        print(f"PROBLEME IDENTIFIE: Trop de verbes de transmission détectés")
        print(f"  - Attendu: ~{stats['total_expected']} isnad")
        print(f"  - Détecté: {stats['isnad_verbs_found']}")
        print(f"\nCauses possibles:")
        print(f"  1. Les verbes 'قال' et 'ذكر' sont trop laxes")
        print(f"  2. Le corpus de référence contient du texte qui ressemble à des isnads")
        print(f"  3. La logique de détection d'isnad est trop permissive\n")

    # Sauvegarder le rapport
    report_path = Path("results/baseline_error_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Analyse des Erreurs de Baseline\n\n")
        f.write("## Statistiques\n")
        f.write(f"- Akhbars détectés: {stats['total_detected']}\n")
        f.write(f"- Akhbars attendus: {stats['total_expected']}\n")
        f.write(f"- Verbes détectés: {stats['isnad_verbs_found']}\n")
        f.write(f"- Ratio: {ratio:.2f}x\n\n")

        f.write("## Verbes les plus fréquents\n")
        for verb, count in list(stats["most_common_verbs"].items())[:10]:
            f.write(f"- '{verb}': {count}\n")

        f.write("\n## Conclusion\n")
        if ratio > 2:
            f.write(f"La baseline sur-segmente **{ratio:.1f}x** trop.\n")
            f.write(f"Les verbes de transmission sont détectés trop largement.\n")

    print(f"Rapport sauvegardé: {report_path}\n")


if __name__ == "__main__":
    main()
