#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construire un corpus de référence à partir des annotations.

Extrait le texte arabe brut de chaque akhbar annoté et crée :
1. Un fichier texte continu avec tous les akhbars concaténés
2. Un fichier JSON avec les boundaries (positions de début/fin de chaque akhbar)
"""

import json
from pathlib import Path


def extract_akhbar_text(akhbar: dict) -> str:
    """Extraire le texte complet d'un akhbar."""
    segments = akhbar.get("content", {}).get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def main():
    """Construire le corpus de référence."""
    # Charger les annotations
    annot_path = Path("data/processed/Kitab_Uqala_al_Majanin_annotated.json")
    with open(annot_path, 'r', encoding='utf-8') as f:
        annot_data = json.load(f)

    akhbars = annot_data.get("akhbar", [])

    print("=" * 70)
    print("CONSTRUCTION DU CORPUS DE RÉFÉRENCE")
    print("=" * 70 + "\n")

    print(f"Traitant {len(akhbars)} akhbars annotés...\n")

    # Extraire les textes et les boundaries
    corpus_text = ""
    boundaries = []
    total_chars = 0

    for idx, akhbar in enumerate(akhbars):
        akhbar_num = akhbar.get("num", idx + 1)
        akhbar_text = extract_akhbar_text(akhbar)

        if not akhbar_text.strip():
            print(f"  [SKIP] Akhbar {akhbar_num}: texte vide")
            continue

        # Ajouter au corpus avec séparateur (double espace)
        start_pos = len(corpus_text)
        corpus_text += akhbar_text + "  "  # Double space comme séparateur
        end_pos = len(corpus_text) - 2  # Avant le séparateur

        # Enregistrer la boundary
        boundaries.append({
            "num": akhbar_num,
            "start": start_pos,
            "end": end_pos,
            "length": end_pos - start_pos,
            "text_length": len(akhbar_text)
        })

        if (idx + 1) % 100 == 0:
            print(f"  [OK] {idx + 1}/{len(akhbars)} akhbars traites")

    # Enlever les doubles espaces finaux
    corpus_text = corpus_text.rstrip()

    print(f"\n{'=' * 70}")
    print(f"RÉSULTATS")
    print(f"{'=' * 70}\n")

    print(f"Akhbars inclus: {len(boundaries)}/{len(akhbars)}")
    print(f"Longueur du corpus: {len(corpus_text):,} caractères")

    # Statistiques
    avg_length = sum(b["length"] for b in boundaries) / len(boundaries)
    print(f"Longueur moyenne par akhbar: {avg_length:.1f} caractères")

    # Sauvegarder le corpus
    corpus_path = Path("data/processed/kitab_uqala_reference_corpus.txt")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(corpus_text)
    print(f"\n[OK] Corpus sauvegardé: {corpus_path}")
    print(f"  Taille: {corpus_path.stat().st_size / 1024:.1f} KB")

    # Sauvegarder les boundaries
    boundaries_path = Path("data/processed/kitab_uqala_boundaries.json")
    with open(boundaries_path, 'w', encoding='utf-8') as f:
        json.dump(boundaries, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Boundaries sauvegardés: {boundaries_path}")

    # Statistiques sur les boundaries
    print(f"\nBoundaries:")
    print(f"  Premiers 3 akhbars:")
    for b in boundaries[:3]:
        print(f"    - Akhbar {b['num']:3d}: chars {b['start']:6d}-{b['end']:6d} (len={b['length']:4d})")

    print(f"\n{'=' * 70}")
    print(f"[OK] CORPUS DE RÉFÉRENCE CRÉÉ")
    print(f"{'=' * 70}\n")
    print(f"Prêt pour validation contre la baseline !")


if __name__ == "__main__":
    main()
