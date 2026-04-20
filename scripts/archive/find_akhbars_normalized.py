#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cherche les akhbars annotés dans le texte nettoyé avec normalisation de diacritiques.
"""

import json
from pathlib import Path


def remove_arabic_diacritics(text: str) -> str:
    """Enlever les diacritiques arabes."""
    arabic_diacritics = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
        '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
        '\u0655', '\u0656', '\u0657', '\u0658', '\u0670',
    ]
    for diacritic in arabic_diacritics:
        text = text.replace(diacritic, '')
    return text


def extract_akhbar_text(akhbar: dict) -> str:
    """Extraire le texte complet d'un akhbar."""
    segments = akhbar.get("content", {}).get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def normalize_for_search(text: str) -> str:
    """Normaliser le texte pour la recherche."""
    # Enlever diacritiques
    text = remove_arabic_diacritics(text)
    # Normaliser espaces
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    """Chercher les akhbars avec normalisation."""
    # Charger annotations
    with open("data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
        annot = json.load(f)

    # Charger texte nettoyé
    with open("data/processed/0406IbnHabib_cleaned.txt", 'r', encoding='utf-8') as f:
        cleaned_text = f.read()

    # Normaliser le texte
    cleaned_norm = normalize_for_search(cleaned_text)

    print("=" * 70)
    print("RECHERCHE AVEC NORMALISATION DE DIACRITIQUES")
    print("=" * 70)

    found_exact = 0
    found_partial = 0
    not_found = []

    akhbars = annot.get("akhbar", [])
    print(f"\nCherchant {len(akhbars)} akhbars...\n")

    for idx, akhbar in enumerate(akhbars):
        akhbar_num = akhbar.get("num", idx + 1)
        akhbar_text = extract_akhbar_text(akhbar)

        if not akhbar_text.strip():
            continue

        # Normaliser le texte de l'akhbar
        akhbar_norm = normalize_for_search(akhbar_text)

        # Exact match
        if akhbar_norm in cleaned_norm:
            found_exact += 1
            if found_exact <= 10:  # Log les 10 premiers trouvés
                print(f"[OK] Akhbar {akhbar_num}: TROUVÉ")
        else:
            # Chercher si au moins les 100 premiers caractères existent
            snippet = akhbar_norm[:100]
            if snippet and snippet in cleaned_norm:
                found_partial += 1
                if found_partial <= 5:
                    print(f"[PARTIAL] Akhbar {akhbar_num}: début trouvé")
            else:
                not_found.append(akhbar_num)

    print(f"\n{'=' * 70}")
    print(f"RÉSULTATS")
    print(f"{'=' * 70}")
    print(f"\nExact matches: {found_exact}/{len(akhbars)}")
    print(f"Partial (début trouvé): {found_partial}/{len(akhbars)}")
    print(f"Non trouvés: {len(not_found)}/{len(akhbars)}")

    if found_exact > 0:
        print(f"\n✓ SUCCESS: {found_exact} akhbars trouvés exactement!")
        if found_exact == len(akhbars):
            print(f"✓ PARFAIT: Tous les akhbars ont été alignés!")
    else:
        print(f"\n✗ Aucun match exact trouvé")
        if found_partial > 0:
            print(f"  Mais {found_partial} akhbars partiellement trouvés")
            print(f"  → Problème possible: diacritiques résiduels, normalisation différente, ou texte modifié")


if __name__ == "__main__":
    main()
