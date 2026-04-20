#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cherche les akhbars annotés dans le texte nettoyé par substring simple.
Plus rapide que fuzzy matching pour diagnostiquer le problème d'alignement.
"""

import json
from pathlib import Path


def extract_akhbar_text(akhbar: dict) -> str:
    """Extraire le texte complet d'un akhbar."""
    segments = akhbar.get("content", {}).get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def main():
    """Chercher les akhbars dans le texte nettoyé."""
    # Charger annotations
    with open("data/processed/Kitab_Uqala_al_Majanin_annotated.json", 'r', encoding='utf-8') as f:
        annot = json.load(f)

    # Charger texte nettoyé
    with open("data/processed/0406IbnHabib_cleaned.txt", 'r', encoding='utf-8') as f:
        cleaned_text = f.read()

    print("=" * 70)
    print("RECHERCHE SIMPLE : SUBSTRING MATCH")
    print("=" * 70)

    found_count = 0
    not_found = []
    partial_found = []  # Akhbars trouvés partiellement

    akhbars = annot.get("akhbar", [])
    print(f"\nCherchant {len(akhbars)} akhbars...\n")

    for idx, akhbar in enumerate(akhbars):
        akhbar_num = akhbar.get("num", idx + 1)
        akhbar_text = extract_akhbar_text(akhbar)

        if not akhbar_text.strip():
            continue

        # Exact match
        if akhbar_text in cleaned_text:
            found_count += 1
            if found_count <= 5:  # Log les 5 premiers trouvés
                print(f"[OK] Akhbar {akhbar_num}: TROUVÉ (exact match, {len(akhbar_text)} chars)")
        else:
            # Chercher si au moins les 50 premiers caractères existent
            snippet = akhbar_text[:100]
            if snippet in cleaned_text:
                partial_found.append((akhbar_num, len(akhbar_text), 100))
            else:
                not_found.append((akhbar_num, len(akhbar_text)))

    print(f"\n{'=' * 70}")
    print(f"RÉSULTATS")
    print(f"{'=' * 70}")
    print(f"\nTrouvés (exact match): {found_count}/{len(akhbars)}")
    print(f"Partiels (début trouvé): {len(partial_found)}/{len(akhbars)}")
    print(f"Non trouvés: {len(not_found)}/{len(akhbars)}")

    if partial_found:
        print(f"\nExemples partiels trouvés (premiers 5):")
        for num, length, match_len in partial_found[:5]:
            print(f"  - Akhbar {num}: {match_len} chars trouvés sur {length}")

    if not_found:
        print(f"\nExemples NON trouvés (premiers 10):")
        for num, length in not_found[:10]:
            akhbar = next((a for a in akhbars if a.get('num') == num), None)
            if akhbar:
                text = extract_akhbar_text(akhbar)
                snippet = text[:80]
                print(f"  - Akhbar {num} ({length} chars): {repr(snippet)}...")

    print(f"\n{'=' * 70}")
    if found_count > 0:
        print(f"SUCCESS: {found_count} akhbars trouvés!")
    else:
        print(f"PROBLEME: Les annotations ne correspondent PAS au texte nettoyé")
        print(f"\nLe problème pourrait être:")
        print(f"  1. Les annotations sont basées sur une autre édition du texte")
        print(f"  2. Le texte a été très modifié (normalize différemment)")
        print(f"  3. Les annotations ne sont PAS complètes ou sont partielles")


if __name__ == "__main__":
    main()
