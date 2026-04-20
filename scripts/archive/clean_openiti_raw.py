#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nettoyage agressif du fichier OpenITI brut.

Enlève :
- En-têtes OpenITI (######OpenITI#, #META#...)
- Marqueurs de page (# PageV01P006)
- Marqueurs de manuscrit (ms001, ms002...)
- Tirets de continuation (~~ en début de ligne)
- Marqueurs de section (###)
- Lignes vides multiples
- Espaces inutiles au début/fin des lignes
"""

import re
from pathlib import Path
import unicodedata


def remove_arabic_diacritics(text: str) -> str:
    """Enlever les diacritiques arabes (tashkeel)."""
    arabic_diacritics = [
        '\u064B',  # FATHATAN
        '\u064C',  # DAMMATAN
        '\u064D',  # KASRATAN
        '\u064E',  # FATHA
        '\u064F',  # DAMMA
        '\u0650',  # KASRA
        '\u0651',  # SHADDA
        '\u0652',  # SUKUN
        '\u0653',  # MADDAH ABOVE
        '\u0654',  # HAMZA ABOVE
        '\u0655',  # HAMZA BELOW
        '\u0656',  # SUBSCRIPT ALEF
        '\u0657',  # INVERTED DAMMA
        '\u0658',  # MARK NOON GHUNNA
        '\u0670',  # SUPERSCRIPT ALEF
    ]
    for diacritic in arabic_diacritics:
        text = text.replace(diacritic, '')
    return text


def clean_openiti(text: str) -> str:
    """Nettoyer agressivement le texte OpenITI."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Enlever les métadonnées OpenITI
        if line.startswith('#META#') or line.startswith('######OpenITI'):
            continue

        # Enlever les marqueurs de page (# PageV01P006, etc.)
        if re.match(r'^#\s*PageV\d+P\d+', line):
            continue

        # Enlever les marqueurs de fin de header
        if '#META#Header#End#' in line:
            continue

        # Remplacer les tirets de continuation (~~ au début)
        line = re.sub(r'^~~\s*', '', line)
        line = re.sub(r'\s*~~\s*', ' ', line)

        # Enlever les marqueurs de manuscrit (ms001, ms002, etc.)
        line = re.sub(r'\s*ms\d+\s*', ' ', line)

        # Enlever les marqueurs de section (### au début de ligne)
        line = re.sub(r'^###\s*\|\s*', '', line)
        line = re.sub(r'^###\s*', '', line)

        # Enlever les # isolés au début de ligne (marqueurs de continuité)
        line = re.sub(r'^#\s+', '', line)

        # Normaliser les espaces
        line = re.sub(r'\s+', ' ', line).strip()

        # Ajouter seulement les lignes non vides
        if line:
            cleaned_lines.append(line)

    # Joindre avec des espaces simples
    text = ' '.join(cleaned_lines)

    # IMPORTANT : Enlever les diacritiques arabes pour normaliser
    text = remove_arabic_diacritics(text)

    # Normaliser les espaces multiples finaux
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    """Nettoyer le fichier OpenITI et sauvegarder."""
    input_path = Path("data/raw/0406IbnHabibNaysaburi.CuqalaMajanin.Shamela0001593-ara1")
    output_path = Path("data/processed/0406IbnHabib_cleaned.txt")

    print(f"[*] Lecture: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    input_size = len(text)
    print(f"    Taille avant: {input_size:,} caractères")

    print(f"[*] Nettoyage agressif...")
    cleaned_text = clean_openiti(text)

    output_size = len(cleaned_text)
    print(f"    Taille après: {output_size:,} caractères")
    print(f"    Réduction: {100 * (1 - output_size / input_size):.1f}%")

    print(f"[*] Sauvegarde: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"\n[OK] Nettoyage terminé")
    print(f"    Fichier: {output_path}")

    # Afficher un extrait
    print(f"\n[*] Aperçu (premier 500 caractères):")
    print("-" * 70)
    print(cleaned_text[:500])
    print("-" * 70)


if __name__ == "__main__":
    main()
