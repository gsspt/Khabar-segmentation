#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline de segmentation des akhbars basée sur la détection d'isnad.

Approche :
1. Détecte le début d'une chaîne d'isnad (verbes de transmission)
2. Identifie la transition vers le récit (passage à la 1ère personne, action narrative)
3. Détecte la fin du khabar (ponctuation majeure, nouveau isnad)

Usage:
    python scripts/baseline_isnad_segmentation.py --input data/raw/text.txt [--output results/]
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import sys


# ============================================================================
# DICTIONNAIRES LINGUISTIQUES ARABES
# ============================================================================

# Verbes de transmission fiables (hautes confiance)
ISNAD_VERBS_RELIABLE = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها', 'حدثهم', 'حدثهن',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها', 'أخبرهم', 'أخبرهن',
    'روى', 'رويت', 'روينا',
    'أنبأ', 'أنبأني', 'أنبأنا',
    'سمعت', 'سمعنا',
}

# Verbes ambigus (trop génériques, créent trop de faux positifs)
ISNAD_VERBS_AMBIGUOUS = {
    'قال', 'قالت', 'قالوا', 'قلن',  # Trop génériques - peuvent être du discours rapporté
    'ذكر', 'ذكرت', 'ذكرنا',  # "mentionné" - trop vague
    'نقل', 'نقلت', 'نقلنا',  # "rapporté" - trop vague
    'وجدت', 'وجدنا',  # "trouvé" - trop vague
    'قرأت', 'قرأنا',  # "lu" - trop vague
}

# Tous les verbes (pour compatibilité backward)
ISNAD_VERBS = ISNAD_VERBS_RELIABLE | ISNAD_VERBS_AMBIGUOUS

# Par défaut, utiliser seulement les verbes fiables
ISNAD_VERBS_STRICT = ISNAD_VERBS_RELIABLE

# Marqueurs de transmission (connecteurs)
TRANSMISSION_MARKERS = {
    'عن',      # de (transmission)
    'من',      # de (source)
    'أن',      # que
    'إن',      # que
    'لما',     # quand
    'قال',     # dit
    'فقال',    # et dit
}

# Verbes d'action narrative (marquent le début du récit)
NARRATIVE_VERBS = {
    'دخل', 'دخلت', 'دخلنا',
    'خرج', 'خرجت', 'خرجنا',
    'ذهب', 'ذهبت', 'ذهبنا',
    'جاء', 'جاءت', 'جاءنا',
    'قال', 'قالت', 'قالوا',  # (quand c'est le narrateur)
    'فعل', 'فعلت', 'فعلنا',
    'رأى', 'رأيت', 'رأينا',
    'سمع', 'سمعت', 'سمعنا',
    'اتخذ', 'اتخذت', 'اتخذنا',
    'أخذ', 'أخذت', 'أخذنا',
    'أراد', 'أرادت', 'أردنا',
    'أسلم', 'أسلمت', 'أسلمنا',
    'أسلف', 'أسلفت', 'أسلفنا',
    'أصاب', 'أصابت', 'أصابنا',
    'كان', 'كانت', 'كانوا',
}

# Pronoms de 1ère personne (singulier et pluriel)
FIRST_PERSON_PRONOUNS = {
    'ي', 'ني', 'نا', 'ن',     # suffixes
    'أنا', 'نحن',             # pronoms indépendants
}

# Marqueurs de fin de khabar potentielle
KHABAR_END_MARKERS = {
    '۔',      # point final arabe
    '.',      # point
    '،',      # virgule arabe
    ';',      # point-virgule
}


# ============================================================================
# PREPROCESSING ET TOKENISATION
# ============================================================================

def normalize_arabic_text(text: str) -> str:
    """Normalise le texte arabe pour le traitement."""
    # Supprimer les diacritiques
    diacritics = 'ًٌٍَُِّْ'
    text = ''.join(c for c in text if c not in diacritics)

    # Normaliser les variantes de hamza
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # Normaliser les taa
    text = text.replace('ة', 'ه')

    return text


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenisation simple basée sur l'espace et la ponctuation.
    Pour une meilleure tokenisation, utiliser camel-tools ou farasa.
    """
    # Ajouter espaces autour de la ponctuation
    text = re.sub(r'([،؛۔\.\?!])', r' \1 ', text)

    # Splitter par espaces
    tokens = text.split()

    return tokens


def extract_root(word: str) -> str:
    """
    Extraction très simplifiée de racine.
    Pour un usage réel, utiliser un vrai stemmer arabe.
    """
    # Supprime les affixes courants
    word = re.sub(r'^(ال|و|ف|ب|ل|ك)', '', word)
    word = re.sub(r'(ها|ه|ة|ين|ون|ان|ات|نا|ني|ها)$', '', word)
    return word


# ============================================================================
# DÉTECTION DE STRUCTURE ISNAD ET TRANSITIONS
# ============================================================================

def is_isnad_verb(word: str, strict: bool = True) -> bool:
    """
    Vérife si le mot est un verbe de transmission.

    Args:
        word: Mot à vérifier
        strict: Si True, utiliser seulement les verbes fiables.
                Si False, inclure aussi les verbes ambigus.
    """
    normalized = normalize_arabic_text(word)
    if strict:
        return normalized in ISNAD_VERBS_STRICT
    else:
        return normalized in ISNAD_VERBS


def is_first_person_form(word: str) -> bool:
    """Vérifie si le mot contient un marqueur de 1ère personne."""
    normalized = normalize_arabic_text(word)

    # Vérifier les suffixes
    for pronoun in FIRST_PERSON_PRONOUNS:
        if normalized.endswith(pronoun):
            return True

    # Vérifier les pronoms indépendants
    if normalized in FIRST_PERSON_PRONOUNS:
        return True

    return False


def is_narrative_verb(word: str) -> bool:
    """Vérifie si le mot est un verbe d'action narrative."""
    normalized = normalize_arabic_text(word)
    root = extract_root(normalized)

    # Chercher le verbe exact ou sa racine
    for verb in NARRATIVE_VERBS:
        if normalized.startswith(verb) or root == extract_root(verb):
            return True

    return False


def detect_isnad_span(tokens: List[str], start_idx: int) -> Tuple[int, int]:
    """
    Détecte la fin d'une chaîne d'isnad à partir d'un indice de départ.
    Retourne (début, fin) de l'isnad.

    L'isnad se termine généralement quand:
    - On rencontre un verbe de narration suivi de 1ère personne
    - On rencontre une ponctuation majeure
    - On rencontre un nouveau verbe d'isnad
    """
    end_idx = start_idx
    first_person_idx = None

    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        normalized = normalize_arabic_text(token)

        # Si on trouve un verbe de narration suivi de 1ère personne
        if is_narrative_verb(token):
            if first_person_idx is None and i > start_idx:
                first_person_idx = i

        # Si on trouve une forme de 1ère personne
        if is_first_person_form(token):
            if first_person_idx is None:
                first_person_idx = i

        # Transition détectée: verbe de narration + 1ère personne détectée
        if first_person_idx is not None and is_narrative_verb(token):
            end_idx = i
            break

        # Alternative: ponctuation majeure
        if token in KHABAR_END_MARKERS and i > start_idx + 2:
            end_idx = i
            break

    return start_idx, end_idx


def detect_khabar_end(tokens: List[str], start_idx: int) -> int:
    """
    Détecte la fin d'un khabar (récit) à partir d'un indice de départ.

    Le khabar se termine quand:
    - On rencontre un nouveau verbe d'isnad (nouvel akhbar)
    - On rencontre une très longue succession de mots sans 1ère personne
    - On atteint la fin du texte
    """
    end_idx = len(tokens) - 1

    for i in range(start_idx, len(tokens)):
        token = tokens[i]

        # Nouveau isnad détecté = fin du khabar précédent
        if is_isnad_verb(token) and i > start_idx + 5:
            # Chercher le dernier point/ponctuation avant
            for j in range(i - 1, start_idx, -1):
                if tokens[j] in KHABAR_END_MARKERS:
                    end_idx = j
                    break
            else:
                end_idx = i - 1
            break

    return end_idx


# ============================================================================
# SEGMENTATION PRINCIPALE
# ============================================================================

def segment_akhbars(text: str) -> List[Dict]:
    """
    Segmente le texte en akhbars selon la structure isnad→récit.

    Retourne une liste de dictionnaires avec:
    - text: le contenu du khabar
    - start_char: indice de départ dans le texte original
    - end_char: indice de fin dans le texte original
    - type: "ISNAD" ou "KHABAR" ou "MIXED"
    - isnad_end: indice où l'isnad se termine
    """
    # Normalisation et tokenisation
    normalized = normalize_arabic_text(text)
    tokens = simple_tokenize(normalized)

    # Mapper les tokens aux indices de caractères du texte original
    # (simplifié: on recalcule les positions)

    akhbars = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # Chercher un début d'isnad (verbe de transmission)
        if is_isnad_verb(token):
            khabar_start = i

            # Détecter la fin de l'isnad
            isnad_start, isnad_end = detect_isnad_span(tokens, i)

            # Détecter la fin du khabar complet
            khabar_end = detect_khabar_end(tokens, isnad_end)

            # Assembler le texte (approximé avec les tokens)
            isnad_text = ' '.join(tokens[isnad_start:isnad_end + 1])
            khabar_text = ' '.join(tokens[isnad_end + 1:khabar_end + 1])
            full_text = ' '.join(tokens[isnad_start:khabar_end + 1])

            akhbars.append({
                'text': full_text,
                'isnad': isnad_text,
                'khabar': khabar_text,
                'start_token': isnad_start,
                'end_token': khabar_end,
                'isnad_end_token': isnad_end,
                'type': 'ISNAD_KHABAR',
            })

            i = khabar_end + 1
        else:
            i += 1

    return akhbars


# ============================================================================
# ANALYSE ET RAPPORT
# ============================================================================

def analyze_segmentation(akhbars: List[Dict], text: str) -> Dict:
    """Analyse les résultats de segmentation."""
    coverage = sum(len(ak['text'].split()) for ak in akhbars)
    total_words = len(simple_tokenize(text))

    isnad_lengths = [len(ak['isnad'].split()) for ak in akhbars]
    khabar_lengths = [len(ak['khabar'].split()) for ak in akhbars]

    return {
        'total_akhbars': len(akhbars),
        'coverage_words': coverage,
        'total_words': total_words,
        'coverage_pct': (coverage / total_words * 100) if total_words > 0 else 0,
        'avg_isnad_length': sum(isnad_lengths) / len(isnad_lengths) if isnad_lengths else 0,
        'avg_khabar_length': sum(khabar_lengths) / len(khabar_lengths) if khabar_lengths else 0,
        'min_isnad': min(isnad_lengths) if isnad_lengths else 0,
        'max_isnad': max(isnad_lengths) if isnad_lengths else 0,
    }


def print_report(analysis: Dict, target: int = 613):
    """Affiche un rapport d'analyse."""
    print("\n" + "=" * 70)
    print("RAPPORT DE SEGMENTATION DES AKHBARS")
    print("=" * 70)
    print(f"Nombre total d'akhbars detectes: {analysis['total_akhbars']}")
    print(f"Cible attendue: {target}")
    print(f"Ecart: {abs(analysis['total_akhbars'] - target)} ({((analysis['total_akhbars'] - target) / target * 100):.1f}%)")
    print()
    print(f"Couverture du texte: {analysis['coverage_words']} / {analysis['total_words']} mots ({analysis['coverage_pct']:.1f}%)")
    print()
    print("Longueurs moyennes:")
    print(f"  - Isnad: {analysis['avg_isnad_length']:.1f} mots (min: {analysis['min_isnad']}, max: {analysis['max_isnad']})")
    print(f"  - Khabar (recit): {analysis['avg_khabar_length']:.1f} mots")
    print("=" * 70 + "\n")


def sample_akhbars(akhbars: List[Dict], n: int = 5):
    """Affiche des exemples d'akhbars segmentés."""
    print(f"\n[EXEMPLES] Des {min(n, len(akhbars))} premiers akhbars:\n")

    for idx, ak in enumerate(akhbars[:n], 1):
        try:
            print(f"--- AKHBAR {idx} ---")
            print(f"Isnad ({len(ak['isnad'].split())} mots): [Texte arabe]")
            print(f"Khabar ({len(ak['khabar'].split())} mots): [Texte arabe]")
            print()
        except UnicodeEncodeError:
            print(f"--- AKHBAR {idx} ---")
            print(f"Isnad ({len(ak['isnad'].split())} mots): [Texte arabe non affichable]")
            print(f"Khabar ({len(ak['khabar'].split())} mots): [Texte arabe non affichable]")
            print()


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_text_from_file(filepath: str) -> str:
    """Charge un texte depuis un fichier."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_text_from_openiti(book_name: str) -> str:
    """
    Tente de charger un texte depuis OpenITI (nécessite internet).
    Format: "Aqala_al_Majanin" ou similaire.
    """
    try:
        import urllib.request

        # OpenITI structure: https://raw.githubusercontent.com/OpenITI/
        # Exemple: corpus/0300AH/Aqala/Aqala.Majanin_txt_NORMALIZED.txt
        base_url = "https://raw.githubusercontent.com/OpenITI/RELEASED-TEXTS/master/"

        # Chercher le texte (approche simplifiée)
        urls = [
            f"{base_url}texts_normalized/{book_name}.txt",
            f"{base_url}data/{book_name}/{book_name}.txt",
        ]

        for url in urls:
            try:
                print(f"📡 Tentative de chargement depuis: {url}")
                with urllib.request.urlopen(url, timeout=10) as response:
                    text = response.read().decode('utf-8')
                    print(f"[OK] Texte charge avec succes ({len(text)} caracteres)")
                    return text
            except:
                continue

        raise Exception(f"Impossible de trouver '{book_name}' sur OpenITI")

    except ImportError:
        raise Exception("urllib non disponible pour télécharger depuis OpenITI")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline de segmentation des akhbars basée sur isnad"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Fichier texte ou nom du livre OpenITI (ex: Aqala.Majanin)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/segmentation_baseline.txt',
        help='Fichier de sortie pour les résultats'
    )
    parser.add_argument(
        '--target',
        type=int,
        default=613,
        help='Nombre cible d\'akhbars attendus'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Nombre d\'exemples à afficher'
    )

    args = parser.parse_args()

    # Charger le texte
    if not args.input:
        print("[ERREUR] --input requis (fichier local ou nom OpenITI)")
        print("\nUsage:")
        print("  python scripts/baseline_isnad_segmentation.py --input 'path/to/text.txt'")
        print("  python scripts/baseline_isnad_segmentation.py --input 'Aqala.Majanin'  # depuis OpenITI")
        sys.exit(1)

    if Path(args.input).exists():
        print(f"[FICHIER] Chargement du fichier: {args.input}")
        text = load_text_from_file(args.input)
    else:
        print(f"[OpenITI] Tentative de chargement depuis OpenITI: {args.input}")
        text = load_text_from_openiti(args.input)

    print(f"[OK] Texte charge: {len(text)} caracteres, {len(simple_tokenize(text))} mots")

    # Segmenter
    print("\n[TRAITEMENT] Segmentation en cours...")
    akhbars = segment_akhbars(text)

    # Analyser
    analysis = analyze_segmentation(akhbars, text)

    # Afficher le rapport
    print_report(analysis, args.target)

    # Afficher des exemples
    if akhbars:
        sample_akhbars(akhbars, args.samples)

    # Sauvegarder les résultats
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RÉSULTATS DE SEGMENTATION DES AKHBARS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Nombre d'akhbars: {analysis['total_akhbars']} / {args.target} (cible)\n")
        f.write(f"Couverture: {analysis['coverage_pct']:.1f}%\n")
        f.write(f"Longueur moyenne isnad: {analysis['avg_isnad_length']:.1f} mots\n")
        f.write(f"Longueur moyenne khabar: {analysis['avg_khabar_length']:.1f} mots\n\n")

        for idx, ak in enumerate(akhbars, 1):
            f.write(f"\n--- AKHBAR {idx} ---\n")
            f.write(f"ISNAD:\n{ak['isnad']}\n\n")
            f.write(f"KHABAR:\n{ak['khabar']}\n")

    print(f"[OK] Resultats sauvegardes dans: {output_path}")


if __name__ == '__main__':
    main()
