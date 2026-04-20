#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline v2 avec filtrage et heuristiques améliorées.

Améliorations:
1. Filtre les isnads très courts (< MIN_ISNAD_LENGTH)
2. Utilise un scoring pour prioriser les verbes "forts" (حدثنا, أخبرنا)
3. Détecte mieux les transitions isnad→khabar
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_ISNAD_LENGTH = 6  # Seuil minimum de longueur d'isnad (en mots)
STRONG_TRANSMISSION_VERBS = {
    'حدثنا', 'حدثني', 'حدثه', 'حدثها',
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرها',
    'أنبأنا', 'أنبأني', 'أنبأه', 'أنبأها',
}

WEAK_TRANSMISSION_VERBS = {
    'قال', 'قالت', 'قالوا',
    'ذكر', 'ذكرت', 'ذكرنا',
    'روى', 'رويت', 'روينا',
    'نقل', 'نقلت', 'نقلنا',
    'سمعت', 'سمعنا',
}

NARRATIVE_VERBS = {
    'دخل', 'دخلت', 'دخلنا',
    'خرج', 'خرجت', 'خرجنا',
    'ذهب', 'ذهبت', 'ذهبنا',
    'جاء', 'جاءت', 'جاءنا',
    'فعل', 'فعلت', 'فعلنا',
    'رأى', 'رأيت', 'رأينا',
    'اتخذ', 'اتخذت', 'اتخذنا',
    'أخذ', 'أخذت', 'أخذنا',
    'أراد', 'أرادت', 'أردنا',
    'كان', 'كانت', 'كانوا',
}

FIRST_PERSON_MARKERS = {'ي', 'ني', 'نا', 'ن', 'أنا', 'نحن', 'دخلت', 'رأيت', 'سمعت'}


def normalize_arabic_text(text: str) -> str:
    """Normalise le texte arabe."""
    diacritics = 'ًٌٍَُِّْ'
    text = ''.join(c for c in text if c not in diacritics)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    return text


def simple_tokenize(text: str) -> List[str]:
    """Tokenise simplement le texte."""
    text = re.sub(r'([،؛۔\.\?!])', r' \1 ', text)
    return text.split()


def is_strong_transmission_verb(word: str) -> bool:
    """Verbe fort de transmission (confiance élevée)."""
    normalized = normalize_arabic_text(word)
    return normalized in STRONG_TRANSMISSION_VERBS


def is_transmission_verb(word: str) -> bool:
    """Verbe de transmission (fort ou faible)."""
    normalized = normalize_arabic_text(word)
    return normalized in STRONG_TRANSMISSION_VERBS or normalized in WEAK_TRANSMISSION_VERBS


def is_narrative_verb(word: str) -> bool:
    """Verbe d'action narrative."""
    normalized = normalize_arabic_text(word)
    return any(normalized.startswith(v) for v in NARRATIVE_VERBS)


def is_proper_noun(word: str) -> bool:
    """Heuristique simpliste: mot long ou contenant des patterns de noms arabes."""
    normalized = normalize_arabic_text(word)
    # Noms propres typiquement plus longs et contiennent certains patterns
    return len(normalized) > 2 and not any(c in normalized for c in 'اله')


def detect_isnad_span_v2(tokens: List[str], start_idx: int) -> Tuple[int, int]:
    """
    Détecte la fin d'une chaîne d'isnad avec heuristiques améliorées.
    Retourne (début, fin) ou None si l'isnad est trop court.
    """
    end_idx = start_idx
    transition_detected = False

    for i in range(start_idx, min(len(tokens), start_idx + 100)):  # Max 100 tokens par isnad
        token = tokens[i]

        # Chercher une transition: verbe narratif suivi de 1ère personne
        if is_narrative_verb(token) and i > start_idx + 2:
            transition_detected = True
            end_idx = i
            break

        # Ou: nouveau verbe de transmission fort = fin de l'isnad
        if is_strong_transmission_verb(token) and i > start_idx + 1:
            end_idx = i - 1
            break

        # Ou: verbe faible suivi d'un nom propre multiple (pattern: "قال X قال Y")
        if token == 'قال' and i > start_idx + 3:
            # Regarder avant et après
            if i + 2 < len(tokens):
                next_pattern = f"{tokens[i-1]} {tokens[i]} {tokens[i+1]}"
                if tokens[i+1] != 'قال':  # Pas un double قال
                    end_idx = i
                    transition_detected = True
                    break

    # Vérifier la longueur minimale
    isnad_length = end_idx - start_idx + 1
    if isnad_length < MIN_ISNAD_LENGTH:
        return None, None  # Isnad trop court, rejeter

    return start_idx, end_idx


def detect_khabar_end_v2(tokens: List[str], start_idx: int) -> int:
    """Détecte la fin du khabar avec heuristiques améliorées."""
    end_idx = len(tokens) - 1
    consecutive_names = 0

    for i in range(start_idx, len(tokens)):
        token = tokens[i]

        # Nouveau verbe fort = fin du khabar précédent
        if is_strong_transmission_verb(token) and i > start_idx + 10:
            end_idx = i - 1
            break

        # Verbe faible en début de position forte = peut être fin
        if is_transmission_verb(token) and i > start_idx + 20 and consecutive_names > 3:
            end_idx = i - 1
            break

        # Compter les noms propres consécutifs (signe d'isnad)
        if is_proper_noun(token):
            consecutive_names += 1
        else:
            consecutive_names = 0

    return end_idx


def segment_akhbars_v2(text: str) -> List[Dict]:
    """Segmente avec heuristiques améliorées."""
    normalized = normalize_arabic_text(text)
    tokens = simple_tokenize(normalized)

    akhbars = []
    i = 0
    rejected = 0

    while i < len(tokens):
        token = tokens[i]

        # Chercher un verbe fort en priorité
        if is_strong_transmission_verb(token) or (is_transmission_verb(token) and i < 100):
            isnad_start, isnad_end = detect_isnad_span_v2(tokens, i)

            if isnad_start is None:
                # Isnad rejeté (trop court)
                rejected += 1
                i += 1
                continue

            khabar_end = detect_khabar_end_v2(tokens, isnad_end)

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
                'isnad_length': isnad_end - isnad_start + 1,
            })

            i = khabar_end + 1
        else:
            i += 1

    return akhbars, rejected


def analyze_segmentation_v2(akhbars: List[Dict], text: str) -> Dict:
    """Analyse les résultats."""
    coverage = sum(len(ak['text'].split()) for ak in akhbars)
    total_words = len(simple_tokenize(text))

    isnad_lengths = [ak['isnad_length'] for ak in akhbars]
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


def print_report_v2(analysis: Dict, rejected: int, target: int = 613):
    """Affiche le rapport comparatif."""
    print("\n" + "=" * 70)
    print("RAPPORT V2 - SEGMENTATION AVEC HEURISTIQUES AMELIOREES")
    print("=" * 70)
    print(f"Nombre total d'akhbars detectes: {analysis['total_akhbars']}")
    print(f"Akhbars rejetes (isnad < {MIN_ISNAD_LENGTH} mots): {rejected}")
    print(f"Cible attendue: {target}")
    delta = analysis['total_akhbars'] - target
    pct = (delta / target * 100) if target > 0 else 0
    print(f"Ecart: {abs(delta)} ({pct:+.1f}%)")
    print()
    print(f"Couverture du texte: {analysis['coverage_words']} / {analysis['total_words']} mots ({analysis['coverage_pct']:.1f}%)")
    print()
    print("Longueurs moyennes:")
    print(f"  - Isnad: {analysis['avg_isnad_length']:.1f} mots (min: {analysis['min_isnad']}, max: {analysis['max_isnad']})")
    print(f"  - Khabar (recit): {analysis['avg_khabar_length']:.1f} mots")
    print("=" * 70 + "\n")


def load_text_from_file(filepath: str) -> str:
    """Charge un texte depuis un fichier."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Baseline v2: segmentation avec heuristiques ameliorees"
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='Fichier texte')
    parser.add_argument('--output', '-o', type=str, default='results/segmentation_baseline_v2.txt',
                       help='Fichier de sortie')
    parser.add_argument('--target', type=int, default=613, help='Nombre cible d\'akhbars')
    parser.add_argument('--min-isnad', type=int, default=MIN_ISNAD_LENGTH,
                       help=f'Longueur minimale d\'isnad (defaut: {MIN_ISNAD_LENGTH})')

    args = parser.parse_args()

    print(f"[FICHIER] Chargement: {args.input}")
    text = load_text_from_file(args.input)
    print(f"[OK] Texte charge: {len(text)} caracteres, {len(simple_tokenize(text))} mots")

    print(f"\n[TRAITEMENT] Segmentation avec MIN_ISNAD_LENGTH={args.min_isnad}...")
    akhbars, rejected = segment_akhbars_v2(text)

    analysis = analyze_segmentation_v2(akhbars, text)

    print_report_v2(analysis, rejected, args.target)

    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RESULTATS V2 - SEGMENTATION AVEC HEURISTIQUES AMELIOREES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Nombre d'akhbars: {analysis['total_akhbars']} / {args.target} (cible)\n")
        f.write(f"Akhbars rejetes: {rejected}\n")
        f.write(f"Couverture: {analysis['coverage_pct']:.1f}%\n")
        f.write(f"Longueur moyenne isnad: {analysis['avg_isnad_length']:.1f} mots\n")
        f.write(f"Longueur moyenne khabar: {analysis['avg_khabar_length']:.1f} mots\n\n")

        for idx, ak in enumerate(akhbars[:20], 1):  # Premier 20
            f.write(f"\n--- AKHBAR {idx} ---\n")
            f.write(f"ISNAD ({ak['isnad_length']} mots):\n{ak['isnad']}\n\n")
            f.write(f"KHABAR ({len(ak['khabar'].split())} mots):\n{ak['khabar']}\n")

    print(f"[OK] Resultats sauvegardes dans: {output_path}")


if __name__ == '__main__':
    main()
