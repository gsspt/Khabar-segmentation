"""
Fuzzy alignment entre le texte brut OpenITI et les akhbars annotés.
Évalue le degré d'alignement et crée un mapping positions.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from tqdm import tqdm


def load_raw_text(text_path: str) -> str:
    """Charger le texte brut du corpus OpenITI."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def load_annotated_akhbars(json_path: str) -> List[Dict]:
    """Charger les akhbars annotés."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['akhbar']


def extract_akhbar_text(akhbar: Dict) -> str:
    """Extraire le texte complet d'un akhbar."""
    segments = akhbar['content']['segments']
    texts = [seg['text'] for seg in segments]
    return ' '.join(texts)


def normalize_for_matching(text: str) -> str:
    """Normaliser le texte pour le matching fuzzy."""
    # Supprimer espaces excessifs
    text = re.sub(r'\s+', ' ', text)
    # Supprimer ponctuation/diacritiques problématiques
    text = re.sub(r'[،؛:\u064B-\u065F]', '', text)
    return text.strip()


def find_akhbar_in_raw_text(
    akhbar_text: str,
    raw_text: str,
    step_size: int = 200
) -> Dict:
    """
    Trouver un akhbar dans le texte brut avec fuzzy matching (optimisé).
    Utilise une approche par fenêtre avec pas régulier.
    """
    # Normaliser l'akhbar
    akhbar_clean = normalize_for_matching(akhbar_text)

    if len(akhbar_clean) < 10:
        return {
            'found': False,
            'position': -1,
            'score': 0,
            'reason': 'too_short'
        }

    best_match = {
        'found': False,
        'position': -1,
        'score': 0
    }

    # Fenêtre = longueur de l'akhbar avec padding
    window_size = min(len(akhbar_clean) * 1.5, 2000)
    window_size = int(window_size)

    # Chercher avec pas régulier dans le texte
    for i in range(0, len(raw_text) - window_size, step_size):
        snippet = normalize_for_matching(raw_text[i:i+window_size])

        # Utiliser RapidFuzz token_set_ratio (flexible aux réarrangements)
        score = fuzz.token_set_ratio(akhbar_clean, snippet)

        if score > best_match['score']:
            best_match = {
                'found': score >= 75,  # Seuil de confiance: 75
                'position': i,
                'score': score
            }

    return best_match


def align_texts(
    raw_text_path: str,
    json_path: str,
    output_path: str = './results/fuzzy_alignment.json'
) -> Dict:
    """
    Aligner le texte brut avec les akhbars annotés.
    Retourne statistiques et mapping.
    """
    print("[*] Loading texts...")
    raw_text = load_raw_text(raw_text_path)
    akhbars = load_annotated_akhbars(json_path)

    print(f"[*] Raw text length: {len(raw_text)} characters")
    print(f"[*] Number of akhbars: {len(akhbars)}")
    print()

    print("[*] Performing fuzzy alignment...")

    results = {
        'total_akhbars': len(akhbars),
        'found': 0,
        'not_found': 0,
        'avg_score': 0,
        'alignment_map': [],
        'score_distribution': {
            'high': 0,    # score >= 90
            'medium': 0,  # 70-89
            'low': 0,     # 50-69
            'very_low': 0 # < 50
        }
    }

    scores = []

    for akh in tqdm(akhbars, desc="Aligning"):
        akhbar_text = extract_akhbar_text(akh)
        match = find_akhbar_in_raw_text(akhbar_text, raw_text)

        scores.append(match['score'])

        if match['found']:
            results['found'] += 1
        else:
            results['not_found'] += 1

        # Catégoriser par score
        if match['score'] >= 90:
            results['score_distribution']['high'] += 1
        elif match['score'] >= 70:
            results['score_distribution']['medium'] += 1
        elif match['score'] >= 50:
            results['score_distribution']['low'] += 1
        else:
            results['score_distribution']['very_low'] += 1

        # Ajouter au mapping
        results['alignment_map'].append({
            'akhbar_num': akh['num'],
            'position': match['position'],
            'score': match['score'],
            'found': match['found'],
            'text_preview': akhbar_text[:100]
        })

    # Statistiques
    if scores:
        results['avg_score'] = sum(scores) / len(scores)

    results['percentages'] = {
        'found_percent': (results['found'] / results['total_akhbars']) * 100,
        'high_quality': (results['score_distribution']['high'] / results['total_akhbars']) * 100,
        'medium_quality': (results['score_distribution']['medium'] / results['total_akhbars']) * 100,
    }

    # Afficher résultats
    print("\n[RESULTS]")
    print(f"  Found: {results['found']}/{results['total_akhbars']} ({results['percentages']['found_percent']:.1f}%)")
    print(f"  Average match score: {results['avg_score']:.2f}")
    print(f"\n[SCORE DISTRIBUTION]")
    print(f"  High (>=90):   {results['score_distribution']['high']} ({results['percentages']['high_quality']:.1f}%)")
    print(f"  Medium (70-89): {results['score_distribution']['medium']} ({results['percentages']['medium_quality']:.1f}%)")
    print(f"  Low (50-69):    {results['score_distribution']['low']}")
    print(f"  Very low (<50): {results['score_distribution']['very_low']}")

    # Sauvegarder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Results saved to {output_path}")

    return results


def main():
    raw_text_path = Path('../Uqala_NLP/openiti_corpus/data/0406IbnHabibNaysaburi/0406IbnHabibNaysaburi.CuqalaMajanin/0406IbnHabibNaysaburi.CuqalaMajanin.Shamela0001593-ara1')
    json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    if not raw_text_path.exists():
        print(f"[!] Raw text not found: {raw_text_path}")
        return

    if not json_path.exists():
        print(f"[!] JSON not found: {json_path}")
        return

    align_texts(str(raw_text_path), str(json_path))


if __name__ == '__main__':
    main()
