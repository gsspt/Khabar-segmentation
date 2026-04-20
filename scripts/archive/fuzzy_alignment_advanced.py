"""
Fuzzy alignment avancé: combine fragments clés + normalisation agressive + LCS.
Cherche par début/fin d'akhbars pour robustesse maximale.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from rapidfuzz import fuzz
from difflib import SequenceMatcher
from tqdm import tqdm


def normalize_aggressive(text: str) -> str:
    """Normalisation agressive pour l'alignement fuzzy."""
    # Supprimer espaces excessifs
    text = re.sub(r'\s+', ' ', text)

    # Supprimer TOUS les diacritiques arabes (tachkil, hamza, etc.)
    # Plage Unicode des diacritiques arabes: U+064B à U+065F
    text = re.sub(r'[\u064B-\u065F]', '', text)

    # Normaliser les variantes courantes
    text = re.sub(r'ة', 'ه', text)  # ة → ه (ta marbuta → ha)
    text = re.sub(r'[أإآ]', 'ا', text)  # alef variants → alef
    text = re.sub(r'ى', 'ي', text)  # alef maksura → ya

    # Supprimer ponctuation
    text = re.sub(r'[،؛:\.\-\(\)«»""''…]', '', text)

    # Normaliser espaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def longest_common_substring(s1: str, s2: str) -> Tuple[str, float]:
    """Trouver la plus longue sous-chaîne commune et retourner son ratio."""
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))

    if match.size == 0:
        return '', 0.0

    lcs = s1[match.a:match.a + match.size]
    ratio = match.size / max(len(s1), len(s2))
    return lcs, ratio


def extract_key_fragments(text: str) -> Dict[str, str]:
    """Extraire fragments clés d'un akhbar."""
    normalized = normalize_aggressive(text)
    tokens = normalized.split()

    return {
        'full': normalized,
        'beginning': ' '.join(tokens[:min(20, len(tokens))]),  # Premiers tokens
        'ending': ' '.join(tokens[max(0, len(tokens)-20):]),   # Derniers tokens
        'first_20_chars': normalized[:min(50, len(normalized))],
        'last_20_chars': normalized[max(0, len(normalized)-50):],
    }


def find_akhbar_advanced(
    akhbar_text: str,
    raw_text: str,
    raw_normalized: str,
) -> Dict:
    """
    Stratégie multi-niveaux pour trouver un akhbar.

    Niveaux (par ordre de priorité):
    1. LCS (Longest Common Substring)
    2. Fragment matching (début + fin)
    3. Token set matching sur texte entier
    """

    fragments = extract_key_fragments(akhbar_text)
    akhbar_normalized = fragments['full']

    if len(akhbar_normalized) < 10:
        return {
            'found': False,
            'position': -1,
            'score': 0,
            'method': 'too_short'
        }

    best_match = {
        'found': False,
        'position': -1,
        'score': 0,
        'method': 'none',
        'confidence': 0
    }

    # ===== NIVEAU 1: LCS (Longest Common Substring) =====
    lcs_text, lcs_ratio = longest_common_substring(akhbar_normalized, raw_normalized)

    if lcs_ratio > 0.5:  # Au moins 50% du texte retrouvé
        # Trouver position du LCS dans le texte brut
        idx = raw_normalized.find(lcs_text)
        if idx != -1:
            # Estimer position dans texte original
            # (approximation car les positions changeront avec la normalisation)
            best_match = {
                'found': True,
                'position': idx,
                'score': lcs_ratio * 100,
                'method': 'lcs',
                'confidence': min(100, lcs_ratio * 150)
            }
            return best_match

    # ===== NIVEAU 2: Fragment Matching (début + fin) =====

    # Chercher le début
    beginning_score = fuzz.token_set_ratio(
        fragments['beginning'],
        raw_normalized[:min(500, len(raw_normalized))]
    )

    # Chercher la fin
    ending_score = fuzz.token_set_ratio(
        fragments['ending'],
        raw_normalized[max(0, len(raw_normalized)-500):]
    )

    if (beginning_score + ending_score) / 2 > 65:
        # Chercher la position par la fenêtre glissante
        window_size = min(len(akhbar_normalized) * 1.5, 2000)
        window_size = int(window_size)

        for i in range(0, len(raw_normalized) - window_size, 100):
            snippet = raw_normalized[i:i+window_size]
            score = fuzz.token_set_ratio(akhbar_normalized, snippet)

            if score > best_match['score']:
                best_match = {
                    'found': score >= 70,
                    'position': i,
                    'score': score,
                    'method': 'fragment_matching',
                    'confidence': (beginning_score + ending_score) / 2
                }

        if best_match['found']:
            return best_match

    # ===== NIVEAU 3: Token Set Matching (texte entier, moins strict) =====
    best_score = 0
    best_pos = -1

    for i in range(0, len(raw_normalized) - 500, 200):
        snippet = raw_normalized[i:i+2000]
        score = fuzz.token_set_ratio(akhbar_normalized, snippet)

        if score > best_score:
            best_score = score
            best_pos = i

    if best_score >= 60:  # Seuil plus bas pour ce niveau
        best_match = {
            'found': best_score >= 70,
            'position': best_pos,
            'score': best_score,
            'method': 'token_set',
            'confidence': best_score
        }

    return best_match


def analyze_failures(
    results: Dict,
    akhbars: List[Dict],
    raw_text: str,
    raw_normalized: str
) -> Dict:
    """Analyser les cas d'échec pour comprendre les divergences."""

    failed = [r for r in results['alignment_map'] if not r['found']]

    if not failed:
        return {'total_failed': 0, 'analysis': 'All akhbars found!'}

    analysis = {
        'total_failed': len(failed),
        'sample_failures': [],
        'common_issues': {
            'very_short': 0,
            'text_not_in_raw': 0,
            'heavily_modified': 0
        }
    }

    # Analyser les 10 premiers échecs
    for failed_result in failed[:10]:
        akh_num = failed_result['akhbar_num']
        akhbar = next((a for a in akhbars if a['num'] == akh_num), None)

        if not akhbar:
            continue

        akhbar_text = extract_akhbar_text(akhbar)
        akhbar_norm = normalize_aggressive(akhbar_text)

        # Chercher manuellement
        if akhbar_norm in raw_normalized:
            analysis['common_issues']['text_not_in_raw'] += 1
            status = 'Trouvé en cherchant manuellement'
        elif akhbar_text[:100] in raw_text[:10000]:
            analysis['common_issues']['heavily_modified'] += 1
            status = 'Texte fortement modifié'
        else:
            analysis['common_issues']['text_not_in_raw'] += 1
            status = 'Pas trouvé dans le texte'

        analysis['sample_failures'].append({
            'akhbar_num': akh_num,
            'score': failed_result['score'],
            'status': status,
            'text_preview': akhbar_text[:80]
        })

    return analysis


def clean_openiti_text(text: str) -> str:
    """Nettoyer ULTRA-AGRESSIVEMENT: garder SEULEMENT du texte arabe.

    1. Supprime explicitement tous les # (métadonnées OpenITI)
    2. Garde SEULEMENT les caractères arabes et espaces
    """
    # 1. Supprimer TOUS les # (peu importe le contexte)
    text = text.replace('#', ' ')

    # 2. Supprimer TOUS les caractères non-arabes et non-espaces
    # Garder seulement:
    # - U+0600-U+06FF: Arabic
    # - U+0750-U+077F: Arabic Supplement
    # - U+FB50-U+FDFF: Arabic Presentation Forms-A
    # - U+FE70-U+FEFF: Arabic Presentation Forms-B
    # - Espaces et newlines
    text = re.sub(
        r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]',
        ' ',
        text
    )

    # 3. Normaliser les espaces multiples et newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def load_raw_text(text_path: str) -> str:
    """Charger et nettoyer le texte brut OpenITI."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Appliquer le nettoyage agressif
    text = clean_openiti_text(text)
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


def align_texts_advanced(
    raw_text_path: str,
    json_path: str,
    output_path: str = './results/fuzzy_alignment_advanced.json'
) -> Dict:
    """Alignment avancé combinant tous les niveaux."""

    print("[*] Loading texts...")
    raw_text = load_raw_text(raw_text_path)
    akhbars = load_annotated_akhbars(json_path)

    print(f"[*] Raw text length: {len(raw_text)} characters")
    print(f"[*] Number of akhbars: {len(akhbars)}")

    print("[*] Normalizing raw text (this may take a moment)...")
    raw_normalized = normalize_aggressive(raw_text)
    print(f"[*] Normalized text length: {len(raw_normalized)} characters")
    print()

    print("[*] Performing advanced fuzzy alignment...")

    results = {
        'total_akhbars': len(akhbars),
        'found': 0,
        'not_found': 0,
        'avg_score': 0,
        'alignment_map': [],
        'score_distribution': {
            'high': 0,      # score >= 85
            'medium': 0,    # 70-84
            'low': 0,       # 50-69
            'very_low': 0   # < 50
        },
        'method_distribution': {}
    }

    scores = []
    methods = []

    for akh in tqdm(akhbars, desc="Aligning"):
        akhbar_text = extract_akhbar_text(akh)
        match = find_akhbar_advanced(akhbar_text, raw_text, raw_normalized)

        scores.append(match['score'])
        methods.append(match['method'])

        if match['found']:
            results['found'] += 1
        else:
            results['not_found'] += 1

        # Catégoriser par score
        if match['score'] >= 85:
            results['score_distribution']['high'] += 1
        elif match['score'] >= 70:
            results['score_distribution']['medium'] += 1
        elif match['score'] >= 50:
            results['score_distribution']['low'] += 1
        else:
            results['score_distribution']['very_low'] += 1

        # Compter les méthodes
        method = match['method']
        results['method_distribution'][method] = results['method_distribution'].get(method, 0) + 1

        results['alignment_map'].append({
            'akhbar_num': akh['num'],
            'position': match['position'],
            'score': match['score'],
            'found': match['found'],
            'method': match['method'],
            'confidence': match.get('confidence', 0),
            'text_preview': akhbar_text[:80]
        })

    # Statistiques
    if scores:
        results['avg_score'] = sum(scores) / len(scores)

    results['percentages'] = {
        'found_percent': (results['found'] / results['total_akhbars']) * 100,
        'high_quality': (results['score_distribution']['high'] / results['total_akhbars']) * 100,
        'medium_quality': (results['score_distribution']['medium'] / results['total_akhbars']) * 100,
        'low_quality': (results['score_distribution']['low'] / results['total_akhbars']) * 100,
    }

    # Analyser les failures
    failure_analysis = analyze_failures(results, akhbars, raw_text, raw_normalized)
    results['failure_analysis'] = failure_analysis

    # Afficher résultats
    print("\n" + "="*70)
    print("[RÉSULTATS ALIGNMENT AVANCÉ]")
    print("="*70)
    print(f"  Found: {results['found']}/{results['total_akhbars']} ({results['percentages']['found_percent']:.1f}%)")
    print(f"  Average match score: {results['avg_score']:.2f}")

    print(f"\n[DISTRIBUTION PAR SCORE]")
    print(f"  High (>=85):    {results['score_distribution']['high']} ({results['percentages']['high_quality']:.1f}%)")
    print(f"  Medium (70-84): {results['score_distribution']['medium']} ({results['percentages']['medium_quality']:.1f}%)")
    print(f"  Low (50-69):    {results['score_distribution']['low']} ({results['percentages']['low_quality']:.1f}%)")
    print(f"  Very low (<50): {results['score_distribution']['very_low']}")

    print(f"\n[MÉTHODES D'ALIGNEMENT UTILISÉES]")
    for method, count in sorted(results['method_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / results['total_akhbars']) * 100
        print(f"  {method}: {count} ({pct:.1f}%)")

    if failure_analysis.get('total_failed', 0) > 0:
        print(f"\n[ANALYSE DES ÉCHECS]")
        print(f"  Total failed: {failure_analysis['total_failed']}")
        for issue, count in failure_analysis['common_issues'].items():
            print(f"    {issue}: {count}")

        if failure_analysis.get('sample_failures'):
            print(f"\n  Exemples:")
            for sample in failure_analysis['sample_failures'][:3]:
                print(f"    Akhbar {sample['akhbar_num']}: {sample['status']} (score: {sample['score']:.2f})")

    print("="*70)

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

    align_texts_advanced(str(raw_text_path), str(json_path))


if __name__ == '__main__':
    main()
