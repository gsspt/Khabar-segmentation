#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyser la précision des frontières (borders) détectées par la baseline
comparées aux frontières de référence des annotations.

Compare position par position pour identifier les faiblesses de la baseline.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from baseline_isnad_segmentation import (
    normalize_arabic_text, simple_tokenize, is_isnad_verb,
    detect_isnad_span, detect_khabar_end
)


def load_reference_data():
    """Charger le corpus et les boundaries de référence."""
    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        boundaries = json.load(f)

    return corpus, boundaries


def map_tokens_to_char_positions(original_text: str, normalized_text: str, tokens: List[str]) -> Dict[int, Tuple[int, int]]:
    """
    Créer une mapping entre positions de tokens et positions de caractères.

    Retourne un dict: token_idx -> (char_start, char_end) dans le texte original.
    """
    # Chercher chaque token dans le texte normalisé pour obtenir sa position
    token_positions = {}
    search_pos = 0

    for i, token in enumerate(tokens):
        # Chercher le token à partir de search_pos
        idx = normalized_text.find(token, search_pos)
        if idx >= 0:
            token_positions[i] = (idx, idx + len(token))
            search_pos = idx + len(token)

    return token_positions


def segment_akhbars_with_positions(text: str, strict: bool = True) -> List[Dict]:
    """
    Segmente le texte et retourne les boundaries en positions de caractères.
    """
    normalized = normalize_arabic_text(text)
    tokens = simple_tokenize(normalized)

    # Créer une mapping token position -> char position
    # (simplifiée: on assume que normalized ≈ original à cause de la normalisation)

    akhbars = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if is_isnad_verb(token, strict=strict):
            isnad_start, isnad_end = detect_isnad_span(tokens, i)
            khabar_end = detect_khabar_end(tokens, isnad_end)

            # Pour chaque akhbar détecté, on garde les indices des tokens
            # On calcule les positions approximatives en caractères
            akhbars.append({
                'start_token': isnad_start,
                'end_token': khabar_end,
                'isnad_end_token': isnad_end,
                'tokens': tokens[isnad_start:khabar_end + 1],
                'isnad_tokens': tokens[isnad_start:isnad_end + 1],
                'khabar_tokens': tokens[isnad_end + 1:khabar_end + 1] if isnad_end + 1 <= khabar_end else [],
            })

            i = khabar_end + 1
        else:
            i += 1

    return akhbars, tokens


def find_text_in_corpus(text_tokens: List[str], corpus: str, start_search: int = 0) -> Tuple[int, int]:
    """
    Chercher une séquence de tokens dans le corpus original.

    Retourne (start_pos, end_pos) en caractères, ou (-1, -1) si non trouvé.
    """
    # Assembler le texte avec espaces
    assembled_text = ' '.join(text_tokens)

    # Chercher des variations (avec espaces variables)
    # Chercher d'abord avec regex pour être plus flexible
    pattern = r'\s*'.join(re.escape(token) for token in text_tokens[:min(5, len(text_tokens))])

    match = re.search(pattern, corpus[start_search:])
    if match:
        # Trouver le début et la fin plus précisément
        start_idx = start_search + match.start()

        # Chercher le dernier token pour trouver la fin
        last_token = text_tokens[-1]
        end_idx = corpus.find(last_token, start_idx)
        if end_idx >= 0:
            return start_idx, end_idx + len(last_token)

    return -1, -1


def analyze_boundaries(corpus: str, ref_boundaries: List[Dict],
                      detected_akhbars: List[Dict], tokens: List[str]) -> Dict:
    """
    Analyser la correspondance entre boundaries détectés et référence.
    """
    matches = []
    unmatched_detected = []
    matched_ref_indices = set()

    current_search_pos = 0

    # Pour chaque akhbar détecté, essayer de trouver sa position dans le corpus
    for det_akh in detected_akhbars:
        start, end = find_text_in_corpus(det_akh['tokens'], corpus, current_search_pos)

        if start >= 0:
            # Chercher le match le plus proche dans la référence
            best_idx = -1
            best_error = float('inf')

            for ref_idx, ref in enumerate(ref_boundaries):
                ref_start = ref['start']
                ref_end = ref['end']

                error = abs(start - ref_start) + abs(end - ref_end)

                if error < best_error and error < 500:  # Tolerance: 500 chars
                    best_error = error
                    best_idx = ref_idx

            if best_idx >= 0:
                ref = ref_boundaries[best_idx]
                matches.append({
                    'detected_pos': (start, end),
                    'reference_pos': (ref['start'], ref['end']),
                    'ref_idx': best_idx,
                    'start_error': start - ref['start'],
                    'end_error': end - ref['end'],
                    'length_error': (end - start) - (ref['end'] - ref['start']),
                })
                matched_ref_indices.add(best_idx)
                current_search_pos = end
            else:
                unmatched_detected.append((start, end))
                current_search_pos = end
        else:
            unmatched_detected.append((-1, -1))

    # Trouver les ref sans match
    unmatched_ref = [i for i in range(len(ref_boundaries)) if i not in matched_ref_indices]

    return {
        'matches': matches,
        'unmatched_detected': unmatched_detected,
        'unmatched_ref': unmatched_ref,
    }


def main():
    """Analyser la précision des frontières."""
    print("=" * 80)
    print("ANALYSE DE LA PRÉCISION DES FRONTIÈRES (BOUNDARIES)")
    print("=" * 80 + "\n")

    # Charger données
    print("[1] Chargement du corpus et des boundaries de référence...")
    corpus, ref_boundaries = load_reference_data()
    print(f"    Corpus: {len(corpus):,} chars")
    print(f"    Boundaries de référence: {len(ref_boundaries)}\n")

    # Exécuter la baseline améliorée
    print("[2] Exécution de la baseline améliorée (strict=True)...")
    akhbars, tokens = segment_akhbars_with_positions(corpus, strict=True)
    print(f"    Akhbars détectés: {len(akhbars)}\n")

    # Analyser les matches
    print("[3] Analyse de la correspondance...")
    result = analyze_boundaries(corpus, ref_boundaries, akhbars, tokens)
    matches = result['matches']
    unmatched_detected = result['unmatched_detected']
    unmatched_ref = result['unmatched_ref']

    print(f"    Matches trouvés: {len(matches)}/{len(akhbars)}")
    print(f"    Détectés sans match: {len(unmatched_detected)}")
    print(f"    Référence sans match: {len(unmatched_ref)}\n")

    if not matches:
        print("\n[ERREUR] Aucun match trouvé. Impossible de continuer l'analyse.\n")
        return

    # Analyser les erreurs de position
    print("[4] Analyse des erreurs de position...\n")

    start_errors = [m['start_error'] for m in matches]
    end_errors = [m['end_error'] for m in matches]
    length_errors = [m['length_error'] for m in matches]

    # Statistiques
    print(f"    Erreur START (décalage en chars):")
    print(f"      Moyenne: {sum(start_errors)/len(start_errors):.1f}")
    print(f"      Médiane: {sorted(start_errors)[len(start_errors)//2]:.1f}")
    print(f"      Max: {max(abs(e) for e in start_errors):.0f}")
    print(f"      Exact (±0): {sum(1 for e in start_errors if e == 0)}/{len(start_errors)}")
    print(f"      ±10 chars: {sum(1 for e in start_errors if abs(e) <= 10)}/{len(start_errors)}")

    print(f"\n    Erreur END (décalage en chars):")
    print(f"      Moyenne: {sum(end_errors)/len(end_errors):.1f}")
    print(f"      Médiane: {sorted(end_errors)[len(end_errors)//2]:.1f}")
    print(f"      Max: {max(abs(e) for e in end_errors):.0f}")
    print(f"      Exact (±0): {sum(1 for e in end_errors if e == 0)}/{len(end_errors)}")
    print(f"      ±10 chars: {sum(1 for e in end_errors if abs(e) <= 10)}/{len(end_errors)}")

    print(f"\n    Erreur LONGUEUR (différence de taille):")
    print(f"      Moyenne: {sum(length_errors)/len(length_errors):.1f}")
    print(f"      Médiane: {sorted(length_errors)[len(length_errors)//2]:.1f}")
    print(f"      Max: {max(abs(e) for e in length_errors):.0f}")
    print(f"      Exact (±0): {sum(1 for e in length_errors if e == 0)}/{len(length_errors)}")
    print(f"      ±10 chars: {sum(1 for e in length_errors if abs(e) <= 10)}/{len(length_errors)}\n")

    # Catégoriser les erreurs
    print("[5] Catégorisation des erreurs...\n")

    exact_matches = sum(1 for m in matches if m['start_error'] == 0 and m['end_error'] == 0)
    small_errors = sum(1 for m in matches if abs(m['start_error']) <= 10 and abs(m['end_error']) <= 10)
    medium_errors = sum(1 for m in matches if 10 < max(abs(m['start_error']), abs(m['end_error'])) <= 50)
    large_errors = sum(1 for m in matches if max(abs(m['start_error']), abs(m['end_error'])) > 50)

    print(f"    Matches exacts: {exact_matches} ({exact_matches/len(matches)*100:.1f}%)")
    print(f"    Petites erreurs (±10): {small_errors} ({small_errors/len(matches)*100:.1f}%)")
    print(f"    Erreurs moyennes (±50): {medium_errors} ({medium_errors/len(matches)*100:.1f}%)")
    print(f"    Grandes erreurs (>50): {large_errors} ({large_errors/len(matches)*100:.1f}%)\n")

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80 + "\n")

    if exact_matches > len(matches) * 0.8:
        verdict = "[EXCELLENT] La baseline détecte les frontières avec grande précision"
    elif small_errors > len(matches) * 0.7:
        verdict = "[BON] La baseline détecte les frontières avec petites variations"
    elif medium_errors > len(matches) * 0.7:
        verdict = "[ACCEPTABLE] La baseline a des erreurs modérées de positionnement"
    else:
        verdict = "[FAIBLE] La baseline a des erreurs importantes de positionnement"

    print(verdict + "\n")

    print(f"Résumé:")
    print(f"  - {len(matches)} boundaries matchés sur {len(akhbars)} détectés ({len(matches)/len(akhbars)*100:.1f}%)")
    print(f"  - {len(unmatched_ref)} boundaries de référence non trouvés ({len(unmatched_ref)}/613 = {len(unmatched_ref)/613*100:.1f}%)")
    print(f"  - Erreur moyenne START: {sum(start_errors)/len(start_errors):.1f} chars")
    print(f"  - Erreur moyenne END: {sum(end_errors)/len(end_errors):.1f} chars")
    print(f"  - {exact_matches} matches exacts ({exact_matches/len(matches)*100:.1f}%)\n")

    # Sauvegarder le rapport
    report_path = Path("../results/boundary_accuracy_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Analyse de la Précision des Frontières\n\n")
        f.write("## Configuration\n")
        f.write(f"- Corpus: Reference corpus (kitab_uqala) - {len(corpus):,} chars\n")
        f.write(f"- Boundaries de référence: {len(ref_boundaries)}\n")
        f.write(f"- Baseline: Améliorée (verbes fiables uniquement, strict=True)\n\n")

        f.write("## Résultats de Détection\n")
        f.write(f"- Akhbars détectés: {len(akhbars)}\n")
        f.write(f"- Boundaries matchés: {len(matches)} ({len(matches)/len(akhbars)*100:.1f}%)\n")
        f.write(f"- Non matchés (détectés): {len(unmatched_detected)}\n")
        f.write(f"- Non matchés (référence): {len(unmatched_ref)} ({len(unmatched_ref)/len(ref_boundaries)*100:.1f}%)\n\n")

        f.write("## Précision des Frontières\n\n")
        f.write("### Erreurs de Position (START)\n")
        f.write(f"- Moyenne: {sum(start_errors)/len(start_errors):.1f} chars\n")
        f.write(f"- Médiane: {sorted(start_errors)[len(start_errors)//2]:.1f} chars\n")
        f.write(f"- Max: {max(abs(e) for e in start_errors):.0f} chars\n")
        f.write(f"- Exact: {sum(1 for e in start_errors if e == 0)}/{len(start_errors)} ({sum(1 for e in start_errors if e == 0)/len(start_errors)*100:.1f}%)\n")
        f.write(f"- ±10 chars: {sum(1 for e in start_errors if abs(e) <= 10)}/{len(start_errors)} ({sum(1 for e in start_errors if abs(e) <= 10)/len(start_errors)*100:.1f}%)\n\n")

        f.write("### Erreurs de Position (END)\n")
        f.write(f"- Moyenne: {sum(end_errors)/len(end_errors):.1f} chars\n")
        f.write(f"- Médiane: {sorted(end_errors)[len(end_errors)//2]:.1f} chars\n")
        f.write(f"- Max: {max(abs(e) for e in end_errors):.0f} chars\n")
        f.write(f"- Exact: {sum(1 for e in end_errors if e == 0)}/{len(end_errors)} ({sum(1 for e in end_errors if e == 0)/len(end_errors)*100:.1f}%)\n")
        f.write(f"- ±10 chars: {sum(1 for e in end_errors if abs(e) <= 10)}/{len(end_errors)} ({sum(1 for e in end_errors if abs(e) <= 10)/len(end_errors)*100:.1f}%)\n\n")

        f.write("### Erreurs de Longueur\n")
        f.write(f"- Moyenne: {sum(length_errors)/len(length_errors):.1f} chars\n")
        f.write(f"- Médiane: {sorted(length_errors)[len(length_errors)//2]:.1f} chars\n")
        f.write(f"- Max: {max(abs(e) for e in length_errors):.0f} chars\n\n")

        f.write("## Catégorisation des Erreurs\n")
        f.write(f"- Matches exacts: {exact_matches} ({exact_matches/len(matches)*100:.1f}%)\n")
        f.write(f"- Petites erreurs (±10): {small_errors} ({small_errors/len(matches)*100:.1f}%)\n")
        f.write(f"- Erreurs moyennes (±50): {medium_errors} ({medium_errors/len(matches)*100:.1f}%)\n")
        f.write(f"- Grandes erreurs (>50): {large_errors} ({large_errors/len(matches)*100:.1f}%)\n\n")

        f.write("## Verdict\n")
        f.write(verdict + "\n")

    print(f"Rapport: {report_path}\n")


if __name__ == "__main__":
    main()
