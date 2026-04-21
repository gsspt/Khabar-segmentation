#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation interactive du corpus avec annotations de segmentation.
Affiche les boundaries détectés par CAMeL-BERT, Baseline v4 et le gold standard.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def load_corpus(corpus_path: str) -> str:
    """Charger le texte du corpus."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_gold_standard(boundaries_path: str) -> List[Dict]:
    """Charger les boundaries du gold standard."""
    with open(boundaries_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_camelbert_segments(segments_path: str) -> List[Dict]:
    """Charger les segments extraits par CAMeL-BERT."""
    with open(segments_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('segmentation', {}).get('segments', [])


def generate_baseline_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Générer les boundaries Baseline v4 en détectant les transitions isnad→prose.
    Utilise une détection simple des verbes d'isnad et des patterns de transition.
    """
    # Verbes et patterns d'isnad classiques
    isnad_verbs = r'\b(حدثنا|أخبرنا|قال|سمعت|أنشدني|سألت|قلت)\b'

    boundaries = []
    last_boundary = 0

    # Chercher les transitions entre isnads et prose
    for match in re.finditer(isnad_verbs, text):
        # Vérifier si c'est le début d'un isnad
        pos = match.start()

        # Si ce n'est pas trop proche du dernier boundary, marquer une transition
        if pos - last_boundary > 50:  # Au moins 50 caractères entre les boundaries
            boundaries.append((last_boundary, pos))
            last_boundary = pos

    # Ajouter le dernier segment jusqu'à la fin
    if last_boundary < len(text):
        boundaries.append((last_boundary, len(text)))

    return boundaries


def extract_word_at_position(text: str, pos: int, context_chars: int = 15) -> str:
    """Extraire le mot autour d'une position avec contexte."""
    start = max(0, pos - context_chars)
    end = min(len(text), pos + context_chars)
    return text[start:end].replace('\n', ' ')[:50]


def create_color_classes() -> str:
    """Retourner le CSS pour les différentes couleurs."""
    return """
    <style>
        body {
            direction: rtl;
            font-family: 'Noto Naskh Arabic', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            line-height: 1.8;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            color: #333;
            text-align: center;
        }

        .legend {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 10px;
        }

        .legend-box {
            width: 20px;
            height: 20px;
            margin-left: 8px;
            border-radius: 3px;
        }

        .text-container {
            background-color: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            line-height: 2;
            word-wrap: break-word;
            text-align: justify;
        }

        /* Gold standard - boundaries vraies */
        .gold-boundary {
            background-color: rgba(0, 0, 255, 0.2);
            border-left: 3px solid #0000ff;
            padding-left: 2px;
        }

        /* CAMeL-BERT - isnads détectés */
        .camelbert-isnad {
            background-color: rgba(255, 165, 0, 0.25);
            border-left: 3px solid #ffa500;
            padding-left: 2px;
        }

        /* CAMeL-BERT - prose */
        .camelbert-prose {
            background-color: rgba(144, 238, 144, 0.15);
        }

        /* Baseline v4 - transitions détectées */
        .baseline-boundary {
            background-color: rgba(0, 128, 0, 0.15);
            border-left: 3px solid #008000;
            padding-left: 2px;
        }

        .tooltip {
            position: relative;
            cursor: help;
            border-bottom: 1px dotted #666;
        }

        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 0;
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            white-space: nowrap;
            z-index: 1;
            font-size: 12px;
            direction: ltr;
        }

        .stats {
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }

        .stats p {
            margin: 5px 0;
            font-size: 14px;
        }

        .comparison-summary {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff9e6;
            border-left: 4px solid #ff9800;
            border-radius: 3px;
        }
    </style>
    """


def create_html_visualization(
    text: str,
    gold_boundaries: List[Dict],
    camelbert_segments: List[Dict],
    baseline_boundaries: List[Tuple[int, int]],
    output_path: str
) -> None:
    """Créer la visualisation HTML."""

    # Convertir gold boundaries à (start, end)
    gold_ranges = [(b['start'], b['end']) for b in gold_boundaries]

    # Créer les ranges CAMeL-BERT
    camelbert_ranges = {
        'isnad': [(s['start'], s['end']) for s in camelbert_segments if s.get('type') == 'isnad'],
        'prose': [(s['start'], s['end']) for s in camelbert_segments if s.get('type') == 'prose']
    }

    # HTML header
    html_parts = [
        "<!DOCTYPE html>",
        "<html dir='rtl' lang='ar'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Visualisation Segmentation Kitab Uqala</title>",
        "<link href='https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap' rel='stylesheet'>",
        create_color_classes(),
        "</head>",
        "<body>",
        "<div class='container'>",
        "<h1>Visualisation de la Segmentation - Kitab Uqala</h1>",
    ]

    # Légende
    html_parts.extend([
        "<div class='legend'>",
        "<div class='legend-item'>",
        "    <span>Gold Standard (Boundaries Vraies)</span>",
        "    <div class='legend-box' style='background-color: rgba(0, 0, 255, 0.2); border-left: 3px solid #0000ff;'></div>",
        "</div>",
        "<div class='legend-item'>",
        "    <span>CAMeL-BERT Isnads</span>",
        "    <div class='legend-box' style='background-color: rgba(255, 165, 0, 0.25); border-left: 3px solid #ffa500;'></div>",
        "</div>",
        "<div class='legend-item'>",
        "    <span>Baseline v4 Transitions</span>",
        "    <div class='legend-box' style='background-color: rgba(0, 128, 0, 0.15); border-left: 3px solid #008000;'></div>",
        "</div>",
        "</div>",
    ])

    # Statistiques
    html_parts.extend([
        f"<div class='stats'>",
        f"    <p><strong>Taille du corpus:</strong> {len(text):,} caractères</p>",
        f"    <p><strong>Gold standard:</strong> {len(gold_boundaries)} boundaries</p>",
        f"    <p><strong>CAMeL-BERT:</strong> {len(camelbert_segments)} segments (isnads: {len(camelbert_ranges['isnad'])}, prose: {len(camelbert_ranges['prose'])})</p>",
        f"    <p><strong>Baseline v4:</strong> {len(baseline_boundaries)} transitions</p>",
        f"</div>",
    ])

    # Créer les spans avec surlignage
    html_parts.append("<div class='text-container'>")

    # Marquer tous les positions importantes
    markers = []

    # Ajouter les gold boundaries
    for start, end in gold_ranges:
        markers.append(('gold_start', start, 'start'))
        markers.append(('gold_end', end, 'end'))

    # Ajouter les CAMeL-BERT isnads
    for start, end in camelbert_ranges['isnad']:
        markers.append(('camelbert_isnad_start', start, 'start'))
        markers.append(('camelbert_isnad_end', end, 'end'))

    # Ajouter les Baseline v4 transitions
    for start, end in baseline_boundaries:
        markers.append(('baseline_start', start, 'start'))
        markers.append(('baseline_end', end, 'end'))

    # Trier les markers par position
    markers.sort(key=lambda x: (x[1], x[2]))

    # Construire le texte avec les tags
    text_spans = []
    last_pos = 0
    active_classes = set()

    for marker_type, pos, marker_side in markers:
        # Ajouter le texte jusqu'à cette position
        if pos > last_pos:
            text_chunk = text[last_pos:pos]
            # Remplacer les newlines et échapper les caractères spéciaux
            text_chunk = text_chunk.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            if active_classes:
                class_str = ' '.join(sorted(active_classes))
                text_spans.append(f"<span class='{class_str}'>{text_chunk}</span>")
            else:
                text_spans.append(text_chunk)

            last_pos = pos

        # Mettre à jour les classes actives
        if 'start' in marker_side:
            if 'gold' in marker_type:
                active_classes.add('gold-boundary')
            elif 'camelbert_isnad' in marker_type:
                active_classes.add('camelbert-isnad')
            elif 'baseline' in marker_type:
                active_classes.add('baseline-boundary')
        else:  # end
            if 'gold' in marker_type:
                active_classes.discard('gold-boundary')
            elif 'camelbert_isnad' in marker_type:
                active_classes.discard('camelbert-isnad')
            elif 'baseline' in marker_type:
                active_classes.discard('baseline-boundary')

    # Ajouter le reste du texte
    if last_pos < len(text):
        text_chunk = text[last_pos:]
        text_chunk = text_chunk.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if active_classes:
            class_str = ' '.join(sorted(active_classes))
            text_spans.append(f"<span class='{class_str}'>{text_chunk}</span>")
        else:
            text_spans.append(text_chunk)

    html_parts.append(''.join(text_spans))
    html_parts.append("</div>")

    # Fermer le HTML
    html_parts.extend([
        "</div>",
        "</body>",
        "</html>",
    ])

    # Écrire le fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print(f"Visualisation générée: {output_path}")


def main():
    # Chemins
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    boundaries_path = "data/processed/kitab_uqala_boundaries.json"
    camelbert_path = "results/camelbert_kitab_uqala_segments.json"
    output_path = "results/visualization_segmentation.html"

    print("Chargement des données...")
    text = load_corpus(corpus_path)
    gold_boundaries = load_gold_standard(boundaries_path)
    camelbert_segments = load_camelbert_segments(camelbert_path)

    print("Génération des boundaries Baseline v4...")
    baseline_boundaries = generate_baseline_boundaries(text)

    print("Création de la visualisation HTML...")
    create_html_visualization(
        text,
        gold_boundaries,
        camelbert_segments,
        baseline_boundaries,
        output_path
    )

    print(f"Done! Ouvrir {output_path} dans un navigateur.")


if __name__ == "__main__":
    main()
