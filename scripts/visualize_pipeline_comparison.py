#!/usr/bin/env python3
"""
Create interactive HTML visualization comparing three pipelines.

Displays:
- Full text with 3 color-coded segment overlays (CAMeL-BERT, Baseline, Gold)
- Toggle buttons to show/hide each pipeline
- Hover tooltips with segment metadata
- Statistics panel showing metrics
- Legend and summary

Usage:
    python scripts/visualize_pipeline_comparison.py \
      --text data/processed/[TEXT]_clean.txt \
      --camelbert results/camelbert_[TEXT]_narrative_units.json \
      --baseline results/baseline_v4_[TEXT]_narrative_units.json \
      --gold results/gold_standard_[TEXT]_deepseek.json \
      --output results/visualization_[TEXT]_comparison.html
"""

import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text(text_file: str) -> str:
    """Load text from file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_units(units_file: str) -> list[dict]:
    """Load narrative units from JSON."""
    with open(units_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('narrative_units', data.get('units', []))


def create_html_visualization(
    text: str,
    camelbert_units: list[dict],
    baseline_units: list[dict],
    gold_units: list[dict],
    output_path: str
) -> None:
    """
    Create interactive HTML visualization.

    Args:
        text: Full corpus text
        camelbert_units: CAMeL-BERT narrative units
        baseline_units: Baseline v4 narrative units
        gold_units: Gold standard units
        output_path: Path to save HTML file
    """

    # Escape HTML special characters in text
    import html
    text_escaped = html.escape(text)

    # Create segment indices for each pipeline
    camelbert_segments = [(u['char_start'], u['char_end'], 'camelbert', u) for u in camelbert_units]
    baseline_segments = [(u['char_start'], u['char_end'], 'baseline', u) for u in baseline_units]
    gold_segments = [(u['char_start'], u['char_end'], 'gold', u) for u in gold_units]

    all_segments = camelbert_segments + baseline_segments + gold_segments
    all_segments.sort(key=lambda x: x[0])

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Comparison Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
        }}

        .stat {{
            padding: 10px;
            border-left: 4px solid #007bff;
        }}

        .stat-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }}

        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}

        .toggle-btn {{
            padding: 10px 15px;
            border: 2px solid;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }}

        .toggle-btn.camelbert {{
            border-color: #e74c3c;
            color: #e74c3c;
        }}

        .toggle-btn.camelbert.active {{
            background: #e74c3c;
            color: white;
        }}

        .toggle-btn.baseline {{
            border-color: #3498db;
            color: #3498db;
        }}

        .toggle-btn.baseline.active {{
            background: #3498db;
            color: white;
        }}

        .toggle-btn.gold {{
            border-color: #2ecc71;
            color: #2ecc71;
        }}

        .toggle-btn.gold.active {{
            background: #2ecc71;
            color: white;
        }}

        .legend {{
            display: flex;
            gap: 30px;
            margin-bottom: 30px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            opacity: 0.3;
        }}

        .legend-color.camelbert {{
            background: #e74c3c;
        }}

        .legend-color.baseline {{
            background: #3498db;
        }}

        .legend-color.gold {{
            background: #2ecc71;
        }}

        .text-container {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            line-height: 1.8;
            direction: rtl;
            text-align: right;
            word-break: break-word;
            white-space: pre-wrap;
            position: relative;
            font-family: 'Arabic Typesetting', 'Arial Unicode MS', Arial, sans-serif;
        }}

        .segment {{
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
        }}

        .segment:hover {{
            opacity: 1 !important;
        }}

        .segment.camelbert {{
            background-color: rgba(231, 76, 60, 0.3);
            border-bottom: 2px solid #e74c3c;
        }}

        .segment.baseline {{
            background-color: rgba(52, 152, 219, 0.3);
            border-bottom: 2px solid #3498db;
        }}

        .segment.gold {{
            background-color: rgba(46, 204, 113, 0.3);
            border-bottom: 2px solid #2ecc71;
        }}

        .segment.hidden {{
            display: none;
        }}

        .tooltip {{
            display: none;
            position: absolute;
            bottom: 100%;
            left: 0;
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            white-space: nowrap;
            font-size: 12px;
            z-index: 1000;
            margin-bottom: 5px;
            pointer-events: none;
        }}

        .segment:hover .tooltip {{
            display: block;
        }}

        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}

            .controls {{
                flex-direction: column;
            }}

            .toggle-btn {{
                width: 100%;
            }}

            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipeline Comparison - Khabar Segmentation</h1>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Text Length</div>
                <div class="stat-value">{len(text):,} chars</div>
            </div>
            <div class="stat">
                <div class="stat-label">CAMeL-BERT Units</div>
                <div class="stat-value">{len(camelbert_units)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Baseline v4 Units</div>
                <div class="stat-value">{len(baseline_units)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Gold Standard Units</div>
                <div class="stat-value">{len(gold_units)}</div>
            </div>
        </div>

        <div class="controls">
            <button class="toggle-btn camelbert active" onclick="togglePipeline('camelbert')">
                CAMeL-BERT
            </button>
            <button class="toggle-btn baseline active" onclick="togglePipeline('baseline')">
                Baseline v4
            </button>
            <button class="toggle-btn gold active" onclick="togglePipeline('gold')">
                Gold Standard
            </button>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color camelbert"></div>
                <span>CAMeL-BERT (Red)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color baseline"></div>
                <span>Baseline v4 (Blue)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color gold"></div>
                <span>Gold Standard (Green)</span>
            </div>
        </div>

        <div class="text-container" id="text-container"></div>

        <div class="footer">
            <p>Hover over highlighted segments to see details. Use toggle buttons to show/hide pipelines.</p>
        </div>
    </div>

    <script>
        const allSegments = {json.dumps(all_segments)};
        const textContent = `{text_escaped}`;

        function togglePipeline(pipeline) {{
            const btn = event.target;
            btn.classList.toggle('active');

            const segments = document.querySelectorAll(`.segment.${{pipeline}}`);
            segments.forEach(seg => {{
                if (btn.classList.contains('active')) {{
                    seg.classList.remove('hidden');
                }} else {{
                    seg.classList.add('hidden');
                }}
            }});
        }}

        function renderSegments() {{
            const container = document.getElementById('text-container');
            container.innerHTML = '';

            let lastEnd = 0;
            const segments = allSegments;

            // Group overlapping segments
            const sorted = segments.sort((a, b) => a[0] - b[0]);

            for (let i = 0; i < sorted.length; i++) {{
                const [start, end, pipeline, unit] = sorted[i];

                // Add text before this segment
                if (start > lastEnd) {{
                    const text = textContent.substring(lastEnd, start);
                    container.appendChild(document.createTextNode(text));
                }}

                // Add highlighted segment
                const seg = document.createElement('span');
                seg.className = `segment ${{pipeline}}`;
                seg.textContent = textContent.substring(start, end);

                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = `${{pipeline}}: chars ${{start}}-${{end}}`;
                seg.appendChild(tooltip);

                container.appendChild(seg);
                lastEnd = Math.max(lastEnd, end);
            }}

            // Add remaining text
            if (lastEnd < textContent.length) {{
                const text = textContent.substring(lastEnd);
                container.appendChild(document.createTextNode(text));
            }}
        }}

        // Initialize
        renderSegments();
    </script>
</body>
</html>
"""

    # Save HTML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Saved visualization to {output_path}")
    logger.info(f"File size: {len(html_content) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive pipeline comparison visualization'
    )
    parser.add_argument(
        '--text',
        required=True,
        help='Path to cleaned text'
    )
    parser.add_argument(
        '--camelbert',
        required=True,
        help='Path to CAMeL-BERT narrative_units.json'
    )
    parser.add_argument(
        '--baseline',
        required=True,
        help='Path to Baseline v4 narrative_units.json'
    )
    parser.add_argument(
        '--gold',
        required=True,
        help='Path to gold_standard_deepseek.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save HTML visualization'
    )

    args = parser.parse_args()

    try:
        logger.info("Loading text and units...")
        text = load_text(args.text)
        camelbert_units = load_units(args.camelbert)
        baseline_units = load_units(args.baseline)
        gold_units = load_units(args.gold)

        logger.info(f"Text: {len(text)} chars")
        logger.info(f"CAMeL-BERT: {len(camelbert_units)} units")
        logger.info(f"Baseline: {len(baseline_units)} units")
        logger.info(f"Gold: {len(gold_units)} units")

        logger.info("Creating visualization...")
        create_html_visualization(
            text,
            camelbert_units,
            baseline_units,
            gold_units,
            args.output
        )
        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == '__main__':
    main()
