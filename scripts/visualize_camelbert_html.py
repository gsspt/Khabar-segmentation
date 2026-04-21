#!/usr/bin/env python3
"""
Génère une visualisation HTML du corpus kitab_uqala avec les isnads
détectés par CAMeL-BERT surlignés et les frontières khabar marquées.

Légende :
  - Fond jaune   : segment isnad détecté par CAMeL-BERT
  - Pastille ●   : début de khabar (frontière détectée)
  - Pastille ◆   : frontière gold (annotation manuelle)
  - Fond rouge   : frontière gold manquée (FN)
  - Fond vert    : frontière gold bien détectée (TP)
"""

import json
import html
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Chargement ────────────────────────────────────────────────────────────────

with open(ROOT / "results/camelbert_char_boundaries_v2.json") as f:
    data = json.load(f)

with open(ROOT / "data/processed/kitab_uqala_boundaries.json") as f:
    gold_data = json.load(f)

corpus = (ROOT / "data/processed/kitab_uqala_reference_corpus.txt").read_text(
    encoding="utf-8"
)

gold_bounds = sorted(int(b["start"]) for b in gold_data)
clusters = data["khabar_boundaries"]

TOL = 80

# ── Matching TP/FP/FN ─────────────────────────────────────────────────────────

preds = [c["char_start"] for c in clusters]

matched_gold: set[int] = set()
matched_pred: set[int] = set()
for i, p in enumerate(preds):
    for j, g in enumerate(gold_bounds):
        if j in matched_gold:
            continue
        if abs(p - g) <= TOL:
            matched_gold.add(j)
            matched_pred.add(i)
            break

tp_preds = matched_pred                           # indices clusters qui matchent gold
fp_preds = set(range(len(preds))) - matched_pred  # clusters sans gold proche
fn_golds = set(range(len(gold_bounds))) - matched_gold  # gold non détectés

# ── Construction des intervalles surlignés ────────────────────────────────────

# Chaque cluster isnad = intervalle [char_start, char_end] à surligner en jaune
isnad_spans: list[tuple[int, int, int]] = [
    (c["char_start"], c["char_end"], i) for i, c in enumerate(clusters)
]

# Événements à insérer dans le texte : (position, type, metadata)
# Types : 'isnad_open', 'isnad_close', 'khabar_boundary', 'gold_boundary'
events: list[tuple[int, str, dict]] = []

for i, (cs, ce, cluster_idx) in enumerate(isnad_spans):
    is_fp = cluster_idx in fp_preds
    events.append((cs, "isnad_open", {"fp": is_fp, "idx": cluster_idx, "n_tokens": clusters[cluster_idx]["n_tokens"]}))
    events.append((ce, "isnad_close", {"idx": cluster_idx}))

for i, p in enumerate(preds):
    events.append((p, "khabar_pred", {"fp": i in fp_preds, "idx": i}))

for j, g in enumerate(gold_bounds):
    events.append((g, "gold", {"fn": j in fn_golds, "idx": j}))

# Trier : à position égale, fermetures avant ouvertures, gold avant pred
TYPE_ORDER = {"isnad_close": 0, "gold": 1, "khabar_pred": 2, "isnad_open": 3}
events.sort(key=lambda e: (e[0], TYPE_ORDER.get(e[1], 9)))

# ── Rendu HTML ────────────────────────────────────────────────────────────────

def make_html(corpus: str, events: list) -> str:
    parts: list[str] = []
    pos = 0
    in_isnad = False

    for ev_pos, ev_type, meta in events:
        if ev_pos > len(corpus):
            ev_pos = len(corpus)
        if ev_pos < pos:
            continue

        # Texte avant l'événement
        chunk = corpus[pos:ev_pos]
        escaped = html.escape(chunk).replace("\n", "<br>")
        parts.append(escaped)
        pos = ev_pos

        if ev_type == "isnad_open":
            fp = meta["fp"]
            n = meta["n_tokens"]
            color = "#fff3cd" if not fp else "#ffd6d6"  # jaune ou rouge clair pour FP
            title = f"Isnad détecté ({n} tokens)" + (" — FP" if fp else " — TP")
            parts.append(f'<mark class="isnad{"_fp" if fp else ""}" title="{title}" style="background:{color}">')
            in_isnad = True

        elif ev_type == "isnad_close":
            if in_isnad:
                parts.append("</mark>")
                in_isnad = False

        elif ev_type == "khabar_pred":
            fp = meta["fp"]
            idx = meta["idx"]
            css = "boundary_fp" if fp else "boundary_tp"
            label = f"K{idx+1}"
            tooltip = f"Frontière khabar #{idx+1}" + (" (FP)" if fp else " (TP)")
            parts.append(f'<span class="khabar_marker {css}" title="{tooltip}">{label}</span>')

        elif ev_type == "gold":
            fn = meta["fn"]
            idx = meta["idx"]
            css = "gold_fn" if fn else "gold_tp"
            tooltip = f"Gold #{idx+1}" + (" (FN — manquée)" if fn else " (TP)")
            parts.append(f'<span class="gold_marker {css}" title="{tooltip}">◆</span>')

    # Texte restant
    remaining = corpus[pos:]
    parts.append(html.escape(remaining).replace("\n", "<br>"))

    return "".join(parts)


body = make_html(corpus, events)

# Statistiques globales
n_tp = len(tp_preds)
n_fp = len(fp_preds)
n_fn = len(fn_golds)
prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0
rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

html_doc = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAMeL-BERT — Isnads détectés — Kitab Uqala</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Amiri", "Scheherazade New", "Traditional Arabic", "Arial", serif;
    font-size: 17px;
    line-height: 2.1;
    background: #fafafa;
    color: #1a1a1a;
    direction: rtl;
  }}
  #header {{
    background: #1e293b;
    color: white;
    padding: 18px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
    direction: ltr;
  }}
  #header h1 {{ font-size: 16px; font-weight: 700; flex: 1; }}
  .stat-box {{
    background: rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    white-space: nowrap;
  }}
  .stat-box b {{ font-size: 15px; }}
  #legend {{
    background: white;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px 32px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    font-size: 13px;
    direction: ltr;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{
    display: inline-block;
    width: 22px; height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(0,0,0,0.15);
  }}
  #corpus {{
    max-width: 860px;
    margin: 32px auto;
    padding: 32px 40px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: justify;
  }}
  mark.isnad {{
    background: #fff3cd;
    border-radius: 3px;
    padding: 1px 0;
    border-bottom: 2px solid #f59e0b;
  }}
  mark.isnad_fp {{
    background: #ffd6d6;
    border-radius: 3px;
    padding: 1px 0;
    border-bottom: 2px solid #ef4444;
  }}
  .khabar_marker {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    border-radius: 4px;
    padding: 1px 4px;
    margin: 0 2px;
    vertical-align: middle;
    cursor: default;
    direction: ltr;
  }}
  .boundary_tp {{
    background: #22c55e;
    color: white;
  }}
  .boundary_fp {{
    background: #ef4444;
    color: white;
  }}
  .gold_marker {{
    font-size: 11px;
    margin: 0 1px;
    vertical-align: middle;
    cursor: default;
  }}
  .gold_tp {{ color: #16a34a; }}
  .gold_fn {{ color: #dc2626; }}
  #controls {{
    max-width: 860px;
    margin: 16px auto 0;
    padding: 0 40px;
    direction: ltr;
    font-size: 13px;
    color: #64748b;
  }}
  #controls label {{ margin-left: 16px; cursor: pointer; }}
  #controls input {{ cursor: pointer; }}
</style>
</head>
<body>

<div id="header">
  <h1>CAMeL-BERT · Isnads détectés · Kitab al-Uqala al-Majanin</h1>
  <div class="stat-box">Précision <b>{prec:.1%}</b></div>
  <div class="stat-box">Rappel <b>{rec:.1%}</b></div>
  <div class="stat-box">F1 <b>{f1:.3f}</b></div>
  <div class="stat-box">TP <b>{n_tp}</b> · FP <b>{n_fp}</b> · FN <b>{n_fn}</b></div>
  <div class="stat-box">Tol ±{TOL} chars</div>
</div>

<div id="legend">
  <div class="legend-item">
    <span class="legend-swatch" style="background:#fff3cd;border-bottom:2px solid #f59e0b"></span>
    Isnad détecté (TP)
  </div>
  <div class="legend-item">
    <span class="legend-swatch" style="background:#ffd6d6;border-bottom:2px solid #ef4444"></span>
    Isnad détecté (FP)
  </div>
  <div class="legend-item">
    <span class="khabar_marker boundary_tp" style="font-size:11px">K</span>
    Frontière khabar détectée (TP)
  </div>
  <div class="legend-item">
    <span class="khabar_marker boundary_fp" style="font-size:11px">K</span>
    Frontière khabar détectée (FP)
  </div>
  <div class="legend-item">
    <span class="gold_marker gold_tp">◆</span>
    Gold TP (bien détectée)
  </div>
  <div class="legend-item">
    <span class="gold_marker gold_fn">◆</span>
    Gold FN (manquée)
  </div>
</div>

<div id="controls">
  <label><input type="checkbox" id="toggle_isnad" checked onchange="toggleClass('isnad','hidden')"> Surligner isnads</label>
  <label><input type="checkbox" id="toggle_pred" checked onchange="toggleMarkers('khabar_marker','hidden')"> Frontières prédites</label>
  <label><input type="checkbox" id="toggle_gold" checked onchange="toggleMarkers('gold_marker','hidden')"> Frontières gold</label>
</div>

<div id="corpus">
{body}
</div>

<script>
function toggleClass(cls, hidden) {{
  document.querySelectorAll('.' + cls).forEach(el => el.classList.toggle(hidden));
}}
function toggleMarkers(cls, hidden) {{
  document.querySelectorAll('.' + cls).forEach(el => el.classList.toggle(hidden));
}}
document.querySelector('#toggle_isnad').addEventListener('change', function() {{
  document.querySelectorAll('mark.isnad, mark.isnad_fp').forEach(el => {{
    el.style.background = this.checked ? '' : 'transparent';
    el.style.borderBottom = this.checked ? '' : 'none';
  }});
}});
document.querySelectorAll('.khabar_marker, .gold_marker').forEach(el => {{
  el.addEventListener('click', function() {{
    alert(this.title);
  }});
}});
</script>

</body>
</html>
"""

out_path = ROOT / "results/camelbert_isnads_visualization.html"
out_path.write_text(html_doc, encoding="utf-8")
print(f"✓ HTML généré : {out_path}")
print(f"  {len(corpus):,} chars · {len(clusters)} isnads · {len(gold_bounds)} gold boundaries")
print(f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (TP={n_tp} FP={n_fp} FN={n_fn})")
