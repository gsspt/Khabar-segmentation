#!/usr/bin/env python3
"""
Visualisation du mapping isnad/khabar produit par CAMeL-BERT.

Sorties :
  results/camelbert_isnad_mapping.png   (figure 4 panneaux)
  results/camelbert_isnad_mapping.html  (texte coloré navigable)
"""

import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent

# ── Chargement ────────────────────────────────────────────────────────────────

with open(ROOT / 'results/camelbert_kitab_uqala_segments_FINAL_TOP800.json') as f:
    data = json.load(f)
segments = data['segmentation']['segments']

corpus = (ROOT / 'data/processed/kitab_uqala_reference_corpus.txt').read_text(encoding='utf-8')
N = len(corpus)

# Frontières gold
with open(ROOT / 'data/processed/kitab_uqala_boundaries.json') as f:
    gd = json.load(f)
gold_bounds = sorted(int(b['start']) for b in gd)

# Frontières baseline v4
bl_path = ROOT / 'results/baseline_v4_boundaries.json'
if not bl_path.exists():
    sys.path.insert(0, str(ROOT / 'scripts/baselines'))
    from baseline_v4 import segment
    akhbars = segment(corpus)
    bl_bounds = sorted(a['start'] for a in akhbars)
    json.dump(bl_bounds, open(bl_path, 'w'))
else:
    bl_bounds = json.load(open(bl_path))

print(f"Corpus    : {N:,} chars")
print(f"Segments  : {len(segments)} CAMeL-BERT")
print(f"Gold      : {len(gold_bounds)} frontières")
print(f"Baseline  : {len(bl_bounds)} frontières")

# ── Tableau de labels char-par-char ──────────────────────────────────────────

labels = np.zeros(N, dtype=np.uint8)   # 0=prose, 1=isnad
for seg in segments:
    s, e = seg['start'], min(seg['end'], N)
    if seg['type'] == 'isnad':
        labels[s:e] = 1

# ── Clusters isnad (regroupement des micro-segments consécutifs) ──────────────
# Un cluster = suite de segments isnad dont les gaps sont < GAP_MAX chars

GAP_MAX = 300   # chars de prose entre deux fragments isnad → même cluster
isnad_segs = [s for s in segments if s['type'] == 'isnad']
isnad_segs.sort(key=lambda s: s['start'])

clusters = []
if isnad_segs:
    cur_start = isnad_segs[0]['start']
    cur_end   = isnad_segs[0]['end']
    for seg in isnad_segs[1:]:
        if seg['start'] - cur_end <= GAP_MAX:
            cur_end = seg['end']
        else:
            clusters.append((cur_start, cur_end))
            cur_start = seg['start']
            cur_end   = seg['end']
    clusters.append((cur_start, cur_end))

cluster_starts = [c[0] for c in clusters]
print(f"Clusters  : {len(clusters)} (gap_max={GAP_MAX} chars) → proxy de frontières CAMeL")

# Évaluer les clusters vs gold (tolérance ±80)
TOL = 80
matched_gold = set()
matched_cls  = set()
for i, cs in enumerate(cluster_starts):
    for j, g in enumerate(gold_bounds):
        if j in matched_gold: continue
        if abs(cs - g) <= TOL:
            matched_gold.add(j)
            matched_cls.add(i)
            break
tp = len(matched_gold)
fp = len(cluster_starts) - len(matched_cls)
fn = len(gold_bounds) - tp
p = tp/(tp+fp) if tp+fp else 0
r = tp/(tp+fn) if tp+fn else 0
f1 = 2*p*r/(p+r) if p+r else 0
print(f"Clusters vs gold (tol={TOL}): P={p:.3f} R={r:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

# ── Densité lissée ────────────────────────────────────────────────────────────

WINDOW, STEP = 2000, 200
positions = np.arange(0, N - WINDOW, STEP)
density   = np.array([labels[p:p+WINDOW].mean() for p in positions])
x_dens    = positions + WINDOW // 2

# ── Figure ────────────────────────────────────────────────────────────────────

C_ISNAD = '#f4a261'   # orange
C_PROSE = '#264653'   # bleu-vert
C_GOLD  = '#e9c46a'   # or
C_BL    = '#90e0ef'   # bleu ciel → baseline
C_CLUST = '#ff6b6b'   # rouge → cluster starts
C_BG    = '#1a1a2e'
C_TEXT  = '#edf2f4'
C_DENS  = '#a8dadc'

fig = plt.figure(figsize=(24, 14), facecolor=C_BG)

def style_ax(ax, title='', xlabel=''):
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values(): sp.set_color('#444')
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.yaxis.label.set_color(C_TEXT)
    if title: ax.set_title(title, color=C_TEXT, fontsize=10, pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=C_TEXT, fontsize=9)

# ── P1 : Barcode complet ──────────────────────────────────────────────────────
ax1 = fig.add_axes([0.04, 0.83, 0.92, 0.10], facecolor=C_BG)

prose_bars = [(s['start'], max(s['length'],1)) for s in segments if s['type']=='prose']
isnad_bars = [(s['start'], max(s['length'],1)) for s in segments if s['type']=='isnad']
ax1.broken_barh(prose_bars, (0.5, 1), facecolors=C_PROSE, linewidth=0)
ax1.broken_barh(isnad_bars, (0.5, 1), facecolors=C_ISNAD, linewidth=0)

for b in gold_bounds: ax1.axvline(b, color=C_GOLD, lw=0.35, alpha=0.8, ymin=0, ymax=0.48)
for b in bl_bounds:   ax1.axvline(b, color=C_BL,   lw=0.35, alpha=0.6, ymin=0.52, ymax=1.0)

ax1.set_xlim(0, N); ax1.set_ylim(0, 2); ax1.set_yticks([]); ax1.set_xticks([])
style_ax(ax1, 'Barcode — isnad (orange) / prose (bleu)  |  Bas : gold (or)  Haut : baseline (bleu ciel)')

# ── P2 : Densité + clusters ───────────────────────────────────────────────────
ax2 = fig.add_axes([0.04, 0.57, 0.92, 0.22], facecolor=C_BG)

ax2.fill_between(x_dens, density, alpha=0.30, color=C_DENS)
ax2.plot(x_dens, density, color=C_DENS, lw=1.2)

for b in gold_bounds:  ax2.axvline(b, color=C_GOLD,  lw=0.4, alpha=0.5)
for b in bl_bounds:    ax2.axvline(b, color=C_BL,    lw=0.4, alpha=0.4)
for cs, ce in clusters:
    ax2.axvspan(cs, ce, alpha=0.3, color=C_CLUST, linewidth=0)
    ax2.axvline(cs, color=C_CLUST, lw=0.7, alpha=0.8)

ax2.set_xlim(0, N)
ax2.set_ylim(0, density.max()*1.4)
ax2.set_ylabel('Densité isnad', color=C_TEXT, fontsize=8)
ax2.set_xticks([])
style_ax(ax2, f'Densité isnad (fenêtre {WINDOW} chars) + clusters regroupés (rouge, gap≤{GAP_MAX} chars)')
ax2.text(0.01, 0.92, f'{len(clusters)} clusters = frontières détectées  |  '
         f'P={p:.3f} R={r:.3f} F1={f1:.3f}  (tol ±{TOL} chars)',
         transform=ax2.transAxes, color=C_CLUST, fontsize=8.5, va='top')

# ── P3 : Zoom premier tiers ───────────────────────────────────────────────────
ZOOM = N // 3
ax3 = fig.add_axes([0.04, 0.25, 0.62, 0.27], facecolor=C_BG)

pb3 = [(s['start'], max(s['length'],1)) for s in segments if s['type']=='prose' and s['start']<ZOOM]
ib3 = [(s['start'], max(s['length'],1)) for s in segments if s['type']=='isnad' and s['start']<ZOOM]
ax3.broken_barh(pb3, (0, 1), facecolors=C_PROSE, linewidth=0)
ax3.broken_barh(ib3, (0, 1), facecolors=C_ISNAD, linewidth=0)

for b in [g for g in gold_bounds if g < ZOOM]: ax3.axvline(b, color=C_GOLD, lw=0.8, alpha=0.85)
for b in [b for b in bl_bounds   if b < ZOOM]: ax3.axvline(b, color=C_BL,   lw=0.8, alpha=0.6)
for cs,ce in [(c,e) for c,e in clusters if c < ZOOM]:
    ax3.axvspan(cs, min(ce,ZOOM), alpha=0.25, color=C_CLUST, linewidth=0)

ax3.set_xlim(0, ZOOM); ax3.set_ylim(0, 1); ax3.set_yticks([])
ax3.tick_params(colors=C_TEXT)
style_ax(ax3, f'Zoom premier tiers (0–{ZOOM:,} chars)',
         xlabel='Position dans le corpus (chars)')

# ── P4 : Statistiques + F1 comparatif ────────────────────────────────────────
ax4 = fig.add_axes([0.71, 0.25, 0.25, 0.27], facecolor='#0d1117')
ax4.set_xticks([]); ax4.set_yticks([])
for sp in ax4.spines.values(): sp.set_color('#333')
style_ax(ax4, 'Comparaison des systèmes')

isnad_chars = sum(s['length'] for s in segments if s['type']=='isnad')
prose_chars  = sum(s['length'] for s in segments if s['type']=='prose')

rows = [
    ('', ''),
    ('── Mapping CAMeL-BERT ──', ''),
    ('Segments isnad', f'{sum(1 for s in segments if s["type"]=="isnad")}'),
    ('  chars isnad', f'{isnad_chars:,} ({100*isnad_chars/N:.1f}%)'),
    ('Segments prose', f'{sum(1 for s in segments if s["type"]=="prose")}'),
    ('  chars prose', f'{prose_chars:,} ({100*prose_chars/N:.1f}%)'),
    ('', ''),
    ('── Clusters (gap ≤ 300) ──', ''),
    ('N clusters', f'{len(clusters)}'),
    ('  P (tol ±80)', f'{p:.3f}'),
    ('  R (tol ±80)', f'{r:.3f}'),
    ('  F1', f'{f1:.3f}'),
    ('', ''),
    ('── Référence ──', ''),
    ('Gold standard', f'{len(gold_bounds)}'),
    ('Baseline v4', f'{len(bl_bounds)}'),
    ('  (F1=0.846)', ''),
]

y = 0.97
for label, val in rows:
    if label == '':
        y -= 0.025; continue
    color = C_TEXT if '──' not in label else '#a8dadc'
    ax4.text(0.04, y, label, transform=ax4.transAxes,
             color=color, fontsize=7.5, va='top',
             fontweight='bold' if '──' in label else 'normal')
    if val:
        ax4.text(0.97, y, val, transform=ax4.transAxes,
                 color=C_TEXT, fontsize=7.5, va='top', ha='right')
    y -= 0.055

# ── Légende ───────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C_ISNAD, label='Isnad (boundary tokens)'),
    mpatches.Patch(color=C_PROSE, label='Prose / Khabar'),
    mpatches.Patch(color=C_GOLD,  label=f'Frontières gold ({len(gold_bounds)})'),
    mpatches.Patch(color=C_BL,    label=f'Baseline v4 ({len(bl_bounds)})'),
    mpatches.Patch(color=C_CLUST, label=f'Clusters CAMeL-BERT ({len(clusters)}) — F1={f1:.3f}'),
]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.50, 0.01),
           ncol=5, fontsize=8.5, facecolor='#111', edgecolor='#444',
           labelcolor=C_TEXT, framealpha=0.9)

fig.suptitle('CAMeL-BERT — Mapping isnad / khabar\n'
             'Kitab ʿUqala al-Majanin · IbnHabibNaysaburi (0406 AH)',
             color=C_TEXT, fontsize=13, y=0.99)

out_png = ROOT / 'results/camelbert_isnad_mapping.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f"\n✓ PNG → {out_png}")

# ── HTML ──────────────────────────────────────────────────────────────────────

gold_set    = set(gold_bounds)
cluster_set = set(cluster_starts)
bl_set      = set(bl_bounds)

# Construire label char-par-char
label_char = ['P'] * N
for seg in segments:
    s, e = seg['start'], min(seg['end'], N)
    if seg['type'] == 'isnad':
        for i in range(s, e): label_char[i] = 'I'

html = [
    '<!DOCTYPE html><html lang="ar"><head><meta charset="utf-8">',
    '<title>CAMeL-BERT Mapping</title><style>',
    'body{background:#1a1a2e;color:#edf2f4;font-family:serif;direction:rtl;',
    'font-size:15px;line-height:2;padding:20px 40px;max-width:1300px;margin:auto}',
    'h1{color:#a8dadc;text-align:center;font-size:1.3em}',
    'p.meta{color:#8899aa;text-align:center;font-size:12px;margin:-8px 0 16px}',
    '.I{background:#f4a261;color:#1a1a2e;border-radius:2px;padding:0 1px}',
    '.P{color:#edf2f4}',
    '.gold{border-right:3px solid #e9c46a;padding-right:2px}',
    '.bl{border-right:3px solid #90e0ef;padding-right:2px}',
    '.cl{border-right:3px solid #ff6b6b;padding-right:2px}',
    '.block{background:#0d1117;border-radius:6px;padding:14px;margin:10px 0;',
    'border:1px solid #264653}',
    '.leg{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;',
    'background:#0d1117;padding:10px;border-radius:6px;margin:10px 0;font-size:13px}',
    '.sw{width:16px;height:16px;display:inline-block;border-radius:2px;',
    'vertical-align:middle;margin-left:4px}',
    '</style></head><body>',
    '<h1>CAMeL-BERT — Mapping isnad / khabar</h1>',
    f'<p class="meta">Kitab ʿUqala al-Majanin · {N:,} chars · '
    f'{len(segments)} segments · {len(clusters)} clusters · {len(gold_bounds)} gold</p>',
    '<div class="leg">',
    '<span><span class="sw" style="background:#f4a261"></span> Isnad (boundary)</span>',
    '<span><span class="sw" style="background:#264653;border:1px solid #a8dadc"></span> Prose/Khabar</span>',
    '<span><span class="sw" style="background:#e9c46a"></span> Frontière gold ⟨★⟩</span>',
    '<span><span class="sw" style="background:#90e0ef"></span> Baseline v4 ⟨▶⟩</span>',
    '<span><span class="sw" style="background:#ff6b6b"></span> Cluster CAMeL ⟨●⟩</span>',
    '</div>',
]

def escape(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

# Découper en blocs de ~5000 chars
CHUNK = 5000
for block_start in range(0, min(N, N//2), CHUNK):   # premier moitié seulement
    block_end = min(block_start + CHUNK, N)
    html.append(f'<div class="block"><p style="color:#8899aa;font-size:11px">'
                f'[{block_start:,} – {block_end:,}]</p>')

    # Grouper par spans continus de même type
    spans = []
    cur_type = label_char[block_start]
    cur_s = block_start
    for i in range(block_start, block_end):
        if label_char[i] != cur_type:
            spans.append((cur_s, i, cur_type))
            cur_type = label_char[i]
            cur_s = i
    spans.append((cur_s, block_end, cur_type))

    buf = []
    for s, e, t in spans:
        txt = escape(corpus[s:e])
        # Marqueurs en début de span
        marks = ''
        if s in gold_set:    marks += '<span style="color:#e9c46a;font-weight:bold">⟨★⟩</span>'
        if s in bl_set:      marks += '<span style="color:#90e0ef;font-weight:bold">⟨▶⟩</span>'
        if s in cluster_set: marks += '<span style="color:#ff6b6b;font-weight:bold">⟨●⟩</span>'
        buf.append(f'<span class="{t}">{marks}{txt}</span>')

    html.append(''.join(buf))
    html.append('</div>')

html.append('</body></html>')

out_html = ROOT / 'results/camelbert_isnad_mapping.html'
out_html.write_text(''.join(html), encoding='utf-8')
print(f"✓ HTML → {out_html}")
print(f"\n{'='*55}")
print(f"Clusters CAMeL-BERT (gap≤{GAP_MAX}) vs gold (tol ±{TOL}):")
print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
print(f"  TP={tp}  FP={fp}  FN={fn}")
print(f"  Baseline v4 : F1=0.846 (référence)")
