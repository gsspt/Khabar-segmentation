#!/usr/bin/env python3
"""
Convertit les boundary_indices (espace token) du fichier boundary_tokens_clean.json
en positions caractères dans le texte original non-tokenisé.

Stratégie :
  1. Extraire tous les vrais tokens (non-PAD/CLS/SEP) du raw_inference avec leurs
     offsets caractères (déjà globaux dans le fichier).
  2. Déduplication : garder la première occurrence de chaque position char_start.
  3. Numéroter séquentiellement → mapping token_seq_idx → char_offset.
  4. Appliquer aux boundary_indices du fichier clean.
  5. Regrouper les boundary tokens consécutifs/proches en clusters → frontières khabar.
  6. Sauvegarder results/camelbert_char_boundaries.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent

# ── Chargement ────────────────────────────────────────────────────────────────

print("Chargement des fichiers...")
with open(ROOT / 'results/camelbert_kitab_uqala_raw_inference.json') as f:
    raw = json.load(f)['inference_results']

with open(ROOT / 'results/camelbert_boundary_tokens_clean.json') as f:
    clean = json.load(f)

corpus_text = (ROOT / 'data/processed/kitab_uqala_reference_corpus.txt').read_text(encoding='utf-8')
N = len(corpus_text)

boundary_indices = clean['boundary_indices']  # token seq indices → à convertir
boundary_tokens  = clean['boundary_tokens']   # token strings correspondants
print(f"  Corpus        : {N:,} chars")
print(f"  Raw inference : {raw['total_tokens']:,} tokens (avec overlaps/padding)")
print(f"  Boundary idx  : {len(boundary_indices)} (min={min(boundary_indices)} max={max(boundary_indices)})")

# ── Étape 1 : extraire tous les vrais tokens avec offsets ─────────────────────

print("\nConstruction de la séquence dédupliquée...")
SPECIAL = {'[PAD]', '[CLS]', '[SEP]', '[UNK]'}

all_tokens  = raw['tokens']
all_offsets = raw['offsets']

# Collecter les vrais tokens : (char_start, char_end, token_string)
real_tokens = []
seen_char_starts = set()

for tok, off in zip(all_tokens, all_offsets):
    if tok in SPECIAL:
        continue
    cs, ce = off
    if cs == 0 and ce == 0:
        continue          # token spécial ou padding avec offset nul
    if cs in seen_char_starts:
        continue          # dédupliqué : garder première occurrence
    seen_char_starts.add(cs)
    real_tokens.append((cs, ce, tok))

# Trier par position char pour obtenir la séquence naturelle du corpus
real_tokens.sort(key=lambda x: x[0])

print(f"  Tokens réels dédupliqués : {len(real_tokens)}")
print(f"  Attendu (corpus)         : {clean['metadata']['corpus_size_tokens']}")

# Vérifier la couverture des chars
if real_tokens:
    covered_start = real_tokens[0][0]
    covered_end   = real_tokens[-1][1]
    print(f"  Couverture chars         : {covered_start} → {covered_end} / {N}")

# Afficher les premiers tokens pour vérifier
print("\n  Premiers tokens dédupliqués :")
for i, (cs, ce, tok) in enumerate(real_tokens[:10]):
    print(f"    [{i:6d}] char [{cs:6d}-{ce:6d}]  tok={tok}")
print("  ...")
print(f"    [{len(real_tokens)-1:6d}] char [{real_tokens[-1][0]:6d}-{real_tokens[-1][1]:6d}]  tok={real_tokens[-1][2]}")

# ── Étape 2 : mapping seq_idx → char_start ────────────────────────────────────

# seq_idx_to_char[i] = (char_start, char_end) pour le i-ème token dédupliqué
seq_idx_to_char = [(cs, ce) for cs, ce, _ in real_tokens]

# ── Étape 3 : convertir les boundary_indices ──────────────────────────────────

print(f"\nConversion des {len(boundary_indices)} boundary indices...")
max_seq_idx = len(seq_idx_to_char) - 1

boundary_chars = []
skipped = 0
for i, (bidx, btok) in enumerate(zip(boundary_indices, boundary_tokens)):
    if bidx > max_seq_idx:
        skipped += 1
        continue
    cs, ce = seq_idx_to_char[bidx]
    boundary_chars.append({
        'seq_idx':    bidx,
        'token':      btok,
        'char_start': cs,
        'char_end':   ce,
        'text_context': corpus_text[max(0, cs-20):ce+20],
    })

print(f"  Convertis : {len(boundary_chars)}")
print(f"  Ignorés (hors plage) : {skipped}")

# ── Étape 4 : regrouper en clusters ────────────────────────────────────────────

print("\nRegroupement en clusters (GAP ≤ 300 chars)...")
GAP_MAX = 300
boundary_chars.sort(key=lambda b: b['char_start'])

clusters = []
if boundary_chars:
    cur = [boundary_chars[0]]
    for b in boundary_chars[1:]:
        if b['char_start'] - cur[-1]['char_end'] <= GAP_MAX:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)

cluster_starts = [c[0]['char_start'] for c in clusters]
cluster_ends   = [c[-1]['char_end']  for c in clusters]

print(f"  Clusters : {len(clusters)}")

# ── Étape 5 : évaluer vs gold ─────────────────────────────────────────────────

with open(ROOT / 'data/processed/kitab_uqala_boundaries.json') as f:
    gold_data = json.load(f)
gold_bounds = sorted(int(b['start']) for b in gold_data)

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
p  = tp / (tp + fp) if (tp + fp) else 0
r  = tp / (tp + fn) if (tp + fn) else 0
f1 = 2*p*r / (p+r) if (p+r) else 0

print(f"\n  Évaluation clusters vs gold (tol ±{TOL}):")
print(f"    P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
print(f"    TP={tp}  FP={fp}  FN={fn}")
print(f"    Baseline v4 référence : F1=0.846")

# ── Étape 6 : afficher quelques exemples de conversion ───────────────────────

print("\n=== Exemples de conversion ===")
for b in boundary_chars[:15]:
    ctx = b['text_context'].replace('\n', ' ')
    print(f"  seq_idx={b['seq_idx']:6d}  char=[{b['char_start']:6d}-{b['char_end']:5d}]"
          f"  tok={b['token']:15s}  ctx=…{ctx}…")

# ── Étape 7 : sauvegarder ─────────────────────────────────────────────────────

output = {
    'metadata': {
        'source_boundary_tokens': 'results/camelbert_boundary_tokens_clean.json',
        'source_raw_inference':   'results/camelbert_kitab_uqala_raw_inference.json',
        'corpus':                 'data/processed/kitab_uqala_reference_corpus.txt',
        'corpus_chars':           N,
        'dedup_tokens':           len(real_tokens),
        'boundary_tokens_converted': len(boundary_chars),
        'clusters_gap_max':       GAP_MAX,
        'n_clusters':             len(clusters),
        'evaluation': {
            'gold_boundaries': len(gold_bounds),
            'tolerance_chars': TOL,
            'precision': round(p, 4),
            'recall':    round(r, 4),
            'f1':        round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
        }
    },
    'boundary_tokens_with_char_offsets': boundary_chars,
    'clusters': [
        {
            'cluster_id':   i,
            'char_start':   c[0]['char_start'],
            'char_end':     c[-1]['char_end'],
            'n_tokens':     len(c),
            'tokens':       [x['token'] for x in c],
            'text_context': corpus_text[max(0, c[0]['char_start']-10):c[-1]['char_end']+50],
        }
        for i, c in enumerate(clusters)
    ],
}

out_path = ROOT / 'results/camelbert_char_boundaries.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✓ Sauvegardé : {out_path}")
print(f"  {len(boundary_chars)} boundary tokens avec positions char")
print(f"  {len(clusters)} clusters = frontières khabar candidates")
