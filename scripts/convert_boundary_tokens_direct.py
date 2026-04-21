#!/usr/bin/env python3
"""
Conversion correcte des boundary tokens CAMeL-BERT en positions caractères.

Approche : extraction directe depuis raw_inference (offsets globaux),
sans passer par la séquence dédupliquée séquentielle qui causait des décalages.

1. Pour chaque token dans raw_inference avec pred=1, lire son offset global char.
2. Dédupliquer par char_start en gardant la probabilité max.
3. Trier par char_start → séquence de positions boundary dans le corpus.
4. Détecter les khabar boundaries : premier boundary token après un long gap
   (= section sans boundary tokens = matn/prose).
5. Évaluer vs gold standard.
6. Sauvegarder results/camelbert_char_boundaries_v2.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

print("Chargement des fichiers...")
with open(ROOT / "results/camelbert_kitab_uqala_raw_inference.json") as f:
    raw = json.load(f)["inference_results"]

with open(ROOT / "data/processed/kitab_uqala_boundaries.json") as f:
    gold_data = json.load(f)

corpus_text = (ROOT / "data/processed/kitab_uqala_reference_corpus.txt").read_text(
    encoding="utf-8"
)
N = len(corpus_text)
gold_bounds = sorted(int(b["start"]) for b in gold_data)

print(f"  Corpus        : {N:,} chars")
print(f"  Gold bounds   : {len(gold_bounds)}")
print(f"  Total tokens  : {raw['total_tokens']:,}")

# ── Étape 1 : extraction directe par char_start ─────────────────────────────

SPECIAL = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

tokens = raw["tokens"]
offsets = raw["offsets"]
preds = raw["predictions"]
probs = raw["probabilities"]

# Pour chaque char_start unique, garder la proba max et la prédiction associée
char_to_prob = {}
char_to_pred = {}
char_to_tok = {}
char_to_end = {}

for tok, off, pred, prob in zip(tokens, offsets, preds, probs):
    if tok in SPECIAL:
        continue
    cs, ce = off
    if cs == 0 and ce == 0:
        continue
    if cs not in char_to_prob or prob > char_to_prob[cs]:
        char_to_prob[cs] = prob
        char_to_pred[cs] = pred
        char_to_tok[cs] = tok
        char_to_end[cs] = ce

total_unique = len(char_to_prob)
boundary_chars_all = sorted(
    cs for cs, pred in char_to_pred.items() if pred == 1
)
n_boundary = len(boundary_chars_all)

print(f"\n  Positions uniques        : {total_unique:,}")
print(f"  Boundary tokens (pred=1) : {n_boundary:,}")
print(f"  Taux boundary            : {n_boundary/total_unique*100:.1f}%")

# Vérification : gold boundary #2 à char=746
print("\n  Premiers boundary tokens :")
for cs in boundary_chars_all[:20]:
    tok = char_to_tok[cs]
    prob = char_to_prob[cs]
    ctx = corpus_text[max(0, cs - 5) : cs + 30].replace("\n", " ")
    print(f"    char={cs:6d}  prob={prob:.4f}  tok={tok:15s}  …{ctx}…")

# ── Étape 2 : regrouper en clusters de boundary tokens ──────────────────────

print("\nRegroupement des boundary tokens en clusters (GAP_CLUSTER ≤ 50 chars)...")
GAP_CLUSTER = 50  # gap max entre tokens d'un même cluster isnad
clusters = []
if boundary_chars_all:
    cur_start = boundary_chars_all[0]
    cur_end = char_to_end[boundary_chars_all[0]]
    cur_toks = [boundary_chars_all[0]]

    for cs in boundary_chars_all[1:]:
        if cs - cur_end <= GAP_CLUSTER:
            cur_end = max(cur_end, char_to_end[cs])
            cur_toks.append(cs)
        else:
            clusters.append(
                {
                    "char_start": cur_start,
                    "char_end": cur_end,
                    "n_tokens": len(cur_toks),
                    "tokens": [char_to_tok[c] for c in cur_toks[:10]],
                    "text_context": corpus_text[
                        max(0, cur_start - 10) : cur_end + 60
                    ],
                }
            )
            cur_start = cs
            cur_end = char_to_end[cs]
            cur_toks = [cs]
    clusters.append(
        {
            "char_start": cur_start,
            "char_end": cur_end,
            "n_tokens": len(cur_toks),
            "tokens": [char_to_tok[c] for c in cur_toks[:10]],
            "text_context": corpus_text[max(0, cur_start - 10) : cur_end + 60],
        }
    )

print(f"  Clusters isnad : {len(clusters)}")

# Longueurs moyennes
avg_cluster_len = sum(c["char_end"] - c["char_start"] for c in clusters) / max(1, len(clusters))
print(f"  Longueur moy. cluster : {avg_cluster_len:.0f} chars")

# ── Étape 3 : extraire khabar boundaries (début de chaque cluster) ───────────

# Le début de chaque cluster isnad = potentielle frontière de khabar
khabar_boundaries = [c["char_start"] for c in clusters]

# ── Étape 4 : évaluation ─────────────────────────────────────────────────────


def evaluate(predictions: list[int], gold: list[int], tol: int) -> dict:
    """F1 au niveau des frontières avec tolérance ±tol chars."""
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    for i, p in enumerate(predictions):
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            if abs(p - g) <= tol:
                matched_gold.add(j)
                matched_pred.add(i)
                break
    tp = len(matched_gold)
    fp = len(predictions) - len(matched_pred)
    fn = len(gold) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": prec, "R": rec, "F1": f1}


print("\n=== Évaluation khabar boundaries ===")
for tol in (50, 80, 150):
    r = evaluate(khabar_boundaries, gold_bounds, tol)
    print(
        f"  tol=±{tol:3d}  P={r['P']:.3f}  R={r['R']:.3f}  F1={r['F1']:.3f}"
        f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}"
    )

print(f"\n  Référence Baseline v4   : F1=0.846")
print(f"  Prédictions : {len(khabar_boundaries)}   Gold : {len(gold_bounds)}")

# ── Étape 5 : analyse de la distribution des gaps ────────────────────────────

print("\nDistribution des gaps entre clusters (= longueur des khabars) :")
gaps = []
for i in range(1, len(clusters)):
    gap = clusters[i]["char_start"] - clusters[i - 1]["char_end"]
    gaps.append(gap)

if gaps:
    gaps_sorted = sorted(gaps)
    print(f"  Min gap : {min(gaps):,} chars")
    print(f"  Médiane : {gaps_sorted[len(gaps_sorted)//2]:,} chars")
    print(f"  Max gap : {max(gaps):,} chars")
    print(f"  Gaps > 100 chars : {sum(1 for g in gaps if g > 100)}")
    print(f"  Gaps > 500 chars : {sum(1 for g in gaps if g > 500)}")

# ── Étape 6 : exemples de correspondances gold/prédiction ───────────────────

print("\n=== Correspondances gold ↔ prédiction (tol ±80) ===")
r80 = evaluate(khabar_boundaries, gold_bounds, 80)
print(f"  (TP={r80['tp']}, FP={r80['fp']}, FN={r80['fn']})")
print("\n  Premiers gold bounds et correspondances :")
for j, g in enumerate(gold_bounds[:15]):
    best = min(khabar_boundaries, key=lambda p: abs(p - g)) if khabar_boundaries else -1
    dist = abs(best - g)
    match = "✓" if dist <= 80 else "✗"
    ctx = corpus_text[max(0, g - 5) : g + 30].replace("\n", " ")
    print(f"  [{j:3d}] gold={g:6d}  best_pred={best:6d}  dist={dist:4d}  {match}  …{ctx}…")

# ── Étape 7 : sauvegarde ─────────────────────────────────────────────────────

eval_80 = evaluate(khabar_boundaries, gold_bounds, 80)

output = {
    "metadata": {
        "method": "direct_char_extraction_from_raw_inference",
        "description": (
            "Extraction directe des boundary tokens (pred=1) via les offsets "
            "globaux du raw_inference. Déduplication par char_start (proba max). "
            "Clustering en segments isnad (GAP_CLUSTER≤50 chars). "
            "Début de chaque cluster = frontière khabar candidate."
        ),
        "source_raw_inference": "results/camelbert_kitab_uqala_raw_inference.json",
        "corpus": "data/processed/kitab_uqala_reference_corpus.txt",
        "corpus_chars": N,
        "unique_char_positions": total_unique,
        "boundary_tokens_count": n_boundary,
        "gap_cluster_max": GAP_CLUSTER,
        "n_clusters_isnad": len(clusters),
        "n_khabar_boundaries": len(khabar_boundaries),
        "evaluation_tol80": {
            "gold_boundaries": len(gold_bounds),
            "tolerance_chars": 80,
            "precision": round(eval_80["P"], 4),
            "recall": round(eval_80["R"], 4),
            "f1": round(eval_80["F1"], 4),
            "tp": eval_80["tp"],
            "fp": eval_80["fp"],
            "fn": eval_80["fn"],
        },
        "baseline_v4_f1": 0.846,
    },
    "khabar_boundaries": [
        {
            "boundary_id": i,
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "n_tokens": c["n_tokens"],
            "tokens": c["tokens"],
            "text_context": c["text_context"],
        }
        for i, c in enumerate(clusters)
    ],
}

out_path = ROOT / "results/camelbert_char_boundaries_v2.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✓ Sauvegardé : {out_path}")
print(f"  {len(clusters)} clusters isnad → {len(khabar_boundaries)} khabar boundaries")
print(f"  F1={eval_80['F1']:.3f}  P={eval_80['P']:.3f}  R={eval_80['R']:.3f}  (tol ±80)")
