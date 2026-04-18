#!/usr/bin/env python3
"""
Classifieur second-pass : filtre TP vs FP parmi les candidats de baseline_v4.

Principe : le système rule-based génère 575 candidats (502 TP + 73 FP).
Un classifieur binaire (LR / RF) est entraîné sur les features de chaque
candidat pour distinguer les vrais akhbars des faux positifs.

Évaluation : cross-validation 5-fold + prédictions out-of-fold (OOF).
La métrique cible est le F1 sur les frontières, pas l'accuracy du classifieur.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline_v4 import (
    norm, segment, find_isnad_starts, particle_density,
    _is_chain_verb, _build_ref_boundaries,
)

# ── Catégories de verbes par qualité ─────────────────────────────────────────

_HIGH_VERBS = frozenset([
    'اخبرنا', 'اخبرني', 'واخبرنا', 'واخبرني',
    'وبهذا', 'ومنهم', 'ومنها', 'انشدنا', 'ولبعضهم',
])
_MED_VERBS = frozenset([
    'حدثنا', 'انشدني', 'وسمعت', 'وحكي', 'وبلغني',
    'وحدثني', 'وانشدتنا', 'وانشدنا', 'وانشدني', 'بلغنا',
])

FEATURE_NAMES = [
    'verb_tier',
    'log_isnad_len', 'log_khabar_len',
    'vtrans_count', 'vtrans_density',
    'an_count',
    'has_qal', 'qal_pos_ratio',
    'name_markers',
    'part_khabar',
    'n_isnad_toks', 'n_khabar_toks',
    'pos_frac',
]


# ── Extraction de features ────────────────────────────────────────────────────

def extract_features(akhbar: dict, verb: str, corpus_len: int) -> list:
    ti = norm(akhbar['isnad'])
    tk = norm(akhbar['khabar'])
    toks_i = ti.split()
    toks_k = tk.split()
    ni = max(len(toks_i), 1)
    nk = max(len(toks_k), 1)

    verb_tier = 2 if verb in _HIGH_VERBS else (1 if verb in _MED_VERBS else 0)

    il = len(ti)
    kl = len(tk)

    # Verbes de transmission dans l'isnad (profondeur de chaîne)
    vtc = sum(1 for w in toks_i if _is_chain_verb(w))
    vtd = vtc / ni

    # Préposition عن (maillon de chaîne typique)
    anc = toks_i.count('عن')

    # قال en fin d'isnad (frontière isnad→matn)
    has_qal = int('قال' in ti[-100:] or 'يقول' in ti[-100:])
    qal_ratio = ti.rfind('قال') / max(il, 1)

    # Indicateurs de noms propres dans l'isnad
    nm = sum(1 for m in ['ابن', 'ابو', 'بن', 'ابنه', 'بنت'] if m in toks_i)

    # Densité de particules narratives dans le khabar
    pk = particle_density(tk)

    # Position dans le corpus (début vs fin du texte)
    pf = akhbar['start'] / corpus_len

    return [
        verb_tier,
        np.log1p(il), np.log1p(kl),
        vtc, vtd,
        anc,
        has_qal, qal_ratio,
        nm, pk,
        ni, nk,
        pf,
    ]


# ── Étiquetage TP/FP ─────────────────────────────────────────────────────────

def label_candidates(akhbars: list, ref_boundaries: list, tol: int = 80) -> np.ndarray:
    """1 = TP (frontière à ±tol d'une référence), 0 = FP."""
    pred = sorted(a['start'] for a in akhbars)
    ref  = sorted(ref_boundaries)
    matched_ref = set()
    matched_pred = set()
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if j in matched_ref:
                continue
            if abs(p - r) <= tol:
                matched_ref.add(j)
                matched_pred.add(i)
                break
    return np.array([1 if i in matched_pred else 0 for i in range(len(akhbars))])


# ── Métriques F1 frontières ───────────────────────────────────────────────────

def boundary_f1(akhbars: list, mask, ref_boundaries: list, total_ref: int, tol: int = 80):
    """
    Calcule P/R/F1 sur les frontières après filtrage par `mask`.
    `total_ref` inclut les FN fixes du système rule-based.
    """
    selected = [a for a, s in zip(akhbars, mask) if s]
    pred = sorted(a['start'] for a in selected)
    ref  = sorted(ref_boundaries)

    matched_ref = set()
    matched_pred = set()
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if j in matched_ref:
                continue
            if abs(p - r) <= tol:
                matched_ref.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_ref)
    fp = len(pred) - len(matched_pred)
    fn = total_ref - tp
    p  = tp / (tp + fp) if (tp + fp) else 0
    r  = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return p, r, f1, tp, fp, fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    corpus_path = Path('data/processed/kitab_uqala_reference_corpus.txt')
    annot_path  = Path('data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    corpus = corpus_path.read_text(encoding='utf-8')
    annot  = json.loads(annot_path.read_text(encoding='utf-8'))
    ref_boundaries = _build_ref_boundaries(annot, corpus)
    total_ref = annot['total']

    # ── Candidats rule-based ──────────────────────────────────────────────────
    t = norm(corpus)
    start_to_verb = {p: v for p, v in find_isnad_starts(t)}
    akhbars = segment(corpus)
    verbs   = [start_to_verb.get(a['start'], 'وقال') for a in akhbars]
    labels  = label_candidates(akhbars, ref_boundaries, tol=80)

    n_tp = int(labels.sum())
    n_fp = len(labels) - n_tp
    print(f"Candidats rule-based : {len(akhbars)}  TP={n_tp}  FP={n_fp}")

    # ── Baseline sans classifieur ─────────────────────────────────────────────
    p0, r0, f0, tp0, fp0, fn0 = boundary_f1(
        akhbars, [True] * len(akhbars), ref_boundaries, total_ref
    )
    print(f"Baseline rule-based  : P={p0:.3f}  R={r0:.3f}  F1={f0:.3f}"
          f"  TP={tp0}  FP={fp0}  FN={fn0}\n")

    # ── Features ─────────────────────────────────────────────────────────────
    X = np.array([extract_features(a, v, len(corpus))
                  for a, v in zip(akhbars, verbs)])
    y = labels

    # ── Modèles ──────────────────────────────────────────────────────────────
    models = {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced', max_iter=1000, C=0.5,
            )),
        ]),
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, class_weight='balanced',
                max_depth=6, random_state=42,
            )),
        ]),
    }

    # ── Cross-validation 5-fold (OOF) ────────────────────────────────────────
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print('─' * 60)
    print('Cross-validation 5-fold — prédictions out-of-fold (OOF)')
    print('─' * 60)

    best_name = None
    best_f1   = 0.0
    best_oof  = None

    for name, model in models.items():
        oof_pred = np.zeros(len(akhbars), dtype=bool)

        for train_idx, test_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            for local_i, global_i in enumerate(test_idx):
                oof_pred[global_i] = bool(y_pred[local_i])

        p, r, f1, tp, fp, fn = boundary_f1(
            akhbars, oof_pred.tolist(), ref_boundaries, total_ref
        )
        print(f'\n{name} (OOF):')
        print(f'  P={p:.3f}  R={r:.3f}  F1={f1:.3f}'
              f'  TP={tp}  FP={fp}  FN={fn}')
        delta = f1 - f0
        sign  = '+' if delta >= 0 else ''
        print(f'  Δ F1 vs baseline : {sign}{delta:+.3f}')

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_oof  = oof_pred.copy()

    # ── Sweep de seuil (meilleur modèle, entraîné sur tout le corpus) ─────────
    print(f'\n{"─"*60}')
    print(f'Sweep de seuil — {best_name} (entraîné sur tout le corpus)')
    print('(attention : évaluation in-sample, optimiste)')
    print('─' * 60)

    best_model = models[best_name]
    best_model.fit(X, y)

    if hasattr(best_model[-1], 'predict_proba'):
        proba = best_model.predict_proba(X)[:, 1]
    else:
        proba = best_model.predict(X).astype(float)

    print(f'{"Seuil":>6}  {"P":>6}  {"R":>6}  {"F1":>6}  {"TP":>4}  {"FP":>4}  {"FN":>4}')
    for thresh in np.arange(0.1, 0.91, 0.1):
        mask = (proba >= thresh).tolist()
        p, r, f1, tp, fp, fn = boundary_f1(
            akhbars, mask, ref_boundaries, total_ref
        )
        marker = '  ← OOF optimal' if abs(thresh - 0.5) < 0.05 else ''
        print(f'  {thresh:.1f}   {p:.3f}  {r:.3f}  {f1:.3f}'
              f'  {tp:>4}  {fp:>4}  {fn:>4}{marker}')

    # ── Importance des features (RF) ──────────────────────────────────────────
    if 'Forest' in best_name:
        clf = best_model.named_steps['clf']
        print(f'\nImportance des features ({best_name}) :')
        ranked = sorted(zip(clf.feature_importances_, FEATURE_NAMES), reverse=True)
        for imp, fname in ranked:
            bar = '█' * int(imp * 80)
            print(f'  {fname:<20} {imp:.3f}  {bar}')

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'RÉSUMÉ')
    print(f'{"="*60}')
    p_oof, r_oof, f_oof, tp_oof, fp_oof, fn_oof = boundary_f1(
        akhbars, best_oof.tolist(), ref_boundaries, total_ref
    )
    print(f'Baseline rule-based : F1={f0:.3f}  P={p0:.3f}  R={r0:.3f}')
    print(f'+ {best_name} (OOF)  : F1={f_oof:.3f}  P={p_oof:.3f}  R={r_oof:.3f}')
    print(f'Gain OOF            : {f_oof - f0:+.3f} F1')


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
