#!/usr/bin/env python3
"""
Baseline v4 — segmentation isnad-khabar sans dépendance à la ponctuation éditoriale.

Améliorations vs v3.5 :
1. Détection de fin d'isnad par analyse de la chaîne de transmission :
   قال/يقول suivi d'un verbe de transmission = maillon de chaîne (continuer).
   قال/يقول suivi d'un non-verbe = frontière réelle.
2. Densité de particules narratives comme validation du khabar.
3. Fallback robuste pour isnads sans قال (أنشدني X + contenu poétique).
"""

import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ── Normalisation ─────────────────────────────────────────────────────────────

def norm(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # diacritiques
    for ch in 'أإآٱ':
        text = text.replace(ch, 'ا')
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    return text


# ── Lexique des verbes de transmission (forme normalisée) ────────────────────
# Tout verbe ici, trouvé après قال, signifie un maillon de chaîne (continuer).

CHAIN_VERB_STARTS: Tuple[str, ...] = (
    'حدثن', 'حدثي', 'حدثه', 'حدثك',        # حدّث
    'اخبرن', 'اخبري', 'اخبره',             # أخبر
    'سمعت', 'سمعن', 'سمعه',               # سمع
    'انشدن', 'انشدي',                      # أنشد
    'انبان', 'انباي',                      # أنبأ
    'حكين', 'حكيت',                        # حكى
    'بلغن', 'بلغي',                        # بلغ
    'رواه', 'روين', 'روان',                # روى
    'كتبن', 'كتبي',                        # كتب
    'يقول', 'يروي',                        # idem
    'ذكرن', 'ذكري',                        # ذكر
    # formes préfixées ـو
    'وحدث', 'واخبر', 'وسمعت', 'وسمعن',
    'وانشد', 'وانبا', 'وحكي', 'وبلغ',
    'وروى', 'ورواه',
)

# Verbes d'ouverture d'isnad (premier mot du segment)
ISNAD_OPENERS: Tuple[str, ...] = (
    # حدّث/أخبر — forme directe (حدثني retiré : 9 FP 0 TP dans ce corpus)
    'حدثنا', 'حدثه', 'حدثها', 'حدثهم',
    'اخبرنا', 'اخبرني', 'اخبره', 'اخبرها', 'اخبرهم',
    'سمعت', 'سمعنا',
    # أنشد — وانشدت retiré : 4 FP 0 TP
    'انشدني', 'انشدنا', 'انشدتنا',
    'انبانا', 'انباني',
    # حكى — standalone retiré (14 FP 2 TP) ; waw-forms gardées car meilleur ratio
    'روى', 'رواه', 'روينا',
    # بلغ — بلغني standalone retiré (17 FP 2 TP) ; forme waw gardée
    'بلغنا',
    # وقال : filtré par _waqal_is_akhbar_opener() dans find_isnad_starts
    'وقال',
    'وحدثنا', 'وحدثني', 'واخبرنا', 'واخبرني',
    'وسمعت', 'وسمعنا', 'وانشدني', 'وانشدنا', 'وانشدتنا',
    'وبلغني', 'وحكى', 'وحكي', 'ومنهم', 'ومنها', 'وبهذا',
    # وله / ولغيره : attribution poétique (وله = "et pour lui [ce poème]")
    'وله', 'ولغيره', 'ولبعض',
    'ولبعضهم',
)

# Particules narratives (marqueurs de matn)
NARRATIVE_PARTICLES = frozenset([
    'ف', 'فـ', 'فقال', 'فقالت', 'فقلت', 'فلما',
    'ثم', 'حتى', 'حتي', 'إذ', 'إذا', 'لما', 'بينا', 'بينما',
])

QAL_RE = re.compile(r'قال[ا]?|يقول')

# Verbes dont l'occurrence après قال indique obligatoirement un maillon de chaîne
# (ces verbes peuvent aussi ouvrir un isnad, mais pas immédiatement après قال)
_CHAIN_SENSITIVE = frozenset([
    'حدثنا', 'حدثني', 'اخبرنا', 'اخبرني', 'سمعت', 'سمعنا',
    'انشدني', 'انشدنا', 'انشدتنا', 'انشدت',
    'وقال', 'وحدثنا', 'وحدثني', 'واخبرنا', 'واخبرني',
    'وسمعت', 'وسمعنا', 'وانشدني', 'وانشدنا',
    'وبلغني', 'وحكى', 'وحكي',
])


# ── Détection de fin d'isnad ──────────────────────────────────────────────────

def _word_after(text: str, pos: int, skip: int = 3) -> str:
    """Premier mot significatif à partir de pos+skip (ignore espaces/ponct)."""
    chunk = text[pos + skip: pos + skip + 60]
    chunk = re.sub(r'^[\s:،؛.«»\"\'\u200c\u200d]+', '', chunk)
    words = chunk.split()
    return words[0] if words else ''


def _is_chain_verb(word: str) -> bool:
    w = norm(word)
    return any(w.startswith(v) for v in CHAIN_VERB_STARTS)


def find_isnad_end(
    text_norm: str,
    isnad_start: int,
    window: int = 900,
    min_offset: int = 6,
) -> int:
    """
    Retourne la position (dans text_norm) juste après la fin de l'isnad.

    Stratégie : parcourir les occurrences de قال/يقول dans la fenêtre.
    - Si le mot qui suit est un verbe de transmission → maillon, continuer.
    - Si le mot qui suit est autre chose → frontière isnad/matn.

    Fallback : si aucun قال trouvé (isnad poétique type أنشدني X),
    retourner la position après le dernier verbe de transmission.

    min_offset : ignorer les قال dans les premiers chars (évite l'auto-match
    si l'ouvreur contient قال).
    """
    end = min(isnad_start + window, len(text_norm))
    segment = text_norm[isnad_start:end]

    last_trans_end = -1       # fin du dernier verbe de transmission vu
    boundary_pos = -1

    for m in QAL_RE.finditer(segment, min_offset):
        w = _word_after(segment, m.start(), skip=len(m.group()))
        if _is_chain_verb(w):
            # Maillon de chaîne — noter la position pour le fallback
            last_trans_end = isnad_start + m.end()
        else:
            # Premier قال non suivi d'un verbe de transmission = frontière
            boundary_pos = isnad_start + m.end()
            break

    if boundary_pos > 0:
        return boundary_pos

    # Fallback: pas de قال final clair
    # → chercher le dernier verbe de transmission et prendre ce qui suit
    if last_trans_end > 0:
        return last_trans_end

    # Fallback ultime: fenêtre fixe
    return isnad_start + min(300, window // 3)


# ── Filtre وقال → ouvreur d'akhbar vs dialogue interne ──────────────────────

# Marqueurs de nom propre : حدثنا = akhbar, ابن/ابو/بن précèdent un nom
_NAME_MARKERS = frozenset(['ابن', 'ابو', 'ابي', 'بن', 'بنت', 'ابنه'])
# Indices que ce qui suit est du discours direct (non attribution)
_WAQAL_REJECT = frozenset([
    'ت', 'وا', 'نا', 'تم', 'تن', 'ن',        # suffixes verbaux
    'لي', 'لنا', 'له', 'لها', 'لهم', 'لك',    # pronoms/prépositions
    'بعضهم', 'بعضهن', 'بعضها',               # attributions vagues
    'الاخر', 'الاخري',                        # "l'autre"
])


def _waqal_is_akhbar_opener(text_norm: str, pos: int) -> bool:
    """
    Renvoie True si وقال à `pos` ouvre un nouvel akhbar (suivi d'un nom propre),
    False s'il est du discours narratif interne (suivi d'un pronom, d'une
    forme verbale, ou d'une attribution ordinale comme للثاني).

    Cette fonction est appelée UNIQUEMENT pour وقال (4 chars).
    """
    chunk = re.sub(
        r'^[\s،؛.«»"\'\u200c\u200d]+', '',
        text_norm[pos + 4: pos + 70],
    )
    words = chunk.split()
    if not words:
        return False
    first = words[0]

    # Marqueur de nom → ouvreur
    if first in _NAME_MARKERS:
        return True
    # Rejet explicite
    if first in _WAQAL_REJECT:
        return False
    # Attributions ordinales/prépositionnelles : للـ (للثاني, للرشيد…)
    if first.startswith('لل'):
        return False
    # Mot trop court (particule, pronom)
    if len(first) <= 2:
        return False
    # Préfixe de l'imparfait → verbe, pas un nom
    if first[0] in 'يتن' and len(first) >= 3:
        return False
    # Par défaut : nom propre présumé → accepter
    return True


# ── Filtre de contexte de chaîne ─────────────────────────────────────────────

def is_chain_context(text_norm: str, pos: int, window: int = 14) -> bool:
    """
    Renvoie True si la position pos est précédée par قال/يقول/قالا dans
    la fenêtre `window` chars (après suppression des espaces et ponctuation).
    Cela indique que le verbe à `pos` est un maillon de chaîne, non un
    début d'akhbar.

    Seuls les verbes de _CHAIN_SENSITIVE sont filtrés ainsi ; les ouvertures
    très distinctives (حدثنا en tête absolue, etc.) passent même en contexte
    de chaîne si l'ouvreur vaut la peine.
    """
    snippet = text_norm[max(0, pos - window): pos]
    snippet_clean = re.sub(r'[\s:،؛.«»"\'\u200c\u200d]+', '', snippet)
    return (
        snippet_clean.endswith('قال')
        or snippet_clean.endswith('يقول')
        or snippet_clean.endswith('قالا')
    )


# ── Densité de particules narratives ─────────────────────────────────────────

def particle_density(text_norm: str) -> float:
    """Fraction de tokens qui sont des particules narratives."""
    tokens = text_norm.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in NARRATIVE_PARTICLES)
    return hits / len(tokens)


# ── Segmentation principale ───────────────────────────────────────────────────

def find_isnad_starts(text_norm: str) -> List[Tuple[int, str]]:
    """
    Trouve toutes les positions de départ d'isnad.

    Filtre les maillons de chaîne : si le verbe appartient à _CHAIN_SENSITIVE
    et qu'il est immédiatement précédé par قال/يقول, il s'agit d'un maillon
    interne de chaîne de transmission, non d'un nouvel akhbar.
    """
    starts: List[Tuple[int, str]] = []
    for verb in sorted(ISNAD_OPENERS, key=len, reverse=True):
        pos = 0
        while True:
            idx = text_norm.find(verb, pos)
            if idx < 0:
                break
            # Vérifier que le verbe commence un mot (pas infixe)
            if idx == 0 or not text_norm[idx - 1].isalpha():
                # Filtrer les maillons de chaîne pour les verbes sensibles
                if verb in _CHAIN_SENSITIVE and is_chain_context(text_norm, idx):
                    pos = idx + 1
                    continue
                # وقال : filtre supplémentaire basé sur le mot qui suit
                if verb == 'وقال' and not _waqal_is_akhbar_opener(text_norm, idx):
                    pos = idx + 1
                    continue
                starts.append((idx, verb))
            pos = idx + 1

    # Dédoublonner : garder le plus long verbe à chaque position
    starts.sort(key=lambda x: (x[0], -len(x[1])))
    deduped: List[Tuple[int, str]] = []
    last = -1
    for pos, verb in starts:
        if pos > last:
            deduped.append((pos, verb))
            last = pos + len(verb)
    return deduped


def segment(text: str) -> List[Dict]:
    """Segmente le texte en akhbars (isnad + khabar)."""
    t = norm(text)
    starts = find_isnad_starts(t)
    if not starts:
        return []

    akhbars: List[Dict] = []
    covered = 0

    for i, (istart, verb) in enumerate(starts):
        if istart < covered:
            continue

        # Borne de la prochaine unité
        next_start = starts[i + 1][0] if i + 1 < len(starts) else len(t)

        # Fin de l'isnad
        iend = find_isnad_end(t, istart, window=min(900, next_start - istart + 50))

        # L'isnad ne doit pas dépasser le prochain marqueur
        if iend <= istart or iend > next_start:
            iend = min(istart + 400, next_start)

        isnad_len = iend - istart
        khabar_len = next_start - iend

        # Filtre de sanité minimal : isnad trop court = faux positif
        if isnad_len < 8:
            continue

        isnad_text = text[istart:iend].strip()
        khabar_text = text[iend:next_start].strip()

        if not isnad_text:
            continue

        # Validation optionnelle par densité de particules
        # (désactivée par défaut pour ne pas filtrer la poésie)
        # p_density = particle_density(norm(khabar_text))

        akhbars.append({
            'num': len(akhbars) + 1,
            'isnad': isnad_text,
            'khabar': khabar_text,
            'start': istart,
            'isnad_end': iend,
            'end': next_start,
        })
        covered = next_start

    return akhbars


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(akhbars: List[Dict], text: str, ref_count: int) -> Dict:
    """Métriques de base."""
    n = len(akhbars)
    coverage = sum(a['end'] - a['start'] for a in akhbars)
    gap = n - ref_count
    return {
        'detected': n,
        'reference': ref_count,
        'gap': gap,
        'gap_pct': 100 * gap / ref_count if ref_count else 0,
        'coverage_pct': 100 * coverage / len(text) if text else 0,
        'avg_isnad_chars': (
            sum(a['isnad_end'] - a['start'] for a in akhbars) / n if n else 0
        ),
        'avg_khabar_chars': (
            sum(a['end'] - a['isnad_end'] for a in akhbars) / n if n else 0
        ),
    }


def boundary_precision_recall(
    akhbars: List[Dict],
    ref_boundaries: List[int],
    tol: int = 80,
) -> Dict:
    """
    Calcule précision/rappel sur les frontières de début d'akhbar.
    Une frontière prédite est correcte si elle est à ±tol chars d'une vraie frontière.
    """
    pred = sorted(a['start'] for a in akhbars)
    ref  = sorted(ref_boundaries)

    matched_ref  = set()
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
    fn = len(ref)  - len(matched_ref)

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    corpus_path = Path('data/processed/kitab_uqala_reference_corpus.txt')
    annot_path  = Path('data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    print('=' * 70)
    print('BASELINE V4 — Détection de chaîne sans ponctuation éditoriale')
    print('=' * 70)

    with open(corpus_path, encoding='utf-8') as f:
        corpus = f.read()
    print(f'\nCorpus : {len(corpus):,} chars')

    # Références depuis le JSON annoté
    with open(annot_path, encoding='utf-8') as f:
        annot = json.load(f)
    ref_count = annot['total']
    print(f'Référence : {ref_count} akhbars\n')

    # Construire les positions de référence depuis le JSON
    # en reconstruisant le texte annoté et en alignant sur le corpus
    ref_boundaries = _build_ref_boundaries(annot, corpus)

    # Segmentation
    akhbars = segment(corpus)
    print(f'Akhbars détectés : {len(akhbars)}\n')

    # Métriques de comptage
    m = evaluate(akhbars, corpus, ref_count)
    sign = '+' if m['gap'] >= 0 else ''
    print(f"Écart vs référence : {sign}{m['gap']} ({sign}{m['gap_pct']:.1f}%)")
    print(f"Couverture corpus  : {m['coverage_pct']:.1f}%")
    print(f"Longueur isnad moy : {m['avg_isnad_chars']:.0f} chars")
    print(f"Longueur khabar moy: {m['avg_khabar_chars']:.0f} chars")

    # Métriques de frontière
    if ref_boundaries:
        br = boundary_precision_recall(akhbars, ref_boundaries, tol=80)
        print(f"\nPrécision frontières : {br['precision']:.3f}")
        print(f"Rappel   frontières  : {br['recall']:.3f}")
        print(f"F1       frontières  : {br['f1']:.3f}")
        print(f"  TP={br['tp']}  FP={br['fp']}  FN={br['fn']}")

    # Verdict
    print('\n' + '=' * 70)
    gap_abs = abs(m['gap_pct'])
    if gap_abs < 5:
        verdict = 'EXCELLENT (< 5%)'
    elif gap_abs < 10:
        verdict = 'TRÈS BON (< 10%)'
    elif gap_abs < 20:
        verdict = 'BON (< 20%)'
    else:
        verdict = 'ACCEPTABLE'
    print(f'VERDICT : {verdict}')
    print('=' * 70)


def _build_ref_boundaries(annot: Dict, corpus: str) -> List[int]:
    """
    Reconstruit les positions de frontières dans le corpus continu
    en cherchant le début de chaque isnad annoté.
    """
    corpus_norm = norm(corpus)
    boundaries: List[int] = []
    search_from = 0

    for akhbar in annot['akhbar']:
        segs = akhbar['content']['segments']
        if not segs:
            continue
        # Prendre les 25 premiers chars du premier segment
        first_text = norm(segs[0]['text'])[:25].strip()
        if not first_text:
            continue
        idx = corpus_norm.find(first_text, search_from)
        if idx >= 0:
            boundaries.append(idx)
            search_from = idx + len(first_text)

    return boundaries


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
