#!/usr/bin/env python3
"""
Teste la généralisation de baseline_v4 sur des textes OpenITI externes.
Pas de gold standard disponible → évaluation qualitative + statistiques.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baseline_v4 import segment, norm, find_isnad_starts

# ── Nettoyage du format mARkdown OpenITI ─────────────────────────────────────

def clean_openiti(path: str | Path) -> str:
    """
    Convertit un fichier OpenITI mARkdown en texte arabe brut.
    Supprime : header, marqueurs ### |, PageV, ms####, ~~ continuation,
    numéros de notes [1], préfixe # de paragraphe.
    """
    raw = Path(path).read_text(encoding='utf-8')

    # 1. Supprimer le header jusqu'à #META#Header#End#
    if '#META#Header#End#' in raw:
        raw = raw[raw.index('#META#Header#End#') + len('#META#Header#End#'):]

    # 2. Supprimer les marqueurs de section ### | ... et # ...
    raw = re.sub(r'^###[^\n]*\n', '', raw, flags=re.MULTILINE)

    # 3. Joindre les lignes de continuation ~~ avec la ligne précédente
    raw = re.sub(r'\n~~', ' ', raw)

    # 4. Supprimer le préfixe # de chaque paragraphe
    raw = re.sub(r'^#+\s*', '', raw, flags=re.MULTILINE)

    # 5. Supprimer les marqueurs de page PageV01P001
    raw = re.sub(r'PageV\d+P\d+', '', raw)

    # 6. Supprimer les marqueurs de manuscrit ms\d+
    raw = re.sub(r'\bms\d+\b', '', raw)

    # 7. Supprimer les numéros de notes [1], [2a], etc.
    raw = re.sub(r'\[\d+[a-z]?\]', '', raw)

    # 8. Normaliser les espaces multiples et lignes vides
    raw = re.sub(r'[ \t]+', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)

    return raw.strip()


# ── Statistiques sur les akhbars détectés ────────────────────────────────────

def describe(corpus: str, akhbars: list, label: str) -> None:
    from collections import Counter
    import random

    words = len(norm(corpus).split())
    n = len(akhbars)
    density = n / (words / 1000) if words else 0

    isnad_lens = [len(norm(a['isnad']).split()) for a in akhbars]
    khabar_lens = [len(norm(a['khabar']).split()) for a in akhbars]

    avg_i = sum(isnad_lens) / n if n else 0
    avg_k = sum(khabar_lens) / n if n else 0

    # Khabars très courts (< 10 mots) → proxy de FP potentiels
    short_k = sum(1 for l in khabar_lens if l < 10)
    short_pct = 100 * short_k / n if n else 0

    # Verbes d'ouverture via find_isnad_starts
    corpus_norm = norm(corpus)
    start_to_verb = {p: v for p, v in find_isnad_starts(corpus_norm)}
    verbs = Counter(start_to_verb.get(a['start'], '?') for a in akhbars)
    top5 = '  '.join(f"{v}:{c}" for v, c in verbs.most_common(5))

    print(f'\n{"─"*60}')
    print(f'  {label}')
    print(f'{"─"*60}')
    print(f'  Mots          : {words:,}')
    print(f'  Akhbars       : {n}')
    print(f'  Densité       : {density:.1f} akhbars / 1000 mots')
    print(f'  Isnad moy.    : {avg_i:.1f} mots   Khabar moy. : {avg_k:.1f} mots')
    print(f'  Khabar < 10 m : {short_k} ({short_pct:.0f}%)  ← proxy FP potentiels')
    print(f'  Verbes top 5  : {top5}')

    # 3 exemples aléatoires
    random.seed(42)
    samples = random.sample(akhbars, min(3, n))
    print(f'\n  Exemples :')
    for i, a in enumerate(samples, 1):
        verb = start_to_verb.get(a['start'], '?')
        isnad_preview = norm(a['isnad'])[:90].replace('\n', ' ')
        khabar_preview = norm(a['khabar'])[:90].replace('\n', ' ')
        print(f'  [{i}] verb={verb}')
        print(f'      ISNAD : {isnad_preview}…')
        print(f'      KHABAR: {khabar_preview}…')


# ── Main ──────────────────────────────────────────────────────────────────────

TEXTS = [
    (
        'data/external/IbnQutayba_CuyunAkhbar.txt',
        "Ibn Qutayba — ʿUyūn al-Akhbār (0276 AH)",
    ),
    (
        'data/external/Baladhuri_AnsabAshraf.txt',
        "Baladhuri — Ansab al-Ashraf (0279 AH)",
    ),
    (
        'data/external/IbnAbiDunya_Manamat.txt',
        "Ibn Abi Dunya — al-Manamat (0281 AH)",
    ),
]

# Référence : notre texte de développement
REFERENCE = (
    'data/processed/kitab_uqala_reference_corpus.txt',
    "IbnHabibNaysaburi — ʿUqala al-Majanin (0406 AH)  [DEV]",
)

def main():
    import os
    os.chdir(Path(__file__).parent.parent)

    print('=' * 60)
    print('GÉNÉRALISATION baseline_v4 — textes OpenITI externes')
    print('=' * 60)

    # Texte de référence (déjà propre)
    ref_path, ref_label = REFERENCE
    ref_corpus = Path(ref_path).read_text(encoding='utf-8')
    ref_akhbars = segment(ref_corpus)
    describe(ref_corpus, ref_akhbars, ref_label)

    # Textes externes
    for path, label in TEXTS:
        if not Path(path).exists():
            print(f'\n⚠ Fichier introuvable : {path}')
            continue
        corpus = clean_openiti(path)
        akhbars = segment(corpus)
        describe(corpus, akhbars, label)

    print(f'\n{"="*60}')
    print('NOTE : sans gold standard, la densité et les exemples sont')
    print('les indicateurs principaux de généralisation.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
