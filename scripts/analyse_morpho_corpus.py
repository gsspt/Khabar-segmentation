"""
Analyse morpho-lexicale du corpus Kitab Uqala al-Majanin par type de segment.
Utilise camel-tools (normalisation) + pyarabic (tokenisation).

Usage :
    python scripts/analyse_morpho_corpus.py --corpus data/processed/Kitab_Uqala_al_Majanin_annotated.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

import pyarabic.araby as araby
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar, normalize_alef_maksura_ar

# ── Constantes morphologiques ─────────────────────────────────────────────────

ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

TRANSMISSION_VERBS = {
    "اخبرنا", "حدثنا", "حدثني", "اخبرني", "سمعت", "سمعنا",
    "انشدني", "انشدنا", "روى", "رواه", "بلغني", "بلغنا",
    "وسمعت", "واخبرنا", "وانشدنا", "وانشدني", "وحكي", "وبلغني",
    "حكى", "حكي", "وبهذا", "ذكر", "كتب",
}

NAME_MARKERS = {"ابن", "ابي", "ابو", "بن", "بنت", "ابنه", "ابنة"}
NARRATIVE_VERBS = {
    "قال", "فقال", "قالت", "كان", "كانت", "فكان", "رايت", "رات",
    "دخلت", "دخل", "خرجت", "خرج", "مررت", "قدم", "لقيت", "بلغني",
}
DISCOURSE_PARTICLES = {"ف", "ثم", "فلما", "حتي", "حتى", "إذ", "إذا", "لما", "بينما", "بينا"}
ISNAD_CLOSERS = {"قال", "يقول", "فقال", "وقال"}
NISBA = re.compile(r"[ا-ي]{4,}ي$")
FIRST_PERSON_PAST = re.compile(r".+[^ا]ت$")


def normalize(text: str) -> str:
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    return text.replace("ة", "ه")


def classify_token(word: str) -> str:
    w = normalize(word)
    if w in TRANSMISSION_VERBS:
        return "V_TRANS"
    if w in NAME_MARKERS:
        return "NAME_MARK"
    if w in NARRATIVE_VERBS:
        return "V_NARR"
    if w in DISCOURSE_PARTICLES:
        return "DISC_PART"
    if w.startswith("ال") and len(w) > 3:
        return "NOUN_DEF"
    if re.match(r"^[أيتن].{3,}$", w):
        return "V_MUDARI"
    if len(w) <= 2:
        return "PARTICLE"
    return "NOUN_VERB"


def analyse(corpus_path: Path) -> None:
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    segments: dict[str, list[str]] = collections.defaultdict(list)
    for akhbar in data["akhbar"]:
        for seg in akhbar["content"]["segments"]:
            segments[seg["type"]].append(seg["text"])

    # Distribution POS par type
    print(f"\n{'─'*60}")
    print("DISTRIBUTION POS PAR TYPE DE SEGMENT")
    print(f"{'─'*60}")
    for stype in ["isnad", "matn", "prose", "poetry"]:
        counter: collections.Counter = collections.Counter()
        total = 0
        for text in segments[stype]:
            for w in araby.tokenize(text):
                if len(w) >= 2:
                    counter[classify_token(w)] += 1
                    total += 1
        print(f"\n[{stype}] — {total} tokens, {len(segments[stype])} segments")
        for cls, cnt in counter.most_common(8):
            print(f"  {cls:12s}  {cnt:5d}  ({100*cnt/total:.1f}%)")

    # Taux de couverture des marqueurs
    print(f"\n{'─'*60}")
    print("COUVERTURE DES MARQUEURS D'OUVERTURE")
    print(f"{'─'*60}")
    for stype, openers in [
        ("isnad", TRANSMISSION_VERBS),
        ("matn", NARRATIVE_VERBS),
    ]:
        covered = total_segs = 0
        for text in segments[stype]:
            tokens = [normalize(w) for w in araby.tokenize(text) if len(w) >= 2]
            total_segs += 1
            if tokens and tokens[0] in openers:
                covered += 1
        print(f"  {stype:6s} : {covered}/{total_segs} = {100*covered/total_segs:.1f}%")

    # Taux de clôture d'isnad sur قال/يقول
    isnad_closed = sum(
        1 for text in segments["isnad"]
        if (toks := [normalize(w) for w in araby.tokenize(text) if len(w) >= 2])
        and toks[-1] in ISNAD_CLOSERS
    )
    print(f"  isnad closes قال/يقول : {isnad_closed}/{len(segments['isnad'])} = "
          f"{100*isnad_closed/len(segments['isnad']):.1f}%")

    # Top 20 premiers mots par type
    print(f"\n{'─'*60}")
    print("PREMIERS MOTS PAR TYPE DE SEGMENT (top 15)")
    print(f"{'─'*60}")
    for stype in ["isnad", "matn", "prose", "poetry"]:
        counter = collections.Counter()
        for akhbar in data["akhbar"]:
            for seg in akhbar["content"]["segments"]:
                if seg["type"] == stype:
                    toks = [normalize(w) for w in araby.tokenize(seg["text"]) if len(w) >= 2]
                    if toks:
                        counter[toks[0]] += 1
        n = sum(counter.values())
        print(f"\n[{stype}]")
        for w, c in counter.most_common(15):
            print(f"  {w:20s}  {c:4d}  ({100*c/n:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse morpho-lexicale par type de segment")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/Kitab_Uqala_al_Majanin_annotated.json"),
    )
    args = parser.parse_args()
    analyse(args.corpus)
