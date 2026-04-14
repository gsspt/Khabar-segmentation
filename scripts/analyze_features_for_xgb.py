"""
Analyser le Kitab Uqala al-Majanin pour identifier les meilleures features XGB.
Identifie les patterns qui distinguent B-KHABAR, I-KHABAR, et O.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normaliser le texte arabe."""
    # Supprimer diacritiques
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text.strip()


def extract_root(word: str) -> str:
    """Extraire la racine approximative d'un mot arabe (simplifié)."""
    # Pour une analyse simplifiée, retourner les 3-4 premiers caractères
    word = normalize_text(word)
    if len(word) >= 3:
        return word[:3]
    return word


def is_connector(word: str) -> bool:
    """Vérifier si le mot est un connecteur discursif arabe."""
    connectors = {
        'قال', 'قالت', 'يقول', 'تقول', 'قالوا', 'قلت',
        'أخبر', 'أخبرنا', 'أنبأ', 'أنبأنا', 'حدث', 'حدثنا',
        'روي', 'رويت', 'رووا', 'رواه', 'رواها',
        'ذكر', 'ذكرت', 'ذكروا', 'ذكرنا',
        'نقل', 'نقلنا', 'نقلت',
        'سمع', 'سمعنا', 'سمعت', 'سمعوا',
        'كتب', 'كتبنا', 'كتبت',
        'وجد', 'وجدنا', 'وجدت',
    }
    word_norm = normalize_text(word).strip()
    return word_norm in connectors


def is_particle(word: str) -> bool:
    """Vérifier si c'est une particule arabe."""
    particles = {'و', 'ف', 'ب', 'ل', 'ال', 'إن', 'أن', 'لا', 'ما', 'قد', 'قد', 'لقد'}
    return normalize_text(word) in particles


def get_pos_tag_simple(word: str) -> str:
    """Estimation simple du POS basée sur patterns."""
    word_norm = normalize_text(word)

    # Verbes courants (3ème personne perfectif)
    if word_norm.endswith(('ا', 'ت', 'ه')):
        if len(word_norm) >= 4:
            return 'VERB'

    # Noms avec article
    if word_norm.startswith('ال'):
        return 'NOUN'

    # Pronoms
    if word_norm in {'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم'}:
        return 'PRON'

    return 'OTHER'


def analyze_akhbars():
    """Analyser les 613 akhbars du Kitab Uqala al-Majanin."""

    json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    if not json_path.exists():
        print(f"[!] File not found: {json_path}")
        return

    print("[*] Loading annotated akhbars...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    akhbars = data['akhbar']
    print(f"[OK] Loaded {len(akhbars)} akhbars\n")

    # Statistiques générales
    print("="*70)
    print("[GENERAL STATISTICS]")
    print("="*70)

    total_tokens = sum(len(seg['text'].split()) for akh in akhbars for seg in akh['content']['segments'])
    avg_tokens = total_tokens / len(akhbars)
    lengths = [len(' '.join([seg['text'] for seg in akh['content']['segments']]).split()) for akh in akhbars]

    print(f"Total akhbars: {len(akhbars)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per akhbar: {avg_tokens:.1f}")
    print(f"Min tokens: {min(lengths)}")
    print(f"Max tokens: {max(lengths)}")
    print(f"Median tokens: {np.median(lengths):.0f}")

    # Analyser les premiers tokens des akhbars (B-KHABAR)
    print("\n" + "="*70)
    print("[FIRST TOKENS (B-KHABAR) - WHAT STARTS AN AKHBAR?]")
    print("="*70)

    first_tokens = []
    first_roots = []
    first_connectors = []
    first_pos = Counter()

    for akh in tqdm(akhbars, desc="Analyzing first tokens"):
        full_text = ' '.join([seg['text'] for seg in akh['content']['segments']])
        tokens = full_text.split()

        if tokens:
            first_token = tokens[0]
            first_tokens.append(normalize_text(first_token))
            first_roots.append(extract_root(first_token))

            if is_connector(first_token):
                first_connectors.append(normalize_text(first_token))

            pos = get_pos_tag_simple(first_token)
            first_pos[pos] += 1

    print(f"\n[TOP 20 FIRST TOKENS]")
    for token, count in Counter(first_tokens).most_common(20):
        pct = count / len(first_tokens) * 100
        is_conn = " [CONNECTOR]" if token in first_connectors else ""
        print(f"  {token:20s} → {count:3d} ({pct:5.1f}%){is_conn}")

    print(f"\n[TOP 15 FIRST ROOTS]")
    for root, count in Counter(first_roots).most_common(15):
        pct = count / len(first_roots) * 100
        print(f"  {root:10s} → {count:3d} ({pct:5.1f}%)")

    print(f"\n[FIRST TOKEN POS DISTRIBUTION]")
    for pos, count in first_pos.most_common():
        pct = count / len(akhbars) * 100
        print(f"  {pos:10s} → {count:3d} ({pct:5.1f}%)")

    print(f"\n[CONNECTORS AT AKHBAR START]")
    connector_counts = Counter(first_connectors)
    for conn, count in connector_counts.most_common():
        pct = count / len(akhbars) * 100
        print(f"  {conn:20s} → {count:3d} ({pct:5.1f}%)")

    # Analyser les patterns de longueur
    print("\n" + "="*70)
    print("[LENGTH PATTERNS]")
    print("="*70)

    print(f"\nAkhbars by length:")
    print(f"  Very short (< 50 tokens): {sum(1 for l in lengths if l < 50)}")
    print(f"  Short (50-100 tokens): {sum(1 for l in lengths if 50 <= l < 100)}")
    print(f"  Medium (100-300 tokens): {sum(1 for l in lengths if 100 <= l < 300)}")
    print(f"  Long (300-500 tokens): {sum(1 for l in lengths if 300 <= l < 500)}")
    print(f"  Very long (> 500 tokens): {sum(1 for l in lengths if l > 500)}")

    # Analyser les patterns internes (I-KHABAR)
    print("\n" + "="*70)
    print("[INTERNAL PATTERNS (I-KHABAR)]")
    print("="*70)

    internal_words = Counter()
    internal_roots = Counter()

    for akh in tqdm(akhbars, desc="Analyzing internal patterns"):
        full_text = ' '.join([seg['text'] for seg in akh['content']['segments']])
        tokens = full_text.split()[1:]  # Skip first token (B-KHABAR)

        for token in tokens:
            token_norm = normalize_text(token)
            internal_words[token_norm] += 1
            internal_roots[extract_root(token)] += 1

    print(f"\n[TOP 20 INTERNAL WORDS (frequency)]")
    for word, count in internal_words.most_common(20):
        pct = count / sum(internal_words.values()) * 100
        print(f"  {word:20s} → {count:6d} ({pct:5.2f}%)")

    # Analyser les patterns de segment (prose vs isnad vs matn)
    print("\n" + "="*70)
    print("[SEGMENT TYPES ANALYSIS]")
    print("="*70)

    segment_types = Counter()
    segment_positions = defaultdict(list)

    for akh in akhbars:
        for i, seg in enumerate(akh['content']['segments']):
            seg_type = seg.get('segment_type', 'unknown')
            segment_types[seg_type] += 1
            segment_positions[seg_type].append(i)

    print(f"\nSegment types distribution:")
    for seg_type, count in segment_types.most_common():
        pct = count / sum(segment_types.values()) * 100
        print(f"  {seg_type:20s} → {count:4d} ({pct:5.1f}%)")

    print(f"\nSegment positions (average):")
    for seg_type in segment_types.keys():
        positions = segment_positions[seg_type]
        if positions:
            avg_pos = np.mean(positions)
            print(f"  {seg_type:20s} → position {avg_pos:.2f} (avg)")

    # Recommandations
    print("\n" + "="*70)
    print("[RECOMMENDED XGB FEATURES]")
    print("="*70)

    features = {
        'Token Features': [
            '✓ is_connector (قال، أخبرنا، روي، etc.)',
            '✓ is_particle (و، ف، ال، etc.)',
            '✓ pos_tag (VERB, NOUN, PRON, OTHER)',
            '✓ token_length (caractères)',
            '✓ starts_with_alef_lam (ال)',
        ],
        'Context Features': [
            '✓ previous_token_is_connector',
            '✓ next_token_is_connector',
            '✓ tokens_since_last_connector',
            '✓ position_in_text (0-100%)',
        ],
        'Linguistic Features': [
            '✓ is_verbal_form',
            '✓ ends_with_alef_maqsura',
            '✓ has_ta_marbuta',
            '✓ root (première 3 lettres)',
        ],
        'Morphological Features': [
            '✓ word_length',
            '✓ consonant_count',
            '✓ vowel_count (diacritiques)',
        ],
        'Segment Features': [
            '✓ segment_type (prose/isnad/matn)',
            '✓ position_in_segment',
            '✓ segment_index',
        ]
    }

    for category, feats in features.items():
        print(f"\n{category}:")
        for feat in feats:
            print(f"  {feat}")

    print("\n" + "="*70)
    print("[FEATURE IMPORTANCE HYPOTHESIS]")
    print("="*70)
    print("""
    Basé sur l'analyse, l'ordre d'importance estimé pour XGB:

    1. is_connector (TRÈS IMPORTANT)
       → Les connecteurs discursifs marquent fortement les débuts d'akhbars
       → قال، أخبرنا، روي: très prédictifs de B-KHABAR

    2. position_in_text / position_in_segment
       → Les akhbars sont structurés sequentiellement

    3. pos_tag (VERB vs NOUN vs PRON)
       → Les akhbars commencent souvent par un verbe

    4. previous_token_is_connector
       → Les connecteurs précédents signalent une transition

    5. token_length, starts_with_alef_lam
       → Features morphologiques secondaires

    6. segment_type
       → Isnad vs matn donne des indices sur la structure
    """)


if __name__ == '__main__':
    analyze_akhbars()
