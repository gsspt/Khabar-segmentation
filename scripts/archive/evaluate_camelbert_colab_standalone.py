"""
Évaluer CAMeL-BERT sur corpus OpenITI ciblé - À exécuter en Colab
Travaille directement depuis Google Drive sans copier le repo
"""

import os
import sys
import json
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import pipeline
import numpy as np

# CONFIGURATION (à adapter selon vos chemins)
DRIVE_BASE = Path('/content/drive/MyDrive')
MODEL_PATH = DRIVE_BASE / 'Thèse/khabar-segmentation/checkpoints/camelbert_akhbars'
CORPUS_PATH = DRIVE_BASE / 'Thèse/khabar-segmentation/openiti_targeted/0255Jahiz/0255Jahiz.Bukhala/'
OUTPUT_PATH = DRIVE_BASE / 'Thèse/khabar-segmentation/evaluation_openiti_targeted.json'

def clean_openiti_text(text: str) -> str:
    """Clean OpenITI text: keep ONLY Arabic."""
    # Remove all # symbols
    text = text.replace('#', ' ')

    # Keep ONLY Arabic characters and spaces
    text = re.sub(
        r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]',
        ' ',
        text
    )

    # Normalize spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 300):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def extract_akhbars_from_text(text: str, nlp, confidence_threshold: float = 0.7):
    """Extract akhbars from text using token-level predictions."""
    # Clean text
    text = clean_openiti_text(text)

    if len(text) < 100:
        return []

    # Split into chunks
    chunks = split_into_chunks(text, chunk_size=1500, overlap=300)

    akhbars = []
    seen_texts = set()

    for chunk in chunks:
        if not chunk.strip():
            continue

        # Limit chunk size for pipeline
        try:
            predictions = nlp(chunk[:2000])
        except Exception as e:
            continue

        # Reconstruct akhbars from BIO tags
        current_akhbar = []
        current_scores = []

        for pred in predictions:
            token = pred.get('word', '')
            entity = pred.get('entity', 'O')
            score = pred.get('score', 0.0)

            # Handle AraBERT continuation tokens
            if token.startswith('##'):
                if current_akhbar:
                    current_akhbar[-1] += token[2:]
                continue

            if entity == 'B-KHABAR' and score >= confidence_threshold:
                # Save previous akhbar
                if current_akhbar:
                    akhbar_text = ' '.join(current_akhbar)
                    if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                        avg_score = np.mean(current_scores)
                        akhbars.append({
                            'text': akhbar_text,
                            'confidence': float(avg_score),
                            'length': len(current_akhbar)
                        })
                        seen_texts.add(akhbar_text)

                # Start new akhbar
                current_akhbar = [token]
                current_scores = [score]

            elif entity == 'I-KHABAR' and score >= confidence_threshold and current_akhbar:
                current_akhbar.append(token)
                current_scores.append(score)

            else:
                # O or low confidence: end current akhbar
                if current_akhbar:
                    akhbar_text = ' '.join(current_akhbar)
                    if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                        avg_score = np.mean(current_scores)
                        akhbars.append({
                            'text': akhbar_text,
                            'confidence': float(avg_score),
                            'length': len(current_akhbar)
                        })
                        seen_texts.add(akhbar_text)
                    current_akhbar = []
                    current_scores = []

        # Save last akhbar
        if current_akhbar:
            akhbar_text = ' '.join(current_akhbar)
            if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                avg_score = np.mean(current_scores)
                akhbars.append({
                    'text': akhbar_text,
                    'confidence': float(avg_score),
                    'length': len(current_akhbar)
                })
                seen_texts.add(akhbar_text)

    return akhbars


def main():
    """Main evaluation function."""

    print("="*70)
    print("[KHABAR SEGMENTATION - CAMeL-BERT EVALUATION]")
    print("="*70)

    # Check paths
    print(f"\n[PATHS]")
    print(f"  Model: {MODEL_PATH} {'✓' if MODEL_PATH.exists() else '✗'}")
    print(f"  Corpus: {CORPUS_PATH} {'✓' if CORPUS_PATH.exists() else '✗'}")

    if not MODEL_PATH.exists():
        print(f"[!] Model not found at {MODEL_PATH}")
        return

    if not CORPUS_PATH.exists():
        print(f"[!] Corpus not found at {CORPUS_PATH}")
        return

    # Load model
    print(f"\n[*] Loading CAMeL-BERT model...")
    nlp = pipeline(
        'token-classification',
        model=str(MODEL_PATH),
        tokenizer='CAMeL-Lab/bert-base-arabic-camelbert-ca',
        device=0 if torch.cuda.is_available() else -1
    )
    print("[OK] Model loaded!")

    # Find text files (OpenITI formats: *.ara1, *-ara1, *.txt, etc.)
    text_files = []

    # Look for all common OpenITI file formats
    for item in CORPUS_PATH.rglob('*'):
        if item.is_file():
            name = item.name.lower()
            # Include files that end with -ara1, .ara1, .txt, or no extension
            if name.endswith('-ara1') or name.endswith('.ara1') or name.endswith('.txt'):
                text_files.append(item)
            elif '.' not in name and item.stat().st_size > 1000:  # Files without extension, >1KB
                text_files.append(item)

    print(f"\n[*] Found {len(text_files)} text files in corpus")
    if text_files:
        print(f"   Examples:")
        for f in text_files[:3]:
            print(f"     - {f.relative_to(CORPUS_PATH)}")

    # Evaluate
    print(f"\n[*] Extracting akhbars...")

    results = {
        'total_files': len(text_files),
        'total_akhbars': 0,
        'total_tokens': 0,
        'files_processed': 0,
        'avg_confidence': 0,
        'file_results': []
    }

    confidences = []

    for file_path in tqdm(text_files, desc="Processing files"):
        try:
            # Load file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if len(text) < 100:
                continue

            # Extract akhbars
            akhbars = extract_akhbars_from_text(text, nlp, confidence_threshold=0.7)

            if akhbars:
                results['files_processed'] += 1
                results['total_akhbars'] += len(akhbars)
                results['total_tokens'] += sum(ak['length'] for ak in akhbars)

                for ak in akhbars:
                    confidences.append(ak['confidence'])

                results['file_results'].append({
                    'file': file_path.name,
                    'akhbars_found': len(akhbars),
                    'tokens': sum(ak['length'] for ak in akhbars),
                    'avg_confidence': float(np.mean([ak['confidence'] for ak in akhbars]))
                })

        except Exception as e:
            continue

    if confidences:
        results['avg_confidence'] = float(np.mean(confidences))

    # Display results
    print("\n" + "="*70)
    print("[EVALUATION RESULTS]")
    print("="*70)
    print(f"Total files processed: {results['files_processed']}/{results['total_files']}")
    print(f"Total akhbars extracted: {results['total_akhbars']}")
    print(f"Total tokens: {results['total_tokens']}")
    print(f"Average confidence: {results['avg_confidence']:.4f}")

    print(f"\n[TOP FILES]")
    for fr in sorted(results['file_results'], key=lambda x: x['akhbars_found'], reverse=True)[:10]:
        print(f"  {fr['file']:40s} → {fr['akhbars_found']:3d} akhbars ({fr['avg_confidence']:.3f})")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Results saved to: {OUTPUT_PATH}")
    print("="*70)


if __name__ == '__main__':
    main()
