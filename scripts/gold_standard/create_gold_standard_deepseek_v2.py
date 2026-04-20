#!/usr/bin/env python3
"""
Crée un gold standard robuste avec API Deepseek pour textes longs.

Gère les coupures au milieu des khabars via:
1. Chunking intelligent avec contexte de chevauchement
2. Marquage des fragments incomplets
3. Fusion post-traitement des chunks
4. Reconstruction avec positions exactes dans le texte original

Output: JSON complet avec tous les akhbars fusionnés et positions exactes.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Charger les variables d'environnement
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


# ── STRUCTURES DE DONNÉES ─────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """Un chunk avec contexte avant/après"""
    chunk_id: int
    content: str  # Le chunk principal
    context_before: str  # Contexte du chunk précédent (300 chars)
    context_after: str  # Contexte du chunk suivant (300 chars)
    start_pos: int  # Position dans le texte original
    end_pos: int  # Position dans le texte original

@dataclass
class SegmentFragment:
    """Un segment qui peut être fragmenté entre chunks"""
    id: str
    type: str  # with_isnad, poetry, prose, continuation
    isnad: Optional[str] = None
    content: str = ""
    confidence: float = 0.0
    notes: str = ""
    is_fragment_start: bool = False  # Commence dans un autre chunk
    is_fragment_end: bool = False  # Finit dans un autre chunk
    chunk_id: int = -1


# ── NETTOYAGE DU FORMAT OPENITI ──────────────────────────────────────────────

def clean_openiti_text(text: str) -> str:
    """Nettoie un texte au format mARkdown OpenITI."""
    if '#META#Header#End#' in text:
        text = text[text.index('#META#Header#End#') + len('#META#Header#End#'):]

    text = re.sub(r'^###[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n~~', ' ', text)
    text = re.sub(r'PageV\d+P\d+', '', text)
    text = re.sub(r'\bms\d+\b', '', text)
    text = re.sub(r'\[\d+[a-z]?\]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── CHUNKING INTELLIGENT ──────────────────────────────────────────────────────

def smart_chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 300
) -> List[TextChunk]:
    """
    Divise le texte en chunks avec contexte de chevauchement.

    Stratégie:
    1. Découper à taille fixe (pas de limite logique pour l'arabe)
    2. Préserver overlap pour contexte
    3. Marquer les positions exactes
    """
    chunks = []
    text_len = len(text)
    chunk_id = 0

    for i in range(0, text_len, chunk_size - overlap):
        # Chunk principal
        start_pos = i
        end_pos = min(i + chunk_size, text_len)
        content = text[start_pos:end_pos]

        if not content.strip():
            continue

        # Contexte avant (derniers 300 chars du chunk précédent)
        context_before_start = max(0, start_pos - overlap)
        context_before = text[context_before_start:start_pos]

        # Contexte après (premiers 300 chars du chunk suivant)
        context_after_end = min(text_len, end_pos + overlap)
        context_after = text[end_pos:context_after_end]

        chunk = TextChunk(
            chunk_id=chunk_id,
            content=content,
            context_before=context_before,
            context_after=context_after,
            start_pos=start_pos,
            end_pos=end_pos
        )
        chunks.append(chunk)
        chunk_id += 1

        if end_pos >= text_len:
            break

    return chunks


# ── LOGGING ──────────────────────────────────────────────────────────────────

class DebugLogger:
    """Enregistre les réponses brutes pour debug"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'deepseek_raw_responses.jsonl'

    def log_response(self, chunk_id: int, request_payload: Dict, response_content: str,
                     json_extracted: Optional[str], error: Optional[str]):
        """Enregistrer une réponse brute"""
        entry = {
            'chunk_id': chunk_id,
            'request_model': request_payload.get('model'),
            'request_temp': request_payload.get('temperature'),
            'response_content': response_content[:500],  # Premiers 500 chars
            'response_length': len(response_content),
            'json_extracted': json_extracted[:200] if json_extracted else None,
            'error': error
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def save_full_response(self, chunk_id: int, response_content: str):
        """Sauvegarder la réponse complète dans un fichier séparé"""
        response_file = self.output_dir / f'chunk_{chunk_id:02d}_response.txt'
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response_content)


# ── API DEEPSEEK ─────────────────────────────────────────────────────────────

def call_deepseek_api(
    chunk: TextChunk,
    debug_logger: Optional[DebugLogger] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> Optional[Dict]:
    """
    Appelle l'API Deepseek pour annoter un chunk avec retry logic.

    Args:
        chunk: Le chunk à annoter
        debug_logger: Logger pour debug
        max_retries: Nombre maximum de tentatives (default: 3)
        initial_backoff: Délai initial de retry en secondes (default: 2.0)

    Instructions spéciales:
    - N'annote que le contenu PRINCIPAL
    - Marque les fragments qui s'étendent dans le contexte
    - Retry automatique sur timeout avec backoff exponentiel
    """

    system_prompt = """أنت متخصص في تحليل النصوص العربية التاريخية.

مهمتك: تقسيم النص بين [=== MAIN START ===] و [=== MAIN END ===] إلى أخبار.

أنواع:
1. with_isnad: إسناد + متن (حدثنا...، أخبرنا...)
2. poetry: شعر
3. prose: نثر بدون إسناد
4. continuation: استمرار من قبل

التعليمات الهامة:
- الحق ONLY ما هو بين MAIN START/END
- إذا امتد الخبر خارج MAIN: is_fragment_start=true أو is_fragment_end=true
- تجاهل ما هو في BEFORE/AFTER فقط

أرجع JSON فقط، بدون نص آخر.
"""

    # Construire le user prompt
    context_before_marker = f"[=== BEFORE CONTEXT ===]\n{chunk.context_before}\n[=== BEFORE END ===]\n" if chunk.context_before.strip() else ""
    context_after_marker = f"\n[=== AFTER CONTEXT ===]\n{chunk.context_after}\n[=== AFTER END ===]" if chunk.context_after.strip() else ""

    user_prompt = f"""{context_before_marker}
[=== MAIN CONTENT START ===]
{chunk.content}
[=== MAIN CONTENT END ===]
{context_after_marker}

Étapes:
1. Ignorer le contenu des contextes BEFORE et AFTER (utiliser seulement pour contexte)
2. Annoter uniquement ce qui est entre [=== MAIN CONTENT START ===] et [=== MAIN CONTENT END ===]
3. Si un segment s'étend au-delà du MAIN:
   - Vers AFTER: marquer is_fragment_end=true
   - Vers BEFORE: marquer is_fragment_start=true
4. Utiliser IDs uniques par chunk: chunk{chunk.chunk_id}_item{{i}}
5. Retourner UNIQUEMENT le JSON valide, sans explications.
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    # Retry loop with exponential backoff
    current_backoff = initial_backoff
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']

                # DEBUG : Enregistrer la réponse brute
                if debug_logger:
                    debug_logger.save_full_response(chunk.chunk_id, content)

                # Extraire JSON - meilleure stratégie
                json_str = None
                json_extracted = None
                parsed_data = None

                # Chercher d'abord les blocs ```json ... ```
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    json_str = json_match.group(1)
                    json_extracted = json_str[:100]
                else:
                    # Sinon chercher un array JSON directement
                    array_match = re.search(r'\[[\s\S]*\]', content)
                    if array_match:
                        json_str = array_match.group(0)
                        json_extracted = json_str[:100]
                    else:
                        # Chercher un objet JSON
                        obj_match = re.search(r'\{[\s\S]*\}', content)
                        if obj_match:
                            json_str = obj_match.group(0)
                            json_extracted = json_str[:100]

                if json_str:
                    parsed_data = json.loads(json_str)

                    # Normaliser les deux formats de réponse Deepseek:
                    # Format 1: {"chunks": [{...}]}
                    # Format 2: [{...}]
                    if isinstance(parsed_data, dict) and 'chunks' in parsed_data:
                        items = parsed_data['chunks']
                    elif isinstance(parsed_data, list):
                        items = parsed_data
                    else:
                        # Format invalide
                        error_msg = f'Unexpected JSON structure: {type(parsed_data)}'
                        if debug_logger:
                            debug_logger.log_response(chunk.chunk_id, payload, content,
                                                    json_extracted, error_msg)
                        return {
                            'success': False,
                            'chunk_id': chunk.chunk_id,
                            'error': error_msg
                        }

                    # Normaliser chaque item: convertir "text" → "content" si nécessaire
                    normalized_items = []
                    for item in items:
                        normalized = dict(item)
                        # Fusionner "text" et "content" en un seul champ "content"
                        if 'text' in normalized and 'content' not in normalized:
                            normalized['content'] = normalized.pop('text')
                        # Assurer qu'on a un champ content
                        if 'content' not in normalized:
                            normalized['content'] = ''
                        normalized_items.append(normalized)

                    # Retourner dans le format attendu par merge_chunk_results
                    normalized_data = {'akhbars': normalized_items}

                    # DEBUG : Log succès
                    if debug_logger:
                        debug_logger.log_response(chunk.chunk_id, payload, content,
                                                json_extracted, None)
                    return {
                        'success': True,
                        'chunk_id': chunk.chunk_id,
                        'chunk_pos': (chunk.start_pos, chunk.end_pos),
                        'data': normalized_data
                    }
                else:
                    error_msg = 'No JSON found in response'
                    # DEBUG : Log erreur
                    if debug_logger:
                        debug_logger.log_response(chunk.chunk_id, payload, content,
                                                None, error_msg)
                    return {
                        'success': False,
                        'chunk_id': chunk.chunk_id,
                        'error': error_msg
                    }
            else:
                error_msg = 'Could not parse response'
                if debug_logger:
                    debug_logger.log_response(chunk.chunk_id, payload, str(result),
                                            None, error_msg)
                return {
                    'success': False,
                    'chunk_id': chunk.chunk_id,
                    'error': error_msg
                }

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as timeout_err:
            # Timeout ou erreur de connexion: retry avec backoff exponentiel
            last_error = str(timeout_err)
            if attempt < max_retries - 1:
                # Attendre avant de retry
                time.sleep(current_backoff)
                current_backoff *= 2  # Backoff exponentiel
                continue
            else:
                # Dernière tentative échouée
                if debug_logger:
                    debug_logger.log_response(chunk.chunk_id, payload, last_error,
                                            None, f'Timeout after {max_retries} retries')
                return {
                    'success': False,
                    'chunk_id': chunk.chunk_id,
                    'error': f'Timeout after {max_retries} retries: {last_error}'
                }

        except Exception as e:
            # Autres erreurs: ne pas retry
            error_msg = str(e)
            if debug_logger:
                debug_logger.log_response(chunk.chunk_id, payload, error_msg,
                                        None, error_msg)
            return {
                'success': False,
                'chunk_id': chunk.chunk_id,
                'error': error_msg
            }

    # Tous les retries échouées
    return {
        'success': False,
        'chunk_id': chunk.chunk_id,
        'error': f'Failed after {max_retries} attempts: {last_error}'
    }


# ── FUSION POST-TRAITEMENT ────────────────────────────────────────────────────

def merge_chunk_results(
    results: List[Dict],
    chunks: List[TextChunk],
    original_text: str
) -> List[Dict]:
    """
    Fusionne les résultats des chunks avec déduplication d'overlaps.

    Stratégie SIMPLIFIÉE:
    1. Collecter tous les segments dans l'ordre des chunks
    2. Fusionner les fragments fragment_end + fragment_start
    3. Dédupliquer le contenu qui apparaît dans plusieurs chunks (overlaps)
    4. Assigner les IDs séquentiels
    5. NE PAS recalculer les positions - utiliser les positions des chunks
    """

    all_segments = []

    # Étape 1: Collecter les segments EN ORDER DES CHUNKS
    # Trier results par chunk_id pour garantir l'ordre
    sorted_results = sorted(
        [r for r in results if r['success']],
        key=lambda r: r['chunk_id']
    )

    for result in sorted_results:
        chunk_id = result['chunk_id']
        chunk_start, chunk_end = result['chunk_pos']

        for akh in result['data'].get('akhbars', []):
            # Normaliser le contenu
            akh['chunk_id'] = chunk_id
            akh['chunk_start'] = chunk_start
            akh['chunk_end'] = chunk_end
            if 'confidence' not in akh:
                akh['confidence'] = 0.5
            all_segments.append(akh)

    # Étape 2: Fusionner les fragments (fragment_end + fragment_start du même type)
    merged_segments = []
    skip_indices = set()

    for i in range(len(all_segments)):
        if i in skip_indices:
            continue

        seg = all_segments[i]

        # Chercher un fragment_start suivant du même type
        if seg.get('is_fragment_end') and i + 1 < len(all_segments):
            next_seg = all_segments[i + 1]
            if (next_seg.get('is_fragment_start') and
                next_seg['type'] == seg['type'] and
                i + 1 not in skip_indices):

                # Fusionner
                if seg['type'] == 'with_isnad':
                    seg['content'] = seg.get('content', '') + ' ' + next_seg.get('content', '')
                    seg['isnad'] = seg.get('isnad', '') + ' ' + next_seg.get('isnad', '')
                else:
                    seg['content'] = seg.get('content', '') + ' ' + next_seg.get('content', '')

                seg['is_fragment_end'] = False
                seg['confidence'] = (seg.get('confidence', 0.5) + next_seg.get('confidence', 0.5)) / 2
                seg['chunk_end'] = next_seg['chunk_end']  # Étendre jusqu'au fin du next chunk

                merged_segments.append(seg)
                skip_indices.add(i + 1)
                continue

        merged_segments.append(seg)

    # Étape 3: Dédupliquer les segments exacts (de l'overlap)
    deduplicated = []
    seen_content_signatures = set()

    for seg in merged_segments:
        # Créer une signature du contenu (ignorant les espaces excessifs)
        content = seg.get('content', '').strip()
        if seg['type'] == 'with_isnad':
            content = (seg.get('isnad', '') + ' ' + content).strip()

        # Normaliser: supprimer espaces multiples
        sig = ' '.join(content.split())[:100]  # Première 100 chars normalisés

        # Si on n'a pas vu ce segment (à l'identique), le garder
        if sig not in seen_content_signatures:
            seen_content_signatures.add(sig)
            deduplicated.append(seg)

    # Étape 4: Assigner les IDs finaux
    current_id = 1
    for seg in deduplicated:
        seg['id'] = current_id

        # Position: utiliser chunk_start/chunk_end comme référence
        chunk_start = seg.get('chunk_start', 0)
        content_len = len(seg.get('content', ''))
        if seg['type'] == 'with_isnad':
            content_len += len(seg.get('isnad', ''))

        seg['text_position'] = {
            'chunk_id': seg.get('chunk_id'),
            'chunk_start': chunk_start,
            'chunk_end': seg.get('chunk_end', chunk_start + content_len)
        }

        current_id += 1

    return deduplicated


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

NUM_WORKERS = 8  # Nombre de workers parallèles pour l'API Deepseek
CHUNK_SIZE = 2000  # Taille des chunks en caractères
OVERLAP = 300  # Chevauchement entre chunks


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(num_workers: int = NUM_WORKERS):
    # Configuration
    openiti_file = Path('openiti_corpus/data/0392IbnIsmacilMisri/0392IbnIsmacilMisri.CuqalaMajanin/0392IbnIsmacilMisri.CuqalaMajanin.Shamela0027093-ara1')
    output_dir = Path('data/gold_standard')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Créer logger debug
    debug_logger = DebugLogger(output_dir / 'debug')
    print(f"[DEBUG] Enregistrement dans: {debug_logger.output_dir}")

    # Vérifier que le fichier existe
    if not openiti_file.exists():
        print(f"[ERROR] File not found: {openiti_file}")
        return

    print(f"[INFO] Loading OpenITI file: {openiti_file.name}")
    raw_text = openiti_file.read_text(encoding='utf-8')

    # Nettoyer
    print(f"[INFO] Cleaning text...")
    cleaned_text = clean_openiti_text(raw_text)
    print(f"  Raw size: {len(raw_text):,} chars")
    print(f"  Cleaned size: {len(cleaned_text):,} chars")
    print(f"  Word count: {len(cleaned_text.split()):,} words")

    # Chunking intelligent
    print(f"\n[INFO] Creating chunks with context...")
    chunks = smart_chunk_text(cleaned_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"  Number of chunks: {len(chunks)}")

    # Annoter avec Deepseek (en parallèle)
    print(f"\n[INFO] Calling Deepseek API for annotation ({num_workers} workers)...")
    print(f"  [WARNING] This will consume API credits!\n")

    all_results = [None] * len(chunks)  # Pré-allouer pour préserver l'ordre
    chunk_time_start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Soumettre tous les chunks
        futures = {
            executor.submit(call_deepseek_api, chunk, debug_logger): chunk.chunk_id
            for chunk in chunks
        }

        # Traiter les résultats au fur et à mesure avec barre de progression
        with tqdm(as_completed(futures), total=len(futures), desc="API Calls") as pbar:
            for future in pbar:
                chunk_id = futures[future]
                try:
                    result = future.result()
                    all_results[chunk_id] = result

                    if result['success']:
                        num_akhbars = len(result['data'].get('akhbars', []))
                        num_fragments = sum(1 for a in result['data'].get('akhbars', [])
                                          if a.get('is_fragment_start') or a.get('is_fragment_end'))
                        pbar.set_postfix({
                            'chunk': chunk_id,
                            'akhbars': num_akhbars,
                            'fragments': num_fragments
                        })
                    else:
                        pbar.set_postfix({'chunk': chunk_id, 'error': result.get('error', 'Unknown')})

                except Exception as e:
                    all_results[chunk_id] = {
                        'success': False,
                        'chunk_id': chunk_id,
                        'error': str(e)
                    }
                    pbar.set_postfix({'chunk': chunk_id, 'error': 'Exception'})

    chunk_time_elapsed = time.time() - chunk_time_start
    print(f"\n[OK] All chunks processed in {chunk_time_elapsed:.1f}s")

    # Fusionner les résultats
    print(f"\n[INFO] Merging chunk results...")
    final_akhbars = merge_chunk_results(all_results, chunks, cleaned_text)

    # Sauvegarder le gold standard
    gold_standard = {
        'metadata': {
            'file': str(openiti_file.name),
            'model': MODEL,
            'raw_size': len(raw_text),
            'cleaned_size': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
        },
        'processing': {
            'total_chunks': len(chunks),
            'successful_chunks': sum(1 for r in all_results if r['success']),
            'chunk_size': 2000,
            'overlap': 300,
        },
        'results': {
            'total_akhbars': len(final_akhbars),
            'akhbars': final_akhbars
        }
    }

    output_file = output_dir / 'gold_standard_0392IbnIsmacil_v2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Gold standard saved to {output_file}")

    # Statistiques par type
    print(f"\n[SUMMARY]")
    type_counts = {}
    for akh in final_akhbars:
        seg_type = akh.get('type', 'unknown')
        type_counts[seg_type] = type_counts.get(seg_type, 0) + 1

    successful_chunks = sum(1 for r in all_results if r and r['success'])

    print(f"  Total akhbars: {len(final_akhbars)}")
    print(f"  By type:")
    for seg_type, count in sorted(type_counts.items()):
        print(f"    - {seg_type}: {count}")
    print(f"\n  Execution:")
    print(f"    - Workers used: {num_workers}")
    print(f"    - Total chunks: {len(chunks)}")
    print(f"    - Successful API calls: {successful_chunks}/{len(chunks)}")
    print(f"    - Processing time: {chunk_time_elapsed:.1f}s")
    print(f"    - Avg time per chunk: {chunk_time_elapsed/len(chunks):.2f}s")
    print(f"    - Throughput: {len(chunks)/chunk_time_elapsed:.1f} chunks/sec")

    # Afficher exemples
    if final_akhbars:
        print(f"\n[EXAMPLES] First 3 akhbars:")
        for akh in final_akhbars[:3]:
            print(f"\n  [{akh['id']}] Type: {akh['type']}")
            print(f"      Confidence: {akh['confidence']:.2f}")
            if akh['type'] == 'with_isnad':
                print(f"      Isnad: {akh.get('isnad', '')[:80]}...")
                print(f"      Content: {akh.get('content', '')[:80]}...")
            else:
                print(f"      Content: {akh.get('content', '')[:100]}...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create gold standard with Deepseek API')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                       help=f'Number of parallel workers (default: {NUM_WORKERS})')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                       help=f'Chunk size in characters (default: {CHUNK_SIZE})')
    parser.add_argument('--overlap', type=int, default=OVERLAP,
                       help=f'Overlap between chunks (default: {OVERLAP})')

    args = parser.parse_args()

    # Override globals
    NUM_WORKERS = args.workers
    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap

    main(num_workers=args.workers)
