#!/usr/bin/env python3
"""
Crée un gold standard en utilisant l'API Deepseek pour annoter des textes arabes.

Approche:
1. Charger un texte OpenITI
2. Nettoyer le format mARkdown
3. Diviser en chunks managéables
4. Appeler l'API Deepseek pour segmenter en isnads + khabars
5. Sauvegarder les résultats

Gold standard : vérité de référence pour comparer :
  - Baseline v4 (rule-based)
  - CAMeL-BERT (fine-tuned)
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Charger les variables d'environnement
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

# Configuration Deepseek
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


# ── NETTOYAGE DU FORMAT OPENITI ──────────────────────────────────────────────

def clean_openiti_text(text: str) -> str:
    """Nettoie un texte au format mARkdown OpenITI."""
    # Supprimer le header
    if '#META#Header#End#' in text:
        text = text[text.index('#META#Header#End#') + len('#META#Header#End#'):]

    # Supprimer marqueurs de section
    text = re.sub(r'^###[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Joindre lignes de continuation
    text = re.sub(r'\n~~', ' ', text)

    # Supprimer marqueurs de page, manuscrit, notes
    text = re.sub(r'PageV\d+P\d+', '', text)
    text = re.sub(r'\bms\d+\b', '', text)
    text = re.sub(r'\[\d+[a-z]?\]', '', text)

    # Normaliser espaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Divise le texte en chunks avec chevauchement."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


# ── API DEEPSEEK ─────────────────────────────────────────────────────────────

def call_deepseek_api(text: str, chunk_id: int = 0) -> Optional[Dict]:
    """
    Appelle l'API Deepseek pour segmenter le texte en isnads et khabars.

    Utilise un prompt en arabe pour de meilleurs résultats.
    """

    system_prompt = """أنت متخصص في تحليل النصوص العربية التاريخية وتقسيمها إلى وحدات سردية (أخبار).

مهمتك: تقسيم النص إلى أخبار من عدة أنواع:

[نوع 1] with_isnad: خبر مع إسناد (الأكثر شيوعاً)
- isnad: سلسلة الرواة (حدثنا...، أخبرنا...، سمعت...)
- matn: محتوى الخبر
- مثال: isnad="حدثنا محمد قال" matn="أخبرنا زيد قال رأيت النبي"

[نوع 2] poetry: مقاطع شعرية
- content: النص الشعري
- مثال: "يا ليل الصبح كم أتمنى / أن يرجع القلب من الشقا"

[نوع 3] prose: نص نثري منفصل (بدون إسناد)
- content: النص
- مثال: "وفي هذا السنة حدثت أحداث عظيمة"

[نوع 4] continuation: متن مسترسل من الخبر السابق
- content: النص المستمر
- مثال: "فلما انتهى من الخطبة قام الناس والتفوا حوله"

الحدود بين الأخبار:
- نهاية إسناد واضحة (قال + فعل ناقل = انتقال)
- فراغ طبيعي في النص
- تغيير الموضوع أو الشخصيات الرئيسية
- بدء إسناد جديد

قدم النتيجة كـ JSON صحيح فقط:
{
  "akhbars": [
    {
      "id": 1,
      "type": "with_isnad" | "poetry" | "prose" | "continuation",
      "isnad": "نص_الإسناد" (فقط إذا كان النوع with_isnad),
      "content": "نص المتن أو المحتوى",
      "confidence": درجة_ثقة (0.0-1.0),
      "notes": "ملاحظات اختيارية"
    }
  ]
}

مثال الإخراج:
{
  "akhbars": [
    {
      "id": 1,
      "type": "with_isnad",
      "isnad": "حدثنا محمد",
      "content": "قال رأيت النبي يصلي",
      "confidence": 0.95,
      "notes": "إسناد واضح وسهل التحديد"
    },
    {
      "id": 2,
      "type": "continuation",
      "content": "ثم التفت الناس حوله",
      "confidence": 0.80,
      "notes": "استمرار من الخبر السابق"
    }
  ]
}

ركز على:
- دقة تحديد نهاية الإسناد وبداية المتن
- تمييز الأنواع المختلفة بوضوح
- درجات ثقة واقعية (أقل للحالات الغامضة)
"""

    user_prompt = f"""قسّم النص التالي إلى أخبار (مع الأنواع والثقة):

---TEXT START---
{text}
---TEXT END---

قواعد مهمة:
1. كل خبر يجب أن يكون وحدة منطقية منفصلة
2. لا تدمج أخبار منفصلة حتى لو تتابعت
3. استخدم "continuation" فقط للنصوص التي تستمر من سابقة مباشرة
4. درجات ثقة عالية (>0.9) للحالات الواضحة، منخفضة (<0.7) للغامضة

أرِني JSON فقط، بدون تفسير."""

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
        "temperature": 0.3,  # Lower temperature للنتائج أكثر استقراراً
        "max_tokens": 4000,
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()

        # استخراج النص من الرد
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']

            # محاولة استخراج JSON من النص
            # (في بعض الحالات Deepseek يضع JSON في ```json ... ```)
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
                akhbars_data = json.loads(json_str)
                return {
                    'success': True,
                    'chunk_id': chunk_id,
                    'data': akhbars_data
                }

        return {
            'success': False,
            'chunk_id': chunk_id,
            'error': 'Could not parse response'
        }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'chunk_id': chunk_id,
            'error': str(e)
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'chunk_id': chunk_id,
            'error': f'JSON decode error: {e}'
        }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Configuration
    openiti_file = Path('openiti_corpus/data/0392IbnIsmacilMisri/0392IbnIsmacilMisri.CuqalaMajanin/0392IbnIsmacilMisri.CuqalaMajanin.Shamela0027093-ara1')
    output_dir = Path('data/gold_standard')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier que le fichier existe
    if not openiti_file.exists():
        print(f"[ERROR] File not found: {openiti_file}")
        print(f"[INFO] Checking available paths:")
        for p in Path('openiti_corpus').rglob('*ara1')[:5]:
            print(f"  - {p}")
        return

    print(f"[INFO] Loading OpenITI file: {openiti_file.name}")
    raw_text = openiti_file.read_text(encoding='utf-8')

    # Nettoyer
    print(f"[INFO] Cleaning text...")
    cleaned_text = clean_openiti_text(raw_text)

    print(f"  Raw size: {len(raw_text)} chars")
    print(f"  Cleaned size: {len(cleaned_text)} chars")
    print(f"  Reduction: {100 * (1 - len(cleaned_text)/len(raw_text)):.1f}%")

    # Diviser en chunks
    print(f"\n[INFO] Splitting into chunks...")
    chunks = split_into_chunks(cleaned_text, chunk_size=1500, overlap=300)
    print(f"  Number of chunks: {len(chunks)}")

    # Annoter avec Deepseek
    print(f"\n[INFO] Calling Deepseek API for annotation...")
    print(f"  [WARNING] This will consume API credits!\n")

    all_results = []

    for i, chunk in enumerate(tqdm(chunks, desc="Chunks")):
        result = call_deepseek_api(chunk, chunk_id=i)
        all_results.append(result)

        if result['success']:
            print(f"\n[OK] Chunk {i}: {len(result['data'].get('akhbars', []))} akhbars")
        else:
            print(f"\n[ERROR] Chunk {i}: {result['error']}")

    # Fusionner les résultats
    print(f"\n[INFO] Merging results...")
    merged_akhbars = []
    for result in all_results:
        if result['success'] and 'akhbars' in result['data']:
            merged_akhbars.extend(result['data']['akhbars'])

    # Sauvegarder le gold standard
    gold_standard = {
        'file': str(openiti_file),
        'model': MODEL,
        'total_chunks': len(chunks),
        'successful_chunks': sum(1 for r in all_results if r['success']),
        'total_akhbars': len(merged_akhbars),
        'akhbars': merged_akhbars
    }

    output_file = output_dir / 'gold_standard_0392IbnIsmacil.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Gold standard saved to {output_file}")
    print(f"\n[SUMMARY]")
    print(f"  Total akhbars: {len(merged_akhbars)}")
    print(f"  Successful API calls: {sum(1 for r in all_results if r['success'])}/{len(chunks)}")

    # Afficher quelques exemples
    if merged_akhbars:
        print(f"\n[EXAMPLES] First 3 akhbars:")
        for akh in merged_akhbars[:3]:
            print(f"\n  Type: {akh.get('type', 'unknown')}")
            print(f"  ID: {akh.get('id', 'N/A')}")
            print(f"  Confidence: {akh.get('confidence', 'N/A'):.2f}")

            if akh.get('type') == 'with_isnad':
                print(f"  Isnad: {akh.get('isnad', '')[:80]}...")
                print(f"  Content: {akh.get('content', '')[:80]}...")
            else:
                print(f"  Content: {akh.get('content', '')[:100]}...")

            if akh.get('notes'):
                print(f"  Notes: {akh.get('notes')}")


if __name__ == '__main__':
    main()
