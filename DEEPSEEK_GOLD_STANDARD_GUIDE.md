# Deepseek Gold Standard Generation Guide

**Status**: Enhanced solution for smart chunking + API calls  
**Script**: `scripts/generate_gold_standard_deepseek_chunked.py`

---

## Problem

The original `generate_gold_standard_deepseek.py` sent entire texts to the API:
- Works for small texts (alDarrab: 15K chars)
- Fails for large texts (Kitab Uqala: 268K chars)
- Risk of context truncation or incomplete segmentation
- Can be expensive in API tokens

---

## Solution: Smart Chunking

### Strategy Overview

```
1. Split text into manageable chunks (~3.5K chars each)
2. Add overlap between chunks (500 chars) to preserve context
3. Discover smart break points (newlines, punctuation)
4. Send each chunk to Deepseek API independently
5. Convert chunk-relative positions → global positions
6. Merge results, deduplicating overlapping regions
7. Validate final boundaries
```

### Visual Example

```
Original text (100 chars):
[A...50 chars...B...50 chars...C]
                 ↓ Split with overlap
Chunk 1 (55 chars):
[A...50 chars...B⚬⚬⚬⚬⚬]  (B + 5 overlap)
                    ↓ Deepseek API

Chunk 2 (55 chars):
            [⚬⚬⚬⚬⚬B...50 chars...C]  (5 overlap + C)
                    ↓ Deepseek API

Result:
Merged units with positions adjusted to global coordinates
```

---

## Parameters Explained

### `--chunk-size` (default: 3500)
- Size of each text chunk in characters
- Larger chunks: fewer API calls, but more tokens per call
- Smaller chunks: more API calls, but less context loss
- **Recommendation**: 3000-4000 for optimal balance

**Impact on Deepseek API**:
- ~3500 chars ≈ 900-1200 tokens (reasonable prompt size)
- ~5000 chars ≈ 1300-1600 tokens
- Model supports up to ~4000 input tokens for `deepseek-chat`

### `--overlap` (default: 500)
- Overlap between consecutive chunks
- Allows API to see context of boundaries
- Prevents units from being cut in half
- **Recommendation**: 400-600 chars (10-15% of chunk)

**Why overlap is crucial**:
```
Without overlap:
Chunk 1: [...isnad 1 complete]
Chunk 2: [khabar 1 content...][isnad 2 starts...]  ← khabar 1 incomplete!

With overlap:
Chunk 1: [...isnad 1 complete][khabar 1 starts...]
         └─── 500 chars overlap ───┘
Chunk 2: [khabar 1 complete][isnad 2 complete...]  ← All units intact!
```

---

## Usage Examples

### Basic Usage (alDarrab)
```bash
export DEEPSEEK_API_KEY="your_api_key_here"

python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/alDarrab_clean.txt \
  --output results/gold_standard_alDarrab_deepseek.json
```

### Large Text (Kitab Uqala)
```bash
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/kitab_uqala_reference_corpus.txt \
  --output results/Kitab_Uqala_al_Majanin/gold_standard_kitab_uqala_deepseek.json \
  --chunk-size 4000 \
  --overlap 600
```

### Aggressive Chunking (for cost optimization)
```bash
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/kitab_uqala_reference_corpus.txt \
  --output results/gold_standard_kitab_uqala_deepseek.json \
  --chunk-size 5000 \
  --overlap 300
```

---

## How It Works: Step-by-Step

### 1. Smart Text Splitting

**Logic**:
- Divide text into ~3500 char chunks
- Look for break points (newline `\n`, Arabic comma `،`, period `.`, space)
- Preserve logical boundaries to avoid cutting isnads/khabars

**Output**:
```
Chunk 0: chars 0-3500
Chunk 1: chars 3000-6500    (500 char overlap)
Chunk 2: chars 6000-9500    (500 char overlap)
...
```

### 2. API Calls

**For each chunk**:
- Send chunk text to Deepseek API
- Request JSON response with units and positions
- Units have `char_start`, `char_end` RELATIVE TO CHUNK

**Prompt guides**:
- Identifies isnads (حدثنا، أخبرنا، قال، etc.)
- Extracts transmission chains
- Returns JSON with narrative units

### 3. Position Conversion

**From chunk-relative → global**:
```python
global_start = chunk_global_start + unit_relative_start
global_end = chunk_global_start + unit_relative_end
```

**Example**:
```
Chunk starts at global position 3000
Unit in chunk: char_start=50, char_end=500
Global position: char_start=3050, char_end=3500
```

### 4. Deduplication

**When chunks overlap** (e.g., Chunk 1 and 2):
- Deepseek may identify same unit in both chunks
- Both have same global positions → detected as duplicate
- Keep only one, using the one with most metadata

**Tolerance**: 50 chars position difference = same unit

### 5. Final Validation

**Checks**:
- All positions within corpus bounds
- No zero-length units
- Units sorted by position
- Unit IDs sequential

---

## Output Format

**JSON structure**:
```json
{
  "metadata": {
    "source": "deepseek-api-chunked",
    "model": "deepseek-chat",
    "text_length_chars": 268540,
    "total_chunks": 78,
    "chunk_size": 3500,
    "overlap": 500,
    "total_units": 485,
    "units_with_isnad": 412
  },
  "narrative_units": [
    {
      "unit_id": 0,
      "char_start": 0,
      "char_end": 245,
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد عن علي..."
    },
    {
      "unit_id": 1,
      "char_start": 245,
      "char_end": 523,
      "has_isnad": false,
      "isnad_text": null
    }
  ]
}
```

---

## Comparison with Other Pipelines

This output is directly comparable with:
- `camelbert_[TEXT]_narrative_units.json`
- `baseline_v4_[TEXT]_narrative_units.json`

**Using `compare_two_pipelines.py`**:
```bash
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_[TEXT]_deepseek.json \
  --pipeline2 results/[TEXT]_narrative_units_camelbert.json \
  --name1 deepseek \
  --name2 camelbert \
  --output results/comparison_deepseek_vs_camelbert_[TEXT].json
```

---

## Cost Estimation

### API Token Usage

**Deepseek pricing** (approximate):
- Input: $0.14 / 1M tokens
- Output: $0.28 / 1M tokens

**Example for Kitab Uqala** (268K chars, 78 chunks):

| Scenario | Chunks | Input Tokens | Output Tokens | Cost |
|----------|--------|--------------|---------------|------|
| chunk-size=3500, overlap=500 | 78 | ~15K | ~8K | ~$3 |
| chunk-size=4000, overlap=300 | 68 | ~14K | ~7K | ~$2.50 |
| Full text (single call) | 1 | ~70K | 4K | ~$10 |

**Recommendation**: Use chunked approach for cost savings.

---

## Troubleshooting

### Issue: "Could not parse JSON from response"
**Cause**: Deepseek returned malformed JSON  
**Fix**:
1. Check API key validity
2. Reduce chunk size (more context per unit)
3. Increase temperature: `temperature=0.5` (more creative, less structured)

### Issue: Inconsistent unit count between chunks
**Cause**: Normal - overlaps handle this  
**Check**: Verify deduplication worked by examining output JSON

### Issue: Units extend beyond text length
**Cause**: API generated invalid boundaries  
**Fix**: Script automatically clamps to text length

### Issue: "API call failed"
**Cause**: Network error, rate limit, or invalid key  
**Fix**:
1. Verify `DEEPSEEK_API_KEY` is set
2. Check rate limits (Deepseek: ~60 RPM)
3. Retry with exponential backoff

---

## Recommendations

### For alDarrab (15K chars)
```bash
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/alDarrab_clean.txt \
  --output results/gold_standard_alDarrab_deepseek.json \
  --chunk-size 3500 \
  --overlap 500
```
Expected: 5-8 chunks, ~20-30 min, ~$0.30

### For Kitab Uqala (268K chars)
```bash
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/kitab_uqala_reference_corpus.txt \
  --output results/Kitab_Uqala_al_Majanin/gold_standard_kitab_uqala_deepseek.json \
  --chunk-size 4000 \
  --overlap 600
```
Expected: 65-75 chunks, ~60-90 min, ~$2-3

### For other OpenITI texts
Adjust `--chunk-size` based on text length:
- Small texts (< 50K): 3500 chars, 500 overlap
- Medium texts (50-200K): 4000 chars, 600 overlap
- Large texts (> 200K): 4500 chars, 700 overlap

---

## Validation Checklist

After generating gold standard:

- [ ] JSON is valid (can be parsed)
- [ ] All units have `char_start < char_end`
- [ ] No units extend beyond text length
- [ ] Unit IDs are sequential (0, 1, 2, ...)
- [ ] Total_units matches length of narrative_units array
- [ ] Can be loaded by `compare_two_pipelines.py`

---

## Integration with Phase 3

**Full workflow**:

```bash
# 1. Generate gold standard
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/alDarrab_clean.txt \
  --output results/gold_standard_alDarrab_deepseek.json \
  --api-key $DEEPSEEK_API_KEY

# 2. Compare with CAMeL-BERT
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_alDarrab_deepseek.json \
  --pipeline2 results/alDarrab_narrative_units_camelbert.json \
  --name1 gold_standard \
  --name2 camelbert \
  --output results/comparison_gold_vs_camelbert.json

# 3. Compare with Baseline
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_alDarrab_deepseek.json \
  --pipeline2 results/alDarrab_narrative_units_baseline.json \
  --name1 gold_standard \
  --name2 baseline \
  --output results/comparison_gold_vs_baseline.json
```

---

**Status**: Ready for production use with Deepseek API

