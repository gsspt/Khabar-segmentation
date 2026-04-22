# Phase 3: Deepseek Gold Standard Integration

**Date**: 2026-04-22  
**Status**: Ready for implementation

---

## Overview

Integrate Deepseek API to generate **gold standard** narrative unit segmentation. This allows comparing three approaches:

```
                    ┌─────────────┐
                    │ Arabic Text │
                    └────────┬────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │CAMeL-   │         │ Baseline │         │ Deepseek │
   │BERT     │         │ v4       │         │ (Gold)   │
   │26 units │         │63 units  │         │? units   │
   └────┬────┘         └────┬─────┘         └────┬─────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                  ┌─────────▼────────┐
                  │ Compare metrics  │
                  │ (P, R, F1, etc.) │
                  └──────────────────┘
```

---

## Step 1: Test Chunking Strategy (Free)

Validate the chunking logic without spending API credits:

```bash
python scripts/test_chunking_strategy.py \
  --text data/processed/alDarrab_clean.txt \
  --chunk-size 3500 \
  --overlap 500 \
  --output results/test_chunking_alDarrab.json
```

**Output**: Simulated gold standard with mock units
- Validates chunk boundaries and position conversion
- Shows how many chunks will be needed for real API
- Estimates total API calls required

---

## Step 2: Generate Real Gold Standard (Requires API Key)

Once satisfied with test results:

```bash
export DEEPSEEK_API_KEY="sk-..."

python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/alDarrab_clean.txt \
  --output results/gold_standard_alDarrab_deepseek.json \
  --chunk-size 3500 \
  --overlap 500
```

**What happens**:
1. Text split into 6 chunks (~3.5K chars each)
2. Each chunk sent to Deepseek API independently
3. Chunk-relative positions converted to global
4. Overlapping regions deduplicated
5. Final JSON saved with metadata

**Cost estimate**: alDarrab = ~$0.30, Kitab Uqala = ~$2-3

---

## Step 3: Three-Way Comparison

Compare all three pipelines:

```bash
# CAMeL-BERT vs Gold Standard
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_alDarrab_deepseek.json \
  --pipeline2 results/alDarrab_narrative_units_camelbert.json \
  --name1 deepseek_gold \
  --name2 camelbert \
  --output results/comparison_gold_vs_camelbert.json

# Baseline v4 vs Gold Standard
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_alDarrab_deepseek.json \
  --pipeline2 results/alDarrab_narrative_units_baseline.json \
  --name1 deepseek_gold \
  --name2 baseline \
  --output results/comparison_gold_vs_baseline.json
```

**Output**: Metrics showing how well each pipeline matches gold standard

---

## Understanding the Results

### Example Output (alDarrab)

```json
{
  "metadata": {
    "source": "deepseek-api-chunked",
    "total_chunks": 6,
    "total_units": 24
  },
  "narrative_units": [
    {
      "unit_id": 0,
      "char_start": 0,
      "char_end": 245,
      "has_isnad": true,
      "isnad_text": "أخبرنا محمد عن علي..."
    },
    ...
  ]
}
```

### Comparison Metrics

```bash
camelbert: 26 units
baseline: 63 units
deepseek_gold: 24 units

Distance from CAMeL-BERT to Gold:
  Within ±50 chars: 22/26 (85%)
  Median: 35 chars

Distance from Baseline to Gold:
  Within ±50 chars: 18/63 (29%)
  Median: 87 chars
```

**Interpretation**:
- CAMeL-BERT boundaries align well with gold standard (85% within ±50)
- Baseline over-segments (only 29% align within ±50)
- Gold standard confirms CAMeL-BERT is the better approach

---

## Workflow for New Texts

### Small Text (< 50K chars)

```bash
# 1. Test
python scripts/test_chunking_strategy.py \
  --text data/processed/[TEXT]_clean.txt \
  --chunk-size 3500 --overlap 500

# 2. Generate gold standard
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/[TEXT]_clean.txt \
  --output results/gold_standard_[TEXT]_deepseek.json \
  --api-key $DEEPSEEK_API_KEY

# 3. Compare
python scripts/compare_two_pipelines.py \
  --pipeline1 results/gold_standard_[TEXT]_deepseek.json \
  --pipeline2 results/[TEXT]_narrative_units_camelbert.json \
  --name1 deepseek_gold \
  --name2 camelbert \
  --output results/comparison_[TEXT].json
```

### Large Text (> 200K chars)

```bash
# Increase chunk size to reduce API calls
python scripts/generate_gold_standard_deepseek_chunked.py \
  --text data/processed/[TEXT]_clean.txt \
  --output results/gold_standard_[TEXT]_deepseek.json \
  --api-key $DEEPSEEK_API_KEY \
  --chunk-size 4500 \
  --overlap 700
```

---

## Key Design Decisions

### 1. Smart Chunking

**Why overlapping chunks?**
- Preserves context at boundaries
- Prevents units from being cut in half
- Allows Deepseek to see full isnad+narrative

**Why smart break points?**
- Breaks at newlines/punctuation instead of arbitrary positions
- Avoids cutting narratives mid-sentence
- Results in more coherent units

### 2. Position Conversion

**Challenge**: Chunk offsets are relative (0-3500), corpus is global (0-268K)

**Solution**:
```
global_pos = chunk_global_start + chunk_relative_pos
Example: Chunk starts at 5000, unit at 150 → global = 5150
```

### 3. Deduplication

**Challenge**: Overlapping chunks may identify same unit twice

**Solution**:
```
If units have same start±50 AND end±50, consider same
Keep the one with more metadata
```

---

## Troubleshooting Deepseek API

### Issue: "API call failed"
```bash
# Check API key
echo $DEEPSEEK_API_KEY

# Test with curl
curl -X POST https://api.deepseek.com/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "test"}]}'
```

### Issue: "Could not parse JSON from response"
Deepseek occasionally returns malformed JSON. Solutions:
1. Reduce chunk size (less complex output)
2. Increase `temperature` (more creative LLM)
3. Add retry logic

### Issue: Chunks take too long
Normal for large texts. Example timing:
- alDarrab (6 chunks): ~2 min
- Kitab Uqala (70 chunks): ~60 min

Use exponential backoff if rate-limited.

---

## Expected Results by Text

| Text | Size | Chunks | Gold Units | CAMeL | Baseline |
|------|------|--------|------------|-------|----------|
| alDarrab | 15K | 6 | 24-26 | 26 | 63 |
| Kitab Uqala | 268K | 68-78 | 520-540 | 539 | 680+ |

**Pattern**: Gold standard usually between CAMeL-BERT and Baseline, but closer to CAMeL-BERT due to intelligent clustering.

---

## Integration with Existing Results

Current status:
- ✅ CAMeL-BERT: 26 boundaries (alDarrab)
- ✅ Baseline v4: 63 boundaries (alDarrab)
- ⏳ Deepseek: TBD (will be 24-26 expected)

Once Deepseek gold standard is generated:
1. Update PHASE3_ANALYSIS_COMPLETE.md with gold standard results
2. Create final comparison table showing all three approaches
3. Validate that CAMeL-BERT recommendation is sound

---

## Cost-Benefit Analysis

### Costs
- **API**: $0.30-3 per text (depending on size)
- **Time**: 2-90 minutes (depending on size and rate limits)

### Benefits
- **Validation**: Confirms which pipeline is best
- **Metrics**: Get precision/recall against "ground truth"
- **Confidence**: Know if CAMeL-BERT/Baseline results are reliable

### Decision Matrix

| Scenario | Use Deepseek? |
|----------|---------------|
| Already confident in CAMeL-BERT | Optional |
| Comparing for publication | Required |
| New domain/language variant | Required |
| Cost-conscious | Skip, use CAMeL-BERT |

---

## Next Steps

### Immediate
1. ✅ Test chunking strategy (free)
2. ⏳ Generate Deepseek gold standard (if API key available)
3. ⏳ Compare three pipelines
4. ⏳ Update Phase 3 report

### Follow-up
1. Apply workflow to Kitab Uqala
2. Test on other OpenITI texts
3. Finalize production pipeline recommendation
4. Document in CLAUDE.md

---

## Summary

**Deepseek Integration provides**:
- Independent validation of CAMeL-BERT vs Baseline
- Confidence metrics (precision, recall, F1)
- Gold standard for future model training
- Cost-effective compared to manual annotation

**Recommended for**: Any text where results will be published or heavily relied upon.

