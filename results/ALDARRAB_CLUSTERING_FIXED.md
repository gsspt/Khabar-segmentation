# alDarrab Clustering: Bug Fix & Results

**Date**: 2026-04-22  
**Issue**: Chunk offsets were relative, not absolute to corpus  
**Solution**: Use stride=500 to convert relative → absolute positions  
**Status**: FIXED ✓

---

## The Problem

Initial inference from Colab produced offsets relative to each 500-char chunk:
- Chunk 0: chars 0-550 (corpus positions)
- Chunk 1: chars 0-550 (chunk-relative) → should be 500-1050 (corpus absolute)
- Chunk 2: chars 0-550 (chunk-relative) → should be 1000-1550 (corpus absolute)
- ... and so on for 32 chunks

Our deduplication logic treated all offsets as absolute, causing them to collapse into the first 550 chars.

## The Fix

**Convert relative chunk offsets to absolute corpus positions:**

```python
CHUNK_SIZE = 500      # from metadata
STRIDE = 500          # chunks processed sequentially with this stride

chunk_starts = [i * STRIDE for i in range(num_chunks)]

# For each token, convert to absolute position:
chunk_id = (token_index / tokens_per_chunk)  # which chunk this token belongs to
absolute_char_pos = relative_offset + chunk_starts[chunk_id]
```

This assumes chunks are processed with stride=CHUNK_SIZE (no overlap), which is correct for the Colab inference.

---

## Results: alDarrab with Corrected Clustering

### Raw Statistics
| Metric | Value |
|--------|-------|
| **Corpus size** | 15,689 chars |
| **Unique positions** | 4,837 |
| **Boundary tokens** | 773 (pred=1) |
| **Boundary percentage** | 16.0% |

### Clustering Results (gap=20)

| Metric | Value |
|--------|-------|
| **Clusters detected** | **26** |
| **Avg cluster span** | 108 chars |
| **Boundaries per cluster** | ~30 tokens |

### Khabar Distribution

| Metric | Value |
|--------|-------|
| **Min gap** | 71 chars |
| **Median gap** | 330 chars |
| **Max gap** | 1,692 chars |
| **Gaps > 100 chars** | 23 |
| **Gaps > 500 chars** | 10 |

### Spatial Distribution

```
26 boundaries spread across 15,689 chars:
  Boundary  0: char   2- 212 (  210 chars)
  Boundary  1: char 1248-1334 (   86 chars)
  Boundary  2: char 1517-1693 (  176 chars)
  ...
  Boundary 25: char 15281-15317 (   36 chars)

Coverage: All major sections of corpus covered
```

---

## Interpretation

**26 khabars** detected across alDarrab text:

1. **Boundary spacing**: 330 chars median = typical isnad (50-100 chars) + narrative content (200-300 chars)
2. **Large gaps** (500-1600 chars): Long narrative passages
3. **Small gaps** (50-100 chars): Short narrative segments or rapid-fire stories
4. **Distribution**: Evenly spread throughout corpus (not concentrated)

---

## Lesson Learned

**Always verify offset interpretation:**
- ❌ Assumption: All offsets are absolute corpus positions
- ✓ Reality: Tokenizer offsets are chunk-relative when using batched inference
- ✓ Fix: Multiply chunk_id by stride to get absolute positions

---

## Files Generated

- `results/camelbert_alDarrab_char_boundaries.json` — 26 boundaries (corrected)
- `results/ALDARRAB_CLUSTERING_FIXED.md` — This report

---

## Next Steps

Now that clustering is fixed for alDarrab, we can:
1. Compare with Baseline v4 results
2. Evaluate against manual gold standard (if available)
3. Test the same fix on Kitab Uqala (re-run with stride=500)
4. Apply to other OpenITI texts

**Status**: Ready for Phase 3 multi-pipeline comparison
