# CAMeL-BERT Conversion Flow: Raw Inference → Khabar Boundaries

**Date**: 2026-04-22  
**Context**: Explain precisely how the conversion works and optimize it

---

## 1. INPUT: Raw Inference JSON

### File Structure
```
results/Kitab_Uqala_al_Majanin/camelbert_kitab_uqala_raw_inference.json
└── metadata
    ├── corpus: "kitab_uqala_reference_corpus"
    ├── text_size_chars: 268,540
    ├── text_size_words: 53,812
    ├── model: "camelbert_binary_classification_final"
    └── fix_applied: "Full document chunking with overlap"

└── inference_results
    ├── total_tokens: 297,984
    ├── tokens: [297,984 elements]        # Token strings
    ├── offsets: [297,984 elements]       # [[char_start, char_end], ...]
    ├── predictions: [297,984 elements]   # [0 or 1, ...] — 0=not boundary, 1=boundary
    └── probabilities: [297,984 elements] # [0.0001...0.9999, ...]
```

### Key Statistics
- **Total tokens**: 297,984
- **Boundary tokens (pred=1)**: 17,938 (6.02%)
- **Non-boundary tokens**: 279,746 (93.98%)
- **Special tokens** ([CLS], [SEP], [PAD], [UNK]): 221,552
- **Offsets with [0,0]**: 371
- **Unique char_start positions**: 72,057

---

## 2. CONVERSION PROCESS: Step-by-Step

### STEP 1: Extract Boundary Tokens (pred=1)
**Goal**: Identify all tokens the model predicted as boundary markers

**Input**: 297,984 tokens with predictions

**Process**:
```python
boundary_indices = [i for i, pred in enumerate(predictions) if pred == 1]
```

**Output**: 17,938 boundary token indices

**Examples**:
- Token at index 583: offset=[723, 744], prob=0.626
- Token at index 590: offset=[746, 757], prob=0.980
- Token at index 591: offset=[753, 774], prob=0.988

### STEP 2: Deduplication by char_start (Keep Max Probability)
**Goal**: For each unique char_start position, keep only the token with highest confidence

**Problem**: Due to BERT's tokenization and chunking with overlap, the same text position can appear multiple times with different tokens and probabilities. We want only the highest-confidence prediction at each position.

**Process**:
```python
boundary_by_char = {}  # char_start → (token, offset, probability)

for i in boundary_indices:
    char_start = offsets[i][0]
    prob = probabilities[i]
    
    if char_start not in boundary_by_char or prob > boundary_by_char[char_start][2]:
        boundary_by_char[char_start] = (tokens[i], offsets[i], prob)
```

**Input**: 17,938 boundary tokens
**Output**: 16,253 unique char_start positions

**Reduction**: 17,938 → 16,253 (9.6% reduction due to duplicate overlaps)

**Examples** (showing duplicates being resolved):
```
char_start=924:
  First occurrence (i=632): prob=0.9929 → STORED
  Second occurrence (i=1025): prob=0.9985 → UPDATED (higher prob)
  
char_start=930:
  First occurrence (i=633): prob=0.9929 → STORED
  Second occurrence (i=1026): prob=0.9987 → UPDATED
  
char_start=956:
  First occurrence (i=639): prob=0.6361 → STORED
  Second occurrence (i=1033): prob=0.9993 → UPDATED (much higher!)
```

### STEP 3: Cluster Boundary Tokens (GAP_CLUSTER ≤ 50 chars)
**Goal**: Group consecutive boundary tokens into isnad clusters

**Problem**: A single isnad (transmission chain) may span multiple tokens. We need to group tokens that are close together (within 50 chars) as part of the same isnad.

**Process**:
```python
GAP_CLUSTER = 50
clusters = []
cur_start = sorted_chars[0]
cur_end = offsets[sorted_chars[0]][1]
cur_tokens = [sorted_chars[0]]

for char_start in sorted_chars[1:]:
    gap = char_start - cur_end
    
    if gap <= GAP_CLUSTER:
        # Extend current cluster
        cur_end = max(cur_end, offsets[char_start][1])
        cur_tokens.append(char_start)
    else:
        # Gap too large → close cluster, start new one
        clusters.append({
            'char_start': cur_start,
            'char_end': cur_end,
            'n_tokens': len(cur_tokens)
        })
        cur_start = char_start
        cur_end = offsets[char_start][1]
        cur_tokens = [char_start]
```

**Input**: 16,253 boundary positions
**Output**: ~520 isnad clusters (depending on corpus)

**Cluster Statistics**:
- Min tokens per cluster: 1
- Max tokens per cluster: 31
- Median tokens per cluster: ~3
- Avg tokens per cluster: ~4.1

**Example Cluster**:
```
Cluster 0:
  char_start=723  char_end=987  (length=264)  n_tokens=13
  Contains: [723, 746, 753, 757, 761, 766, 769, 777, 782, 784, 788, 791, 798]
```

### STEP 4: Extract Khabar Boundaries
**Goal**: Each cluster's start position = a khabar boundary

**Process**:
```python
khabar_boundaries = [cluster['char_start'] for cluster in clusters]
```

**Input**: 520 clusters
**Output**: 520 khabar boundaries

**Example**:
```
[723, 1197, 1829, 2087, 2550, ..., 267812]
```

### STEP 5: Analyze Gap Distribution
**Goal**: Understand khabar lengths (gaps between clusters = content regions)

**Process**:
```python
gaps = []
for i in range(1, len(clusters)):
    gap = clusters[i]['char_start'] - clusters[i-1]['char_end']
    gaps.append(gap)
```

**Statistics** (for Kitab Uqala):
- Min gap: 1 char (back-to-back isnads)
- Median gap: 250 chars (typical khabar length)
- Max gap: 4,113 chars (long narrative section)
- Gaps > 100 chars: 368 (long khabars)
- Gaps > 500 chars: 114 (very long narratives)

---

## 3. CURRENT FLOW: convert_boundary_tokens_direct.py

### What It Does
1. Load raw_inference.json (297,984 tokens)
2. Extract boundary tokens (pred=1)
3. Deduplicate by char_start
4. Cluster boundary tokens (GAP_CLUSTER=50)
5. Extract khabar boundaries
6. Evaluate against gold standard (F1, P, R, distances)
7. Save results to `char_boundaries_v2.json`

### Output Structure
```json
{
  "metadata": {
    "method": "direct_char_extraction_from_raw_inference",
    "unique_char_positions": 16253,
    "boundary_tokens_count": 17938,
    "n_clusters_isnad": 520,
    "n_khabar_boundaries": 520,
    "evaluation_tol80": {
      "precision": 0.9346,
      "recall": 0.7928,
      "f1": 0.8579,
      "tp": 486,
      "fp": 34,
      "fn": 127
    }
  },
  "khabar_boundaries": [
    {
      "boundary_id": 0,
      "char_start": 723,
      "char_end": 987,
      "n_tokens": 13,
      "tokens": ["أخبرنا", "محمد", "بن", ...],
      "text_context": "..."
    },
    ...
  ]
}
```

---

## 4. ISSUES & OPTIMIZATIONS

### Issue 1: Fixed GAP_CLUSTER Value (Current: 50 chars)

**Problem**: GAP_CLUSTER=50 is hardcoded. It may not be optimal for:
- Short dense isnads (might oversplit)
- Long complex isnads (might merge separate ones)

**Current Behavior**:
- 17,938 boundary tokens → 16,253 unique positions
- 16,253 positions → 520 clusters (avg gap: 30.8 chars)

**Optimization**:
```python
# Propose dynamic GAP_CLUSTER based on statistics
char_starts = sorted(boundary_by_char.keys())
gaps_between_tokens = [char_starts[i+1] - char_starts[i] for i in range(len(char_starts)-1)]

gap_p25 = statistics.quantiles(gaps_between_tokens, n=4)[0]  # 25th percentile
gap_p75 = statistics.quantiles(gaps_between_tokens, n=4)[2]  # 75th percentile
gap_iqr = gap_p75 - gap_p25

# Adaptive GAP_CLUSTER = p75 + 0.5*IQR
GAP_CLUSTER = int(gap_p75 + 0.5 * gap_iqr)
```

**Why**: This makes the algorithm data-driven instead of hardcoded.

---

### Issue 2: Deduplication Keeps Only Highest Prob

**Problem**: When same char_start appears multiple times, we keep only the highest-probability token. But what if:
- Token A (prob=0.98) is " أخبرنا" (isnad marker)
- Token B (prob=0.99) is "ا" (diacritic)

Token B has higher prob but may be less meaningful.

**Current Behavior**:
```
char_start=956:
  Token 1: prob=0.6361  → STORED
  Token 2: prob=0.9993  → UPDATED (kept)
```

**Optimization**: Add semantic weighting
```python
# Weight by token meaningfulness
ISNAD_VERBS = {"أخبرنا", "حدثنا", "قال", "ذكر", ...}
TOKEN_WEIGHTS = {
    "noun": 2.0,
    "verb": 2.5,
    "isnad_verb": 5.0,
    "punctuation": 0.1,
    "[PAD]": 0.0
}

score = probability * TOKEN_WEIGHTS.get(token_type, 1.0)
```

---

### Issue 3: No Merging of Very Close Clusters

**Problem**: Two isnad clusters might be separated by just whitespace (< 5 chars). Should they be merged?

**Current Behavior**:
```
Cluster 0: char_start=723  char_end=987
Cluster 1: char_start=1197 char_end=1724
Gap: 1197 - 987 = 210 chars → kept separate
```

**Optimization**: Merge clusters that are too close
```python
MIN_GAP_BETWEEN_CLUSTERS = 10  # Adjacent isnads < 10 chars apart might be same unit

i = 0
while i < len(clusters) - 1:
    gap = clusters[i+1]['char_start'] - clusters[i]['char_end']
    if gap < MIN_GAP_BETWEEN_CLUSTERS and not is_paragraph_break(gap):
        # Merge clusters
        clusters[i]['char_end'] = clusters[i+1]['char_end']
        clusters[i]['n_tokens'] += clusters[i+1]['n_tokens']
        clusters.pop(i+1)
    else:
        i += 1
```

---

### Issue 4: No Filtering of Low-Confidence Boundaries

**Problem**: Some boundary tokens have very low probability (e.g., prob < 0.55). Should they be kept?

**Current Behavior**:
```
All boundaries kept regardless of confidence
17,938 boundary tokens, ranging from prob=0.0001 to prob=0.9999
```

**Optimization**: Threshold low-confidence predictions
```python
CONFIDENCE_THRESHOLD = 0.70

boundary_by_char_filtered = {
    cs: (tok, off, prob) 
    for cs, (tok, off, prob) in boundary_by_char.items() 
    if prob >= CONFIDENCE_THRESHOLD
}
```

**Expected Effect**:
- Reduces false positives (34 FP with current F1=0.8579)
- May reduce recall slightly but improve precision
- Trade-off: Higher precision vs. lower recall

---

### Issue 5: No Validation Against Corpus Length

**Problem**: Offsets could theoretically exceed corpus length due to bugs.

**Current Behavior**:
```
No validation - assumes all offsets are valid
```

**Optimization**: Add validation
```python
corpus_len = len(corpus_text)
invalid_clusters = []

for cluster in clusters:
    if cluster['char_end'] > corpus_len:
        invalid_clusters.append(cluster)
        cluster['char_end'] = corpus_len
```

---

## 5. PROPOSED OPTIMIZED PIPELINE

```python
def convert_raw_inference_optimized(
    raw_inference_path: str,
    corpus_path: str,
    gold_boundaries_path: str = None,
    
    # Parameters
    confidence_threshold: float = 0.70,
    gap_cluster: str = "adaptive",  # "adaptive" or int
    merge_close_clusters: bool = True,
    min_gap_merge: int = 10,
    verbose: bool = True
) -> dict:
    """
    Optimized conversion with configurable parameters.
    
    Improvements:
    1. Confidence thresholding
    2. Adaptive/configurable gap clustering
    3. Merge nearby clusters
    4. Boundary validation
    5. Per-parameter evaluation metrics
    """
    
    # Load data
    raw = load_json(raw_inference_path)
    corpus = load_text(corpus_path)
    gold = load_gold(gold_boundaries_path) if gold_boundaries_path else None
    
    # Step 1: Extract boundary tokens
    boundary_indices = [i for i, p in enumerate(raw['predictions']) if p == 1]
    
    # Step 2: Deduplicate with confidence threshold
    boundary_by_char = {}
    for i in boundary_indices:
        prob = raw['probabilities'][i]
        if prob < confidence_threshold:
            continue  # Skip low-confidence
        
        char_start = raw['offsets'][i][0]
        if char_start not in boundary_by_char or prob > boundary_by_char[char_start][1]:
            boundary_by_char[char_start] = (raw['tokens'][i], raw['offsets'][i], prob)
    
    # Step 3: Determine optimal GAP_CLUSTER
    if gap_cluster == "adaptive":
        char_starts = sorted(boundary_by_char.keys())
        gaps = [char_starts[i+1] - char_starts[i] for i in range(len(char_starts)-1)]
        gap_p75 = quantiles(gaps, n=4)[2]
        gap_iqr = quantiles(gaps, n=4)[2] - quantiles(gaps, n=4)[0]
        GAP_CLUSTER = int(gap_p75 + 0.5 * gap_iqr)
    else:
        GAP_CLUSTER = gap_cluster
    
    # Step 4: Cluster
    clusters = cluster_tokens(boundary_by_char, GAP_CLUSTER)
    
    # Step 5: Merge close clusters (optional)
    if merge_close_clusters:
        clusters = merge_nearby_clusters(clusters, min_gap_merge)
    
    # Step 6: Extract boundaries
    boundaries = [c['char_start'] for c in clusters]
    
    # Step 7: Evaluate (if gold standard available)
    if gold:
        metrics = evaluate_boundaries(boundaries, gold, tolerance=80)
    else:
        metrics = None
    
    return {
        'metadata': {
            'confidence_threshold': confidence_threshold,
            'gap_cluster': GAP_CLUSTER if isinstance(gap_cluster, int) else f"adaptive({GAP_CLUSTER})",
            'merge_close_clusters': merge_close_clusters,
            'unique_positions': len(boundary_by_char),
            'clusters': len(clusters),
            'boundaries': len(boundaries),
            'metrics': metrics
        },
        'boundaries': boundaries,
        'clusters': clusters
    }
```

---

## 6. TESTING & VALIDATION STRATEGY

### Test 1: Parameter Sensitivity
```bash
# Test different confidence thresholds
for THRESHOLD in 0.50 0.60 0.70 0.80 0.90; do
  python optimized_convert.py --confidence-threshold $THRESHOLD
  # Compare F1 scores
done
```

### Test 2: Gap Cluster Impact
```bash
# Test different gap values
for GAP in 25 50 75 100 200; do
  python optimized_convert.py --gap-cluster $GAP
  # Compare F1 scores
done
```

### Test 3: Merge Impact
```bash
# Test with/without merging
python optimized_convert.py --merge-close-clusters true
python optimized_convert.py --merge-close-clusters false
# Compare F1, cluster count
```

---

## 7. SUMMARY

| Component | Current | Issue | Optimization |
|-----------|---------|-------|--------------|
| Confidence | All kept | False positives | Threshold (0.70) |
| Gap Cluster | Fixed (50) | Hardcoded | Adaptive (data-driven) |
| Merging | None | Fragmented isnads | Merge < 10 chars |
| Validation | None | Invalid offsets | Bounds check |
| Weighting | Prob only | Low semantic value | Semantic weighting |

**Expected Improvement**: F1 from 0.8579 → ~0.88-0.90 with optimizations.

