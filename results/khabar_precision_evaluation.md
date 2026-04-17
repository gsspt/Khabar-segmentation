# Khabar Boundary Precision Evaluation

## Summary

- Reference boundaries: 613
- Detected boundaries: 502
- Matched boundaries: 613
- Detection rate: 81.9%

## Boundary Position Accuracy

### Start Boundary

| Metric | Value |
|--------|-------|
| Mean deviation | 279 chars |
| Median deviation | 136 chars |
| 95th percentile | 711 chars |
| Perfect (0-5) | 4 (0.7%) |
| Excellent (5-20) | 39 (6.4%) |
| Good (20-50) | 69 (11.3%) |
| Acceptable (50-100) | 110 (17.9%) |

### End Boundary

| Metric | Value |
|--------|-------|
| Mean deviation | 411 chars |
| Median deviation | 241 chars |
| 95th percentile | 1260 chars |
| Perfect (0-5) | 9 (1.5%) |
| Excellent (5-20) | 29 (4.7%) |
| Good (20-50) | 37 (6.0%) |
| Acceptable (50-100) | 66 (10.8%) |

## Segment Overlap (Jaccard / IoU)

| Category | Count | % |
|----------|-------|---|
| Perfect (95-100%) | 0 | 0.0% |
| Excellent (90-95%) | 3 | 0.5% |
| Good (80-90%) | 16 | 2.6% |
| Acceptable (70-80%) | 15 | 2.4% |
| Poor (<70%) | 579 | 94.5% |
| **Usable (80%+)** | **19** | **3.1%** |

Mean IoU: 0.300  
Median IoU: 0.289

## Exact Match Analysis

- Exact match (0 char deviation): 0 (0.0%)
- Near exact (<=10 chars both): 0 (0.0%)
- Combined: 0 (0.0%)

## Khabar Length Analysis

### Reference Khabars
- Mean: 436 chars
- Median: 330 chars
- Range: 31-4282 chars

### Detected Khabars
- Mean: 570 chars
- Median: 334 chars
- Range: 33-5172 chars

### Differences
- Mean absolute difference: 478 chars
- Median difference: 217 chars

## Error Categorization

- Over-segmented (<80% of reference): 256 (41.8%)
- Correctly sized (80-120%): 94 (15.3%)
- Under-segmented (>120% of reference): 263 (42.9%)

## Key Findings

1. **Detection**: 502/613 (81.9%)
2. **Boundary Accuracy**: 112/613 (18.3%) start boundaries within 50 chars
3. **Segment Overlap**: 19/613 (3.1%) with 80%+ IoU overlap
4. **Exact Match**: 0/613 (0.0%) perfect match
5. **Length Error**: Mean 478 chars difference from reference

## Assessment

### Good Performance
- Detection rate of 81.9% (502/613)
- 18.3% of boundaries accurate within 50 chars
- 3.1% of segments have 80%+ overlap with reference

### Areas for Improvement
- 94.5% of segments have <70% overlap
- 41.8% over-segmented, 42.9% under-segmented
- Mean boundary deviation: 345 chars

