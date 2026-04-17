# TASK 1.4: Boundary Accuracy & Position Error Analysis

## Executive Summary

- Reference boundaries: 613
- Detected boundaries: 396
- Matched boundaries: 613
- Unmatched reference: 0

## Deviation Statistics

- **Average deviation**: 500 chars
- **Median deviation**: 180 chars
- **Max deviation**: 8864 chars
- **Std Dev**: 1252 chars

## Detection Direction

- **Exact matches** (0 offset): 1
- **Over-detected** (too early): 274 (44.7%)
- **Under-detected** (too late): 338 (55.1%)

## Accuracy Tiers

| Tier | Range | Count | % |
|------|-------|-------|---|
| Perfect | 0-5 | 5 | 0.8% |
| Excellent | 5-20 | 34 | 5.5% |
| Good | 20-50 | 48 | 7.8% |
| Acceptable | 50-100 | 93 | 15.2% |
| Poor | 100-200 | 148 | 24.1% |
| Very Poor | 200-500 | 179 | 29.2% |
| Failed | 500-inf | 106 | 17.3% |

## Key Findings

### 1. Boundary Position Accuracy
- **High accuracy** (< 50 chars): 87 (14.2%)
- **Poor accuracy** (> 200 chars): 285 (46.5%)

### 2. Systematic Errors
- **Bias toward over-detection**: False
- **Average signed deviation**: -257 chars
  (Negative = detected too early, Positive = detected too late)

### 3. Segment Length Distributions
- **Isnad lengths**: 12-405 chars (avg 99)
- **Khabar lengths**: 21-9171 chars (avg 516)

## Recommendations

1. **Fix boundary-finding algorithm**
   - Large median deviation (500 chars) suggests algorithm issues
   - Consider: missing 'qal' marker? overlapping isnads?

2. **Focus on high-frequency error patterns**
   - 29.2% have 200-500 char errors
   - These likely represent systematic detection failures

3. **Improve isnad boundary detection**
   - Add missing prefixed verbs
   - Handle isnads without 'qal' marker

