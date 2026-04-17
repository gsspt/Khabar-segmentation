# TASK 1.3: Pattern Discovery & Edge Case Analysis

## Isnad Clustering Patterns

- Total isnad starts found in corpus: 1121
- Clusters with 2+ isnads within 200 chars: 295
- Maximum cluster size: 28 isnads
- Analysis: Dense clustering suggests overlapping/adjacent isnads may cause detection failures

## Verb-Specific Detection Rates

| Verb | Total | Detected | Undetected | Rate |
|------|-------|----------|------------|------|
| حدثنا | 648 | 203 | 445 | 31.3% |
| سمعت | 416 | 162 | 254 | 38.9% |
| حدثني | 34 | 19 | 15 | 55.9% |
| حدثهم | 4 | 0 | 4 | 0.0% |
| وسمعت | 10 | 7 | 3 | 70.0% |
| وحدثنا | 3 | 1 | 2 | 33.3% |
| حدثه | 1 | 0 | 1 | 0.0% |
| روى | 1 | 0 | 1 | 0.0% |
| حدثت | 1 | 1 | 0 | 100.0% |
| سمعنا | 2 | 2 | 0 | 100.0% |
| رويت | 1 | 1 | 0 | 100.0% |

## Critical Finding: Most Frequent Verb Analysis

- Verb (represented as index in list): 5 chars
- Total occurrences: 648
- Detected: 203 (31.3%)
- Undetected: 445 (68.7%)

**Key Insight**: Most frequent verb may have lower detection rate due to:
- Boundary detection failure (no 'قال' found?)
- Length validation issues
- Overlapping detections

## Prefixed Verbs (و + verb)

- Unique prefixed verbs found: 3
- Top 10 prefixed verbs (by occurrence count):

  1. Length 5: 11 occurrences
  2. Length 6: 3 occurrences
  3. Length 6: 1 occurrences

**Key Insight**: Prefixed verbs (wa+verb) are NOT in ISNAD_START_VERBS list!
Adding prefixed variants could recover significant undetected isnads.

## Boundary Position Deviation Analysis

- Average absolute deviation: 500 chars
- Median deviation: 180 chars
- Max deviation: 8864 chars

Deviations by Range:

-   0- 10 chars:  19 (  3.1%)
-  10- 50 chars:  68 ( 11.1%)
-  50-100 chars:  93 ( 15.2%)
- 100-200 chars: 148 ( 24.1%)
- 200-500 chars: 179 ( 29.2%)

## Non-Verb Openers (Edge Cases)

Non-verb words that appear at isnad start:

- :: 13 times
- ل: 12 times
- د: 9 times
- ا: 8 times
- قال: 7 times
- ن: 6 times
- ه: 6 times
- الله: 4 times
- بن: 4 times
- نا: 4 times

## Recommendations

1. **Add prefixed verb variants** (و + ISNAD_START_VERBS)
   - Impact: Could recover 200+ undetected isnads

2. **Debug أخبرنا detection failure**
   - Why is most-frequent verb most-undetected?
   - Check: missing 'قال' marker? Length validation? Overlaps?

3. **Analyze clustering effects**
   - Dense clusters may cause overlap issues
   - Consider interval-based filtering

4. **Handle non-verb openers**
   - Proper names (الحسن, علي) sometimes start isnads
   - Need context-aware fallback
