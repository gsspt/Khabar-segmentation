# Phase 2.5: Remaining Undetected Akhbars Analysis

## Summary

- Reference boundaries: 613
- Detected by v3: 452 (73.7%)
- Undetected (100 char tolerance): 402 (65.6%)

## Categorization of Undetected

| Category | Count | % | Interpretation |
|----------|-------|---|----------------|
| Verb in v3 list | 229 | 57.0% | Boundary detection failure |
| Verb NOT in list | 173 | 43.0% | Missing verbs to add |
| No clear verb | 0 | 0.0% | Edge cases |

## Top 20 First Words in Undetected

| Rank | Count | In V3? | Notes |
|------|-------|--------|-------|
| 1 | 144 | ✓ | Length: 6 |
| 2 | 19 | ✓ | Length: 4 |
| 3 | 16 | ✓ | Length: 4 |
| 4 | 15 | ✗ | Length: 3 |
| 5 | 10 | ✓ | Length: 5 |
| 6 | 9 | ✓ | Length: 6 |
| 7 | 9 | ✓ | Length: 5 |
| 8 | 5 | ✓ | Length: 6 |
| 9 | 4 | ✓ | Length: 5 |
| 10 | 3 | ✓ | Length: 5 |
| 11 | 3 | ✗ | Length: 3 |
| 12 | 3 | ✗ | Length: 5 |
| 13 | 2 | ✓ | Length: 7 |
| 14 | 2 | ✓ | Length: 3 |
| 15 | 2 | ✗ | Length: 5 |
| 16 | 2 | ✓ | Length: 5 |
| 17 | 2 | ✗ | Length: 4 |
| 18 | 1 | ✗ | Length: 3 |
| 19 | 1 | ✗ | Length: 4 |
| 20 | 1 | ✗ | Length: 8 |

## High-Priority Verbs to Add

Top candidates with count >= 3 and not in v3:

- Word (count=15): 15 undetected akhbars
- Word (count=3): 3 undetected akhbars
- Word (count=3): 3 undetected akhbars

## Recommendations

1. **Recover boundary detection failures** (229 akhbars)
   - These have verbs in v3 list but aren't detected
   - Likely issues: missing قال marker, length validation, clustering
   - Action: Debug a sample to understand failure mode

2. **Add missing verbs** (173 akhbars)
   - High-priority: verbs with count >= 3
   - Estimated recovery: 21 akhbars

3. **Handle edge cases** (0 akhbars)
   - Non-verb openers (proper names, titles)
   - Contextual patterns
