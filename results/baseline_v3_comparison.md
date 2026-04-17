# Baseline V2 vs V3 Comparison

## Detection Results

| Metric | V2 | V3 | Change |
|--------|----|----|--------|
| Detected | 396/613 | 452/613 | +56 |
| Detection Rate | 64.6% | 73.7% | +14.1% |
| Coverage | 90.6% | 65.7% | -24.9% |

## Boundary Accuracy

| Tier | V2 | V3 | Change |
|------|----|----|--------|
| Perfect (0-5) | 0.8% | 0.7% | -0.2% |
| Excellent (5-20) | 5.5% | 5.4% | -0.2% |
| Good (20-50) | 7.8% | 10.8% | +2.9% |
| Acceptable (50-100) | 15.2% | 17.1% | +2.0% |
| Poor (100-200) | 24.1% | 25.9% | +1.8% |
| Very Poor (200-500) | 29.2% | 27.1% | -2.1% |
| Failed (500+) | 17.3% | 13.1% | -4.2% |

## Key Improvements

1. **Detection**: +56 akhbars (+14.1%)
2. **Boundary Accuracy**: 14.2% → 16.8% with < 50 chars error
3. **Coverage**: 90.6% → 65.7%
