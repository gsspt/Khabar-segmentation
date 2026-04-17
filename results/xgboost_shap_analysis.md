# SHAP Feature Importance Analysis

## Global Feature Importance (Mean |SHAP|)

Feature importance shows how much each feature contributes to model predictions.
Higher values = more important for determining isnad boundaries.

| Rank | Feature | Importance |
|------|---------|-------------|
| 1 | qal_distance | 60.2561 |
| 2 | reference_in_early_window | 13.9177 |
| 3 | distance_to_next_isnad | 7.9449 |
| 4 | narrative_verb_count | 2.8631 |
| 5 | window_char_count | 2.2404 |
| 6 | has_qal | 1.5793 |
| 7 | pronoun_count | 1.3102 |
| 8 | avg_word_length | 1.1056 |
| 9 | window_word_count | 1.0952 |
| 10 | qal_in_valid_range | 0.9954 |
| 11 | punctuation_count | 0.8778 |
| 12 | has_faqal | 0.6669 |
| 13 | has_end_suffix | 0.1368 |
| 14 | has_waqal | 0.1004 |
| 15 | has_halath | 0.0000 |
| 16 | qal_to_reference_diff | 0.0000 |
| 17 | reference_in_late_window | 0.0000 |

## Interpretation

### Top Features Explained

**1. qal_distance**: Distance to 'قال' from isnad start (direct position hint)

**2. reference_in_early_window**: Feature indicates boundary characteristics

**3. distance_to_next_isnad**: Distance to next isnads (bounds the boundary)

**4. narrative_verb_count**: Count of narrative verbs (marks khabar beginning)

**5. window_char_count**: Total characters in search window (size indicator)

