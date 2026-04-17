# CAMeL Tools Analysis in WSL — Setup & Usage Guide

## Overview

This guide explains how to use CAMeL Tools to lemmatize and POS tag isnads and khabars separately, to identify linguistic patterns for baseline refinement.

**Key goals:**
- Extract lemmas and POS tags for isnads vs khabars
- Find distinctive linguistic patterns (verb roots, POS sequences, etc.)
- Identify refinement opportunities for the baseline segmentation

---

## Prerequisites

### 1. WSL Installation (Windows)

If you don't have WSL installed:

```bash
# Open PowerShell as Administrator and run:
wsl --install

# This installs WSL2 with Ubuntu by default
# Reboot after installation

# Verify installation:
wsl --list --verbose
```

### 2. Python in WSL

```bash
# In WSL terminal:
wsl

# Install Python 3.10+
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

# Verify
python3 --version
```

---

## Setup

### From Windows (using `wsl` prefix)

```bash
# Navigate to project in WSL
cd /mnt/c/Users/augus/Desktop/Khabar-segmentation

# Make script executable
chmod +x scripts/run_cameltools_wsl.sh

# Run it
bash scripts/run_cameltools_wsl.sh
```

### From WSL Terminal Directly

```bash
# Start WSL
wsl

# Navigate to project
cd /mnt/c/Users/augus/Desktop/Khabar-segmentation

# Make script executable
chmod +x scripts/run_cameltools_wsl.sh

# Run it
bash scripts/run_cameltools_wsl.sh
```

---

## Manual Setup (if auto setup doesn't work)

```bash
wsl

cd /mnt/c/Users/augus/Desktop/Khabar-segmentation

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install camel-tools
pip install camel-tools

# Run analysis
python3 scripts/analyze_with_cameltools.py
```

---

## What the Script Does

### 1. **Extract isnads and khabars** from baseline results
   - Parses `results/segmentation_baseline.txt`
   - Separates ISNAD and KHABAR sections
   - Cleans artifacts (page markers, strikethrough)

### 2. **Tokenize and analyze** with CAMeL Tools
   - Normalizes Unicode
   - Tokenizes text into words
   - Extracts lemma (stem/root)
   - Guesses POS tag (V, N, PREP, PRON, PART)

### 3. **Analyze linguistic patterns**
   - Most common lemmas in isnads vs khabars
   - POS distributions
   - Opening words (first lemmas)
   - Pattern differences

### 4. **Generate report** with findings
   - Text report: `results/cameltools_pattern_comparison.txt`
   - JSON details: `results/cameltools_isnad_analysis.json`
   - JSON details: `results/cameltools_khabar_analysis.json`

---

## Output Files

After running, you'll get:

### `results/cameltools_pattern_comparison.txt`
High-level comparison of linguistic patterns:
```
[ISNADS]
Total tokens: 6,234
Most common lemmas:
  حدثنا: 847 (13.6%)
  أخبرنا: 623 (10.0%)
  ...
POS distribution:
  V: 2,140 (34.3%)
  N: 3,200 (51.3%)
  ...

[KHABARS]
...

[PATTERN DIFFERENCES]
Isnad POS profile: {'V', 'N', 'PREP'}
Khabar POS profile: {'N', 'PREP', 'PART'}
Distinctive to Isnad: {'V'}
Distinctive to Khabar: {'PART'}
```

### `results/cameltools_isnad_analysis.json`
Detailed token analysis for isnads (first 10 as sample):
```json
{
  "total": 711,
  "tokens_analyzed": 6234,
  "sample_items": [
    {
      "id": 0,
      "tokens": [
        {"word": "حدثنا", "lemma": "حدث", "pos": "V"},
        {"word": "ابو", "lemma": "ابو", "pos": "N"},
        ...
      ]
    }
  ]
}
```

### `results/cameltools_khabar_analysis.json`
Similar for khabars.

---

## Analyzing the Results

### Key Insights to Look For

1. **POS Differences**
   - Isnads likely have high verb frequency (transmission verbs)
   - Khabars likely have more diverse POS (narrative, description)

2. **First Lemma Patterns**
   - Isnads typically start with: حدثنا، أخبرنا، روى، سمعت
   - Khabars typically start with: قال، كان، فعل، رأى

3. **Lemma Frequency**
   - Isnads: repetitive (same transmission verbs)
   - Khabars: diverse vocabulary (narrative content)

### Using These Patterns to Refine

Once you have the patterns, you can:

1. **Improve verb detection** — use discovered isnad verbs
2. **Refine boundaries** — use POS sequences to detect transitions
3. **Filter false positives** — exclude short isnads (< 5 tokens)
4. **Add confidence scoring** — weight by observed frequency

Example refinement:
```python
# From analysis, discover these are strong isnad markers:
ISNAD_SPECIFIC_VERBS = {'حدثنا', 'أخبرنا', 'روى', 'سمعت'}

# From analysis, discover these POS sequences mark isnad start:
ISNAD_START_PATTERNS = [
    'V N PREP N',  # حدثنا ابو من ...
    'V PREP N N',  # أخبرنا عن ابو ...
]
```

---

## Troubleshooting

### Error: "CAMeL Tools not found"
```bash
# Make sure it's installed in the virtual environment
source .venv/bin/activate
pip install camel-tools

# If still failing, try:
pip install --upgrade camel-tools
```

### Error: "File not found: segmentation_baseline.txt"
- Make sure you've run the baseline segmentation first:
  ```bash
  python3 scripts/baseline_isnad_segmentation.py --input data/raw/[filename]
  ```

### WSL Issues

**If running from Windows CMD/PowerShell:**
```bash
wsl bash scripts/run_cameltools_wsl.sh
```

**If inside WSL:**
```bash
bash scripts/run_cameltools_wsl.sh
```

**If permissions denied:**
```bash
wsl
chmod +x scripts/run_cameltools_wsl.sh
bash scripts/run_cameltools_wsl.sh
```

### Path Issues

WSL paths are:
- Windows: `C:\Users\augus\Desktop\Khabar-segmentation`
- WSL: `/mnt/c/Users/augus/Desktop/Khabar-segmentation`

The scripts handle this automatically.

---

## Next Steps

After analysis:

1. **Review the pattern comparison report** — identify distinctive features
2. **Update baseline rules** — incorporate discovered patterns
3. **Test refined baseline** — run `baseline_isnad_segmentation.py` with new rules
4. **Compare metrics** — check if gap reduced from 15.5%

Example workflow:
```bash
# 1. Run analysis
bash scripts/run_cameltools_wsl.sh

# 2. Review results
cat results/cameltools_pattern_comparison.txt

# 3. Update scripts/baseline_isnad_segmentation.py with discoveries

# 4. Test refined version
python3 scripts/baseline_isnad_segmentation.py --input data/raw/[filename]

# 5. Compare outputs
diff results/segmentation_baseline.txt results/segmentation_baseline_v2.txt
```

---

## Advanced: Enabling Full CAMeL Disambiguation

If you want full morphological analysis (lemmatization) instead of simplified extraction:

```bash
# In WSL, after installing camel-tools:
python3 -c "from camel_tools.disambiguators.mle import MLEDisambiguator; print('MLE available')"

# If available, edit analyze_with_cameltools.py to use:
# disambiguator = MLEDisambiguator.pretrained()
# morphs = disambiguator.disambiguate(tokens)
```

This requires downloaded models (automatic via pip).

---

## References

- **CAMeL Tools docs**: https://camel-tools.readthedocs.io/
- **Baseline script**: `scripts/baseline_isnad_segmentation.py`
- **Baseline results**: `results/segmentation_baseline.txt`
- **CLAUDE.md**: Project setup and conventions

---

## Questions?

- Check `CLAUDE.md` for project context
- Review `results/cameltools_pattern_comparison.txt` for detailed analysis
- Run with `--help` for script options
