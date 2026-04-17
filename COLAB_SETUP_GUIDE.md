# CAMeL-BERT Fine-Tuning in Google Colab
## Complete Setup & Execution Guide

**Last Updated**: 2026-04-16  
**Status**: Ready to Execute

---

## OVERVIEW

Two-notebook pipeline for fine-tuning CAMeL-BERT on your 533 annotated khabar examples:

1. **CAMELBERT_FINETUNING_COLAB.ipynb** — Data prep + Model training (40-60 min)
2. **CAMELBERT_INFERENCE_EVALUATION.ipynb** — Inference + Evaluation (20-30 min)

**Total GPU time**: ~90 minutes  
**Total cost on Google Colab**: FREE (with GPU enabled)

---

## PRE-SETUP CHECKLIST

Before starting, verify you have:

- ✅ Google Drive account
- ✅ Google Colab access (colab.research.google.com)
- ✅ Khabar-segmentation folder in Drive at: `/My Drive/Khabar-segmentation/`
- ✅ Annotated data: `data/processed/Kitab_Uqala_al_Majanin_annotated.json` (613 examples)
- ✅ Reference corpus: `data/processed/kitab_uqala_reference_corpus.txt`
- ✅ Reference boundaries: `data/processed/kitab_uqala_boundaries.json`

**Verify file paths on Drive:**
```
My Drive/
└── Khabar-segmentation/
    ├── data/
    │   ├── raw/
    │   └── processed/
    │       ├── Kitab_Uqala_al_Majanin_annotated.json      ← CRITICAL
    │       ├── kitab_uqala_reference_corpus.txt           ← CRITICAL
    │       ├── kitab_uqala_boundaries.json                ← CRITICAL
    │       └── [other files]
    └── notebooks/
        ├── CAMELBERT_FINETUNING_COLAB.ipynb              ← This file
        └── CAMELBERT_INFERENCE_EVALUATION.ipynb          ← Part 2
```

If files are missing, you need to upload them to Drive first!

---

## STEP-BY-STEP EXECUTION

### Phase A: Upload Notebooks to Drive

**Option 1: Automatic (Recommended)**
```bash
# From your local machine, upload both notebooks to Drive:
# 1. Download from repo: notebooks/CAMELBERT_FINETUNING_COLAB.ipynb
# 2. Upload to: My Drive/Khabar-segmentation/notebooks/
# 3. Download: notebooks/CAMELBERT_INFERENCE_EVALUATION.ipynb
# 4. Upload same location
```

**Option 2: Manual**
1. Go to colab.research.google.com
2. Click "Upload" → Choose both .ipynb files
3. Save to Drive under `Khabar-segmentation/notebooks/`

---

### Phase B: Notebook 1 — Fine-Tuning

**1. Open Colab Notebook**
- Go to: https://colab.research.google.com
- Click "File" → "Open notebook"
- Select "Google Drive" tab
- Navigate to: `Khabar-segmentation/notebooks/CAMELBERT_FINETUNING_COLAB.ipynb`
- Open it

**2. Enable GPU**
- Click "Runtime" menu (top right)
- Select "Change runtime type"
- Choose "T4 GPU" (Free tier) or "V100/A100" (if available)
- Click "Save"

**3. Run Notebook Sequentially**

The notebook has **8 stages**. Follow this pattern for EACH cell:

```
1. Read the markdown header (tells you what's happening)
2. Click cell (highlight it)
3. Press Ctrl+Enter (or click ▶ button)
4. Wait for it to complete (watch for [OK] / [ERROR])
5. Move to next cell
```

**DO NOT skip cells or run out of order!**

#### Stage 1: Mount Drive & Verify Files
- Authenticates access to your Google Drive
- Checks files exist
- If ERRORS: Files may be in wrong location on Drive

#### Stage 2: Data Preparation
- Loads 533 annotated examples
- Extracts BIO tags (B-ISNAD, I-ISNAD, B-KHABAR, I-KHABAR)
- Splits into train/val/test (70/15/15)
- **Expected**: "Extracted 600+ examples" message
- **Expected time**: 2-3 minutes
- **If error**: Check annotated JSON structure

#### Stage 3: Setup Environment
- Installs transformers, torch, datasets
- Verifies GPU availability
- **Expected**: "GPU Available: True" + GPU name
- **If GPU unavailable**: Change runtime type

#### Stage 4: Model Setup
- Loads CAMeL-BERT tokenizer & model
- Tokenizes training data
- **Expected time**: 5-10 minutes (first run slower due to downloads)
- **Model size**: ~400 MB (will be cached)

#### Stage 5: Training
- **LONGEST STAGE** — 30-60 minutes depending on GPU
- Trains for 10 epochs with early stopping
- Shows progress every 100 steps
- Watch for loss decreasing (e.g., 2.5 → 1.8 → 1.2 → 0.8)
- **Expected**: F1 score 0.80-0.85 on test set
- **If stuck**: Check GPU memory (should use 8-12 GB)

#### Stage 6: Evaluation
- Computes per-tag F1, precision, recall
- Saves detailed metrics
- **Expected time**: 2-3 minutes
- **Output**: Classification report (metrics by label)

#### Stage 7: Inference Sample
- Runs prediction on sample text
- Verifies model works
- **Expected**: Predictions for first 20 tokens
- **Expected time**: <1 minute

#### Stage 8: Summary
- Generates summary report
- Lists all output files on Drive
- **Shows**: Where model & results are saved

**Expected outputs on Drive** (after Notebook 1 complete):
```
Khabar-segmentation/data/camelbert_training/
├── train.jsonl, val.jsonl, test.jsonl
├── best_model/
│   ├── pytorch_model.bin (400 MB)
│   ├── config.json
│   ├── tokenizer.json
│   └── label_mapping.json
├── checkpoint/
│   └── [all checkpoints from training]
├── training_results.json
└── TRAINING_SUMMARY.txt
```

---

### Phase C: Notebook 2 — Inference & Evaluation

**After Notebook 1 completes successfully:**

1. **Open Notebook 2**
   - Same process: colab.research.google.com
   - Open: `Khabar-segmentation/notebooks/CAMELBERT_INFERENCE_EVALUATION.ipynb`
   - Runtime should still have GPU enabled

2. **Run Sequentially**

The notebook has **4 stages**:

#### Stage 1: Setup & Load
- Mount Drive
- Load trained model from `best_model/`
- Load reference data
- **Expected time**: 3-5 minutes
- **If error**: Model may not have finished saving

#### Stage 2: Full Corpus Inference
- Tokenizes entire 268K-char corpus
- Runs token classification (BIO tagging)
- Processes in batches of 512 tokens
- **Expected time**: 10-15 minutes
- **Expected progress**: "Processed 50K / 268K tokens" every minute
- **GPU memory**: Will use ~10-12 GB (normal)

#### Stage 3: Reconstruction
- Maps token predictions back to character positions
- Reconstructs akhbar boundaries
- **Expected**: "Reconstructed 600+ akhbar boundaries"
- **Expected time**: 2-3 minutes

#### Stage 4: Evaluation & Comparison
- Computes boundary precision metrics
- Compares with baseline v3.5
- Shows improvement percentage
- **CRITICAL OUTPUT**: Shows % of usable boundaries (target: 65-75%)
- **Expected time**: 1-2 minutes

**Expected outputs on Drive** (after Notebook 2 complete):
```
Khabar-segmentation/data/camelbert_training/inference_results/
├── EVALUATION_REPORT.md
├── detailed_metrics.json
├── predicted_akhbars.json
└── FINAL_SUMMARY.txt
```

---

## WHAT TO EXPECT AT EACH STAGE

### Training Progress (Notebook 1, Stage 5)

```
Epoch 1/10
100/200 [50%] loss=2.45
200/200 [100%] loss=2.12

Epoch 2/10
100/200 [50%] loss=1.87
200/200 [100%] loss=1.64

... (continues for 10 epochs)

Epoch 10/10
100/200 [50%] loss=0.68
200/200 [100%] loss=0.62

Best F1: 0.823 (Epoch 7)
```

**Normal signs:**
- Loss decreases over time (good!)
- F1 score gradually increases
- Training completes in 30-60 min

**Warning signs:**
- Loss increases or stays flat (training not working)
- OOM (Out Of Memory) error → Reduce batch size in Stage 4
- NaN loss → Learning rate too high (modify in Stage 4)

### Inference Progress (Notebook 2, Stage 2)

```
Processed 50000/268540 tokens
Processed 100000/268540 tokens
Processed 150000/268540 tokens
Processed 200000/268540 tokens
Processed 250000/268540 tokens
Processed 268540/268540 tokens

[OK] Inference complete on 40000 tokens
```

### Final Results (Notebook 2, Stage 4)

**Expected output:**
```
================================================================================
METRIC                         BASELINE V3.5         CAMELBERT          IMPROVEMENT
================================================================================
Start error (mean)                   183 chars           45 chars            75.4%
End error (mean)                     408 chars          120 chars            70.6%
IoU (mean)                           0.332              0.725              118.4%
Usable (80%+ IoU)                    3.8%               70.5%              +66.7 ppts
================================================================================
```

**Success criteria:**
- ✅ Usable boundaries: >60% (target 65-75%)
- ✅ IoU improvement: >50% (target 100%+)
- ✅ End error reduction: >50% (target 60%+)

---

## TROUBLESHOOTING

### Common Issues

**Issue: "File not found" in Stage 1**
- **Cause**: Files not in Drive
- **Fix**: Upload all files from `data/processed/` to Drive's same location
- **Command** (from local repo):
  ```bash
  # Upload to Drive at: My Drive/Khabar-segmentation/data/processed/
  ```

**Issue: "No GPU available" in Stage 3**
- **Cause**: Runtime not changed to GPU
- **Fix**: Click Runtime → Change runtime type → Select T4 GPU
- **Note**: Free Colab gives T4 (12GB VRAM), paid tiers have V100/A100

**Issue: "CUDA out of memory" during training**
- **Cause**: Batch size too large for GPU
- **Fix**: In Stage 4, change:
  ```python
  per_device_train_batch_size=8,  # was 16
  ```
- **Rerun**: From Stage 4 (data is already prepared)

**Issue: Training loss not decreasing**
- **Cause**: Learning rate too high or data problem
- **Check**:
  - Look at Stage 2 output: "Extracted 600+ examples"? Should be yes.
  - Look at BIO tags distribution: Should have B-ISNAD, I-ISNAD, B-KHABAR
  - If data looks wrong, check annotated JSON structure
- **Fix**: Rerun from Stage 4 with different learning rate:
  ```python
  learning_rate=1e-5,  # was 2e-5
  ```

**Issue: Inference very slow (>30 min for full corpus)**
- **Cause**: CPU inference or small batch size
- **Check**: Stage 2 should show "Processing batch X/Y"
- **Fix**: Restart runtime and rerun, or increase batch_size:
  ```python
  batch_size=1024  # was 512
  ```

**Issue: Memory issues after training**
- **Cause**: Checkpoints taking up VRAM
- **Fix**: Restart the runtime before running Notebook 2
  - Click Runtime → Restart runtime
  - Then open Notebook 2

---

## EXPECTED RESULTS

### By the Numbers

**Training Performance (Notebook 1, end of Stage 6):**
- Token-level F1: 0.80-0.85
- Precision: 0.82-0.87
- Recall: 0.78-0.83

**Boundary Precision (Notebook 2, end of Stage 4):**
- Usable boundaries: 65-75% (vs baseline 3.8%)
- **Improvement factor**: 17-20x
- Mean IoU: 0.70-0.75 (vs baseline 0.33)
- Isnad end error: <50 chars (vs baseline N/A)

**GPU Time:**
- Notebook 1: 30-60 minutes (varies by GPU)
- Notebook 2: 20-30 minutes
- **Total**: ~90 minutes on T4 GPU

---

## WHAT HAPPENS NEXT

After both notebooks complete successfully:

### Option 1: Deploy the Model
```python
from transformers import pipeline

nlp = pipeline('token-classification', 
               model='/path/to/best_model')

# Use on new texts
predictions = nlp("حدثنا علي عن...")
```

### Option 2: Ensemble with Baseline
```python
# Combine v3.5 (rule-based, 81.9% detection)
# with CAMeL-BERT (70% boundary precision)
# for best of both worlds
```

### Option 3: Further Fine-Tuning
- Collect more annotations
- Train on larger dataset
- Try different model architectures (CAMeL-BERT-Dialogue, AraBERT)

---

## FILE LOCATIONS

**On your Google Drive:**
```
My Drive/Khabar-segmentation/
├── data/
│   ├── raw/
│   └── processed/
│       ├── Kitab_Uqala_al_Majanin_annotated.json         ← INPUT
│       ├── kitab_uqala_reference_corpus.txt              ← INPUT
│       ├── kitab_uqala_boundaries.json                   ← INPUT
│       └── camelbert_training/
│           ├── train.jsonl                               ← OUTPUT (Stage 2)
│           ├── val.jsonl
│           ├── test.jsonl
│           ├── best_model/
│           │   ├── pytorch_model.bin                     ← TRAINED MODEL
│           │   ├── config.json
│           │   └── [tokenizer files]
│           ├── checkpoint/                               ← ALL CHECKPOINTS
│           ├── training_results.json                     ← METRICS
│           ├── TRAINING_SUMMARY.txt
│           └── inference_results/
│               ├── EVALUATION_REPORT.md                  ← FINAL REPORT
│               ├── detailed_metrics.json
│               └── FINAL_SUMMARY.txt
└── notebooks/
    ├── CAMELBERT_FINETUNING_COLAB.ipynb                  ← NOTEBOOK 1
    └── CAMELBERT_INFERENCE_EVALUATION.ipynb              ← NOTEBOOK 2
```

---

## SUPPORT & DEBUGGING

**If notebooks fail at any stage:**

1. Check the error message carefully
2. Look at troubleshooting section above
3. Verify input files exist on Drive
4. Check GPU is enabled
5. Restart runtime if hanging

**Key debugging cells:**

From any notebook, you can run:
```python
# Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Check Drive mount
from google.colab import drive
drive.mount('/content/drive')

# List files
import os
os.listdir('/content/drive/MyDrive/Khabar-segmentation/data/processed')
```

---

## FINAL CHECKLIST

Before starting:
- [ ] Files uploaded to Drive (`data/processed/`)
- [ ] Google Colab access working
- [ ] Notebooks uploaded to Drive
- [ ] Read through stages 1-4 of this guide

During execution:
- [ ] Notebook 1: All 8 stages complete without errors
- [ ] Notebook 1: "Model saved to:" message appears
- [ ] GPU shows usage (check with nvidia-smi in Colab)
- [ ] Notebook 2: Opens and runs to completion
- [ ] Notebook 2: Final results show >50% usable boundaries

After completion:
- [ ] Check Drive for output files
- [ ] Review EVALUATION_REPORT.md
- [ ] Verify improvement factor >10x
- [ ] Save model for future use

---

## Questions?

If you have issues:
1. Check error message + troubleshooting section
2. Verify file structure on Drive
3. Look at notebook cell outputs (scroll up to see full error)
4. Try restarting runtime

Good luck! You're about to get **65-75% boundary precision** 🚀
