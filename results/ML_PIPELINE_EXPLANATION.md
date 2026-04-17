# ML Pipeline for Khabar Boundary Precision

---

## PART 1: Non-Specialist Explanation

### The Problem in Simple Terms

**Current Situation:**
- We have a "keyword finder" that spots where stories start (isnads) - it's 82% accurate at finding them
- But we don't know exactly where each story ends - we just assume it ends when the next one starts
- This causes big mistakes: we're off by an average of **400+ characters** (about 1-2 paragraphs)

**Why Rules Alone Can't Fix It:**
- Arabic text is complex and variable
- Different authors write isnads differently
- A story can end abruptly or continue in unexpected ways
- We'd need to manually list thousands of patterns to handle all cases

### The ML Solution in Everyday Terms

**Think of it like training a teacher:**

1. **Show examples** (Phase 1: Data Preparation)
   - Give the ML system 200 "correctly segmented" stories
   - Mark exactly where each isnad starts and ends
   - Like: "Here's a story. See how it starts? See how it ends?"

2. **Let it learn patterns** (Phase 2: Model Training)
   - The ML system reads all 200 examples
   - It learns patterns: "When I see X, Y usually follows"
   - "Isnads starting with 'حدثنا' tend to be followed by..."
   - "The word 'فقال' usually signals the transition..."
   - Takes 2-4 hours of computer time

3. **Test the teacher** (Phase 3: Validation)
   - Test on stories it hasn't seen before
   - "Did you correctly identify the boundaries?"
   - Measure accuracy: "You got 70% correct"

4. **Use it on new stories** (Phase 4: Deployment)
   - Feed it unlabeled stories
   - It predicts: "This story probably ends here"

### What Improvement Would Look Like

```
Current (Rules):  82% finds stories, but boundaries are wrong
                  Only 3% have usable accuracy

With ML:          ~85% finds stories, AND 70% have good boundaries
                  This means: "In 70% of cases, the detected text 
                             is similar enough to real segments"
```

### Time & Resources Needed

| Phase | Time | Effort | Expertise |
|-------|------|--------|-----------|
| Prepare data | 3-5 days | Manual annotation | Domain expert |
| Train model | 1-2 days | Run script | ML engineer |
| Test & improve | 3-5 days | Adjust & retrain | ML engineer |
| Total | 1-2 weeks | Moderate | 2 people |

**Resources:**
- A laptop/GPU (costs nothing if using cloud free tier)
- 2 people: 1 Arabic expert to annotate, 1 ML engineer to code
- 200 example stories (takes ~1-2 hours per person to annotate)

---

## PART 2: Specialist Explanation

### Problem Formulation

**Current Approach: Rule-Based Isnad Detection**
- Pattern matching on transmission verbs from ISNAD_START_VERBS
- Boundary detection via قال marker (precision: median 180-410 char error)
- IoU against reference: 0.30 (catastrophic)

**Root Cause Analysis:**
1. Isnad/khabar boundary is not binary (start/end) but sequential
2. Many isnads lack قال marker (~10-11%)
3. قال appears in both isnad AND khabar text (disambiguation needed)
4. Context-dependent boundary decisions (multi-token dependencies)

**Why linguistic rules insufficient:**
- Boundary markers are ambiguous without context
- Verb conjugations and variations create false positives/negatives
- Length constraints are arbitrary (min/max bounds on isnad)
- No mechanism to model temporal dependencies (next word, next phrase)

### ML Pipeline Architecture

#### Phase 1: Data Preparation

**Annotation Strategy:**
- Use existing reference boundaries (613 from Kitab_Uqala)
- Prepare ~200 annotated examples via stratified sampling
- Encoding: BIO tagging scheme
  ```
  B-ISNAD: Beginning of isnad
  I-ISNAD: Inside isnad
  B-KHABAR: Beginning of khabar
  I-KHABAR: Inside khabar
  ```

**Data format transformation:**
```
Input: [boundaries.json, corpus.txt]
↓
Process: Tokenize + align with reference boundaries
↓
Output: Token-level labels (token_id, text, boundary_tag, position)
```

**Train/val/test split:**
- Train: 140 examples (~70%)
- Validation: 33 examples (~15%)
- Test: 27 examples (~15%)
- Stratification: Ensure diverse akhbar lengths, verb types, positions

#### Phase 2: Model Selection & Architecture

**Recommended: Transformer-based Token Classifier**

```
INPUT TEXT
    ↓
[TOKENIZER: AraBERT/CAMeL-BERT]
    ↓
TOKENS: [حدثنا, علي, عن, ...]
    ↓
[PRETRAINED ENCODER]
(AraBERT-base or CAMeL-BERT)
    ↓
CONTEXTUAL EMBEDDINGS
[vec1, vec2, vec3, ...]
    ↓
[FINE-TUNING LAYER]
(Dropout + Dense + Softmax)
    ↓
TOKEN-LEVEL PREDICTIONS
[B-ISNAD, I-ISNAD, B-KHABAR, ...]
    ↓
[POST-PROCESSING]
(CRF constraint layer, boundary validation)
    ↓
OUTPUT BOUNDARIES
{start, end, tag_confidence}
```

**Why this approach:**
- Pretrained models already understand Arabic morphology + syntax
- Contextual embeddings capture long-range dependencies
- Token classification directly targets boundary detection
- Transfer learning + fine-tuning = 2-4 weeks training (vs. 6+ months from scratch)

**Model Variants to Consider:**
1. **CAMeL-BERT (Recommended for Classical Arabic)**
   - Specialized for Arabic NLP
   - Better at morphological nuances
   - ~110M parameters (fits on CPU/GPU)

2. **AraBERT-base**
   - Larger pretraining corpus
   - Faster inference
   - Similar performance

3. **Lightweight: RoBERTa-base w/ custom Arabic tokenizer**
   - If latency/deployment critical
   - Trade-off: ~5% accuracy drop

#### Phase 3: Training Strategy

**Loss function:**
```
Loss = Cross-entropy(predicted_tag, true_tag) 
       + λ * CRF_transition_penalty
       + α * focal_loss(hard_examples)
```

**Hyperparameters:**
```
Learning rate: 2e-5 (for fine-tuning)
Batch size: 16-32 (context window ~512 tokens)
Epochs: 5-10 (early stopping on val_loss)
Optimizer: AdamW with weight decay (0.01)
```

**Training Loop:**
```python
for epoch in range(epochs):
    for batch in train_dataloader:
        tokens = batch['input_ids']
        labels = batch['token_labels']
        
        outputs = model(tokens)  # Forward pass
        loss = compute_loss(outputs, labels)
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        validate(model, val_dataloader)  # Early stopping
```

**Expected training time:**
- GPU (NVIDIA V100): 30-60 minutes
- CPU: 2-3 hours
- Fine-tuning focused → relatively fast convergence

#### Phase 4: Evaluation Metrics

**Token-level metrics:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 * (Precision * Recall) / (Precision + Recall)
```

**Segment-level metrics (more relevant):**
```
IoU(detected, reference) = intersection / union
Boundary accuracy = % within X chars of reference
Coverage = total chars correctly segmented / total chars
```

**Success criteria:**
- Token-level F1 ≥ 0.75 (good) / 0.85 (excellent)
- Segment-level IoU ≥ 0.65 (usable) / 0.80 (good)
- Boundary accuracy ≥ 60% within 50 chars
- Improvement: from current 3.1% → target 65-75% usable

#### Phase 5: Post-Processing & Constraints

**CRF Constraint Layer:**
```
Enforce valid state transitions:
  - B-ISNAD can only be followed by I-ISNAD or B-KHABAR
  - B-KHABAR must be preceded by I-ISNAD
  - Reject transitions that violate isnad/khabar structure
```

**Boundary validation:**
```python
def validate_boundary(isnad_start, isnad_end, khabar_end):
    # Hard constraints from linguistic rules
    if isnad_end - isnad_start < 10:  # Min isnad length
        return False
    if khabar_end - isnad_end < 20:  # Min khabar length
        return False
    return True
```

**Ensemble approach (optional):**
```
Combine:
  - Rule-based boundary (from v3.5): 82% detection
  - ML boundary (from fine-tuned model): 70% accuracy
  
Using voting/averaging:
  - If both agree: high confidence (90%+)
  - If conflict: use ML prediction with confidence score
  
Result: Better recall + better precision
```

#### Phase 6: Inference Pipeline

```
NEW TEXT (unlabeled)
  ↓
[Tokenizer: AraBERT]
  ↓
[Pretrained encoder + Fine-tuned head]
  ↓
Token predictions: [B-ISNAD, I-ISNAD, B-KHABAR, ...]
  ↓
[CRF decoding]: Most likely tag sequence
  ↓
[Post-processing]: Validate constraints, compute confidence
  ↓
FINAL BOUNDARIES: 
  {start: 0, isnad_end: 45, end: 320, confidence: 0.92}
  {start: 320, isnad_end: 365, end: 650, confidence: 0.87}
```

### Implementation Details

**Framework: HuggingFace Transformers**
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import TrainingArguments

# Load pretrained model
model = AutoModelForTokenClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv02",
    num_labels=4  # B-ISNAD, I-ISNAD, B-KHABAR, I-KHABAR
)

tokenizer = AutoTokenizer.from_pretrained(
    "aubmindlab/bert-base-arabertv02"
)

# Fine-tune on custom data
training_args = TrainingArguments(
    output_dir="./khabar_classifier",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Expected Results & Trade-offs

**Performance Projections:**
```
Metric                  | Current (v3.5) | ML Target | Improvement
Detection rate (recall) | 81.9%          | 85-88%    | +3-7%
Boundary accuracy       | 3.1% usable    | 65-75%    | +20x
IoU (overlap)          | 0.30           | 0.70-0.75 | +2.3x
Token F1               | N/A            | 0.80-0.85 | Baseline
```

**Inference Cost:**
```
v3.5 (rule-based):     ~50 ms/doc (CPU)
ML model (GPU):        ~20 ms/doc (fine-tuned forward pass)
ML model (CPU):        ~200 ms/doc (4x slower, no GPU)
```

**Data Requirements:**
- 200 annotated documents: acceptable, not ideal
- 500 annotated documents: good
- 1000+ annotated documents: excellent (diminishing returns)

### Failure Modes & Mitigation

**Common issues:**
1. **Class imbalance**: I-ISNAD tokens >> I-KHABAR tokens
   - Mitigation: Weighted loss, focal loss

2. **Short sequences**: Many isnads < 50 tokens
   - Mitigation: Window-based approach with overlap

3. **Ambiguous قال**: Appears in both isnad and khabar
   - Mitigation: Contextual embeddings + CRF should handle

4. **Rare verb patterns**: Not in training data
   - Mitigation: Ensemble with rule-based (fallback)

### Timeline & Resource Estimate

```
Week 1:
  - Data preparation & annotation (140 examples)
  - Setup HuggingFace environment
  - Dataset creation scripts

Week 2:
  - Model training & hyperparameter tuning
  - Validation & early stopping
  - Evaluation metrics implementation

Week 3:
  - Post-processing & CRF constraint layer
  - Error analysis & fine-tuning
  - Inference optimization

Deliverables:
  - Trained model checkpoint (50-100 MB)
  - Inference script
  - Performance report (metrics on test set)
  - Production deployment guide
```

**Team:**
- 1 Specialist (annotator/domain expert): ~40-60 hours annotation
- 1 ML Engineer: ~80-100 hours coding/training/evaluation
- Total: 2-3 weeks, 1-2 FTE

### Comparison: Rule-Based vs ML

| Aspect | Rule-Based (v3.5) | ML Approach |
|--------|---|---|
| **Implementation** | 1000 LOC | 2000 LOC (including training) |
| **Training data needed** | None (rules handcrafted) | 200+ examples |
| **Detection rate** | 81.9% | 85-88% |
| **Boundary precision** | 3.1% | 65-75% |
| **Inference speed** | Fast (50 ms) | Medium (20-200 ms) |
| **Maintenance burden** | High (add rules manually) | Low (retrain on new data) |
| **Adaptation** | Hard (need linguistic expertise) | Easy (add examples, retrain) |
| **Cost to reach 70% boundary accuracy** | N/A (plateau at 3.1%) | 1-2 weeks + annotation |

---

## Decision Framework

**Use v3.5 (Rule-Based) if:**
- You need rough segmentation (82% is acceptable)
- You tolerate ±400 char boundary errors
- You have limited ML resources
- Fast inference is critical

**Switch to ML if:**
- Accurate boundaries are critical (medical, legal, downstream tasks)
- You can invest 2-3 weeks
- You have an annotator for 200 examples
- 70%+ boundary accuracy is required

**Hybrid Approach (Recommended for MVP):**
- Start with v3.5 (deployed today)
- Collect annotations of 200 examples in parallel
- Train ML model (2 weeks)
- Deploy ensemble (rule-based + ML) for best of both

