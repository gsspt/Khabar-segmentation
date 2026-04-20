"""
Fine-tuner AraBERT pour la segmentation d'akhbars.
Version 2 : Simplifié, sans Hydra, avec tqdm pour suivi en terminal.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm


# Label encoding
LABEL2ID = {
    'O': 0,
    'B-KHABAR': 1,
    'I-KHABAR': 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_examples_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Charger les exemples depuis un fichier JSONL."""
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {Path(jsonl_path).name}"):
            if line.strip():
                examples.append(json.loads(line))
    return examples


def tokenize_and_align_labels(
    examples: Dict,
    tokenizer,
    max_length: int = 512
) -> Dict:
    """Tokenizer et aligner les labels avec les tokens du tokenizer."""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        return_overflowing_tokens=False,
    )

    labels = []
    for i, label_seq in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(label_seq):
                    label_ids.append(LABEL2ID.get(label_seq[word_idx], 0))
                else:
                    label_ids.append(-100)
            else:
                if word_idx < len(label_seq):
                    label = label_seq[word_idx]
                    if label.startswith('B-'):
                        label = 'I-' + label[2:]
                    label_ids.append(LABEL2ID.get(label, 0))
                else:
                    label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def prepare_dataset(
    train_path: str,
    val_path: str,
    tokenizer,
    max_length: int = 512,
) -> tuple:
    """Preparer les datasets pour le fine-tuning."""

    print("[*] Loading training examples...")
    train_examples = load_examples_from_jsonl(train_path)
    val_examples = load_examples_from_jsonl(val_path)
    print(f"    Train: {len(train_examples)} examples")
    print(f"    Val: {len(val_examples)} examples")

    # Convertir en Dataset HF
    train_dataset = Dataset.from_dict({
        'tokens': [ex['tokens'] for ex in train_examples],
        'labels': [ex['labels'] for ex in train_examples],
    })
    val_dataset = Dataset.from_dict({
        'tokens': [ex['tokens'] for ex in val_examples],
        'labels': [ex['labels'] for ex in val_examples],
    })

    # Tokenizer et aligner
    print("[*] Tokenizing and aligning labels...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=['tokens', 'labels'],
        desc="Train dataset"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=['tokens', 'labels'],
        desc="Val dataset"
    )

    return train_dataset, val_dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Calculer les metriques d'evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        for pred, lbl in zip(prediction, label):
            if lbl != -100:
                true_predictions.append(ID2LABEL[pred])
                true_labels.append(ID2LABEL[lbl])

    print("\n[METRICS]")
    print(classification_report(true_labels, true_predictions))

    return {
        'f1': f1_score(true_labels, true_predictions, average='weighted'),
        'precision': precision_score(true_labels, true_predictions, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, true_predictions, average='weighted', zero_division=0),
    }


def main():
    """Fine-tuner le modele."""

    print("[*] Configuration")
    print("    Model: aubmindlab/bert-base-arabertv2")
    print("    Task: Token classification (BIO) for akhbar segmentation")
    print("    Epochs: 10")
    print("    Batch size: 16")
    print("    Learning rate: 2e-5")
    print()

    # Chemins
    data_dir = Path('./data/processed')
    train_path = data_dir / 'train_akhbars_bio.jsonl'
    val_path = data_dir / 'val_akhbars_bio.jsonl'
    output_dir = Path('./checkpoints/arabertv2_akhbars_v2')

    # Verifier les fichiers
    if not train_path.exists() or not val_path.exists():
        print("[!] Dataset files not found!")
        print(f"    Expected: {train_path} and {val_path}")
        print("[*] Run prepare_dataset_v2.py first:")
        print("    python scripts/prepare_dataset_v2.py")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le tokenizer et le modele
    print("[*] Loading tokenizer: aubmindlab/bert-base-arabertv2...")
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')

    print("[*] Loading model: aubmindlab/bert-base-arabertv2...")
    model = AutoModelForTokenClassification.from_pretrained(
        'aubmindlab/bert-base-arabertv2',
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Preparer les datasets
    print()
    train_dataset, val_dataset = prepare_dataset(
        str(train_path),
        str(val_path),
        tokenizer,
        max_length=512,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy='steps',
        save_strategy='steps',
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        report_to=[],
    )

    # Trainer
    print()
    print("[*] Starting training...")
    print(f"    Output dir: {output_dir}")
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tuner
    trainer.train()

    print()
    print(f"[*] Saving best model to {output_dir}...")
    trainer.save_model(str(output_dir))

    print(f"[OK] Training complete!")
    print(f"[OK] Model saved to {output_dir}")

    # Sauvegarder les labels
    with open(output_dir / 'label2id.json', 'w') as f:
        json.dump(LABEL2ID, f)
    with open(output_dir / 'id2label.json', 'w') as f:
        json.dump(ID2LABEL, f)

    print("[OK] Labels saved!")


if __name__ == '__main__':
    main()
