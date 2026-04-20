"""
Fine-tuner AraBERT pour la segmentation d'akhbars (token classification).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import hydra
from omegaconf import DictConfig, OmegaConf


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
        for line in f:
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
                # Token special (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Nouveau mot -> utiliser le label du mot
                if word_idx < len(label_seq):
                    label_ids.append(LABEL2ID.get(label_seq[word_idx], 0))
                else:
                    label_ids.append(-100)
            else:
                # Meme mot, token suivant du meme mot -> I- au lieu de B-
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
) -> Tuple[Dataset, Dataset]:
    """Preparer les datasets pour le fine-tuning."""

    # Charger les donnees
    print("[*] Loading training examples...")
    train_examples = load_examples_from_jsonl(train_path)
    val_examples = load_examples_from_jsonl(val_path)
    print(f"    Train: {len(train_examples)}")
    print(f"    Val: {len(val_examples)}")

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
        remove_columns=['tokens', 'labels']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=['tokens', 'labels']
    )

    return train_dataset, val_dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Calculer les metriques d'evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Filtrer les labels = -100 (tokens speciaux)
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        for pred, lbl in zip(prediction, label):
            if lbl != -100:
                true_predictions.append(ID2LABEL[pred])
                true_labels.append(ID2LABEL[lbl])

    # Classification report
    print("\n[METRICS] Classification Report:")
    print(classification_report(true_labels, true_predictions))

    return {
        'f1': f1_score(true_labels, true_predictions, average='weighted'),
        'precision': precision_score(true_labels, true_predictions, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, true_predictions, average='weighted', zero_division=0),
    }


@hydra.main(config_path='../configs', config_name='finetune_arabertv2', version_base=None)
def main(config: DictConfig):
    """Fine-tuner le modele."""
    print(OmegaConf.to_yaml(config))

    # Chemins
    data_dir = Path(config.data.dir)
    train_path = data_dir / config.data.train_file
    val_path = data_dir / config.data.val_file
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le tokenizer et le modele
    print(f"[*] Loading tokenizer: {config.model.name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    print(f"[*] Loading model: {config.model.name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.model.name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Preparer les datasets
    print("\n[*] Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(
        str(train_path),
        str(val_path),
        tokenizer,
        max_length=config.data.max_length,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        seed=config.training.seed,
        report_to=['tensorboard'],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tuner
    print("\n[*] Starting training...")
    trainer.train()

    print(f"\n[OK] Training complete! Model saved to {output_dir}")

    # Sauvegarder les labels
    with open(output_dir / 'label2id.json', 'w') as f:
        json.dump(LABEL2ID, f)
    with open(output_dir / 'id2label.json', 'w') as f:
        json.dump(ID2LABEL, f)


if __name__ == '__main__':
    main()
