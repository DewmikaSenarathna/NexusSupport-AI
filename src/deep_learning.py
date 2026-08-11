"""
deep_learning.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from preprocessing import load_and_clean_tickets, split_dataset

MODEL_NAME = "distilbert-base-uncased"


def prepare_hf_datasets(train, val, test, label2id):
    """Converts pandas DataFrames into Hugging Face Dataset objects with
    integer labels, which is the format the Trainer API expects."""
    def to_hf(df):
        return Dataset.from_dict({
            "text": df["text_clean"].tolist(),
            "label": [label2id[c] for c in df["category"]],
        })
    return to_hf(train), to_hf(val), to_hf(test)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_transformer(csv_path: str, output_dir: str = "models/distilbert_ticket_clf"):
    df = load_and_clean_tickets(csv_path)
    train_df, val_df, test_df = split_dataset(df)

    categories = sorted(df["category"].unique())
    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds, val_ds, test_ds = prepare_hf_datasets(train_df, val_df, test_df, label2id)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,          # small LR: we're nudging pretrained weights, not overwriting them
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Test set performance ===")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved fine-tuned model to {output_dir}")
    return trainer, id2label


def predict(text: str, model_dir: str = "models/distilbert_ticket_clf") -> dict:
    """Inference helper used by the FastAPI app / agent."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(probs))
    return {
        "category": model.config.id2label[pred_id],
        "confidence": round(float(probs[pred_id]), 3),
    }


if __name__ == "__main__":
    train_transformer(csv_path="data/support_tickets.csv")
