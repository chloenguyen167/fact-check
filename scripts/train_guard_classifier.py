"""Fine-tune PhoBERT classifier cho Guard tầng 3.

Ví dụ:
  python scripts/train_guard_classifier.py \
    --data data/guard_train.csv \
    --out models/guard_phobert \
    --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import Dataset
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from guard.classifier import LABELS, build_input
from guard.ner import GazetteerNER


def parse_args():
    p = argparse.ArgumentParser(description="Train Guard PhoBERT classifier")
    p.add_argument("--data", default="data/guard_train.csv")
    p.add_argument("--out", default="models/guard_phobert")
    p.add_argument("--base", default="vinai/phobert-base-v2")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV cần cột text,label")

    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise SystemExit(f"Nhãn lạ trong CSV: {unknown}")

    ner = GazetteerNER()
    texts = []
    for text in df["text"].astype(str):
        feats = ner.features(text)
        fused = build_input(text, feats)
        texts.append(ViTokenizer.tokenize(fused))

    labels = [label2id[lab] for lab in df["label"]]
    train_x, val_x, train_y, val_y = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = Dataset.from_dict({"text": train_x, "label": train_y}).map(
        tokenize, batched=True
    )
    val_ds = Dataset.from_dict({"text": val_x, "label": val_y}).map(
        tokenize, batched=True
    )
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(
        classification_report(
            val_y, y_pred, target_names=LABELS, digits=3, zero_division=0
        )
    )

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Đã lưu model → {out}")


if __name__ == "__main__":
    main()
