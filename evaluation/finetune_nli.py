"""
Fine-tune NLI model trên PUBHEALTH-Vi (+ optional XNLI Vietnamese).

Hỗ trợ 2 base model:
  - mDeBERTa (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
  - CafeBERT (uitnlp/CafeBERT)

Usage:
  # Fine-tune mDeBERTa (domain adaptation, nhanh nhất):
  python -m evaluation.finetune_nli --base mdeberta --epochs 5

  # Fine-tune CafeBERT (Vietnamese-native):
  python -m evaluation.finetune_nli --base cafebert --epochs 8

  # Dùng thêm XNLI Vietnamese cho CafeBERT (cần học NLI từ đầu):
  python -m evaluation.finetune_nli --base cafebert --use-xnli --epochs 5
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import DATASET_B_DEFAULT
from evaluation.io_utils import read_jsonl
from evaluation.labels import dataset_b_path

MODEL_CONFIGS = {
    "mdeberta": {
        "name": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "max_length": 512,
        "lr": 2e-5,
    },
    "cafebert": {
        "name": "uitnlp/CafeBERT",
        "max_length": 256,
        "lr": 3e-5,
    },
}

LABEL_MAP = {"true": 0, "false": 1, "unproven": 2}
ID2LABEL = {0: "entailment", 1: "contradiction", 2: "neutral"}
LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}

FINETUNE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)


def split_into_chunks(text, max_chars=800, min_chars=50):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else [text]

    paragraphs = re.split(r"\n\s*\n|\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""
    for para in paragraphs:
        if not para.strip():
            continue
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip() if current else para
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
            current = para
    if current and len(current) >= min_chars:
        chunks.append(current)
    return chunks if chunks else [text[:max_chars]]


def prepare_pubhealth_pairs(records, chunk_size=800):
    """Create (evidence_chunk, claim, label) pairs from Dataset B jsonl."""
    pairs = []
    for row in records:
        label = LABEL_MAP.get(str(row["label"]).lower())
        if label is None:
            continue
        claim = str(row["claim"]).strip()
        chunks = split_into_chunks(row["main_text"], max_chars=chunk_size)
        for chunk in chunks:
            pairs.append({"premise": chunk, "hypothesis": claim, "label": label})
    return pairs


def load_xnli_vietnamese(max_samples=5000):
    """Load XNLI Vietnamese from HuggingFace datasets."""
    from datasets import load_dataset

    print("Đang tải XNLI Vietnamese...")
    ds = load_dataset("facebook/xnli", "vi", trust_remote_code=True)

    pairs = []
    for split in ["validation", "test"]:
        if split not in ds:
            continue
        for row in ds[split]:
            pairs.append({
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
            })
            if len(pairs) >= max_samples:
                break

    print(f"  Đã tải {len(pairs)} mẫu XNLI Vietnamese")
    return pairs


class NLIPairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        encoding = self.tokenizer(
            pair["premise"],
            pair["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(pair["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def get_device(force_cpu=False):
    """Pick training device. mDeBERTa is unstable on MPS (NaN gradients)."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # DeBERTa v3 overflows on Apple MPS; use CPU for stable fine-tuning
    if torch.backends.mps.is_available():
        return "cpu"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLI model")
    parser.add_argument(
        "--base",
        choices=["mdeberta", "cafebert"],
        default="mdeberta",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset-b",
        choices=["vifactcheck", "pubhealth"],
        default=DATASET_B_DEFAULT,
        help="Nguồn Dataset B để fine-tune",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Đường dẫn jsonl (ghi đè --dataset-b)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--use-xnli",
        action="store_true",
        help="Dùng thêm XNLI Vietnamese (cần cho CafeBERT)",
    )
    parser.add_argument("--xnli-max", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục lưu model (default: models/nli_finetuned_<base>)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for training (auto: cuda > cpu; mDeBERTa avoids MPS)",
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=None,
        help="Giới hạn số training pairs (để test nhanh)",
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.base]
    model_name = config["name"]
    max_length = config["max_length"]
    lr = args.lr or config["lr"]
    output_dir = args.output_dir or os.path.join(
        FINETUNE_DIR, f"nli_finetuned_{args.base}"
    )

    if args.device == "auto":
        device = get_device()
    elif args.device == "cpu":
        device = "cpu"
    elif args.device == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Fine-tune NLI: {args.base}")
    print(f"  Base model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {lr}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    records_path = args.input or dataset_b_path(args.dataset_b)
    if not os.path.exists(records_path):
        hint = (
            "python -m evaluation.prepare_vifactcheck"
            if "vifactcheck" in records_path
            else "python -m evaluation.prepare_pubhealth"
        )
        raise FileNotFoundError(f"Chưa có {records_path}. Chạy: {hint}")

    records = read_jsonl(records_path)
    print(f"Dataset B ({records_path}): {len(records)} records")

    train_records, test_records = train_test_split(
        records,
        test_size=args.test_size,
        random_state=42,
        stratify=[r["label"] for r in records],
    )
    print(f"  Train: {len(train_records)}, Test: {len(test_records)}")

    test_path = os.path.join(os.path.dirname(output_dir), "dataset_b_test_split.jsonl")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved test split -> {test_path}")

    train_pairs = prepare_pubhealth_pairs(train_records, chunk_size=args.chunk_size)
    test_pairs = prepare_pubhealth_pairs(test_records, chunk_size=args.chunk_size)

    if args.max_train_pairs and len(train_pairs) > args.max_train_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_pairs), args.max_train_pairs, replace=False)
        train_pairs = [train_pairs[i] for i in sorted(idx)]
        print(f"  (giới hạn train pairs -> {len(train_pairs)})")
    print(f"\nTraining pairs: {len(train_pairs)} (from {len(train_records)} records)")
    print(f"Test pairs: {len(test_pairs)} (from {len(test_records)} records)")

    if args.use_xnli:
        xnli_pairs = load_xnli_vietnamese(max_samples=args.xnli_max)
        train_pairs = xnli_pairs + train_pairs
        print(f"Total training pairs (XNLI + Dataset B): {len(train_pairs)}")

    from collections import Counter
    label_dist = Counter(p["label"] for p in train_pairs)
    print(f"Train label dist: {dict(label_dist)}")

    print(f"\nLoading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = NLIPairDataset(train_pairs, tokenizer, max_length)
    test_dataset = NLIPairDataset(test_pairs, tokenizer, max_length)

    use_fp16 = device == "cuda"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nBắt đầu training ({len(train_dataset)} samples, {args.epochs} epochs)...")
    trainer.train()

    print("\n=== Kết quả trên test set ===")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    print(f"  Loss: {eval_results['eval_loss']:.4f}")

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    meta = {
        "base_model": model_name,
        "base_key": args.base,
        "epochs": args.epochs,
        "lr": lr,
        "chunk_size": args.chunk_size,
        "train_samples": len(train_pairs),
        "test_samples": len(test_pairs),
        "use_xnli": args.use_xnli,
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_f1_macro": eval_results["eval_f1_macro"],
        "eval_loss": eval_results["eval_loss"],
        "device": device,
    }
    with open(os.path.join(best_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved -> {best_dir}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.1%}")
    print(f"F1-Macro: {eval_results['eval_f1_macro']:.1%}")
    print("Done!")


if __name__ == "__main__":
    main()
