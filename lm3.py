import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Label mapping
LABEL2ID = {"NOT": 0, "OFF": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------- Robust .pipe reader ----------------------
def read_pipe_file(path):
    """
    Read a .pipe-like file robustly. Accepts lines where the text may contain '|' or tabs.
    Strategy: split on the LAST '|' first; if absent, try the LAST tab.
    Skips a header line like 'message|label' or 'text|label' if present.
    Returns (DataFrame, has_labels_bool)
    """
    texts, labels = [], []
    has_label = True

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue

            low = line.strip().lower()
            if low in {"message|label", "text|label"}:
                # Skip header if any
                continue

            if "|" in line:
                left, right = line.rsplit("|", 1)
                lbl = right.strip().upper()
                text = left.strip()
                if lbl in LABEL2ID:
                    texts.append(text)
                    labels.append(lbl)
                else:
                    # Not a valid label → mark as unlabeled
                    texts.append(line.strip())
                    has_label = False
            elif "\t" in line:
                left, right = line.rsplit("\t", 1)
                lbl = right.strip().upper()
                text = left.strip()
                if lbl in LABEL2ID:
                    texts.append(text)
                    labels.append(lbl)
                else:
                    texts.append(line.strip())
                    has_label = False
            else:
                texts.append(line.strip())
                has_label = False

    if has_label and len(texts) == len(labels):
        df = pd.DataFrame({"text": texts, "label": labels})
        return df, True
    else:
        df = pd.DataFrame({"text": texts})
        return df, False


def df_to_dataset(df, has_label: bool):
    """Convert pandas DataFrame to HF Dataset. If has_label map to ints."""
    if df is None:
        return None
    if has_label:
        df = df.copy()
        df["label"] = df["label"].astype(str).str.strip().str.upper()
        df = df[df["label"].isin(LABEL2ID.keys())].reset_index(drop=True)
        df["label"] = df["label"].map(LABEL2ID).astype(int)
        return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    else:
        return Dataset.from_pandas(df[["text"]], preserve_index=False)


# ---------------------- Metrics & saving ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "precision": p, "recall": r}


def save_results(output_dir, split_name, logits, labels):
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)
    cls_rep = classification_report(labels, preds, target_names=[ID2LABEL[0], ID2LABEL[1]], zero_division=0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "robertaResults.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"=== {split_name.upper()} RESULTS @ {datetime.now().isoformat()} ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {p_macro:.4f}\nMacro Recall: {r_macro:.4f}\nMacro F1: {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weight:.4f}\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(pd.DataFrame(cm, index=[ID2LABEL[0], ID2LABEL[1]],
                             columns=[ID2LABEL[0], ID2LABEL[1]]).to_string())
        f.write("\n\nClassification Report:\n")
        f.write(cls_rep + "\n\n")
    print(f"[{split_name}] Saved metrics to {out_path}")


def write_predictions(output_dir, split_name, texts, pred_ids):
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{split_name}_predictions.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("label\ttext\n")
        for t, p in zip(texts, pred_ids):
            f.write(f"{ID2LABEL[int(p)]}\t{t}\n")
    print(f"[{split_name}] Wrote predictions to {out}")


# ---------------------- TrainingArguments (backward-compatible) ----------------------
def build_training_args(args, use_fp16: bool, use_bf16: bool) -> TrainingArguments:
    """
    Build TrainingArguments using new API if available; otherwise fall back to legacy flags.
    Handles older Transformers where 'evaluation_strategy' etc. don't exist.
    """
    try:
        # Newer API
        return TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            fp16=use_fp16,
            bf16=use_bf16 if hasattr(TrainingArguments, "bf16") else False,
            weight_decay=0.01,
            report_to=["none"],
            dataloader_num_workers=2,
            seed=42,
        )
    except TypeError:
        # Legacy API
        print("[INFO] Falling back to legacy TrainingArguments (old Transformers).")
        try:
            return TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.lr,
                logging_steps=50,
                save_steps=500,
                do_eval=True,
                evaluate_during_training=True,   # legacy flag
                fp16=use_fp16,
                weight_decay=0.01,
                seed=42,
            )
        except TypeError:
            # Ultra-legacy: drop unknown fields
            return TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.lr,
                logging_steps=50,
                save_steps=500,
                do_eval=True,
                fp16=use_fp16,
                weight_decay=0.01,
                seed=42,
            )


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("lm3.py — RoBERTa OFF/NOT classifier for .pipe files (backward compatible)")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--test_path", default=None, help="Optional test.pipe (may have labels or not)")
    parser.add_argument("--model_name", default="roberta-base", help="HF model name")
    parser.add_argument("--output_dir", default="runs/roberta_offense")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if available")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if available")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read files
    print("Loading train/dev files...")
    train_df, train_has_label = read_pipe_file(args.train_path)
    dev_df, dev_has_label = read_pipe_file(args.dev_path)
    if not (train_has_label and dev_has_label):
        raise ValueError("train_path and dev_path must contain labels (OFF/NOT) at the end separated by '|'. Use the cleaner if needed.")

    test_df, test_has_label = (None, False)
    if args.test_path:
        test_df, test_has_label = read_pipe_file(args.test_path)

    # Datasets
    train_ds = df_to_dataset(train_df, has_label=True)
    dev_ds   = df_to_dataset(dev_df,   has_label=True)
    test_ds  = df_to_dataset(test_df,  has_label=test_has_label) if args.test_path else None

    # Tokenizer & tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    dev_tok   = dev_ds.map(tokenize_fn,   batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tokenize_fn,  batched=True, remove_columns=["text"]) if test_ds is not None else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Mixed precision flags
    # Prefer bf16 if user asks and it's available; else fp16 if asked and bf16 isn't used.
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if args.bf16 and torch.cuda.is_available() and not use_bf16 and hasattr(torch.cuda, "get_device_capability"):
        major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = (major >= 8)  # heuristic for Ampere+

    use_fp16 = bool(args.fp16 and torch.cuda.is_available() and not use_bf16)

    # Training args (backward compatible)
    training_args = build_training_args(args, use_fp16=use_fp16, use_bf16=use_bf16)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

    # Evaluate on dev and save metrics
    print("Evaluating on dev set...")
    dev_out = trainer.predict(dev_tok)
    save_results(args.output_dir, "dev", dev_out.predictions, dev_out.label_ids)

    # Optional test
    if test_tok is not None:
        print("Evaluating/predicting on test set...")
        test_out = trainer.predict(test_tok)
        if test_has_label:
            save_results(args.output_dir, "test", test_out.predictions, test_out.label_ids)
        else:
            pred_ids = np.argmax(test_out.predictions, axis=-1)
            write_predictions(args.output_dir, "test", test_df["text"].tolist(), pred_ids)

    print("Done.")


if __name__ == "__main__":
    main()