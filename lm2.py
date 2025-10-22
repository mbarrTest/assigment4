#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm2.py — BiLSTM + FastText for offensive/profanity detection on tweets.

Key behavior:
- Robust .pipe reader: splits each row on the LAST '|' → text|label (extra '|' in text is fine).
- Label mapping: supports numeric 0/1 and common strings (OFF/NOT, toxic/clean, etc).
- BiLSTM with mean+max pooling.
- Early stopping on dev F1; saves best model to lm2_best.pt.
- After training, evaluates the BEST checkpoint on DEV and
  prints + writes ONLY the FINAL metrics to <save_dir>/lstmResults.txt.
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional tokenizer (spaCy); falls back to regex
try:
    import spacy
    _SPACY = True
except Exception:
    _SPACY = False

# Embeddings (Gensim)
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors


# =========================
# Utilities / IO
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_last_pipe(line: str):
    """Split at the LAST '|' only → (text, label_str)."""
    if "|" not in line:
        return line, ""
    left, right = line.rsplit("|", 1)
    return left, right.strip()


def read_table(path: str) -> pd.DataFrame:
    """
    Robust reader for .pipe/.tsv/.csv.
    - If '.pipe' or first line has '|': split each line at the LAST '|' into text|label.
    - Else: use pandas read_csv with detected sep.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()

    looks_pipe = ("|" in head) or path.lower().endswith(".pipe")
    if looks_pipe:
        texts, labels = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().rstrip("\n\r")
            parts = header.rsplit("|", 1)
            has_header = (len(parts) == 2 and parts[1].strip().lower() == "label")
            if not has_header and header.strip():
                t, lbl = _split_last_pipe(header)
                texts.append(t); labels.append(lbl)
            for line in f:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                t, lbl = _split_last_pipe(line)
                texts.append(t); labels.append(lbl)
        return pd.DataFrame({"text": texts, "label": labels})

    sep = "\t" if "\t" in head else ","
    return pd.read_csv(path, sep=sep, quoting=csv.QUOTE_MINIMAL)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def default_tokenize(text: str) -> List[str]:
    # keeps hashtags/mentions/emojis separated
    return re.findall(r"[#@]?\w+|[^\s\w]", text)


class Tokenizer:
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and _SPACY
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            except Exception:
                self.use_spacy = False

    def __call__(self, text: str) -> List[str]:
        text = normalize_text(text)
        if self.use_spacy and self.nlp is not None:
            return [t.text for t in self.nlp(text)]
        return default_tokenize(text)


# =========================
# Label mapping
# =========================
def map_labels_inplace(df: pd.DataFrame, split_name: str):
    """Map df['label'] to {0,1}. Supports numeric and common string labels."""
    assert "label" in df.columns
    lab_raw = df["label"].astype(str).str.strip().str.lower()

    # Numeric?
    num = pd.to_numeric(lab_raw, errors="coerce")
    if num.notna().any() and num.dropna().isin([0, 1]).all():
        before = len(df)
        df["label"] = num.fillna(0).astype(int)
        after = len(df)
        print(f"[info] {split_name}: kept {after}/{before} rows (numeric labels 0/1)")
        return

    # Common text labels
    mapping = {
        "off": 1, "offensive": 1, "abusive": 1, "hate": 1, "toxic": 1, "insult": 1, "offense": 1, "offence": 1,
        "1": 1, "true": 1, "yes": 1,
        "not": 0, "none": 0, "neutral": 0, "clean": 0, "ok": 0, "benign": 0,
        "0": 0, "false": 0, "no": 0
    }
    mapped = lab_raw.map(mapping)
    before = len(df)
    df["label"] = mapped
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    after = len(df)
    print(f"[info] {split_name}: kept {after}/{before} rows after label mapping")


# =========================
# Vocabulary / Dataset
# =========================
PAD = "<pad>"
UNK = "<unk>"

class Vocab:
    def __init__(self, min_freq: int = 1, max_size: Optional[int] = 50000, specials: Optional[List[str]] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.freqs: Dict[str, int] = {}
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.specials = specials if specials is not None else [PAD, UNK]

    def build(self, tokenized_texts: List[List[str]]):
        for toks in tokenized_texts:
            for t in toks:
                self.freqs[t] = self.freqs.get(t, 0) + 1
        self.itos = list(self.specials)
        items = sorted(((t, c) for t, c in self.freqs.items() if c >= self.min_freq),
                       key=lambda x: (-x[1], x[0]))
        if self.max_size is not None:
            items = items[: max(0, self.max_size - len(self.specials))]
        self.itos += [t for t, _ in items]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi.get(UNK, 1)
        return [self.stoi.get(t, unk) for t in tokens]


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, vocab: Vocab, max_len: int = 80):
        assert "text" in df.columns and "label" in df.columns
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int):
        toks = self.tokenizer(self.texts[idx])
        ids = self.vocab.encode(toks)[: self.max_len]
        y = int(self.labels[idx])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.float)


def pad_collate(batch, pad_idx: int):
    ids_list, y_list = zip(*batch)
    lengths = [len(x) for x in ids_list]
    max_len = max(lengths) if lengths else 1
    padded = []
    for ids in ids_list:
        if len(ids) < max_len:
            pad = torch.full((max_len - len(ids),), pad_idx, dtype=torch.long)
            ids = torch.cat([ids, pad])
        padded.append(ids)
    X = torch.stack(padded, dim=0)
    y = torch.stack(y_list, dim=0)
    return X, y, torch.tensor(lengths, dtype=torch.long)


# =========================
# Embeddings
# =========================
def load_embeddings(emb_path: str):
    ext = Path(emb_path).suffix.lower()
    if ext == ".bin":
        print(f"[emb] Loading FastText (bin): {emb_path}")
        kv = load_facebook_vectors(emb_path)  # subword OOV support
        dim = kv.vector_size
        return kv, dim, True
    else:
        print(f"[emb] Loading word vectors (vec): {emb_path}")
        kv = KeyedVectors.load_word2vec_format(emb_path, binary=False)
        dim = kv.vector_size
        return kv, dim, False


def build_embedding_matrix(vocab: Vocab, kv, dim: int, is_ft_bin: bool, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    if PAD in vocab.stoi:
        mat[vocab.stoi[PAD]] = 0.0
    for i, tok in enumerate(vocab.itos):
        if tok in (PAD, UNK):
            continue
        vec = None
        try:
            if is_ft_bin:
                vec = kv.get_vector(tok)
            else:
                if tok in kv.key_to_index:
                    vec = kv.get_vector(tok)
        except KeyError:
            vec = None
        if vec is not None:
            mat[i] = vec
    return torch.tensor(mat)


# =========================
# Model
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embedding_matrix: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        if embedding_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        dir_mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * dir_mult * 2, 1)  # mean+max concat

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor):
        # ids: (B,T), lengths: (B,)
        emb = self.embedding(ids)  # (B,T,E)

        # LSTM over packed sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H*dir)

        # boolean mask for non-PAD tokens; shape (B,T,1) for broadcast
        mask = (ids != self.embedding.padding_idx).unsqueeze(-1)  # bool, (B,T,1)

        # mean pooling over valid time steps
        out_masked_zero = out.masked_fill(~mask, 0.0)             # (B,T,H*dir)
        lengths_f = lengths.clamp(min=1).unsqueeze(1).to(out.dtype)  # (B,1)
        mean_pool = out_masked_zero.sum(dim=1) / lengths_f        # (B,H*dir)

        # max pooling over valid time steps
        out_masked_neg = out.masked_fill(~mask, -1e9)             # (B,T,H*dir)
        max_pool, _ = out_masked_neg.max(dim=1)                   # (B,H*dir)

        # concatenate, dropout, classify
        feat = torch.cat([mean_pool, max_pool], dim=1)            # (B, 2*H*dir)
        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(1)                         # (B,)
        return logits


# =========================
# Train / Eval
# =========================
def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds, golds = [], []

    for ids, labels, lengths in loader:
        ids = ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(ids, lengths)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * ids.size(0)
        probs = torch.sigmoid(logits)
        preds.append((probs >= 0.5).long().cpu().numpy())
        golds.append(labels.long().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=int)
    golds = np.concatenate(golds) if golds else np.zeros(0, dtype=int)
    m = metrics_binary(golds, preds) if len(golds) > 0 else {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "tp":0, "fp":0, "tn":0, "fn":0}
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, m


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="BiLSTM + FastText for offensive language detection")
    # Data
    parser.add_argument("--train", type=str, required=True, help="Path to train .pipe/.tsv/.csv")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev/validation file")
    parser.add_argument("--test", type=str, default=None, help="Optional test file (unused in final metrics)")
    # Embeddings
    parser.add_argument("--emb_path", type=str, required=True, help=".vec or FastText .bin path")
    parser.add_argument("--freeze_embeddings", type=int, default=1, help="1 freeze, 0 finetune")
    # Vocab/length
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=80)
    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Train
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # Saving
    parser.add_argument("--save_dir", type=str, default="runs/lstm_fasttext")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("[info] Loading data…")
    df_train = read_table(args.train)
    df_dev = read_table(args.dev)
    for df, name in [(df_train, "train"), (df_dev, "dev")]:
        assert "text" in df.columns and "label" in df.columns, f"{name} must have 'text' and 'label' columns"
        df["text"] = df["text"].astype(str).map(normalize_text)
        map_labels_inplace(df, name)

    # Guard: empty splits
    if len(df_train) == 0:
        raise SystemExit("[error] Train split empty after parsing/label mapping.")
    if len(df_dev) == 0:
        raise SystemExit("[error] Dev split empty after parsing/label mapping.")

    # Tokenizer & vocab (build on train only)
    print("[info] Building tokenizer & vocab…")
    tokenizer = Tokenizer(use_spacy=True)  # set to False to silence spaCy warnings
    tok_train = [tokenizer(t) for t in df_train["text"].tolist()]
    vocab = Vocab(min_freq=args.min_freq, max_size=args.max_vocab)
    vocab.build(tok_train)
    print(f"[info] Vocab size = {len(vocab)} (min_freq={args.min_freq}, max={args.max_vocab})")

    # Embeddings
    print("[info] Loading embeddings…")
    kv, emb_dim, is_ft_bin = load_embeddings(args.emb_path)
    emb_matrix = build_embedding_matrix(vocab, kv, emb_dim, is_ft_bin, seed=args.seed)

    # Datasets / loaders
    ds_train = TextDataset(df_train, tokenizer, vocab, max_len=args.max_len)
    ds_dev = TextDataset(df_dev, tokenizer, vocab, max_len=args.max_len)
    collate = lambda batch: pad_collate(batch, pad_idx=vocab.stoi[PAD])

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    model = BiLSTMClassifier(
        num_embeddings=len(vocab),
        embedding_dim=emb_dim,
        padding_idx=vocab.stoi[PAD],
        embedding_matrix=emb_matrix,
        freeze_embeddings=bool(args.freeze_embeddings),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=bool(args.bidirectional),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train with early stopping on Dev F1 (only concise logging)
    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "lm2_best.pt")
    patience_left = args.early_stop_patience

    print("[info] Training…")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_m = run_epoch(model, dl_train, device, optimizer=optimizer, criterion=criterion)
        dv_loss, dv_m = run_epoch(model, dl_dev, device, optimizer=None, criterion=criterion)
        dur = time.time() - start

        print(f"[epoch {epoch:02d}] train loss {tr_loss:.4f} F1 {tr_m['f1']:.3f} || "
              f"dev loss {dv_loss:.4f} F1 {dv_m['f1']:.3f} ({dur:.1f}s)")

        # Early stopping on dev F1
        if dv_m["f1"] > best_f1:
            best_f1 = dv_m["f1"]
            torch.save({"model_state": model.state_dict()}, best_path)
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[info] Early stopping.")
                break

    # -------- FINAL-ONLY METRICS (BEST CHECKPOINT) --------
    print("\n[info] Training complete. Evaluating best model on dev set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dv_loss, dv_m = run_epoch(model, dl_dev, device, optimizer=None, criterion=criterion)

    results_text = (
        "\n========== FINAL EVALUATION (Best on Dev) ==========\n"
        f"Accuracy:  {dv_m['accuracy']:.4f}\n"
        f"Precision: {dv_m['precision']:.4f}\n"
        f"Recall:    {dv_m['recall']:.4f}\n"
        f"F1 Score:  {dv_m['f1']:.4f}\n"
        f"--------------------------------------\n"
        f"Confusion Matrix:\n"
        f"TP={dv_m['tp']}  FP={dv_m['fp']}\n"
        f"TN={dv_m['tn']}  FN={dv_m['fn']}\n"
        "====================================================\n"
    )

    print(results_text)
    results_path = os.path.join(args.save_dir, "lstmResults.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_text)
    print(f"[info] Final metrics written to {results_path}")


if __name__ == "__main__":
    main()

