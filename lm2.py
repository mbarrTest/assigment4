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

# Optional tokenizer (spaCy)
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
    if "|" not in line:
        return line, ""
    left, right = line.rsplit("|", 1)
    return left, right.strip()


def read_table(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()

    looks_pipe = ("|" in head) or path.lower().endswith(".pipe")
    if looks_pipe:
        texts, labels = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t, lbl = _split_last_pipe(line)
                texts.append(t)
                labels.append(lbl)
        return pd.DataFrame({"text": texts, "label": labels})

    sep = "\t" if "\t" in head else ","
    return pd.read_csv(path, sep=sep, quoting=csv.QUOTE_MINIMAL)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def default_tokenize(text: str) -> List[str]:
    return re.findall(r"[#@]?\w+|[^\s\w]", text)


class Tokenizer:
    def __init__(self, use_spacy: bool = False):
        self.use_spacy = use_spacy and _SPACY
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            except Exception:
                self.use_spacy = False

    def __call__(self, text: str) -> List[str]:
        text = normalize_text(text)
        if self.use_spacy and self.nlp:
            return [t.text for t in self.nlp(text)]
        return default_tokenize(text)


# =========================
# Label mapping
# =========================
def map_labels_inplace(df: pd.DataFrame, split_name: str):
    lab_raw = df["label"].astype(str).str.strip().str.lower()

    num = pd.to_numeric(lab_raw, errors="coerce")
    if num.notna().any() and num.dropna().isin([0, 1]).all():
        df["label"] = num.fillna(0).astype(int)
        print(f"[info] {split_name}: numeric 0/1 labels")
        return

    mapping = {
        "off": 1, "offensive": 1, "abusive": 1, "hate": 1, "toxic": 1, "insult": 1,
        "1": 1, "true": 1, "yes": 1,
        "not": 0, "none": 0, "neutral": 0, "clean": 0, "ok": 0, "benign": 0,
        "0": 0, "false": 0, "no": 0
    }
    df["label"] = lab_raw.map(mapping)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    print(f"[info] {split_name}: mapped text labels → numeric")


# =========================
# Vocabulary / Dataset
# =========================
PAD = "<pad>"
UNK = "<unk>"

class Vocab:
    def __init__(self, min_freq=1, max_size=50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.freqs = {}
        self.itos, self.stoi = [], {}

    def build(self, tokenized_texts):
        for toks in tokenized_texts:
            for t in toks:
                self.freqs[t] = self.freqs.get(t, 0) + 1
        items = sorted(self.freqs.items(), key=lambda x: -x[1])
        self.itos = [PAD, UNK] + [t for t, c in items if c >= self.min_freq][: self.max_size]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self): return len(self.itos)
    def encode(self, tokens): return [self.stoi.get(t, self.stoi[UNK]) for t in tokens]


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_len=100):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        toks = self.tokenizer(self.texts[idx])
        ids = self.vocab.encode(toks)[: self.max_len]
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.float)


def pad_collate(batch, pad_idx):
    ids_list, y_list = zip(*batch)
    lengths = [len(x) for x in ids_list]
    max_len = max(lengths)
    padded = []
    for ids in ids_list:
        pad = torch.full((max_len - len(ids),), pad_idx, dtype=torch.long)
        padded.append(torch.cat([ids, pad]))
    return torch.stack(padded), torch.tensor(y_list), torch.tensor(lengths)


# =========================
# Embeddings
# =========================
def load_embeddings(emb_path):
    ext = Path(emb_path).suffix.lower()
    if ext == ".bin":
        kv = load_facebook_vectors(emb_path)
        return kv, kv.vector_size, True
    kv = KeyedVectors.load_word2vec_format(emb_path, binary=False)
    return kv, kv.vector_size, False


def build_embedding_matrix(vocab, kv, dim, is_ft_bin):
    mat = np.random.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    if PAD in vocab.stoi:
        mat[vocab.stoi[PAD]] = 0.0
    for tok, idx in vocab.stoi.items():
        if tok in (PAD, UNK): continue
        if tok in kv.key_to_index:
            mat[idx] = kv.get_vector(tok)
    return torch.tensor(mat)


# =========================
# Model & Metrics
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, emb_matrix, hidden=256, bidir=True, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(emb_matrix)
        self.embedding.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden * (2 if bidir else 1) * 2, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, ids, lengths):
        emb = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (ids != self.embedding.padding_idx).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        maxp, _ = out.masked_fill(~mask, -1e9).max(1)
        x = torch.cat([mean, maxp], 1)
        return self.fc(self.drop(x)).squeeze(1)


def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc, precision=prec, recall=rec, f1=f1)


# =========================
# Train & Evaluate
# =========================
def run_epoch(model, loader, device, opt=None, criterion=None):
    model.train(opt is not None)
    preds, golds = [], []
    total_loss = 0.0
    for ids, y, lengths in loader:
        ids, y, lengths = ids.to(device), y.to(device), lengths.to(device)
        out = model(ids, lengths)
        loss = criterion(out, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(y)
        preds.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy())
        golds.extend(y.long().cpu().numpy())
    m = metrics(golds, preds)
    return total_loss / len(loader.dataset), m


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--emb_path", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--bidirectional", type=int, default=1)
    p.add_argument("--freeze_embeddings", type=int, default=1)
    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--filtered", action="store_true", help="Write to flstmResults.txt if True")
    args = p.parse_args()

    set_seed(42)

    df_train, df_dev, df_test = read_table(args.train), read_table(args.dev), read_table(args.test)
    for df, name in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
        df["text"] = df["text"].astype(str).map(normalize_text)
        map_labels_inplace(df, name)

    tokenizer = Tokenizer(use_spacy=False)
    tok_train = [tokenizer(t) for t in df_train["text"]]
    vocab = Vocab()
    vocab.build(tok_train)
    kv, dim, isft = load_embeddings(args.emb_path)
    emb = build_embedding_matrix(vocab, kv, dim, isft)

    ds_train = TextDataset(df_train, tokenizer, vocab, args.max_len)
    ds_dev = TextDataset(df_dev, tokenizer, vocab, args.max_len)
    ds_test = TextDataset(df_test, tokenizer, vocab, args.max_len)
    collate = lambda b: pad_collate(b, vocab.stoi[PAD])
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(len(vocab), dim, vocab.stoi[PAD], emb,
                             hidden=args.hidden_size, bidir=bool(args.bidirectional),
                             freeze=bool(args.freeze_embeddings)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_f1, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_m = run_epoch(model, dl_train, device, opt, criterion)
        dv_loss, dv_m = run_epoch(model, dl_dev, device, None, criterion)
        print(f"[epoch {ep:02d}] train F1={tr_m['f1']:.3f} dev F1={dv_m['f1']:.3f}")
        if dv_m["f1"] > best_f1:
            best_f1 = dv_m["f1"]
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    print("\n[info] Evaluating best model...")
    dev_loss, dev_m = run_epoch(model, dl_dev, device, None, criterion)
    test_loss, test_m = run_epoch(model, dl_test, device, None, criterion)

    result_text = (
        "\n========== FINAL DEV RESULTS ==========\n"
        f"Accuracy:  {dev_m['accuracy']:.4f}\nPrecision: {dev_m['precision']:.4f}\n"
        f"Recall:    {dev_m['recall']:.4f}\nF1 Score:  {dev_m['f1']:.4f}\n"
        f"TP={dev_m['tp']} FP={dev_m['fp']} TN={dev_m['tn']} FN={dev_m['fn']}\n"
        "=======================================\n"
        "\n========== FINAL TEST RESULTS ==========\n"
        f"Accuracy:  {test_m['accuracy']:.4f}\nPrecision: {test_m['precision']:.4f}\n"
        f"Recall:    {test_m['recall']:.4f}\nF1 Score:  {test_m['f1']:.4f}\n"
        f"TP={test_m['tp']} FP={test_m['fp']} TN={test_m['tn']} FN={test_m['fn']}\n"
        "=======================================\n"
    )

    out_name = "flstmResults.txt" if args.filtered else "lstmResults.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"[info] Results written to {out_name}")


if __name__ == "__main__":
    main()