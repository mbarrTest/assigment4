#!/usr/bin/env python3
# lm1.py — SVM baseline (train + dev) for columns: message|label (PIPE-separated)

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# ---------------------------
# Simple Tweet Preprocessing
# ---------------------------
url_pat = re.compile(r'https?://\S+|www\.\S+')
mention_pat = re.compile(r'@\w+')
hashtag_pat = re.compile(r'#')
emoji_like = re.compile(r'[\u2600-\u27BF\u1F300-\u1F6FF\u1F900-\u1F9FF]')

def clean_tweet(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = url_pat.sub(' URL ', t)
    t = mention_pat.sub(' USER ', t)
    t = hashtag_pat.sub('', t)            # keep token, drop '#'
    t = emoji_like.sub(' EMOJI ', t)      # coarse placeholder
    return t.strip()

# ---------------------------
# Data Loading (PIPE-separated)
# ---------------------------
def load_split(path: Path):
    """
    Load a PIPE-separated file with header: 'message|label'.
    Labels expected: OFF or NOT (case-insensitive also OK).
    """
    df = pd.read_csv(
        path,
        sep="|",                  # <-- PIPE-separated
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn"
    )

    # Validate columns
    expected_cols = {"message", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"{path} must have columns 'message' and 'label'. Found: {list(df.columns)}"
        )

    # Clean text
    df["message"] = df["message"].astype(str).map(clean_tweet)

    # Map labels to {NOT:0, OFF:1}
    label_map = {'OFF': 1, 'NOT': 0, 'off': 1, 'not': 0, 1: 1, 0: 0}
    y = df["label"].map(lambda v: label_map.get(v, v))

    if y.isnull().any():
        bad = df.loc[y.isnull(), "label"].value_counts()
        raise ValueError(
            "Unrecognized labels. Expected {OFF, NOT} (or 1/0). "
            f"Problematic values:\n{bad}"
        )

    X = df["message"].tolist()
    y = y.astype(int)
    return X, y

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='Path to train.pipe (message|label)')
    ap.add_argument('--dev', required=True, help='Path to dev.pipe (message|label)')
    ap.add_argument('--out_model', default='svm_offense.joblib')
    args = ap.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)

    # Load data
    X_train, y_train = load_split(train_path)
    X_dev, y_dev = load_split(dev_path)

    # Class weights for imbalance (explicit 0/1 classes)
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): w for c, w in zip(classes, weights)}

    # Pipeline: TF-IDF word uni/bi-grams + Linear SVM
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True
        )),
        ('svm', LinearSVC(
            C=1.0,
            class_weight=class_weight,
            max_iter=5000
        ))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Dev evaluation
    y_pred = pipe.predict(X_dev)
    print("\n=== DEV RESULTS ===")
    print(classification_report(y_dev, y_pred, target_names=['NOT', 'OFF']))
    print("Macro-F1:", f1_score(y_dev, y_pred, average='macro'))
    print("Confusion matrix:\n", confusion_matrix(y_dev, y_pred))

    # Save model for later tuning
    dump(pipe, args.out_model)
    print(f"\nModel saved to: {args.out_model}")

if __name__ == '__main__':
    main()
