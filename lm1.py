import re
import argparse
from pathlib import Path
import math
import numpy as np

from joblib import dump
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

# ---------- cleaning ----------
url_pat = re.compile(r'https?://\S+|www\.\S+')
mention_pat = re.compile(r'@\w+')
hashtag_pat = re.compile(r'#')
emoji_like = re.compile(r'[\u2600-\u27BF\u1F300-\u1F6FF\u1F900-\u1F9FF]')
long_repeat = re.compile(r'(.)\1{2,}')  # sooooo -> soo

def clean_tweet(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = url_pat.sub(' URL ', t)
    t = mention_pat.sub(' USER ', t)
    t = hashtag_pat.sub(' HASHTAG ', t)   # keep cue
    t = emoji_like.sub(' EMOJI ', t)      # keep cue
    t = long_repeat.sub(r'\1\1', t)       # de-elongate
    return t.strip()

# ---------- numeric feature transformer ----------
class NumericFeats(BaseEstimator, TransformerMixin):
    """
    Extract simple numeric features from raw texts:
      - log(n_chars), log(n_words), avg_word_len
      - log(! count), log(? count)
      - caps_ratio, digit_ratio
      - log(URL/USER/HASHTAG/EMOJI counts)
      - log(lexicon hits) if a lexicon is provided (one term per line, lowercased)
    Returns CSR sparse matrix to combine with TF-IDF in FeatureUnion.
    """
    def __init__(self, lexicon_path=None):
        self.lexicon_path = lexicon_path
        self.lexicon = None

    def fit(self, X, y=None):
        if self.lexicon_path:
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = set(w.strip().lower() for w in f if w.strip())
        else:
            self.lexicon = set()
        return self

    def transform(self, X):
        rows = []
        for s in X:
            s = s if isinstance(s, str) else ""
            n_chars = len(s)
            words = s.split()
            n_words = len(words)
            avg_word_len = (n_chars / max(n_words, 1))

            exclam = s.count('!')
            question = s.count('?')

            caps = sum(1 for c in s if c.isupper())
            caps_ratio = caps / max(n_chars, 1)

            digits = sum(1 for c in s if c.isdigit())
            digit_ratio = digits / max(n_chars, 1)

            url_c = s.count('URL')
            user_c = s.count('USER')
            hash_c = s.count('HASHTAG')
            emoji_c = s.count('EMOJI')

            lex_hits = 0
            if self.lexicon:
                low = s.lower()
                # simple substring presence
                lex_hits = sum(1 for w in self.lexicon if w in low)

            vec = [
                math.log1p(n_chars),
                math.log1p(n_words),
                avg_word_len,
                math.log1p(exclam),
                math.log1p(question),
                caps_ratio,
                digit_ratio,
                math.log1p(url_c),
                math.log1p(user_c),
                math.log1p(hash_c),
                math.log1p(emoji_c),
                math.log1p(lex_hits),
            ]
            rows.append(vec)

        return csr_matrix(np.asarray(rows, dtype=float))

# ---------- robust loader (splits on LAST '|') ----------
def load_split(path: Path):
    msgs, labs = [], []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if '|' not in header or 'message' not in header.lower() or 'label' not in header.lower():
            # header missing or malformed -> treat first line as data
            f.seek(0)
        for raw in f:
            line = raw.rstrip("\n")
            if not line or '|' not in line:
                continue
            msg, lab = line.rsplit('|', 1)
            msg = clean_tweet(msg)
            lab = lab.strip()
            if lab not in {"OFF", "NOT", "off", "not", "0", "1"}:
                continue
            labs.append({'OFF':1, 'NOT':0, 'off':1, 'not':0, '1':1, '0':0}[lab])
            msgs.append(msg)
    if not msgs:
        raise ValueError(f"No valid rows parsed from {path}")
    return msgs, np.array(labs, dtype=int)

def bincount_dict(arr):
    b = np.bincount(arr, minlength=2)
    return {'NOT(0)': int(b[0]), 'OFF(1)': int(b[1])}

def build_pipeline(args):
    vecs = []

    if args.use_word:
        vecs.append((
            'word',
            TfidfVectorizer(
                analyzer='word',
                ngram_range=(args.word_ngram_min, args.word_ngram_max),
                min_df=args.word_min_df,
                max_df=args.word_max_df,
                binary=bool(args.word_binary),
                sublinear_tf=bool(args.word_sublinear),
                lowercase=bool(args.word_lowercase)
            )
        ))

    if args.use_char:
        vecs.append((
            'char',
            TfidfVectorizer(
                analyzer=args.char_analyzer,  # 'char' or 'char_wb'
                ngram_range=(args.char_ngram_min, args.char_ngram_max),
                min_df=args.char_min_df,
                binary=bool(args.char_binary),
                sublinear_tf=bool(args.char_sublinear),
                lowercase=bool(args.char_lowercase)
            )
        ))

    if args.use_numeric:
        vecs.append((
            'num',
            Pipeline([
                ('num', NumericFeats(lexicon_path=args.lexicon)),
                ('scale', MaxAbsScaler())
            ])
        ))

    if not vecs:
        raise ValueError("At least one of --use_word, --use_char, --use_numeric must be enabled.")

    features = FeatureUnion(vecs)

    cw = 'balanced' if args.class_weight == 'balanced' else None
    svm = LinearSVC(C=args.C, class_weight=cw, max_iter=args.max_iter)

    pipe = Pipeline([
        ('features', features),
        ('svm', svm)
    ])
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='Path to train.pipe (message|label)')
    ap.add_argument('--dev', required=True, help='Path to dev.pipe (message|label)')
    ap.add_argument('--out_model', default='svm_offense.joblib')

    # word TF-IDF
    ap.add_argument('--use_word', type=int, default=1, help='1=use word TF-IDF, 0=off')
    ap.add_argument('--word_ngram_min', type=int, default=1)
    ap.add_argument('--word_ngram_max', type=int, default=2)
    ap.add_argument('--word_min_df', type=int, default=2)
    ap.add_argument('--word_max_df', type=float, default=0.95)
    ap.add_argument('--word_binary', type=int, default=0)
    ap.add_argument('--word_sublinear', type=int, default=1)
    ap.add_argument('--word_lowercase', type=int, default=1)

    # char TF-IDF
    ap.add_argument('--use_char', type=int, default=1, help='1=use char TF-IDF, 0=off')
    ap.add_argument('--char_analyzer', default='char_wb', choices=['char', 'char_wb'])
    ap.add_argument('--char_ngram_min', type=int, default=3)
    ap.add_argument('--char_ngram_max', type=int, default=5)
    ap.add_argument('--char_min_df', type=int, default=2)
    ap.add_argument('--char_binary', type=int, default=0)
    ap.add_argument('--char_sublinear', type=int, default=1)
    ap.add_argument('--char_lowercase', type=int, default=1)

    # numeric feats
    ap.add_argument('--use_numeric', type=int, default=1, help='1=add numeric features, 0=off')
    ap.add_argument('--lexicon', default=None, help='Optional offensive terms file (one per line)')

    # SVM
    ap.add_argument('--C', type=float, default=0.5)
    ap.add_argument('--class_weight', default='balanced', choices=['balanced', 'none'])
    ap.add_argument('--max_iter', type=int, default=10000)

    args = ap.parse_args()

    # load data
    train_path, dev_path = Path(args.train), Path(args.dev)
    X_train, y_train = load_split(train_path)
    X_dev, y_dev = load_split(dev_path)
    print("Train class counts:", bincount_dict(y_train))
    print("Dev   class counts:", bincount_dict(y_dev))

    # build + train
    pipe = build_pipeline(args)
    pipe.fit(X_train, y_train)

    # eval
    y_pred = pipe.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    macro_f1 = f1_score(y_dev, y_pred, average='macro')
    weighted_f1 = f1_score(y_dev, y_pred, average='weighted')
    per_p, per_r, per_f1, per_s = precision_recall_fscore_support(
        y_dev, y_pred, labels=[0,1], zero_division=0
    )
    cm = confusion_matrix(y_dev, y_pred, labels=[0,1])

    try:
        scores = pipe.decision_function(X_dev)
        roc_auc = roc_auc_score(y_dev, scores)
    except Exception:
        roc_auc = None

    print("\n=== DEV RESULTS ===")
    print(classification_report(y_dev, y_pred, target_names=['NOT','OFF'], zero_division=0))
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}   (primary)")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:     {roc_auc:.4f}")
    print("\nPer-class (P/R/F1/supp):")
    print(f"NOT (0): P={per_p[0]:.4f} R={per_r[0]:.4f} F1={per_f1[0]:.4f} supp={per_s[0]}")
    print(f"OFF (1): P={per_p[1]:.4f} R={per_r[1]:.4f} F1={per_f1[1]:.4f} supp={per_s[1]}")
    print("\nConfusion matrix [rows=true, cols=pred] (NOT=0, OFF=1):")
    print(cm)

    dump(pipe, args.out_model)
    print(f"\nModel saved to: {args.out_model}")

if __name__ == '__main__':
    main()