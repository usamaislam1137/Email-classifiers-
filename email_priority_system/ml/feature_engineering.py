"""
Feature engineering for the Email Priority Classification System.

Extracted features
------------------
Metadata (13):
    sender_domain, recipient_count, has_cc, has_bcc,
    hour_of_day, day_of_week, is_weekend,
    subject_length, body_length, word_count,
    is_reply, is_forward, forward_count

Urgency (4):
    urgency_keyword_count, has_deadline, has_question, exclamation_count

Structural (5):
    subject_has_re, subject_has_fw, thread_depth,
    attachment_count (proxy from body markers)

TF-IDF (5000):
    Combined subject + body, unigrams & bigrams

BERT embeddings (768):
    DistilBERT [CLS] token from subject + first 256 chars of body

Output
------
features.pkl - dict with keys:
    X_tfidf        : sparse matrix (n_samples, max 5000)
    X_meta         : dense array   (n_samples, n_meta_features)
    X_bert         : dense array   (n_samples, 768) or 0 if use_bert=False
    X_combined     : dense array   (n_samples, n_meta + n_tfidf [+ 768 if BERT])
    y              : array         (n_samples,)
    feature_names  : list[str]
    tfidf_vectorizer: fitted TfidfVectorizer
"""
from __future__ import annotations

import sys
import re
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_CSV,
    FEATURES_PKL,
    TFIDF_VECTORIZER_PKL,
    TFIDF_MAX_FEATURES,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EMBEDDING_DIM,
    CRITICAL_KEYWORDS,
    HIGH_KEYWORDS,
    LOG_FILE,
    LOG_LEVEL,
)

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

ALL_URGENCY_KEYWORDS = CRITICAL_KEYWORDS + HIGH_KEYWORDS


# -- Metadata features ---------------------------------------------------------

def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dense DataFrame of hand-crafted metadata features."""
    feats = pd.DataFrame(index=df.index)

    # Sender domain
    def _domain(sender: str) -> str:
        match = re.search(r"@([\w.\-]+)", str(sender))
        return match.group(1).lower() if match else "unknown"

    feats["sender_domain_hash"] = (
        df["sender"].apply(_domain).apply(lambda d: hash(d) % (2**15))
    )
    feats["recipient_count"] = df.get("recipient_count", pd.Series(0, index=df.index))
    feats["has_cc"] = df.get("has_cc", pd.Series(0, index=df.index))
    feats["has_bcc"] = df.get("has_bcc", pd.Series(0, index=df.index))

    # Time features
    feats["hour_of_day"] = pd.to_numeric(df.get("hour_of_day", -1), errors="coerce").fillna(-1)
    feats["day_of_week"] = pd.to_numeric(df.get("day_of_week", -1), errors="coerce").fillna(-1)
    feats["is_weekend"] = feats["day_of_week"].apply(lambda d: int(d in (5, 6)) if d >= 0 else 0)

    feats["subject_length"] = df["subject"].str.len().fillna(0)
    feats["body_length"] = df["body"].str.len().fillna(0)
    feats["word_count"] = df["body"].str.split().str.len().fillna(0)

    feats["subject_has_re"] = df.get("subject_has_re",
        df["subject"].str.lower().str.startswith("re:").astype(int))
    feats["subject_has_fw"] = df.get("subject_has_fw",
        (df["subject"].str.lower().str.startswith("fw:")
         | df["subject"].str.lower().str.startswith("fwd:")).astype(int))
    feats["is_reply"] = feats["subject_has_re"]
    feats["is_forward"] = feats["subject_has_fw"]
    feats["forward_count"] = df["subject"].apply(
        lambda s: len(re.findall(r"fw[d]?:", str(s).lower()))
    )

    return feats.astype(float)


def extract_urgency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return urgency-signal features."""
    feats = pd.DataFrame(index=df.index)

    def _text(row):
        return (str(row.get("subject", "")) + " " + str(row.get("body", ""))[:500]).lower()

    texts = df.apply(_text, axis=1)

    feats["urgency_keyword_count"] = texts.apply(
        lambda t: sum(1 for kw in ALL_URGENCY_KEYWORDS if kw in t)
    )
    feats["has_deadline"] = texts.str.contains(
        r"\b(?:deadline|due date|by \w+day|by eod|by end of)\b", regex=True
    ).astype(int)
    feats["has_question"] = df["body"].apply(
        lambda b: int("?" in str(b))
    )
    feats["exclamation_count"] = df["body"].apply(
        lambda b: str(b).count("!")
    )
    feats["thread_depth"] = df["subject"].apply(
        lambda s: max(len(re.findall(r"re\s*:", str(s).lower())),
                      len(re.findall(r"fw[d]?\s*:", str(s).lower())))
    )
    feats["attachment_count"] = df["body"].apply(
        lambda b: len(re.findall(
            r"(attachment|attached|please find|enclosed)", str(b).lower()
        ))
    )

    return feats.astype(float)


# -- TF-IDF --------------------------------------------------------------------

def build_tfidf_features(
    df: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
    fit: bool = True,
) -> tuple:
    """
    Return (sparse_matrix, fitted_vectorizer).

    If *vectorizer* is None and fit=True, a new vectorizer is created and fitted.
    If *vectorizer* is provided, fit is ignored (transform only).
    """
    texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).tolist()

    if vectorizer is None and fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
            min_df=2,
            max_df=0.95,
        )
        X_tfidf = vectorizer.fit_transform(texts)
        log.info("TF-IDF fitted: %d features.", X_tfidf.shape[1])
    else:
        if vectorizer is None:
            raise ValueError("A fitted vectorizer must be provided when fit=False.")
        X_tfidf = vectorizer.transform(texts)
        log.info("TF-IDF transformed: %d features.", X_tfidf.shape[1])

    return X_tfidf, vectorizer


# -- BERT embeddings -----------------------------------------------------------

def build_bert_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Compute DistilBERT [CLS] embeddings for each email.

    Uses the subject + first 256 chars of body as input text.
    Falls back to zero vectors if torch/transformers unavailable.
    """
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading DistilBERT tokenizer and model on %s ...", device)

        tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
        model.eval()
        model.to(device)

        texts = (
            df["subject"].fillna("") + " [SEP] " + df["body"].fillna("").str[:256]
        ).tolist()

        embeddings = []
        batch_size = BERT_BATCH_SIZE

        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            # [CLS] token is the first token of the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

        result = np.vstack(embeddings)
        log.info("BERT embeddings shape: %s", result.shape)
        return result

    except ImportError:
        log.warning("torch/transformers not available - using zero BERT embeddings.")
        return np.zeros((len(df), BERT_EMBEDDING_DIM), dtype=np.float32)
    except Exception as exc:
        log.warning("BERT embedding failed (%s) - using zero embeddings.", exc)
        return np.zeros((len(df), BERT_EMBEDDING_DIM), dtype=np.float32)


# -- Combined feature matrix ---------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, use_bert: bool = True) -> dict:
    """
    Build the full feature matrix from a processed email DataFrame.

    Returns a dict suitable for saving to features.pkl.
    """
    log.info("Extracting metadata features ...")
    meta_df = extract_metadata_features(df)
    urgency_df = extract_urgency_features(df)
    meta_combined = pd.concat([meta_df, urgency_df], axis=1)
    X_meta = meta_combined.values.astype(np.float32)
    meta_feature_names = meta_combined.columns.tolist()
    log.info("Metadata features: %d", X_meta.shape[1])

    log.info("Building TF-IDF features ...")
    X_tfidf, vectorizer = build_tfidf_features(df, fit=True)
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()

    if use_bert:
        log.info("Building BERT embeddings ...")
        X_bert = build_bert_embeddings(df)
    else:
        log.info("Skipping BERT embeddings (use_bert=False) — train on meta + TF-IDF only (no BERT block).")
        X_bert = np.zeros((len(df), 0), dtype=np.float32)

    bert_feature_names = [f"bert_{i}" for i in range(X_bert.shape[1])]

    # Combined: meta + tfidf + (optional) bert (dense)
    from scipy.sparse import issparse
    X_tfidf_dense = X_tfidf.toarray() if issparse(X_tfidf) else X_tfidf
    if X_bert.shape[1] > 0:
        X_combined = np.hstack([X_meta, X_tfidf_dense, X_bert]).astype(np.float32)
    else:
        X_combined = np.hstack([X_meta, X_tfidf_dense]).astype(np.float32)
    feature_names = meta_feature_names + tfidf_feature_names + bert_feature_names

    y = df["priority"].values.astype(int)

    log.info("Combined feature matrix: %s", X_combined.shape)

    return {
        "X_tfidf": X_tfidf,
        "X_meta": X_meta,
        "X_bert": X_bert,
        "X_combined": X_combined,
        "y": y,
        "feature_names": feature_names,
        "meta_feature_names": meta_feature_names,
        "tfidf_feature_names": tfidf_feature_names,
        "bert_feature_names": bert_feature_names,
        "tfidf_vectorizer": vectorizer,
        "n_meta": X_meta.shape[1],
        "n_tfidf": X_tfidf.shape[1],
        "n_bert": X_bert.shape[1],
    }


# -- Shipped sklearn models: 16 meta+urgency + TF-IDF (16 + 1444 = 1460) ------
# Current extract produces 15 meta + 6 urgency; we use 10 "base" meta + 6 urgency
# (excludes re/fw / reply/forward/forward_count features that were not in the training run).


def _meta_urg_21_to_16(row21: np.ndarray) -> np.ndarray:
    r = np.asarray(row21, dtype=np.float32).ravel()
    if r.size < 21:
        return r
    return np.concatenate([r[:10], r[15:21]]).astype(np.float32)


def trim_to_sklearn_meta_tfidf(
    x_combined: np.ndarray,
    n_tfidf: int,
) -> np.ndarray:
    """
    If x_combined is [21 + n_tfidf] (new layout), project to [16 + n_tfidf] to match
    shipped .pkl classifiers. If already 16+n_tfidf, return as-is.
    """
    x = np.asarray(x_combined, dtype=np.float32).ravel()
    d = x.size
    if d == 16 + n_tfidf:
        return x
    if d == 21 + n_tfidf:
        m = x[:21]
        t = x[21:]
        m16 = _meta_urg_21_to_16(m)
        return np.concatenate([m16, t]).astype(np.float32)
    return x


def trim_feature_matrix_to_sklearn(
    X: np.ndarray, n_tfidf: int, n_bert: int = 0,
) -> np.ndarray:
    """Batch form: trim 21→16 meta columns; keep TF-IDF and optional BERT tail."""
    X = np.asarray(X, dtype=np.float32)
    if n_bert:
        if X.shape[1] == 16 + n_tfidf + n_bert:
            return X
        if X.shape[1] == 21 + n_tfidf + n_bert:
            m, rest = X[:, :21], X[:, 21:]
            m16 = np.hstack([m[:, :10], m[:, 15:21]])
            return np.hstack([m16, rest]).astype(np.float32)
        return X
    if X.shape[1] == 16 + n_tfidf:
        return X
    if X.shape[1] == 21 + n_tfidf:
        m, t = X[:, :21], X[:, 21:]
        m16 = np.hstack([m[:, :10], m[:, 15:21]])
        return np.hstack([m16, t]).astype(np.float32)
    return X


# -- Single-email feature extraction (for inference) --------------------------

def extract_single_email_features(
    email_dict: dict,
    vectorizer: TfidfVectorizer,
    use_bert: bool = True,
) -> np.ndarray:
    """
    Extract features for a single email dict, compatible with the trained models.

    Args:
        email_dict: keys: sender, recipients, subject, body, date, cc, bcc, etc.
        vectorizer: fitted TfidfVectorizer
        use_bert: whether to compute BERT embeddings

    Returns:
        1D numpy array of features
    """
    df = pd.DataFrame([email_dict])

    # Ensure required columns exist
    for col in ["sender", "recipients", "subject", "body", "cc", "bcc"]:
        if col not in df.columns:
            df[col] = ""

    # Parse date if needed
    if "hour_of_day" not in df.columns:
        try:
            from dateutil import parser as dp
            dt = dp.parse(str(email_dict.get("date", "")))
            df["hour_of_day"] = dt.hour
            df["day_of_week"] = dt.weekday()
        except Exception:
            df["hour_of_day"] = -1
            df["day_of_week"] = -1

    if "recipient_count" not in df.columns:
        df["recipient_count"] = df["recipients"].apply(
            lambda x: len(str(x).split(",")) if x else 0
        )
    if "has_cc" not in df.columns:
        df["has_cc"] = df["cc"].apply(lambda x: int(bool(x and str(x).strip())))
    if "has_bcc" not in df.columns:
        df["has_bcc"] = df["bcc"].apply(lambda x: int(bool(x and str(x).strip())))
    if "subject_has_re" not in df.columns:
        df["subject_has_re"] = df["subject"].str.lower().str.startswith("re:").astype(int)
    if "subject_has_fw" not in df.columns:
        df["subject_has_fw"] = (
            df["subject"].str.lower().str.startswith("fw:")
            | df["subject"].str.lower().str.startswith("fwd:")
        ).astype(int)

    meta_df = extract_metadata_features(df)
    urgency_df = extract_urgency_features(df)
    meta_combined = pd.concat([meta_df, urgency_df], axis=1)
    X_meta = meta_combined.values.astype(np.float32)

    X_tfidf, _ = build_tfidf_features(df, vectorizer=vectorizer, fit=False)
    X_tfidf_dense = X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else X_tfidf

    if use_bert:
        X_bert = build_bert_embeddings(df)
    else:
        # Match disk models: [meta, TF-IDF] only (no 768-dim BERT padding).
        X_bert = np.zeros((1, 0), dtype=np.float32)

    if X_bert.shape[1] > 0:
        X_combined = np.hstack([X_meta, X_tfidf_dense, X_bert]).astype(np.float32)
    else:
        X_combined = np.hstack([X_meta, X_tfidf_dense]).astype(np.float32)
    return X_combined[0]


def feature_names_for_meta_tfidf(vectorizer) -> list[str]:
    """
    Ordered feature names (16 + n_tfidf) matching shipped sklearn / trim_to_sklearn_meta_tfidf.
    """
    df0 = pd.DataFrame(
        [
            {
                "sender": "a@b.com",
                "recipients": "c@d.com",
                "subject": "subj",
                "body": "body",
                "cc": "",
                "bcc": "",
                "hour_of_day": 12,
                "day_of_week": 1,
                "recipient_count": 1,
                "has_cc": 0,
                "has_bcc": 0,
                "subject_has_re": 0,
                "subject_has_fw": 0,
            }
        ]
    )
    meta_c = extract_metadata_features(df0)
    ur_c = extract_urgency_features(df0)
    full = pd.concat([meta_c, ur_c], axis=1)
    n21 = full.columns.tolist()
    n16 = n21[0:10] + n21[15:21]
    return n16 + list(vectorizer.get_feature_names_out())


# -- Persistence ---------------------------------------------------------------

def save_features(feature_dict: dict, path: Path = FEATURES_PKL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(feature_dict, fh, protocol=4)
    log.info("Features saved to %s", path)


def load_features(path: Path = FEATURES_PKL) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# -- CLI entry point -----------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix from processed emails.")
    parser.add_argument("--input", type=Path, default=PROCESSED_CSV)
    parser.add_argument("--output", type=Path, default=FEATURES_PKL)
    parser.add_argument("--no-bert", action="store_true", help="Skip BERT embeddings.")
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Processed CSV not found at %s. Run preprocess.py first.", args.input)
        sys.exit(1)

    log.info("Loading processed emails from %s ...", args.input)
    df = pd.read_csv(args.input)
    log.info("Loaded %d emails.", len(df))

    feature_dict = build_feature_matrix(df, use_bert=not args.no_bert)

    # Also save the TF-IDF vectorizer separately
    with open(TFIDF_VECTORIZER_PKL, "wb") as fh:
        pickle.dump(feature_dict["tfidf_vectorizer"], fh, protocol=4)
    log.info("TF-IDF vectorizer saved to %s", TFIDF_VECTORIZER_PKL)

    save_features(feature_dict, args.output)
    log.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
