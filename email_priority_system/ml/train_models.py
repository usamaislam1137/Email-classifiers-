"""
Train four email priority classification models:

1. Logistic Regression (TF-IDF baseline)
2. Random Forest (full feature matrix)
3. XGBoost (full feature matrix)
4. DistilBERT fine-tuned (transformers Trainer API)

All models are saved to models/ directory.
Training metrics are saved to models/training_results.json.

Usage:
    python train_models.py [--no-bert] [--model {lr,rf,xgb,bert,all}]
"""
from __future__ import annotations

import sys
import json
import time
import pickle
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURES_PKL,
    PROCESSED_CSV,
    MODELS_DIR,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    BERT_MODEL_DIR,
    TFIDF_VECTORIZER_PKL,
    TRAINING_RESULTS_JSON,
    PRIORITY_LABEL_NAMES,
    CV_FOLDS,
    RANDOM_STATE,
    BERT_FINETUNE_EPOCHS,
    BERT_FINETUNE_BATCH_SIZE,
    BERT_FINETUNE_LR,
    BERT_FINETUNE_MAX_STEPS,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    LOG_FILE,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -- Helpers -------------------------------------------------------------------

def _apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """Apply SMOTE oversampling. Handles sparse and dense matrices."""
    if issparse(X):
        X = X.toarray()
    counts = np.bincount(y)
    min_samples = counts.min()
    k = min(5, min_samples - 1)
    if k < 1:
        log.warning("Too few samples for SMOTE (min class=%d). Skipping.", min_samples)
        return X, y
    try:
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
        log.info("SMOTE: %d -> %d samples.", len(y), len(y_res))
        return X_res, y_res
    except Exception as exc:
        log.warning("SMOTE failed (%s). Using original data.", exc)
        return X, y


def _cv_scores(model, X, y, n_splits: int = CV_FOLDS) -> dict:
    """Run stratified k-fold CV and return mean scores."""
    if issparse(X):
        X = X.toarray()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "cv_accuracy_mean": float(np.mean(results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(results["test_f1_macro"])),
    }


def _train_test_split_stratified(X, y, test_size: float = 0.2):
    """Stratified train/test split."""
    from sklearn.model_selection import train_test_split
    if issparse(X):
        X = X.toarray()
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def _eval_metrics(y_true, y_pred) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred,
        target_names=PRIORITY_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {"accuracy": acc, "macro_f1": f1, "classification_report": report}


# -- Model 1: Logistic Regression + TF-IDF ------------------------------------

def train_logistic_regression(features: dict) -> dict:
    log.info("=== Training Logistic Regression (TF-IDF baseline) ===")
    X = features["X_tfidf"]
    y = features["y"]

    X_train, X_test, y_train, y_test = _train_test_split_stratified(X, y)
    X_train, y_train = _apply_smote(X_train, y_train)

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log.info("LR trained in %.1fs", train_time)

    y_pred = model.predict(X_test)
    metrics = _eval_metrics(y_test, y_pred)
    log.info("LR  accuracy=%.4f  macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    cv_scores = _cv_scores(model, X, y)
    log.info("LR CV accuracy=%.4f +/- %.4f", cv_scores["cv_accuracy_mean"], cv_scores["cv_accuracy_std"])

    joblib.dump(model, LR_MODEL_PATH)
    log.info("LR model saved to %s", LR_MODEL_PATH)

    return {
        "model_name": "logistic_regression",
        "train_time_s": round(train_time, 2),
        **metrics,
        **cv_scores,
    }


# -- Model 2: Random Forest ----------------------------------------------------

def train_random_forest(features: dict) -> dict:
    log.info("=== Training Random Forest (full feature matrix) ===")
    X = features["X_combined"]
    y = features["y"]

    X_train, X_test, y_train, y_test = _train_test_split_stratified(X, y)
    X_train, y_train = _apply_smote(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log.info("RF trained in %.1fs", train_time)

    y_pred = model.predict(X_test)
    metrics = _eval_metrics(y_test, y_pred)
    log.info("RF  accuracy=%.4f  macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    cv_scores = _cv_scores(model, X, y)

    joblib.dump(model, RF_MODEL_PATH)
    log.info("RF model saved to %s", RF_MODEL_PATH)

    return {
        "model_name": "random_forest",
        "train_time_s": round(train_time, 2),
        **metrics,
        **cv_scores,
    }


# -- Model 3: XGBoost ---------------------------------------------------------

def train_xgboost(features: dict) -> dict:
    log.info("=== Training XGBoost (full feature matrix) ===")
    X = features["X_combined"]
    y = features["y"]

    X_train, X_test, y_train, y_test = _train_test_split_stratified(X, y)
    X_train, y_train = _apply_smote(X_train, y_train)

    # Class weights for imbalance
    from collections import Counter
    counts = Counter(y_train)
    total = len(y_train)
    scale_pos = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - t0
    log.info("XGB trained in %.1fs", train_time)

    y_pred = model.predict(X_test)
    metrics = _eval_metrics(y_test, y_pred)
    log.info("XGB accuracy=%.4f  macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])

    cv_scores = _cv_scores(model, X, y)

    joblib.dump(model, XGB_MODEL_PATH)
    log.info("XGB model saved to %s", XGB_MODEL_PATH)

    return {
        "model_name": "xgboost",
        "train_time_s": round(train_time, 2),
        **metrics,
        **cv_scores,
    }


# -- Model 4: DistilBERT fine-tuned --------------------------------------------

def train_distilbert(df: pd.DataFrame) -> dict:
    """Fine-tune DistilBERT using the Transformers Trainer API."""
    log.info("=== Fine-tuning DistilBERT ===")
    try:
        import torch
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from datasets import Dataset
        from sklearn.model_selection import train_test_split as sk_split

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Using device: %s", device)

        tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

        # Prepare texts and labels
        texts = (df["subject"].fillna("") + " [SEP] " + df["body"].fillna("").str[:256]).tolist()
        labels = df["priority"].astype(int).tolist()

        train_texts, val_texts, train_labels, val_labels = sk_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=RANDOM_STATE
        )

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=BERT_MAX_LENGTH,
            )

        train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        model = DistilBertForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=4,
            id2label={i: l for i, l in enumerate(PRIORITY_LABEL_NAMES)},
            label2id={l: i for i, l in enumerate(PRIORITY_LABEL_NAMES)},
        )

        BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(BERT_MODEL_DIR),
            num_train_epochs=BERT_FINETUNE_EPOCHS,
            per_device_train_batch_size=BERT_FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=BERT_FINETUNE_BATCH_SIZE,
            learning_rate=BERT_FINETUNE_LR,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            max_steps=BERT_FINETUNE_MAX_STEPS,
            logging_steps=50,
            save_total_limit=2,
            report_to="none",
            no_cuda=(device == "cpu"),
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        t0 = time.time()
        train_result = trainer.train()
        train_time = time.time() - t0
        log.info("DistilBERT fine-tuning complete in %.1fs", train_time)

        eval_result = trainer.evaluate()
        log.info("DistilBERT eval: %s", eval_result)

        trainer.save_model(str(BERT_MODEL_DIR))
        tokenizer.save_pretrained(str(BERT_MODEL_DIR))
        log.info("DistilBERT saved to %s", BERT_MODEL_DIR)

        return {
            "model_name": "distilbert",
            "train_time_s": round(train_time, 2),
            "accuracy": float(eval_result.get("eval_accuracy", 0)),
            "macro_f1": float(eval_result.get("eval_f1_macro", 0)),
            "eval_loss": float(eval_result.get("eval_loss", 0)),
        }

    except ImportError as exc:
        log.warning("DistilBERT training skipped - missing dependency: %s", exc)
        return {
            "model_name": "distilbert",
            "skipped": True,
            "reason": str(exc),
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }
    except Exception as exc:
        log.error("DistilBERT training failed: %s", exc, exc_info=True)
        return {
            "model_name": "distilbert",
            "skipped": True,
            "reason": str(exc),
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }


# -- Main training pipeline ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train email priority classification models.")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "xgb", "bert", "all"],
        default="all",
        help="Which model to train.",
    )
    parser.add_argument("--no-bert", action="store_true", help="Skip DistilBERT fine-tuning.")
    args = parser.parse_args()

    # -- Load features ---------------------------------------------------------
    if not FEATURES_PKL.exists():
        log.error("features.pkl not found. Run feature_engineering.py first.")
        sys.exit(1)

    log.info("Loading features from %s ...", FEATURES_PKL)
    with open(FEATURES_PKL, "rb") as fh:
        features = pickle.load(fh)
    log.info("Feature matrix X_combined: %s", features["X_combined"].shape)

    # Load processed CSV for DistilBERT (needs raw text)
    df = None
    if (args.model in ("bert", "all")) and not args.no_bert:
        if PROCESSED_CSV.exists():
            df = pd.read_csv(PROCESSED_CSV)
        else:
            log.warning("Processed CSV not found; skipping DistilBERT.")

    results = {}
    train_which = args.model

    if train_which in ("lr", "all"):
        results["logistic_regression"] = train_logistic_regression(features)

    if train_which in ("rf", "all"):
        results["random_forest"] = train_random_forest(features)

    if train_which in ("xgb", "all"):
        results["xgboost"] = train_xgboost(features)

    if train_which in ("bert", "all") and not args.no_bert and df is not None:
        results["distilbert"] = train_distilbert(df)

    # -- Save training results -------------------------------------------------
    import datetime
    output = {
        "training_date": datetime.datetime.now().isoformat(),
        "n_samples": int(features["X_combined"].shape[0]),
        "n_features": int(features["X_combined"].shape[1]),
        "models": results,
    }

    with open(TRAINING_RESULTS_JSON, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    log.info("Training results saved to %s", TRAINING_RESULTS_JSON)

    # Summary
    log.info("\n=== Training Summary ===")
    for name, res in results.items():
        acc = res.get("accuracy", 0)
        f1 = res.get("macro_f1", 0)
        skipped = res.get("skipped", False)
        status = "SKIPPED" if skipped else f"accuracy={acc:.4f}  macro_f1={f1:.4f}"
        log.info("  %-25s  %s", name, status)


if __name__ == "__main__":
    main()
