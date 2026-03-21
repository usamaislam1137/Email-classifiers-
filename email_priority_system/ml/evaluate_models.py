"""
Evaluate all trained models, generate SHAP explanations, and determine the best model.

Outputs:
    models/evaluation_report.json  - full evaluation metrics + best model decision
    models/best_model.txt           - name of the best model

Usage:
    python evaluate_models.py
"""
from __future__ import annotations

import sys
import json
import pickle
import logging
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
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
    EVALUATION_REPORT_JSON,
    BEST_MODEL_FILE,
    PRIORITY_LABEL_NAMES,
    ACCURACY_THRESHOLD,
    MACRO_F1_THRESHOLD,
    RANDOM_STATE,
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


# -- Helpers -------------------------------------------------------------------

def _safe_load(path: Path):
    if not path.exists():
        log.warning("Model file not found: %s", path)
        return None
    return joblib.load(path)


def _full_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()
    report = classification_report(
        y_true, y_pred, target_names=PRIORITY_LABEL_NAMES, output_dict=True, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(PRIORITY_LABEL_NAMES):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return result


# -- SHAP explanations ---------------------------------------------------------

def compute_shap_xgb(model, X_sample: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP values for XGBoost using TreeExplainer."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            mean_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        top_indices = np.argsort(mean_abs)[::-1][:20]
        top_features = {
            feature_names[i]: float(mean_abs[i])
            for i in top_indices
            if i < len(feature_names)
        }
        return {"top_features": top_features, "method": "TreeExplainer"}
    except Exception as exc:
        log.warning("SHAP XGB failed: %s", exc)
        return {"error": str(exc)}


def compute_shap_lr(model, X_sample: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP values for Logistic Regression using LinearExplainer."""
    try:
        import shap
        masker = shap.maskers.Independent(X_sample, max_samples=100)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_sample[:200])

        if isinstance(shap_values, list):
            mean_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        top_indices = np.argsort(mean_abs)[::-1][:20]
        top_features = {
            feature_names[i]: float(mean_abs[i])
            for i in top_indices
            if i < len(feature_names)
        }
        return {"top_features": top_features, "method": "LinearExplainer"}
    except Exception as exc:
        log.warning("SHAP LR failed: %s", exc)
        return {"error": str(exc)}


# -- DistilBERT evaluation -----------------------------------------------------

def evaluate_distilbert(df: pd.DataFrame) -> Optional[dict]:
    """Evaluate fine-tuned DistilBERT on a held-out test set."""
    if not BERT_MODEL_DIR.exists():
        log.warning("DistilBERT model directory not found: %s", BERT_MODEL_DIR)
        return None
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        model.eval()
        model.to(device)

        texts = (df["subject"].fillna("") + " [SEP] " + df["body"].fillna("").str[:256]).tolist()
        labels = df["priority"].astype(int).tolist()

        _, test_texts, _, test_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
        )

        y_pred = []
        batch_size = 32
        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())

        y_true = np.array(test_labels[:len(y_pred)])
        y_pred = np.array(y_pred)
        return _full_metrics(y_true, y_pred)
    except Exception as exc:
        log.warning("DistilBERT evaluation failed: %s", exc)
        return None


# -- Main evaluation pipeline --------------------------------------------------

def evaluate_all() -> dict:
    # Load features
    if not FEATURES_PKL.exists():
        log.error("features.pkl not found. Run feature_engineering.py first.")
        sys.exit(1)

    log.info("Loading features ...")
    with open(FEATURES_PKL, "rb") as fh:
        features = pickle.load(fh)

    X_tfidf = features["X_tfidf"]
    X_combined = features["X_combined"]
    y = features["y"]
    feature_names = features.get("feature_names", [])
    tfidf_names = features.get("tfidf_feature_names", [])

    # Use same train/test split as training (same random_state)
    from scipy.sparse import issparse
    X_tfidf_arr = X_tfidf.toarray() if issparse(X_tfidf) else X_tfidf
    X_combined_arr = X_combined if not issparse(X_combined) else X_combined.toarray()

    _, X_tfidf_test, _, y_test = train_test_split(
        X_tfidf_arr, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    _, X_full_test, _, y_test2 = train_test_split(
        X_combined_arr, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_results = {}

    # -- Logistic Regression ---------------------------------------------------
    lr_model = _safe_load(LR_MODEL_PATH)
    if lr_model is not None:
        log.info("Evaluating Logistic Regression ...")
        y_pred_lr = lr_model.predict(X_tfidf_test)
        y_proba_lr = lr_model.predict_proba(X_tfidf_test)
        metrics_lr = _full_metrics(y_test, y_pred_lr, y_proba_lr)
        shap_lr = compute_shap_lr(lr_model, X_tfidf_arr[:500], tfidf_names)
        metrics_lr["shap"] = shap_lr
        model_results["logistic_regression"] = metrics_lr
        log.info("LR  accuracy=%.4f  macro_f1=%.4f", metrics_lr["accuracy"], metrics_lr["macro_f1"])

    # -- Random Forest ---------------------------------------------------------
    rf_model = _safe_load(RF_MODEL_PATH)
    if rf_model is not None:
        log.info("Evaluating Random Forest ...")
        y_pred_rf = rf_model.predict(X_full_test)
        y_proba_rf = rf_model.predict_proba(X_full_test)
        metrics_rf = _full_metrics(y_test2, y_pred_rf, y_proba_rf)
        model_results["random_forest"] = metrics_rf
        log.info("RF  accuracy=%.4f  macro_f1=%.4f", metrics_rf["accuracy"], metrics_rf["macro_f1"])

    # -- XGBoost ---------------------------------------------------------------
    xgb_model = _safe_load(XGB_MODEL_PATH)
    if xgb_model is not None:
        log.info("Evaluating XGBoost ...")
        y_pred_xgb = xgb_model.predict(X_full_test)
        y_proba_xgb = xgb_model.predict_proba(X_full_test)
        metrics_xgb = _full_metrics(y_test2, y_pred_xgb, y_proba_xgb)
        shap_xgb = compute_shap_xgb(xgb_model, X_full_test[:200], feature_names)
        metrics_xgb["shap"] = shap_xgb
        model_results["xgboost"] = metrics_xgb
        log.info("XGB accuracy=%.4f  macro_f1=%.4f", metrics_xgb["accuracy"], metrics_xgb["macro_f1"])

    # -- DistilBERT ------------------------------------------------------------
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
        bert_metrics = evaluate_distilbert(df)
        if bert_metrics:
            model_results["distilbert"] = bert_metrics
            log.info("BERT accuracy=%.4f  macro_f1=%.4f",
                     bert_metrics["accuracy"], bert_metrics["macro_f1"])

    # -- Select best model -----------------------------------------------------
    best_model = None
    best_f1 = -1.0
    for name, metrics in model_results.items():
        f1 = metrics.get("macro_f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    log.info("Best model: %s (macro_f1=%.4f)", best_model, best_f1)

    # -- Threshold check -------------------------------------------------------
    needs_fallback = False
    if best_model:
        best_acc = model_results[best_model].get("accuracy", 0)
        best_mf1 = model_results[best_model].get("macro_f1", 0)
        if best_acc < ACCURACY_THRESHOLD or best_mf1 < MACRO_F1_THRESHOLD:
            needs_fallback = True
            log.warning(
                "Best model '%s' below threshold (accuracy=%.4f, macro_f1=%.4f). "
                "Fallback recommended.",
                best_model, best_acc, best_mf1,
            )

    # -- Build report ----------------------------------------------------------
    report = {
        "evaluation_date": datetime.datetime.now().isoformat(),
        "best_model": best_model,
        "needs_fallback": needs_fallback,
        "thresholds": {
            "accuracy": ACCURACY_THRESHOLD,
            "macro_f1": MACRO_F1_THRESHOLD,
        },
        "models": model_results,
    }

    with open(EVALUATION_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("Evaluation report saved to %s", EVALUATION_REPORT_JSON)

    if best_model:
        BEST_MODEL_FILE.write_text(best_model)
        log.info("Best model name written to %s", BEST_MODEL_FILE)

    return report


def main():
    report = evaluate_all()
    log.info("\n=== Evaluation Summary ===")
    for name, metrics in report["models"].items():
        log.info(
            "  %-25s  accuracy=%.4f  macro_f1=%.4f",
            name,
            metrics.get("accuracy", 0),
            metrics.get("macro_f1", 0),
        )
    log.info("Best model: %s", report["best_model"])
    if report["needs_fallback"]:
        log.warning("Fallback is recommended! Run fallback_model.py")
    else:
        log.info("Performance is above threshold. No fallback needed.")


if __name__ == "__main__":
    main()
