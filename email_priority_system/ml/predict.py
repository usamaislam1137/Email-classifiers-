"""
Inference module for the Email Priority Classification System.

Loads the best trained model (or fallback) and classifies a single email dict.

Returned dict:
    priority          : str   - "critical" | "high" | "normal" | "low"
    priority_index    : int   - 0-3
    confidence        : float - confidence of the predicted class
    confidence_scores : dict  - {class_name: score} for all 4 classes
    shap_values       : dict  - top feature importances (if available)
    model_used        : str   - name of the model that made the prediction
    processing_time_ms: int   - inference time in milliseconds

Usage:
    from predict import classify_email
    result = classify_email({
        "sender": "boss@example.com",
        "recipients": "me@example.com",
        "subject": "URGENT: server down",
        "body": "Please fix immediately!",
        "date": "2025-03-16T09:00:00",
    })
"""
from __future__ import annotations

import sys
import time
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    BERT_MODEL_DIR,
    FALLBACK_MODEL_PATH,
    BEST_MODEL_FILE,
    TFIDF_VECTORIZER_PKL,
    FEATURES_PKL,
    EVALUATION_REPORT_JSON,
    PRIORITY_LABELS,
    PRIORITY_LABEL_NAMES,
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

# -- Global model cache (loaded once, reused) ----------------------------------
_model_cache: dict = {}


def _load_vectorizer() -> Optional[object]:
    if not TFIDF_VECTORIZER_PKL.exists():
        return None
    return joblib.load(TFIDF_VECTORIZER_PKL)


def _load_best_model_name() -> str:
    if BEST_MODEL_FILE.exists():
        return BEST_MODEL_FILE.read_text().strip()
    # Fallback priority order
    if FALLBACK_MODEL_PATH.exists():
        return "fallback"
    return "rule_based"


def _load_model(name: str):
    """Load a model by name. Returns (model, model_type_hint)."""
    model_paths = {
        "logistic_regression": LR_MODEL_PATH,
        "random_forest": RF_MODEL_PATH,
        "xgboost": XGB_MODEL_PATH,
    }
    if name in model_paths and model_paths[name].exists():
        return joblib.load(model_paths[name]), name
    if name in ("fallback",) and FALLBACK_MODEL_PATH.exists():
        with open(FALLBACK_MODEL_PATH, "rb") as fh:
            payload = pickle.load(fh)
        return payload.get("model"), "fallback"
    if name == "distilbert" and BERT_MODEL_DIR.exists():
        return _load_bert_model(), "distilbert"
    # Last resort: rule-based
    from fallback_model import RuleBasedEmailClassifier
    return RuleBasedEmailClassifier(), "rule_based"


def _load_bert_model():
    """Lazy-load the fine-tuned DistilBERT model."""
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        tokenizer = DistilBertTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DistilBertForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        model.eval()
        model.to(device)
        return {"tokenizer": tokenizer, "model": model, "device": device}
    except Exception as exc:
        log.warning("Could not load DistilBERT: %s", exc)
        return None


def get_cached_model():
    """Return (model, model_name, vectorizer) from cache or load fresh."""
    global _model_cache
    if _model_cache:
        return _model_cache["model"], _model_cache["name"], _model_cache["vectorizer"]

    best_name = _load_best_model_name()
    log.info("Loading model: %s", best_name)
    model, actual_name = _load_model(best_name)
    vectorizer = _load_vectorizer()

    _model_cache = {"model": model, "name": actual_name, "vectorizer": vectorizer}
    return model, actual_name, vectorizer


# -- Feature extraction for single email --------------------------------------

def _features_for_email(email: dict, vectorizer) -> np.ndarray:
    """Extract full feature vector for a single email dict."""
    try:
        import pandas as pd
        from feature_engineering import extract_single_email_features, trim_to_sklearn_meta_tfidf
        x = extract_single_email_features(email, vectorizer, use_bert=False)
        n_tfidf = len(vectorizer.get_feature_names_out())
        return trim_to_sklearn_meta_tfidf(x, n_tfidf)
    except Exception as exc:
        log.warning("Feature extraction failed (%s) - using empty feature vector.", exc)
        n_features = 1500  # meta + tfidf (disk models); safe default for a failed build
        return np.zeros(n_features, dtype=np.float32)


# -- SHAP for single prediction ------------------------------------------------

def _logistic_regression_explain(
    model, X_instance: np.ndarray, feature_names: list,
) -> dict:
    """Magnitude of x * w for the predicted class (aligns with meta + TF-IDF input)."""
    x = np.asarray(X_instance, dtype=np.float64).ravel()
    nfi = int(getattr(model, "n_features_in_", len(x)))
    if len(x) != nfi:
        return {}
    if not hasattr(model, "coef_"):
        return {}
    coef = np.asarray(model.coef_)
    proba = model.predict_proba(X_instance.reshape(1, -1))[0]
    pred = int(np.argmax(proba))
    if coef.ndim == 1:
        imp = np.abs(x * coef)
    else:
        imp = np.abs(x * coef[pred])
    top_idx = np.argsort(imp)[::-1][:10]
    result = {}
    for i in top_idx:
        if i < len(imp):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            result[str(name)] = float(imp[i])
    return result


def _compute_shap_single(model, model_name: str, X_instance: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP (or model-specific explanation) and return top-10 features."""
    try:
        if model_name == "logistic_regression":
            return _logistic_regression_explain(model, X_instance, feature_names)
    except Exception as exc:
        log.debug("LR explanation failed: %s", exc)
        return {}

    try:
        import shap
        if model_name in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_instance.reshape(1, -1))
            if isinstance(sv, list):
                mean_abs = np.abs(np.array(sv)).mean(axis=0)[0]
            else:
                mean_abs = np.abs(sv[0])
        else:
            return {}

        top_idx = np.argsort(np.asarray(mean_abs).ravel())[::-1][:10]
        result = {}
        ma = np.asarray(mean_abs).ravel()
        for i in top_idx:
            if i < len(ma):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                result[str(name)] = float(ma[i])
        return result
    except Exception as exc:
        log.debug("SHAP computation failed: %s", exc)
        return {}


# -- DistilBERT inference ------------------------------------------------------

def _bert_predict(bert_payload: dict, email: dict) -> tuple[int, np.ndarray]:
    """Run DistilBERT inference for a single email."""
    try:
        import torch
        tokenizer = bert_payload["tokenizer"]
        model = bert_payload["model"]
        device = bert_payload["device"]
        text = str(email.get("subject", "")) + " [SEP] " + str(email.get("body", ""))[:256]
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(proba))
        return pred, proba
    except Exception as exc:
        log.warning("DistilBERT inference failed: %s", exc)
        return 2, np.array([0.05, 0.15, 0.70, 0.10])


# -- Rule-based SHAP proxy (for fallback) -------------------------------------

def _maybe_bump_urgency_from_rules(
    email: dict, pred_idx: int, proba: np.ndarray, model_name: str
) -> tuple[int, np.ndarray, bool]:
    """
    If a linear/tree model over-predicts 'normal' on clearly urgent / academic
    phrasing, let the rule layer raise priority (keeps ML for typical mail).
    """
    if model_name not in ("logistic_regression", "random_forest", "xgboost"):
        return pred_idx, proba, False
    if pred_idx != 2:  # not "normal" in PRIORITY_LABEL_NAMES
        return pred_idx, proba, False
    from fallback_model import RuleBasedEmailClassifier
    ridx, rproba = RuleBasedEmailClassifier()._classify_item(email)
    if ridx not in (0, 1):
        return pred_idx, proba, False
    if float(rproba[ridx]) < 0.45:
        return pred_idx, proba, False
    blend = 0.55 * proba + 0.45 * rproba
    blend = blend / blend.sum()
    return ridx, blend, True


def _email_text_lower(email: dict) -> str:
    return (str(email.get("subject", "")) + " " + str(email.get("body", ""))[:2000]).lower()


def _maybe_downgrade_distant_planning(
    email: dict, pred_idx: int, proba: np.ndarray
) -> tuple[int, np.ndarray]:
    """
    Far-future-only scheduling (e.g. "viva ... next year") is planning/reminder
    rather than time-critical. Shift probability toward 'normal' unless near-term
    urgency phrases are also present.
    """
    if pred_idx not in (0, 1):
        return pred_idx, proba
    from config import DISTANT_HORIZON_PHRASES, NEAR_TERM_URGENCY_PHRASES

    t = _email_text_lower(email)
    if not any(p in t for p in DISTANT_HORIZON_PHRASES):
        return pred_idx, proba
    if any(p in t for p in NEAR_TERM_URGENCY_PHRASES):
        return pred_idx, proba
    p = np.asarray(proba, dtype=np.float64).copy()
    # Blend toward a mostly-normal distribution (low confidence in critical/high is appropriate)
    target = np.array([0.02, 0.10, 0.78, 0.10], dtype=np.float64)
    alpha = 0.62
    p = (1.0 - alpha) * p + alpha * target
    p = p / p.sum()
    return int(np.argmax(p)), p.astype(np.float64)


def _rule_based_shap(email: dict) -> dict:
    """Return keyword-signal explanation for rule-based classifier."""
    from config import CRITICAL_KEYWORDS, HIGH_KEYWORDS, LOW_KEYWORDS
    text = (str(email.get("subject", "")) + " " + str(email.get("body", ""))[:500]).lower()
    signals = {}
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            signals[f"critical_keyword:{kw}"] = 1.0
    for kw in HIGH_KEYWORDS:
        if kw in text:
            signals[f"high_keyword:{kw}"] = 0.5
    for kw in LOW_KEYWORDS:
        if kw in text:
            signals[f"low_keyword:{kw}"] = 0.3
    # Structural signals
    if "?" in str(email.get("body", "")):
        signals["has_question"] = 0.3
    excl = str(email.get("body", "")).count("!")
    if excl > 0:
        signals["exclamation_count"] = float(excl) * 0.2
    return dict(sorted(signals.items(), key=lambda x: x[1], reverse=True)[:10])


# -- Main classify function ----------------------------------------------------

def classify_email(email: dict) -> dict:
    """
    Classify a single email and return priority, confidence, and explanations.

    Args:
        email: dict with keys sender, recipients, subject, body, date (optional)

    Returns:
        dict with priority, confidence, confidence_scores, shap_values,
             model_used, processing_time_ms
    """
    t_start = time.time()

    model, model_name, vectorizer = get_cached_model()

    # Load feature names for SHAP
    feature_names: list = []
    try:
        if FEATURES_PKL.exists():
            with open(FEATURES_PKL, "rb") as fh:
                feat_data = pickle.load(fh)
            feature_names = feat_data.get("feature_names", [])
    except Exception:
        pass

    # -- Inference -------------------------------------------------------------
    try:
        if model_name == "distilbert" and isinstance(model, dict):
            # DistilBERT
            pred_idx, proba = _bert_predict(model, email)
            shap_vals = {}

        elif model_name in ("rule_based", "fallback") and hasattr(model, "_classify_item"):
            # Rule-based classifier
            pred_idx, proba = model._classify_item(email)
            shap_vals = _rule_based_shap(email)

        elif model_name == "fallback":
            # HFPipelineWrapper or rule-based
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([email])[0]
            else:
                proba = np.array([0.05, 0.15, 0.70, 0.10])
            pred_idx = int(np.argmax(proba))
            shap_vals = _rule_based_shap(email)

        else:
            # Scikit-learn / XGBoost model needs feature vector
            if vectorizer is None:
                log.warning("TF-IDF vectorizer not loaded - using fallback")
                from fallback_model import RuleBasedEmailClassifier
                rb = RuleBasedEmailClassifier()
                pred_idx, proba = rb._classify_item(email)
                shap_vals = _rule_based_shap(email)
                model_name = "rule_based_fallback"
            else:
                X = _features_for_email(email, vectorizer)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X.reshape(1, -1))[0]
                else:
                    proba = np.array([0.05, 0.15, 0.70, 0.10])
                pred_idx = int(np.argmax(proba))
                pred_idx, proba, _ = _maybe_bump_urgency_from_rules(
                    email, pred_idx, proba, model_name
                )
                names = feature_names
                if (not names or len(names) != len(X)) and vectorizer is not None:
                    try:
                        from feature_engineering import feature_names_for_meta_tfidf
                        names = feature_names_for_meta_tfidf(vectorizer)
                    except Exception:
                        names = list(feature_names) if feature_names else []
                try:
                    shap_vals = _compute_shap_single(model, model_name, X, names)
                except Exception as shap_exc:
                    log.debug("Post-hoc explainability skipped: %s", shap_exc)
                    shap_vals = {}

    except Exception as exc:
        log.error("Inference error: %s", exc, exc_info=True)
        # Ultimate fallback
        from fallback_model import RuleBasedEmailClassifier
        rb = RuleBasedEmailClassifier()
        pred_idx, proba = rb._classify_item(email)
        shap_vals = _rule_based_shap(email)
        model_name = "rule_based_emergency_fallback"

    pred_idx, proba = _maybe_downgrade_distant_planning(email, pred_idx, proba)

    processing_ms = int((time.time() - t_start) * 1000)
    confidence_scores = {PRIORITY_LABEL_NAMES[i]: float(proba[i]) for i in range(4)}

    return {
        "priority": PRIORITY_LABEL_NAMES[pred_idx],
        "priority_index": pred_idx,
        "confidence": float(proba[pred_idx]),
        "confidence_scores": confidence_scores,
        "shap_values": shap_vals,
        "model_used": model_name,
        "processing_time_ms": processing_ms,
    }


def invalidate_cache():
    """Clear the in-memory model cache (useful after re-training)."""
    global _model_cache
    _model_cache = {}
    log.info("Model cache invalidated.")


# -- CLI for quick testing -----------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classify a test email.")
    parser.add_argument("--sender", default="boss@company.com")
    parser.add_argument("--subject", default="URGENT: please respond ASAP")
    parser.add_argument("--body", default="This is time sensitive. Action required immediately.")
    parser.add_argument("--recipients", default="me@company.com")
    args = parser.parse_args()

    email = {
        "sender": args.sender,
        "recipients": args.recipients,
        "subject": args.subject,
        "body": args.body,
        "date": "2025-03-16T09:00:00",
    }

    log.info("Classifying email ...")
    result = classify_email(email)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
