"""
Fallback model for Email Priority Classification.

When trained model performance is below threshold:
  1. Attempt to download a pre-trained model from HuggingFace Hub.
  2. If that fails, build a robust rule-based ensemble fallback.

The fallback model is saved to models/fallback_model.pkl and implements
the same predict/predict_proba interface as scikit-learn classifiers.

Usage:
    python fallback_model.py [--force] [--check-only]
"""
from __future__ import annotations

import sys
import json
import pickle
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    FALLBACK_MODEL_PATH,
    EVALUATION_REPORT_JSON,
    BEST_MODEL_FILE,
    ACCURACY_THRESHOLD,
    MACRO_F1_THRESHOLD,
    CRITICAL_KEYWORDS,
    HIGH_KEYWORDS,
    LOW_KEYWORDS,
    DISTANT_HORIZON_PHRASES,
    NEAR_TERM_URGENCY_PHRASES,
    PRIORITY_LABEL_NAMES,
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


# -- Rule-Based Fallback Classifier --------------------------------------------

class RuleBasedEmailClassifier:
    """
    A deterministic rule-based email priority classifier.

    Uses keyword matching, structural signals, and sender heuristics
    to assign priority classes. Implements a scikit-learn-compatible
    predict / predict_proba interface.

    Class mapping: 0=critical, 1=high, 2=normal, 3=low
    """

    CRITICAL_KWS = CRITICAL_KEYWORDS
    HIGH_KWS = HIGH_KEYWORDS
    LOW_KWS = LOW_KEYWORDS
    CSUITE = ["ceo", "cfo", "coo", "cto", "president", "chairman", "vp", "svp", "evp"]

    def _score_text(self, text: str) -> tuple[int, np.ndarray]:
        """
        Return (predicted_class, probability_vector).
        """
        t = text.lower()

        critical_score = sum(2 if kw in t else 0 for kw in self.CRITICAL_KWS)
        high_score = sum(1 if kw in t else 0 for kw in self.HIGH_KWS)
        low_score = sum(1 if kw in t else 0 for kw in self.LOW_KWS)

        # Structural signals
        excl = t.count("!")
        has_question = "?" in t
        has_re = t.startswith("re:")
        has_fw = t.startswith("fw:") or t.startswith("fwd:")

        if excl > 2:
            critical_score += 1
        if has_question:
            high_score += 0.5
        if has_fw:
            low_score += 1
        if has_re:
            high_score += 0.3

        has_distant = any(p in t for p in DISTANT_HORIZON_PHRASES)
        has_near = any(p in t for p in NEAR_TERM_URGENCY_PHRASES)
        if has_distant and not has_near:
            critical_score *= 0.15
            high_score *= 0.15

        # Compute soft probabilities (keep normal baseline low so keywords can win)
        normal_prior = 2.8 if (has_distant and not has_near) else 0.6
        raw = np.array([
            float(critical_score),
            float(high_score),
            float(normal_prior),
            float(low_score),
        ])
        raw = np.clip(raw, 0, None)

        # Ensure normal is competitive baseline
        if raw.sum() < 1e-6:
            raw[2] = 1.0  # default to normal

        proba = raw / raw.sum()
        pred = int(np.argmax(proba))
        return pred, proba

    def predict(self, X) -> np.ndarray:
        """
        X can be:
          - list/array of strings (raw text)
          - list of dicts with keys: subject, body, sender
          - numpy array (feature matrix; will fallback to normal for all)
        """
        results = []
        for item in X:
            pred, _ = self._classify_item(item)
            results.append(pred)
        return np.array(results)

    def predict_proba(self, X) -> np.ndarray:
        probas = []
        for item in X:
            _, proba = self._classify_item(item)
            probas.append(proba)
        return np.array(probas)

    def _classify_item(self, item) -> tuple[int, np.ndarray]:
        if isinstance(item, dict):
            text = (
                str(item.get("subject", ""))
                + " "
                + str(item.get("body", ""))[:500]
                + " "
                + str(item.get("sender", ""))
            )
            # Check sender for C-suite signals
            sender = str(item.get("sender", "")).lower()
            folder = str(item.get("folder", "")).lower()
            pred, proba = self._score_text(text)
            # Override: C-suite sender always bumps to at least high
            if any(title in sender for title in self.CSUITE) and pred > 1:
                proba[1] += 0.3
                proba = proba / proba.sum()
                pred = int(np.argmax(proba))
            # Override: spam/junk folder -> low
            if "spam" in folder or "junk" in folder:
                proba = np.array([0.02, 0.03, 0.05, 0.90])
                pred = 3
            return pred, proba
        elif isinstance(item, str):
            return self._score_text(item)
        elif isinstance(item, (np.ndarray, list)):
            # Feature vector: can't apply rules, default to normal
            return 2, np.array([0.05, 0.15, 0.70, 0.10])
        else:
            return 2, np.array([0.05, 0.15, 0.70, 0.10])

    def __repr__(self):
        return "RuleBasedEmailClassifier()"


# -- HuggingFace pre-trained download attempt ---------------------------------

def _try_download_hf_model() -> Optional[object]:
    """
    Attempt to download a community email classification model from HuggingFace.
    Returns a classifier object or None if unavailable.
    """
    try:
        from huggingface_hub import hf_hub_download, list_models
        from transformers import pipeline

        log.info("Attempting to download pre-trained email classifier from HuggingFace ...")

        # Try a few community models for text classification
        candidate_models = [
            "mrm8488/bert-tiny-finetuned-enron-email-classification",
            "distilbert-base-uncased-finetuned-sst-2-english",  # fallback: sentiment
        ]

        for model_id in candidate_models:
            try:
                log.info("Trying HuggingFace model: %s", model_id)
                clf = pipeline(
                    "text-classification",
                    model=model_id,
                    return_all_scores=True,
                    truncation=True,
                    max_length=128,
                )
                log.info("Successfully loaded HuggingFace model: %s", model_id)
                return clf
            except Exception as exc:
                log.warning("Could not load %s: %s", model_id, exc)
                continue

        return None
    except ImportError as exc:
        log.warning("huggingface_hub/transformers not available: %s", exc)
        return None
    except Exception as exc:
        log.warning("HuggingFace download failed: %s", exc)
        return None


class HFPipelineWrapper:
    """Wraps a HuggingFace text-classification pipeline to scikit-learn interface."""

    def __init__(self, pipeline, label_map: Optional[dict] = None):
        self.pipeline = pipeline
        self.label_map = label_map  # HF label -> priority int

    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        results = []
        for item in X:
            text = item if isinstance(item, str) else str(item.get("subject", "")) + " " + str(item.get("body", ""))[:200]
            try:
                scores = self.pipeline(text[:512])[0]
                # Normalise to 4-class vector
                proba = np.array([0.05, 0.15, 0.70, 0.10], dtype=float)
                for sc in scores:
                    label = sc["label"].lower()
                    score = sc["score"]
                    if "LABEL_0" in label or "critical" in label or "urgent" in label:
                        proba[0] = score
                    elif "LABEL_1" in label or "high" in label or "positive" in label:
                        proba[1] = score
                    elif "LABEL_2" in label or "normal" in label or "negative" in label:
                        proba[2] = score
                    elif "LABEL_3" in label or "low" in label:
                        proba[3] = score
                proba = proba / proba.sum()
                results.append(proba)
            except Exception:
                results.append(np.array([0.05, 0.15, 0.70, 0.10]))
        return np.array(results)

    def __repr__(self):
        return f"HFPipelineWrapper(pipeline={self.pipeline.model.config._name_or_path})"


# -- Threshold check -----------------------------------------------------------

def check_performance_threshold() -> tuple[bool, dict]:
    """
    Read evaluation_report.json and decide if fallback is needed.

    Returns (needs_fallback: bool, best_model_metrics: dict).
    """
    if not EVALUATION_REPORT_JSON.exists():
        log.warning("Evaluation report not found. Assuming fallback is needed.")
        return True, {}

    with open(EVALUATION_REPORT_JSON) as fh:
        report = json.load(fh)

    needs_fallback = report.get("needs_fallback", True)
    best_model = report.get("best_model")
    best_metrics = report.get("models", {}).get(best_model, {}) if best_model else {}

    if needs_fallback:
        acc = best_metrics.get("accuracy", 0)
        f1 = best_metrics.get("macro_f1", 0)
        log.warning(
            "Fallback triggered: best model '%s' accuracy=%.4f (threshold %.2f), "
            "macro_f1=%.4f (threshold %.2f).",
            best_model, acc, ACCURACY_THRESHOLD, f1, MACRO_F1_THRESHOLD,
        )
    else:
        log.info("Performance meets threshold. Fallback not required.")

    return needs_fallback, best_metrics


# -- Create fallback model -----------------------------------------------------

def create_rule_based_fallback() -> RuleBasedEmailClassifier:
    clf = RuleBasedEmailClassifier()
    _save_fallback(clf, reason="rule-based fallback (no HuggingFace model available)")
    return clf


def _save_fallback(model, reason: str = "") -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "reason": reason, "type": type(model).__name__}
    with open(FALLBACK_MODEL_PATH, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Fallback model saved to %s (reason: %s)", FALLBACK_MODEL_PATH, reason)


def load_fallback_model():
    """Load the fallback model from disk."""
    if not FALLBACK_MODEL_PATH.exists():
        return None
    with open(FALLBACK_MODEL_PATH, "rb") as fh:
        payload = pickle.load(fh)
    return payload.get("model")


# -- Main ----------------------------------------------------------------------

def check_and_apply_fallback(force: bool = False) -> bool:
    """
    Checks performance threshold and applies fallback if needed.
    Returns True if fallback was applied.
    """
    needs_fallback, best_metrics = check_performance_threshold()

    if not needs_fallback and not force:
        log.info("No fallback needed.")
        return False

    log.info("Applying fallback model ...")

    # 1) Try HuggingFace pre-trained model
    hf_model = _try_download_hf_model()
    if hf_model is not None:
        wrapper = HFPipelineWrapper(hf_model)
        acc = best_metrics.get("accuracy", 0)
        f1 = best_metrics.get("macro_f1", 0)
        reason = (
            f"HuggingFace fallback (trained model: accuracy={acc:.4f}, macro_f1={f1:.4f}; "
            f"below thresholds acc>={ACCURACY_THRESHOLD}, f1>={MACRO_F1_THRESHOLD})"
        )
        _save_fallback(wrapper, reason=reason)
        log.info("Fallback set to HuggingFace pipeline model.")
        return True

    # 2) Always-available rule-based fallback
    log.info("Using rule-based fallback classifier.")
    acc = best_metrics.get("accuracy", 0)
    f1 = best_metrics.get("macro_f1", 0)
    reason = (
        f"Rule-based fallback (trained model: accuracy={acc:.4f}, macro_f1={f1:.4f}; "
        f"below thresholds acc>={ACCURACY_THRESHOLD}, f1>={MACRO_F1_THRESHOLD})"
    )
    create_rule_based_fallback()
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply fallback model if performance is below threshold.")
    parser.add_argument("--force", action="store_true", help="Force fallback even if performance is OK.")
    parser.add_argument("--check-only", action="store_true", help="Only check threshold, don't apply fallback.")
    args = parser.parse_args()

    if args.check_only:
        needs, metrics = check_performance_threshold()
        print(json.dumps({"needs_fallback": needs, "best_metrics": metrics}, indent=2))
        sys.exit(0 if not needs else 1)

    applied = check_and_apply_fallback(force=args.force)
    if applied:
        log.info("Fallback model is now active at %s", FALLBACK_MODEL_PATH)
    else:
        log.info("Primary trained models will be used.")


if __name__ == "__main__":
    main()
