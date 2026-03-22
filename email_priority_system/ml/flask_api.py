"""
Flask REST API for Email Priority Classification.

Endpoints:
    POST /predict           - classify a single email
    POST /batch_predict     - classify multiple emails
    GET  /health            - health check
    GET  /model_info        - current best model info
    GET  /models            - all models and their performance metrics

Usage:
    python flask_api.py
    # or via gunicorn:
    # gunicorn -w 2 -b 0.0.0.0:5000 flask_api:app
"""
from __future__ import annotations

import sys
import json
import time
import logging
import datetime
from pathlib import Path
from typing import Any

from flask import Flask, request, jsonify, g
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EVALUATION_REPORT_JSON,
    BEST_MODEL_FILE,
    TRAINING_RESULTS_JSON,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    PRIORITY_LABEL_NAMES,
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

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -- Startup: lazy-import predict to avoid heavy model loading at import time --
_classify = None

def _get_classifier():
    global _classify
    if _classify is None:
        try:
            from predict import classify_email
            _classify = classify_email
            log.info("Classifier loaded successfully.")
        except Exception as exc:
            log.error("Failed to load classifier: %s", exc)
    return _classify


# -- Request timing middleware -------------------------------------------------

@app.before_request
def _start_timer():
    g.t_start = time.time()


@app.after_request
def _log_request(response):
    elapsed_ms = int((time.time() - getattr(g, "t_start", time.time())) * 1000)
    log.info(
        "%s %s -> %d  (%dms)",
        request.method, request.path, response.status_code, elapsed_ms
    )
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
    return response


# -- Helpers -------------------------------------------------------------------

def _json_error(message: str, status: int = 400) -> Any:
    return jsonify({"error": message, "status": status}), status


def _validate_email_payload(data: dict) -> tuple[bool, str]:
    """Validate required fields in an email payload."""
    if not data:
        return False, "Request body is empty or not valid JSON."
    if not data.get("subject") and not data.get("body"):
        return False, "At least one of 'subject' or 'body' is required."
    return True, ""


def _load_eval_report() -> dict:
    if EVALUATION_REPORT_JSON.exists():
        with open(EVALUATION_REPORT_JSON) as fh:
            return json.load(fh)
    return {}


def _load_training_results() -> dict:
    if TRAINING_RESULTS_JSON.exists():
        with open(TRAINING_RESULTS_JSON) as fh:
            return json.load(fh)
    return {}


# -- /health -------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    classifier = _get_classifier()
    best_model = BEST_MODEL_FILE.read_text().strip() if BEST_MODEL_FILE.exists() else "unknown"
    report = _load_eval_report()

    return jsonify({
        "status": "ok" if classifier else "degraded",
        "classifier_loaded": classifier is not None,
        "best_model": best_model,
        "models_available": list(report.get("models", {}).keys()),
        "timestamp": datetime.datetime.now().isoformat(),
    })


# -- /model_info ---------------------------------------------------------------

@app.route("/model_info", methods=["GET"])
def model_info():
    """Return info about the currently active best model."""
    report = _load_eval_report()
    training = _load_training_results()

    best_model = report.get("best_model", "unknown")
    model_metrics = report.get("models", {}).get(best_model, {})

    training_date = training.get("training_date") or report.get("evaluation_date", "unknown")

    return jsonify({
        "best_model": best_model,
        "accuracy": model_metrics.get("accuracy", 0),
        "macro_f1": model_metrics.get("macro_f1", 0),
        "per_class": model_metrics.get("per_class", {}),
        "training_date": training_date,
        "evaluation_date": report.get("evaluation_date", "unknown"),
        "needs_fallback": report.get("needs_fallback", False),
        "thresholds": report.get("thresholds", {}),
    })


# -- /models -------------------------------------------------------------------

@app.route("/models", methods=["GET"])
def list_models():
    """List all trained models and their performance metrics."""
    report = _load_eval_report()
    training = _load_training_results()

    models_data = {}
    for name, metrics in report.get("models", {}).items():
        training_info = training.get("models", {}).get(name, {})
        models_data[name] = {
            "accuracy": metrics.get("accuracy", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "per_class": metrics.get("per_class", {}),
            "train_time_s": training_info.get("train_time_s", 0),
            "is_best": name == report.get("best_model"),
        }

    return jsonify({
        "models": models_data,
        "best_model": report.get("best_model", "unknown"),
        "evaluation_date": report.get("evaluation_date", "unknown"),
        "total_models": len(models_data),
    })


# -- /predict ------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single email.

    Request JSON:
        {
            "sender": "...",
            "recipients": "...",
            "subject": "...",
            "body": "...",
            "date": "2025-03-16T09:00:00"  // optional
        }

    Response JSON:
        {
            "priority": "critical",
            "priority_index": 0,
            "confidence": 0.92,
            "confidence_scores": {"critical": 0.92, "high": 0.05, ...},
            "shap_values": {"urgent": 0.41, ...},
            "model_used": "xgboost",
            "processing_time_ms": 42
        }
    """
    data = request.get_json(silent=True)
    valid, err = _validate_email_payload(data)
    if not valid:
        return _json_error(err, 400)

    classifier = _get_classifier()
    if classifier is None:
        return _json_error("Classifier not available. Check server logs.", 503)

    email_input = {
        "sender": data.get("sender", ""),
        "recipients": data.get("recipients", ""),
        "subject": data.get("subject", ""),
        "body": data.get("body", ""),
        "date": data.get("date", datetime.datetime.now().isoformat()),
        "cc": data.get("cc", ""),
        "bcc": data.get("bcc", ""),
    }

    try:
        result = classifier(email_input)
        return jsonify(result)
    except Exception as exc:
        log.error("Prediction error: %s", exc, exc_info=True)
        return _json_error(f"Prediction failed: {str(exc)}", 500)


# -- /batch_predict ------------------------------------------------------------

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Classify multiple emails in one request.

    Request JSON:
        {
            "emails": [
                {"sender": ..., "subject": ..., "body": ...},
                ...
            ]
        }

    Response JSON:
        {
            "results": [...],
            "total": 3,
            "processing_time_ms": 150
        }
    """
    data = request.get_json(silent=True)
    if not data or "emails" not in data:
        return _json_error("Request must contain an 'emails' array.", 400)

    emails = data["emails"]
    if not isinstance(emails, list) or len(emails) == 0:
        return _json_error("'emails' must be a non-empty list.", 400)

    MAX_BATCH = 100
    if len(emails) > MAX_BATCH:
        return _json_error(f"Batch size exceeds maximum ({MAX_BATCH}).", 400)

    classifier = _get_classifier()
    if classifier is None:
        return _json_error("Classifier not available.", 503)

    t_start = time.time()
    results = []
    errors = []

    for idx, email in enumerate(emails):
        valid, err = _validate_email_payload(email)
        if not valid:
            errors.append({"index": idx, "error": err})
            results.append(None)
            continue
        try:
            result = classifier({
                "sender": email.get("sender", ""),
                "recipients": email.get("recipients", ""),
                "subject": email.get("subject", ""),
                "body": email.get("body", ""),
                "date": email.get("date", datetime.datetime.now().isoformat()),
                "cc": email.get("cc", ""),
            })
            results.append(result)
        except Exception as exc:
            log.error("Batch item %d failed: %s", idx, exc)
            errors.append({"index": idx, "error": str(exc)})
            results.append(None)

    total_ms = int((time.time() - t_start) * 1000)
    return jsonify({
        "results": results,
        "total": len(emails),
        "successful": sum(1 for r in results if r is not None),
        "errors": errors,
        "processing_time_ms": total_ms,
    })


# -- Error handlers ------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found.", "status": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed.", "status": 405}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error.", "status": 500}), 500


# -- CLI entry -----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Email Priority Classification API.")
    parser.add_argument("--host", default=FLASK_HOST)
    parser.add_argument("--port", type=int, default=FLASK_PORT)
    parser.add_argument("--debug", action="store_true", default=FLASK_DEBUG)
    args = parser.parse_args()

    # Pre-load classifier at startup
    log.info("Pre-loading classifier ...")
    _get_classifier()

    log.info("Starting Flask API on %s:%d (debug=%s)", args.host, args.port, args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug)
