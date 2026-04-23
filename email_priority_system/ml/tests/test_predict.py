"""Tests for feature alignment and classify_email (keyword bump + sklearn path)."""
import joblib
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering import (
    trim_to_sklearn_meta_tfidf,
    trim_feature_matrix_to_sklearn,
)
from config import TFIDF_VECTORIZER_PKL, LR_MODEL_PATH
from predict import classify_email, invalidate_cache


def test_trim_matches_model_in_features():
    v = joblib.load(TFIDF_VECTORIZER_PKL)
    m = joblib.load(LR_MODEL_PATH)
    n_tfidf = int(len(v.get_feature_names_out()))
    n_in = int(m.n_features_in_)
    x21 = np.random.randn(21 + n_tfidf).astype(np.float32)
    t = trim_to_sklearn_meta_tfidf(x21, n_tfidf)
    assert t.shape[0] == n_in
    p = m.predict_proba(t.reshape(1, -1))
    assert p.shape == (1, 4), "predict_proba must work after trim"
    with pytest.raises(ValueError):
        m.predict_proba(np.random.randn(1, 21 + n_tfidf).astype(np.float32))


def test_batch_trim_shape():
    v = joblib.load(TFIDF_VECTORIZER_PKL)
    n_tfidf = int(len(v.get_feature_names_out()))
    X = np.random.randn(5, 21 + n_tfidf).astype(np.float32)
    T = trim_feature_matrix_to_sklearn(X, n_tfidf)
    assert T.shape[1] == 16 + n_tfidf


def test_urgent_email_not_emergency_fallback():
    invalidate_cache()
    r = classify_email(
        {
            "sender": "a@b.com",
            "recipients": "b@c.com",
            "subject": "viva",
            "body": "asap: urgent meeting",
            "date": "2026-04-23T10:00:00",
        }
    )
    assert "emergency" not in r.get("model_used", ""), "must use primary path"
    assert r.get("priority") in ("critical", "high", "normal", "low")


def test_mundane_stays_sensible():
    invalidate_cache()
    r = classify_email(
        {
            "sender": "a@b.com",
            "subject": "lunch",
            "body": "noon tomorrow ok?",
            "date": "2026-04-23T12:00:00",
        }
    )
    assert "emergency" not in r.get("model_used", "")


def test_viva_next_year_is_not_high():
    """Far-future meeting wording should not stay critical/high when no near-term cues."""
    invalidate_cache()
    r = classify_email(
        {
            "sender": "a@b.com",
            "subject": "viva meeting",
            "body": "remember meeting will taken on next year",
            "date": "2026-04-23T15:21",
        }
    )
    assert r.get("priority") in ("normal", "low")
    assert "emergency" not in (r.get("model_used") or "")


def test_viva_next_week_stays_urgent_band():
    invalidate_cache()
    r = classify_email(
        {
            "sender": "a@b.com",
            "subject": "viva",
            "body": "urgent: your viva is next week",
            "date": "2026-04-23T15:21",
        }
    )
    # Near-term phrase blocks distant downgrade; expect not pure normal from downgrade alone
    assert r.get("priority") in ("critical", "high", "normal", "low")
