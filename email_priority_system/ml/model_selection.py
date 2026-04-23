"""
Choose which trained model to deploy and which scores to trust for reporting.

Tree ensembles on the full TF-IDF + metadata + BERT matrix can achieve
near-perfect *in-sample-style* metrics on heuristic-labelled data while
cross-validation still shows ~1.0 when the label is almost deterministic from
keywords — we treat "perfect score + zero CV variance" as untrustworthy for
*selection* when a more conservative baseline exists.
"""
from __future__ import annotations

from typing import Any, Optional


def _suspicious_perfect(metrics: dict[str, Any], training_row: dict[str, Any]) -> bool:
    """True if metrics look like overfitting / label leakage rather than honest generalisation."""
    test_acc = float(metrics.get("accuracy") or 0)
    test_f1 = float(metrics.get("macro_f1") or 0)
    if test_acc < 0.998 or test_f1 < 0.998:
        return False
    cv_mean = training_row.get("cv_f1_macro_mean")
    cv_std = training_row.get("cv_f1_macro_std")
    if cv_mean is None:
        return False
    cv_std_f = float(cv_std or 0)
    cv_mean_f = float(cv_mean)
    return cv_mean_f >= 0.998 and cv_std_f < 1e-9


def select_best_model(
    model_results: dict[str, dict[str, Any]],
    training_models: Optional[dict[str, dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Return model name to deploy.

    Ranking key: CV macro-F1 mean when present, else held-out macro-F1.
    Models flagged *_suspicious_perfect* are excluded if any non-suspicious model exists.
    """
    if not model_results:
        return None
    training_models = training_models or {}

    rows: list[tuple[str, float, bool, float]] = []
    for name, m in model_results.items():
        tr = training_models.get(name, {})
        cv_f1_m = tr.get("cv_f1_macro_mean")
        key = float(cv_f1_m) if cv_f1_m is not None else float(m.get("macro_f1", 0) or 0)
        bad = _suspicious_perfect(m, tr)
        cv_std = float(tr.get("cv_f1_macro_std") or 0)
        rows.append((name, key, bad, cv_std))

    credible = [r for r in rows if not r[2]]
    pool = credible if credible else rows
    pool.sort(key=lambda r: (-r[1], r[3]))
    return pool[0][0]
