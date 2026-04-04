# Evidence Folder — Email Priority Classification System

**Dissertation:** Automated Email Priority Classification Using Machine Learning
**Author:** Usama Islam
**Pipeline run:** 2026-04-05
**Best model:** Random Forest (accuracy = 100.00%, Macro F1 = 1.0000)

---

## Contents

### Visualisations

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart showing dataset class distribution (critical / high / normal / low) |
| `confusion_matrix_logistic_regression.png` | Confusion matrix for Logistic Regression baseline |
| `confusion_matrix_random_forest.png` | Confusion matrix for Random Forest (**best model**) |
| `confusion_matrix_xgboost.png` | Confusion matrix for XGBoost |
| `model_comparison.png` | Side-by-side bar chart: Accuracy vs Macro F1 for all three models |
| `roc_curves.png` | One-vs-rest ROC curves for all models (AUC per class) |
| `feature_importance_random_forest.png` | Top-20 most important features — Random Forest |
| `feature_importance_xgboost.png` | Top-20 most important features — XGBoost |

### Reports

| File | Description |
|------|-------------|
| `classification_reports.txt` | Per-class precision, recall, F1, support + confusion matrices (plain text) |
| `training_results.json` | Training metadata: dataset size, feature dimensions, train time, CV scores |
| `evaluation_report.json` | Full evaluation metrics: accuracy, macro F1, per-class scores, confusion matrices |
| `training_log.txt` | Chronological pipeline log (all steps, timings, and metrics) |

---

## Model Performance Summary

| Model | Accuracy | Macro F1 | CV Accuracy (5-fold) | Train Time |
|-------|----------|----------|----------------------|------------|
| Logistic Regression | 74.62% | 0.705 | 74.35% ± 1.33% | ~19s |
| Random Forest ⭐ | **100.00%** | **1.000** | — | ~0.7s |
| XGBoost | **100.00%** | **1.000** | — | ~40s |

> **Note:** Random Forest and XGBoost achieve perfect accuracy on this dataset because labels were assigned deterministically via keyword rules — the same rules that TF-IDF features encode, allowing tree ensembles to perfectly learn the decision boundaries. Logistic Regression, being a linear model, achieves a lower but still acceptable 74.6%.

---

## How to Reproduce

```bash
cd email_priority_system/ml

# Step 1 — Generate dataset
python generate_dataset.py

# Step 2 — Run full training pipeline + evidence generation
python run_mock_pipeline.py
```

All artefacts (plots, JSON reports, trained models) will be regenerated.
