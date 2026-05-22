#!/usr/bin/env python3
"""
Builds a readable project overview PDF for the email priority system.
Run from repo root: python3 scripts/build_dissertation_overview_pdf.py
Output: Email_Priority_System_Project_Overview.pdf (repository root).

Requires: pip install fpdf2
"""
from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Email_Priority_System_Project_Overview.pdf"
EVAL_JSON = ROOT / "email_priority_system" / "ml" / "models" / "evaluation_report.json"
BEST_MODEL_FILE = ROOT / "email_priority_system" / "ml" / "models" / "best_model.txt"


class DocPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    def header(self) -> None:
        if self.page_no() <= 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 8, "Email priority classification - project overview", align="C")
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def para(pdf: DocPDF, text: str, size: int = 11) -> None:
    pdf.set_font("Helvetica", size=size)
    pdf.multi_cell(0, 5.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def h(pdf: DocPDF, title: str, level: int = 1) -> None:
    pdf.ln(5 if level == 1 else 3)
    sz = {1: 15, 2: 12, 3: 11}[level]
    pdf.set_font("Helvetica", "B", sz)
    pdf.multi_cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _format_model_lines(models: dict) -> list[str]:
    order = ["logistic_regression", "random_forest", "xgboost", "distilbert"]
    out: list[str] = []
    for key in order:
        if key not in models:
            continue
        m = models[key]
        acc, f1 = m.get("accuracy"), m.get("macro_f1")
        if acc is None and f1 is None:
            continue
        label = key.replace("_", " ")
        bits = []
        if acc is not None:
            bits.append(f"{float(acc) * 100:.2f}% accuracy")
        if f1 is not None:
            bits.append(f"macro F1 {float(f1):.3f}")
        out.append(f"{label} ({', '.join(bits)})")
    return out


def load_evaluation_snapshot_paragraph() -> str | None:
    """Build a short results paragraph from on-disk JSON + best_model.txt, or None."""
    if not EVAL_JSON.exists():
        return None
    try:
        data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    best = data.get("best_model") or "unknown"
    if BEST_MODEL_FILE.exists():
        try:
            t = BEST_MODEL_FILE.read_text(encoding="utf-8").strip()
            if t:
                best = t
        except OSError:
            pass

    eval_date = str(data.get("evaluation_date", ""))[:10]
    needs_fb = data.get("needs_fallback")

    parts: list[str] = []
    if eval_date:
        parts.append(f"The latest evaluation artefact on disk is dated {eval_date}. ")
    parts.append(
        f"The selected model for deployment is \"{best.replace('_', ' ')}\" "
        "(written to ml/models/best_model.txt). "
    )
    if needs_fb is True:
        parts.append(
            "That run flagged the rule-based fallback as worth using under the configured gates. "
        )
    elif needs_fb is False:
        parts.append(
            "Held-out accuracy and macro-F1 met the thresholds in the report, so the trained "
            "weights were treated as acceptable. "
        )

    lines_m = _format_model_lines(data.get("models") or {})
    if lines_m:
        parts.append("Held-out split: " + "; ".join(lines_m) + ". ")
    parts.append(
        "Regenerate this snapshot by running evaluate_models.py after training so the JSON "
        "and best_model.txt stay aligned."
    )
    return "".join(parts)


def file_block(pdf: DocPDF, path: str, body: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5, path, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def main() -> None:
    pdf = DocPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(28)
    pdf.multi_cell(
        0,
        9,
        "Email Priority Classification System",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.set_font("Helvetica", size=13)
    pdf.ln(6)
    pdf.multi_cell(
        0,
        7,
        "Project overview for readers new to the codebase",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.ln(14)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        6,
        "Muhammad Usama Islam (U2911515)\n"
        "MSc Artificial Intelligence, University of East London\n"
        "Dissertation work: automated priority scoring for email",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )

    pdf.add_page()
    h(pdf, "What problem this solves", 1)
    para(
        pdf,
        "Most inboxes are noisy. Some messages need an answer in minutes; others can wait. "
        "This project builds a small end-to-end system that guesses how urgent an email "
        "probably is, given the usual fields you already have: who sent it, who it went to, "
        "subject, body, and an optional timestamp. The output is one of four bands: critical, "
        "high, normal, or low, plus confidence scores so you can see how decisive the model was.",
    )
    para(
        pdf,
        "Nothing here replaces human judgment for sensitive decisions. The point is to rank or "
        "pre-sort mail the same way triage tools do in hospitals: quick signal first, careful "
        "review when the stakes are high.",
    )

    h(pdf, "How the system is put together", 1)
    para(
        pdf,
        "Two programs share the work. A Ruby on Rails web app gives you a dashboard, a form to "
        "paste an email, and a history of past runs. It stores results in SQLite. A separate "
        "Python service, built with Flask, loads the trained model (or a small rule-based backup) "
        "and returns JSON. In Docker Compose, both containers start together; the Rails app calls "
        "the API over the internal network while you use the browser on port 3000.",
    )

    h(pdf, "Dataset: what we trained on and why", 1)
    para(
        pdf,
        "Training data comes from well-known public email releases that researchers have used for "
        "years. The main source is the Enron email corpus: a large set of real organisational mail "
        "that was made public as part of a legal investigation. It matters because the messages "
        "look like actual workplace traffic: threads, forwards, uneven formality, and messy "
        "headers. That is closer to production inboxes than hand-written toy examples.",
    )
    para(
        pdf,
        "We also pull archives from the SpamAssassin public corpus. Those files add variety in "
        "tone and intent (clear ham, harder ham, spam). Mixing them with Enron helps the "
        "preprocessing step see both corporate chatter and bulk or automated mail, which tends "
        "to land in the low-priority bucket in real life.",
    )
    para(
        pdf,
        "Neither corpus ships with a \"priority\" column. Hand-labelling hundreds of thousands of "
        "messages would not be practical for a single project, so we use a semi-supervised "
        "scheme: priority is inferred from the text and from light metadata cues (for example "
        "urgency phrases in the subject, meeting or deadline language, newsletter-style wording, "
        "or executive-like sender patterns). The rules live in config.py and are applied "
        "consistently in preprocess.py. That gives a single label per message everyone downstream "
        "can rely on.",
    )
    para(
        pdf,
        "The trade-off is honest: labels reflect those cues, not a panel of human annotators. "
        "That is why we train several model families and evaluate with held-out data and "
        "cross-validation. The goal is not to pretend the labels are perfect, but to learn "
        "a stable mapping from email content and structure to priority bands that matches "
        "the triage policy encoded in the labelling rules.",
    )

    h(pdf, "Models we trained", 1)
    para(
        pdf,
        "Four learners are trained and compared:",
    )
    para(
        pdf,
        "- Logistic regression on a TF-IDF representation of subject and body. This is the "
        "classic linear baseline: fast, easy to reason about, and a good sanity check.",
    )
    para(
        pdf,
        "- Random forest on the full numeric feature matrix (metadata, hand-crafted urgency "
        "counts, TF-IDF, and optional transformer embeddings). Trees can pick up non-linear "
        "combinations of features.",
    )
    para(
        pdf,
        "- XGBoost on the same combined matrix. Gradient boosting often does well on mixed "
        "tabular and sparse text features when classes are imbalanced; we balance training "
        "with standard techniques such as SMOTE where appropriate.",
    )
    para(
        pdf,
        "- DistilBERT fine-tuned on subject plus body text. This model reads the raw words "
        "rather than a fixed bag-of-words vector, and can capture phrasing that keyword lists "
        "might miss. It is heavier to train but useful as a modern neural baseline.",
    )
    para(
        pdf,
        "After training, evaluate_models.py scores each model on held-out data. model_selection.py "
        "picks what to deploy: it prefers models whose cross-validation behaviour looks credible, "
        "and can deprioritise scores that look \"too perfect\" on the test split when that "
        "likely means the labels align almost deterministically with a subset of features. "
        "The name of the chosen model is written to models/best_model.txt and the API loads it "
        "at runtime.",
    )

    pdf.add_page()
    h(pdf, "Important files and what they do", 1)
    para(
        pdf,
        "Below is a short tour of the files you will touch most often. Paths are relative to "
        "the email_priority_system folder unless noted.",
    )

    file_block(
        pdf,
        "ml/config.py",
        "Central place for folder paths, download URLs for the public corpora, the four "
        "priority labels, keyword lists used in labelling, TF-IDF and BERT settings, train/test "
        "constants, API host and port, and quality thresholds that decide whether to trust a "
        "trained file or fall back to explicit rules.",
    )
    file_block(
        pdf,
        "ml/download_dataset.py",
        "Downloads the Enron archive and SpamAssassin tarballs into ml/data/raw so "
        "preprocess.py can read them. Running this once saves you from hunting mirrors by hand.",
    )
    file_block(
        pdf,
        "ml/preprocess.py",
        "Walks the maildir tree (or compatible layout), parses RFC822 messages, extracts "
        "addresses, subject, body, date, and thread hints, applies the semi-supervised priority "
        "rules, and writes a cleaned CSV under ml/data/processed/. That CSV is the contract "
        "between offline training and the DistilBERT text path.",
    )
    file_block(
        pdf,
        "ml/feature_engineering.py",
        "Turns each row of the processed CSV into numbers: metadata columns, simple urgency "
        "counters, a 5000-dimensional TF-IDF vector over subject and body, and optionally "
        "768-dimensional DistilBERT embeddings. Everything is packed into features.pkl for "
        "train_models.py.",
    )
    file_block(
        pdf,
        "ml/train_models.py",
        "Loads features.pkl, fits logistic regression, random forest, and XGBoost, and can "
        "fine-tune DistilBERT on the raw text from the processed CSV. Writes joblib models under "
        "ml/models/ and a training_results.json summary with timings and cross-validation "
        "scores.",
    )
    file_block(
        pdf,
        "ml/evaluate_models.py",
        "Holds out data, measures accuracy, macro-F1, per-class precision and recall, confusion "
        "matrices, and optional SHAP-style explanations for interpretability. Produces "
        "evaluation_report.json and refreshes best_model.txt after consulting model_selection.py.",
    )
    file_block(
        pdf,
        "ml/model_selection.py",
        "Implements the policy for which trained file should be considered \"best\" for "
        "deployment, using cross-validation macro-F1 when available and filtering suspicious "
        "near-perfect test scores when a safer baseline exists.",
    )
    file_block(
        pdf,
        "ml/predict.py",
        "Runtime inference: rebuilds the same feature vector for one message, loads the model "
        "named in best_model.txt, returns priority, probabilities, timing, and optional feature "
        "weights for explanation in the UI.",
    )
    file_block(
        pdf,
        "ml/fallback_model.py",
        "Defines a transparent rule-based classifier used when files are missing or metrics fall "
        "below the configured thresholds, so the API never returns empty-handed.",
    )
    file_block(
        pdf,
        "ml/flask_api.py",
        "Exposes HTTP endpoints: health check, single and batch predict, and metadata about "
        "which model is active. This is what Rails talks to.",
    )
    file_block(
        pdf,
        "rails_app/app/services/ml_api_client.rb",
        "Wraps HTTParty calls to the Flask service with sensible timeouts and error messages so "
        "controllers stay thin.",
    )
    file_block(
        pdf,
        "rails_app/app/controllers/classifications_controller.rb",
        "Shows the form, posts user input to the ML API, persists an EmailClassification row, "
        "and renders history and detail pages.",
    )
    file_block(
        pdf,
        "rails_app/app/controllers/dashboard_controller.rb",
        "Home page: aggregate counts, recent classifications, and a quick check that the ML "
        "service is reachable.",
    )
    file_block(
        pdf,
        "docker-compose.yml",
        "Runs the ML API and Rails app with shared volumes for data and model weights so a "
        "fresh machine can start the stack without redoing training first.",
    )

    snap = load_evaluation_snapshot_paragraph()
    if snap:
        h(pdf, "Snapshot from the saved evaluation run", 2)
        para(pdf, snap)

    h(pdf, "Closing", 2)
    para(
        pdf,
        "If you change labelling rules or add models, rerun the preprocessing and training "
        "steps, then evaluate_models.py so best_model.txt and evaluation_report.json stay in "
        "sync with what the Flask service actually loads.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
