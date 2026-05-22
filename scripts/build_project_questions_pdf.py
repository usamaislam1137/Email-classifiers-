#!/usr/bin/env python3
"""
Answers five dissertation-style questions about the Email Priority Classification
project as a standalone PDF.

Run from repository root:
  python3 scripts/build_project_questions_pdf.py

Output:
  Dissertation_Email_Priority_Five_Answers.pdf (repository root)

Requires: fpdf2 (pip install fpdf2)
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Dissertation_Email_Priority_Five_Answers.pdf"


class DocPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() <= 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 7, "Email priority classification - five answers (detailed)", align="C")
        self.ln(7)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 7, f"Page {self.page_no()}", align="C")


def para(pdf: DocPDF, text: str, size: int = 10.5, leading: float = 5.0) -> None:
    pdf.set_font("Helvetica", size=size)
    pdf.multi_cell(0, leading, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def heading(pdf: DocPDF, title: str, n: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, f"Question {n}: {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def subh(pdf: DocPDF, t: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 5.8, t, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.5)


def main() -> None:
    pdf = DocPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.ln(20)
    pdf.multi_cell(
        0,
        8,
        "Automated Email Priority Classification",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        7,
        "Five Detailed Answers (PDF briefing)",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.set_font("Helvetica", size=11)
    pdf.ln(10)
    para(
        pdf,
        "Author (per repository README): Muhammad Usama Islam (U2911515), MSc Artificial "
        "Intelligence, University of East London. This document summarizes the codebase in "
        "email_priority_system/ (Python ML pipeline, Flask API, Rails web app, Docker) "
        "and aligns answers with artefacts such as DATASET_INFO.md, evaluation_report.json, "
        "and model_selection.py. Figures in evidence/ complement these answers.",
        size=11,
    )

    # ---- Q1 -----------------------------------------------------------------
    pdf.add_page()
    heading(pdf, "Why I chose this project", "1")

    para(
        pdf,
        "I chose automated email priority classification because it connects every strand of "
        "an AI masters programme to a problem people feel every working day: inboxes overload "
        "attention, delay the wrong messages can have serious consequences (incidents, "
        "finance, deadlines), and deferring noisy mail reduces stress. Unlike toy datasets "
        "with clean labels handed out by instructors, organisational email mixes threads, tone, "
        "power dynamics, newsletters, forwards, automated notices, and one-off crises. Modeling "
        "that mess is academically honest and industrially plausible.",
    )
    para(
        pdf,
        "The dissertation framing in the README and DATASET_INFO is explicit: classify each "
        "message into four operational bands --- critical, high, normal, and low --- not just "
        "spam versus ham. That matches how triage actually works on helpdesks or executive "
        "assistants routing mail: urgency is graded, not binary. Designing for four balanced "
        "classes invites proper multi-class metrics (macro-F1, confusion matrices per class), "
        "handles imbalance with techniques such as stratified splitting and SMOTE in "
        "train_models.py, and forces the UX to expose uncertainty (confidence vectors in the "
        "API responses used by Rails).",
    )
    para(
        pdf,
        "Methodologically the project exploits two large, public corpora everyone can "
        "reproduce --- the CMU Enron mailbox release and Apache SpamAssassin public archives "
        "(URLs and downloader in ml/config.py and download_dataset.py). That choice matters for "
        "academic scrutiny: supervisors and examiners can re-download the same archives, rerun "
        "preprocess.py, regenerate processed_emails.csv, and obtain comparable numbers. Synthetic "
        "or proprietary inboxes rarely offer that reproducibility.",
    )
    para(
        pdf,
        "Engineering completeness is another motive. README.md lays out end-to-end delivery: "
        "feature engineering stacking TF-IDF, metadata, urgency counters, optional DistilBERT "
        "embeddings; comparative training across logistic regression, random forest, XGBoost, "
        "and fine-tuned DistilBERT (train_models.py); evaluation with threshold-based fallback "
        "and Hugging Face hooks (fallback_model.py, config thresholds ACCURACY_THRESHOLD and "
        "MACRO_F1_THRESHOLD); Flask HTTP API for inference; Ruby on Rails 7 dashboard storing "
        "SQLite history; Chart.js visualisations for confidence and attribution; Docker Compose "
        "for turnkey startup with volume-mounted models/data. Showing you can prototype, expose, "
        "and operationalise ML is persuasive for both dissertation defence and portfolios.",
    )
    para(
        pdf,
        "Finally, transparency and critique are baked in semantically-aware keyword corpora "
        "(config.py plus data/keywords/*.txt), SHAP tooling in evaluate_models.py, and explicit "
        "warnings in evidence/EVIDENCE_INDEX.md that tree ensembles scored as perfect mirrors "
        "the deterministic labeller unless human labels arrive. Selecting models with "
        "model_selection.py demonstrates intellectual honesty: deployment choice is justified "
        "by credibility of cross-validation, not headline accuracy alone.",
    )

    # ---- Q2 -----------------------------------------------------------------
    pdf.add_page()
    heading(pdf, "Future of this project", "2")

    subh(pdf, "Near-term productisation")
    para(
        pdf,
        "The repository already documents multi-architecture Docker Hub images and "
        "docker-compose.hub.yml for pull-and-run demos. Immediate future work pushes that outward: "
        "package the Flask service behind API gateways, authenticate tenants, meter usage, "
        "and expose webhooks so Microsoft 365/Gmail-compatible connectors enqueue messages instead "
        "of manual pasting through the Rails classify form.",
    )
    subh(pdf, "Labels grounded in humans, not only heuristics")
    para(
        pdf,
        "Today's labels are semi-supervised: priority columns come from lexical cues, folder "
        "names, executive-like sender aliases, newsletters, distant-horizon downgrades configured "
        "in preprocess.py and config.py. Trees memorise those dictionaries and show perfect "
        "held-out digits even when logistic regression exposes genuine ambiguity (~74.6 percent "
        "accuracy in evaluation_report.json). The highest leverage future milestone is curator-"
        "reviewed labels on a statistically meaningful random sample stratified across domains, "
        "plus dispute resolution workflows so domain experts overturn machine suggestions. That "
        "unlocks calibrated probabilities, unbiased evaluation, and lawful deployment under "
        "EU/UK fairness expectations.",
    )
    subh(pdf, "Model evolution and monitoring")
    para(
        pdf,
        "DistilBERT fine-tuning is implemented but intentionally optional (--no-bert) for weaker "
        "hardware. Extend with multilingual models, longer contexts, attachments as extracted text, "
        "and hierarchical models that thread related messages jointly. Operational ML needs drift "
        "detection comparing live confidence histograms versus training snapshots, alerting when "
        "incoming mail shifts (new marketing templates, outages, regulatory language). Fallback "
        "logic already encodes thresholds; escalation policies can route ambiguous cases back "
        "to humans or retrain subsets monthly.",
    )
    subh(pdf, "Research integrations")
    para(
        pdf,
        "Active learning could prioritise reviewer time on borderline logits. Federated setups "
        "would let enterprises train without pooling raw mailbox contents. Combining graph "
        "signals (organisational chart, SLA timers) atop textual features aligns with intelligent "
        "ticketing startups. Publication-wise, reproducible benchmarks from Enron-derived slices "
        "plus SpamAssassin could join shared leaderboards analogous to toxicity or intent tasks.",
    )

    # ---- Q3 -----------------------------------------------------------------
    pdf.add_page()
    heading(pdf, "Why we use this model (deployment choice versus full portfolio)", "3")

    para(
        pdf,
        "The training stack deliberately fits four families: logistic regression atop TF-IDF + "
        "tabular metadata (baseline), random forest and XGBoost on the stitched numeric matrix "
        "(trees capture non-linear keyword conjunctions plus engineered counts), and "
        "fine-tunable DistilBERT over subject/body tuples for contextual semantics. README.md lists "
        "their roles plainly: baseline sanity check, nonlinear classics, boosted ensemble, neural "
        "sequence model.",
    )
    para(
        pdf,
        "However, WHICH model ships is governed by ml/model_selection.py and evaluate_models.py. "
        "The live selection recorded in ml/models/best_model.txt and evaluation_report.json is "
        "logistic regression (evaluation_date 2026-04-05 snapshot in-repo). Reason: logistic "
        "regression's cross-validation macro-F1 mean hovers ~0.697 with nonzero standard deviation, "
        "whereas random forest/XGBoost both report test accuracy and macro-F1 of 1.0 with CV "
        "means also 1.0 and practically zero variance --- textbook symptoms that features "
        'encode the heuristic labeller almost verbatim. Tree ensembles therefore look "perfect" '
        "without proving safer generalisation; they risk mirroring artefacts of the semi-"
        "supervised rules instead of human judgment on unseen phrasing.",
    )
    para(
        pdf,
        "The selection routine (_suspicious_perfect in model_selection.py) therefore flags those "
        "runs and demotes them when a non-suspicious candidate exists. Logistic regression "
        "remains interpretable (linear weights on TF-IDF dimensions), pairs naturally with SHAP "
        "LinearExplainer paths, and trains quickly (order of tens of seconds in logged runs). "
        "Macro-F1 on CV clears the nominal 0.65 gate; headline accuracy lands in the mid-74 "
        "percent band on this heuristic-labelled corpus (see evaluation_report.json for the split "
        "you ship). evaluate_models.py can recommend fallback_model.py when EITHER configured "
        "threshold fails --- explicit rules mirrored to the UI keep the demo usable.",
    )
    para(
        pdf,
        "In short: multi-model comparison demonstrates breadth of MSc-level competence; logistic "
        "regression is chosen for deployment credibility, auditability, and alignment with cautious "
        "threshold logic rather than leaderboard vanity on auto-generated pseudo-labels.",
    )

    # ---- Q4 -----------------------------------------------------------------
    pdf.add_page()
    heading(pdf, "What dataset we trained this model on", "4")

    para(
        pdf,
        "Primary bulk sources mirror academic practice: tens to hundreds of thousands of Enron "
        "messages obtainable from the CMU mirror (tarball referenced in ml/config.py) plus "
        "multiple SpamAssassin tarballs merging easy ham, harder ham, and spam eras. DOWNLOAD "
        "scripts fetch originals into ml/data/raw; preprocess.py parses RFC822 payloads, derives "
        "thread IDs, computes metadata such as counts of recipients/CC/BCC, word counts, and "
        "calendar fields, assigns folder tags when present, normalises textual fields, THEN "
        "applies the semi-supervised labeler mapping content and sender cues to priority "
        "integers {0,...,3}. README tables summarise rule families: critical merges urgency cues, "
        "C-suite shorthand, outages, legal keywords; high emphasises deadlines, RSVP, procurement, "
        "supervisor academia lexicon expansions customised for postgraduate correspondence; "
        "low hunts FYI/marketing/newsletter/unsubscribe motifs; distant-horizon lists downgrade "
        "future planning mails unless contradictory near-term urgency markers fire.",
    )
    para(
        pdf,
        "The curated artifact checked into git per DATASET_INFO.md is processed_emails.csv with "
        "4,000 rows (600 critical, 1,000 high, 1,800 normal, 600 low --- approximating 15/25/45/15 "
        "percent mix). Each row keeps message_id, addressing fields, subject, body, datetime, "
        "engineered numeric columns, string folder label, and both numeric priority and textual "
        "priority_label. Train/test evaluation uses an 80/20 stratified split (random_state=42) "
        "so every class appears proportionally in train (3,200) and test (800). feature_"
        "engineering.py serialises features.pkl containing TF-IDF matrices (vectoriser saved as "
        "tfidf_vectorizer.pkl up to TFIDF_MAX_FEATURES=5000), metadata arrays, concatenated stacks "
        "for ensembles, aligned labels vector y, plus feature naming metadata SHAP consumes.",
    )
    para(
        pdf,
        "Separate from full-scale downloads generate_dataset.py and run_mock_pipeline.py rebuild "
        "smaller illustrative corpora paired with plotting scripts under evidence/ for defence "
        "slides. Dummy scenarios in repository-root DUMMY_EMAIL_DATA.md intentionally stress-test "
        "API behaviour with synthetic hospital, outage, procurement, promotional mail without "
        "claiming identical ground truth calibration as the heuristic labels.",
    )
    para(
        pdf,
        "Students should plainly disclose limitations in write-ups: labels inherit whichever "
        "biases lurk inside keyword inventories and folder path heuristics, Enron-era language "
        "differs from 2026 enterprise tone, SpamAssassin spam is historical, and stratified splits "
        "still evaluate within the SAME generative storytelling as training (no unseen "
        "organisation). Supplementary qualitative studies or outsourced annotation would quantify "
        "that gap empirically.",
    )

    # ---- Q5 -----------------------------------------------------------------
    pdf.add_page()
    heading(pdf, "Outcome of this project", "5")

    subh(pdf, "Measured performance snapshot (on-disk artefacts)")
    para(
        pdf,
        "The April 2026 evaluation artefact chooses logistic regression with cross-validation macro-"
        "F1 approximately 0.697 (std ~0.016). Held-out test accuracy prints about 74.6 percent; CV "
        "accuracy mean sits near 74.35 percent --- both in the mid-74s, while macro-F1 comfortably "
        "exceeds the 0.65 gate in ml/config.py. Random forest trains in under a second and "
        "XGBoost roughly 40 seconds with perfect splits but CV standard deviation zero,"
        ' signalling memorisation of auto-labels --- see evidence/EVIDENCE_INDEX commentary. '
        "Re-run evaluate_models.py after edits: dual thresholds combine accuracy AND macro-F1, "
        "and needs_fallback reflects that check on the deployed candidate.",
    )
    subh(pdf, "Software artefacts learners and judges can inspect")
    para(
        pdf,
        "Users receive a reproducible codebase: Dockerfile.ml trains or serves inference; "
        "rails_app Dockerfile seeds SQLite via docker-entrypoint; routes expose dashboard, bulk "
        "history, RESTful classifications resource; MlApiClient wraps HTTParty with timeouts "
        "(see config/initializers/ml_api.rb). Flask serves /predict returning priority string, "
        "logit-like confidence maps, latency, optionally SHAP-like maps for logistic/XGB lanes, "
        "consistent JSON contract documented in README. Batch endpoints allow demos up to hundreds "
        "of rows for stress tests.",
    )
    subh(pdf, "Interpretability & trust outcomes")
    para(
        pdf,
        "Scientific upside is methodological clarity: README states goals (85--92 percent target "
        "accuracy band historically proposed) while code transparently critiques when naive "
        "metrics mislead due to heuristic labelling synergy. Keywords live in-repo for audit, SHAP "
        "hooks highlight dominant stems, dashboards visualise probabilities so non-technical users "
        "see model tentativeness. Fallback_model.py assures graceful degradation preventing blank "
        "errors if weights corrupt or thresholds trip.",
    )
    subh(pdf, "Institutional learning outcome")
    para(
        pdf,
        "Completion demonstrates ability to integrate classical ML, gradient boosting, "
        "transformers, interpretability libraries, REST microservices, Ruby MVC stack, container "
        "orchestration, and narrative documentation inside a single dissertation-sized package --- "
        "the tangible outcome is not only percentage points but a defensible engineering story "
        "with candid limitations and a roadmap (question 2) toward production-grade systems.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
