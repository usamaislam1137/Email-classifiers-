#!/usr/bin/env python3
"""
One-off builder: friendly project overview PDF (run from repo root or any cwd).
Output: PROJECT_GUIDE.pdf next to this script's parent (workspace root).
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PROJECT_GUIDE.pdf"


class GuidePDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 8, "Email Priority Classification System - Project Guide", align="C")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def paragraph(pdf: GuidePDF, text: str, size: int = 11) -> None:
    pdf.set_font("Helvetica", size=size)
    pdf.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def heading(pdf: GuidePDF, title: str, level: int = 1) -> None:
    pdf.ln(4 if level == 1 else 2)
    if level == 1:
        hsize = 16
    elif level == 2:
        hsize = 13
    else:
        hsize = 11
    pdf.set_font("Helvetica", "B", hsize)
    pdf.multi_cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def file_row(pdf: GuidePDF, path: str, desc: str) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 4.5, path, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 4.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def main() -> None:
    pdf = GuidePDF()
    pdf.add_page()

    # --- Cover ---
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(36)
    pdf.multi_cell(
        0,
        10,
        "Email Priority Classification System",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.set_font("Helvetica", size=14)
    pdf.ln(6)
    pdf.multi_cell(
        0,
        8,
        "A friendly guide to what this project does\nand what each file is for",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.ln(20)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        6,
        "This document walks you through the whole workspace in plain language: "
        "the web app, the machine learning pipeline, Docker setup, and evidence "
        "from experiments. It is meant for readers who are new to the codebase.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )

    pdf.add_page()
    heading(pdf, "What this project is about", 1)
    paragraph(
        pdf,
        "This is a dissertation-style system that helps decide how urgent an email "
        "probably is. You paste (or send) typical email fields: sender, recipients, "
        "subject, body, and an optional date. The system returns one of four priority "
        "levels: critical, high, normal, or low, plus confidence scores so you can see "
        "how sure the model is.",
    )
    paragraph(
        pdf,
        "Two main parts work together. A Ruby on Rails web application gives you a "
        "simple dashboard and a form to classify emails; it stores past results in a "
        "database. A separate Python service exposes a small HTTP API that does the "
        "actual scoring. In production-style setups, both can run in Docker so they "
        "start together and talk over the internal network.",
    )
    paragraph(
        pdf,
        "Behind the scenes, a training pipeline can build labelled datasets (from "
        "public email archives or from a synthetic generator), engineer features "
        "(metadata plus text statistics and TF-IDF, with an optional deep text model), "
        "train several classifiers, pick a strong default, and save evaluation "
        "reports. The live API loads whichever model is marked as best and falls back "
        "to clear keyword rules if files are missing or quality thresholds are not met.",
    )

    heading(pdf, "How the pieces fit together", 2)
    paragraph(
        pdf,
        "1) User fills the form in Rails. 2) Rails calls POST /predict on the Python "
        "service. 3) Python runs the saved classifier (or fallback rules). 4) JSON "
        "comes back with priority, confidence, optional explanation features, and "
        "timing. 5) Rails saves a row you can revisit on the history and detail pages.",
    )

    heading(pdf, "Runtime folders (not committed as source)", 2)
    paragraph(
        pdf,
        "When you run the ML scripts, they create folders such as ml/data/ (raw and "
        "processed CSVs) and ml/models/ (saved classifiers, vectoriser, logs). These "
        "are working directories, not hand-written source files, but they matter when "
        "you train or deploy.",
    )

    pdf.add_page()
    heading(pdf, "Every file in this repository", 1)
    paragraph(
        pdf,
        "Below is a complete list of tracked source and config files in the workspace, "
        "grouped by folder. Paths are relative to the repository root.",
        10,
    )

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Repository root",
            [
                (
                    ".gitignore",
                    "Tells Git which generated or local files to skip at the top level.",
                ),
                (
                    "DUMMY_EMAIL_DATA.md",
                    "Extra sample emails and JSON snippets for manual testing of the "
                    "form and API beyond the three quick samples in the Rails controller.",
                ),
            ],
        ),
        (
            "email_priority_system/",
            [
                (
                    ".gitignore",
                    "Ignores Python caches, data dumps, trained weights, and other "
                    "generated artefacts inside this subsystem.",
                ),
                (
                    "docker-compose.yml",
                    "Defines the ML API container and Rails app, ports, shared volumes, "
                    "environment variables, and startup order with a health check.",
                ),
                (
                    "Dockerfile.ml",
                    "Container recipe for the Python stack: dependencies, app code, "
                    "and how the API process is launched.",
                ),
                (
                    "Dockerfile.rails",
                    "Container recipe for the Rails app: Ruby, gems, assets, and server.",
                ),
            ],
        ),
        (
            "email_priority_system/ml/",
            [
                (
                    "config.py",
                    "Single place for paths, dataset download URLs, model filenames, "
                    "the four priority labels, keyword lists used for rule-based "
                    "labelling, and training or API tuning constants.",
                ),
                (
                    "requirements.txt",
                    "Pinned Python libraries for preprocessing, training, optional deep "
                    "text training, and the Flask API.",
                ),
                (
                    "download_dataset.py",
                    "Fetches public email corpora (for example Enron and SpamAssassin) "
                    "into the expected raw data layout.",
                ),
                (
                    "preprocess.py",
                    "Walks raw mail folders, parses headers and bodies, assigns priority "
                    "labels using keyword rules, and writes a clean processed CSV.",
                ),
                (
                    "generate_dataset.py",
                    "Builds a synthetic labelled dataset that mirrors the real pipeline "
                    "format, useful for quick demos without large downloads.",
                ),
                (
                    "feature_engineering.py",
                    "Builds numeric features from each email (metadata, urgency cues, "
                    "TF-IDF text vectors, optional transformer embeddings) and saves a "
                    "feature bundle for training.",
                ),
                (
                    "train_models.py",
                    "Trains and compares several classifiers (for example logistic "
                    "regression, tree ensembles, and an optional fine-tuned text model), "
                    "runs cross-validation where configured, and writes models plus "
                    "training_results.json.",
                ),
                (
                    "evaluate_models.py",
                    "Scores trained models on a hold-out set and produces detailed "
                    "metrics and plots for analysis and dissertation evidence.",
                ),
                (
                    "predict.py",
                    "Loads the chosen on-disk model (or a safe fallback), runs "
                    "inference on one email dict, and returns priority, probabilities, "
                    "optional feature attributions, and timing.",
                ),
                (
                    "fallback_model.py",
                    "Supplies a transparent rule-based classifier and related logic "
                    "when automated quality checks suggest relying on explicit keyword "
                    "and sender heuristics instead of a trained file.",
                ),
                (
                    "flask_api.py",
                    "HTTP service: health check, single and batch predict, and "
                    "endpoints that describe which model is active and how it performed.",
                ),
                (
                    "run_pipeline.sh",
                    "Shell script that chains the main offline steps from download "
                    "through training for convenience.",
                ),
                (
                    "run_mock_pipeline.py",
                    "End-to-end demo using the synthetic dataset and copying evidence "
                    "artefacts for reports.",
                ),
            ],
        ),
        (
            "email_priority_system/rails_app/",
            [
                (".ruby-version", "Documents the Ruby version expected for this app."),
                ("Gemfile", "Declares Ruby gem dependencies (Rails, HTTP client, DB, etc.)."),
                (
                    "Gemfile.lock",
                    "Exact resolved versions of every gem for reproducible installs.",
                ),
                ("Rakefile", "Entry point for Rake tasks provided by Rails and plugins."),
                ("config.ru", "Rack configuration that boots the Rails application."),
                (
                    "bin/rails",
                    "Wrapper to run Rails commands inside the bundled environment.",
                ),
                ("bin/rake", "Wrapper to run Rake tasks with the correct bundle."),
                ("bin/bundle", "Bundler executable stub for this project."),
                (
                    "bin/setup",
                    "Script to install gems, prepare the database, and other first-time "
                    "setup steps.",
                ),
                (
                    "config/application.rb",
                    "Defines the Rails application class, autoload paths, and default "
                    "framework settings.",
                ),
                (
                    "config/boot.rb",
                    "Loads Bundler before the rest of the configuration stack.",
                ),
                (
                    "config/environment.rb",
                    "Loads the full Rails environment for console, server, and tasks.",
                ),
                (
                    "config/database.yml",
                    "Database connection settings for development, test, and production.",
                ),
                (
                    "config/puma.rb",
                    "Configures the Puma HTTP server: threads, workers, and bind address.",
                ),
                (
                    "config/importmap.rb",
                    "Maps logical JavaScript module names to files for the browser.",
                ),
                (
                    "config/routes.rb",
                    "Maps URLs to controllers: home dashboard, classifications CRUD, "
                    "and health endpoints.",
                ),
                (
                    "config/environments/development.rb",
                    "Developer-friendly defaults: verbose errors, asset reloading, etc.",
                ),
                (
                    "config/environments/production.rb",
                    "Hardened defaults for deployment: caching, logging, and security.",
                ),
                (
                    "config/environments/test.rb",
                    "Settings used when running the automated test suite.",
                ),
                (
                    "config/initializers/ml_api.rb",
                    "Sets ML_API_URL and timeout so the Rails app knows where to call "
                    "the Python service.",
                ),
                (
                    "config/initializers/inflections.rb",
                    "Optional custom singular or plural word rules for Rails.",
                ),
                (
                    "db/migrate/20250316000001_create_email_classifications.rb",
                    "Migration that creates the table for stored classification results.",
                ),
                (
                    "db/schema.rb",
                    "Current database structure as inferred from migrations (used by "
                    "tools and new clones).",
                ),
                (
                    "app/models/application_record.rb",
                    "Base Active Record model class for shared behaviour.",
                ),
                (
                    "app/models/email_classification.rb",
                    "Model for one saved classification: email fields, priority, "
                    "confidence, JSON score blobs, model name, and timing.",
                ),
                (
                    "app/controllers/application_controller.rb",
                    "Base controller: filters and behaviour shared by all controllers.",
                ),
                (
                    "app/controllers/classifications_controller.rb",
                    "New form, create (calls ML API then saves), index, show, destroy.",
                ),
                (
                    "app/controllers/dashboard_controller.rb",
                    "Home page statistics, recent rows, ML API health, and JSON health "
                    "for monitoring.",
                ),
                (
                    "app/helpers/application_helper.rb",
                    "Small view helpers shared across templates.",
                ),
                (
                    "app/services/ml_api_client.rb",
                    "Wraps HTTParty calls to predict, batch predict, health, and model "
                    "metadata with consistent error handling.",
                ),
                (
                    "app/views/layouts/application.html.erb",
                    "Site-wide HTML shell: title, navigation, yield, flash partial.",
                ),
                (
                    "app/views/shared/_flash.html.erb",
                    "Renders success and error alert banners.",
                ),
                (
                    "app/views/dashboard/index.html.erb",
                    "Dashboard: counts by priority, averages, API status, recent list.",
                ),
                (
                    "app/views/classifications/index.html.erb",
                    "Table of recent saved classifications with links to details.",
                ),
                (
                    "app/views/classifications/new.html.erb",
                    "Form to submit sender, recipients, subject, body, date for scoring.",
                ),
                (
                    "app/views/classifications/show.html.erb",
                    "Detail page for one result: priority, confidence breakdown, top "
                    "explanation features.",
                ),
                (
                    "app/assets/stylesheets/application.css",
                    "Global CSS for layout, typography, and components.",
                ),
                (
                    "app/javascript/application.js",
                    "Front-end JavaScript entry point wired through importmap.",
                ),
            ],
        ),
        (
            "evidence/",
            [
                (
                    "EVIDENCE_INDEX.md",
                    "Index of evaluation outputs: what each plot or report file means "
                    "and how to reproduce the pipeline.",
                ),
                (
                    "classification_reports.txt",
                    "Plain-text per-model classification reports and confusion summaries.",
                ),
                (
                    "evaluation_report.json",
                    "Structured metrics from evaluation runs (accuracy, F1, matrices).",
                ),
                (
                    "training_results.json",
                    "Structured training metadata: sizes, timings, cross-validation.",
                ),
                (
                    "training_log.txt",
                    "Chronological log of pipeline steps for audit or dissertation "
                    "appendices.",
                ),
            ],
        ),
        (
            "scripts/",
            [
                (
                    "scripts/build_project_guide_pdf.py",
                    "This helper script: regenerates PROJECT_GUIDE.pdf from the latest "
                    "repository layout.",
                ),
            ],
        ),
    ]

    for section_title, rows in sections:
        heading(pdf, section_title, 2)
        for path, desc in rows:
            file_row(pdf, path, desc)
        pdf.ln(2)

    heading(pdf, "Closing notes", 2)
    paragraph(
        pdf,
        "If you add new source files, rerun scripts/build_project_guide_pdf.py after "
        "updating the script's file list so the PDF stays complete. For questions about "
        "metrics and charts, start with evidence/EVIDENCE_INDEX.md and the JSON "
        "reports in the same folder.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
