# Email Priority Classification System

**Dissertation Project - Muhammad Usama Islam (U2911515)**
MSc Artificial Intelligence - University of East London

Classifies emails into four priority levels: **Critical**, **High**, **Normal**, **Low** using
Logistic Regression, Random Forest, XGBoost, and fine-tuned DistilBERT.
Includes SHAP-based feature attribution and a full Ruby on Rails web dashboard.

---

## Project Structure

```
email_priority_system/
+-- ml/                         # Python ML pipeline
|   +-- config.py               # All paths, labels, thresholds
|   +-- download_dataset.py     # Download Enron + SpamAssassin datasets
|   +-- preprocess.py           # Parse emails, semi-supervised labelling
|   +-- feature_engineering.py  # TF-IDF, BERT embeddings, metadata
|   +-- train_models.py         # Train 4 classifiers
|   +-- evaluate_models.py      # Evaluate, pick best, compute SHAP
|   +-- fallback_model.py       # Rule-based fallback if below threshold
|   +-- predict.py              # Single-email prediction
|   +-- flask_api.py            # REST API (POST /predict, GET /health, ...)
|   +-- run_pipeline.sh         # One-command pipeline runner
|   +-- requirements.txt
|   +-- data/                   # Sample raw + processed CSVs (tracked for clone-and-run)
|   \-- models/                 # Trained weights + vectoriser + best_model.txt (tracked)
+-- rails_app/                  # Ruby 3.3.4 / Rails 7.2 web app
|   +-- app/
|   |   +-- controllers/        # DashboardController, ClassificationsController
|   |   +-- models/             # EmailClassification
|   |   +-- views/              # Dashboard, classify form, results, history
|   |   +-- services/           # MlApiClient (HTTParty)
|   |   \-- helpers/            # priority_badge, accuracy_color, ...
|   +-- config/
|   |   +-- routes.rb
|   |   \-- initializers/ml_api.rb
|   +-- db/migrate/             # SQLite3 migration
|   \-- Gemfile                 # Ruby 3.3.4, Rails 7.2
+-- Dockerfile.ml
+-- Dockerfile.rails
+-- docker-compose.yml
+-- docker-compose.hub.yml   # pull pre-built images from Docker Hub
+-- bin/push-dockerhub.sh     # tag + push images to your Hub namespace
\-- README.md
```

---

## Quick Start

### Option A - Docker Compose (recommended)

`ml/data` and `ml/models` are included in the repository so a fresh clone can run **without training first**.

```bash
cd email_priority_system
docker compose up --build
```

The Rails image runs `rails db:prepare` on every container start (see `rails_app/bin/docker-entrypoint`) so the SQLite database exists on the named Docker volume. An empty volume would otherwise hide the database files from the image build.

To **re-train** inside the ML container instead:

```bash
docker compose run --rm ml-api bash run_pipeline.sh --no-bert
```

- Rails app -> http://localhost:3000
- ML API (from your machine) -> http://localhost:5001 by default (port **5000** is often taken on macOS by AirPlay). Inside Compose, Rails still calls `http://ml-api:5000`. To map host port 5000 instead: `ML_API_HOST_PORT=5000 docker compose up`

### Docker Hub (pre-built images, multi-arch)

Images are built for **linux/amd64** and **linux/arm64** (Intel/AMD and Apple Silicon) via Buildx and pushed as a single manifest, so any common laptop or server pulls the right variant.

**Push** (after `docker login`):

```bash
cd email_priority_system
./bin/push-dockerhub.sh
```

The script reads your Docker Hub username from the `docker login` credential helper. To override: `export DOCKERHUB_USER=yourhublogin` then run the script.

It publishes `DOCKERHUB_USER/email-priority-ml-api:latest` and `DOCKERHUB_USER/email-priority-rails-app:latest`.

If `docker buildx` reports **overlay2 … input/output error**, the script now defaults to the **`docker` buildx driver** (and removes an old `docker-container` builder). You can still force the old driver with `export BUILDX_DRIVER=docker-container`. If Docker’s disk image is corrupted, use Docker Desktop → Troubleshoot → **Clean / Purge data** (or `docker system prune` after saving what you need).

**Make repositories public** (so `docker pull` works without logging in): on [Docker Hub](https://hub.docker.com) open each repository → **Settings** → **Visibility** → **Public** → Save. New repos sometimes default to private depending on account settings.

**Pull and run** on another machine (clone the repo so `./ml/data` and `./ml/models` exist for volume mounts):

```bash
cd email_priority_system
export DOCKERHUB_USER=your_dockerhub_username   # account that owns the images
docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up
```

---

### Option B - Local Development

#### 1. Python ML Pipeline

**Requirements:** Python 3.10+, ~8 GB RAM for full BERT training (DistilBERT is lighter)

```bash
cd email_priority_system/ml

# Create virtualenv and install dependencies
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the full pipeline (add --no-bert to skip BERT and train faster)
bash run_pipeline.sh

# Or run steps individually:
python download_dataset.py        # ~3 GB download
python preprocess.py
python feature_engineering.py    # add --no-bert to skip BERT embeddings
python train_models.py            # add --no-bert to skip DistilBERT fine-tuning
python evaluate_models.py
python fallback_model.py          # checks threshold, applies fallback if needed

# Start the Flask API
python flask_api.py
# API is now running on http://localhost:5000
```

**Test the API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "ceo@enron.com",
    "recipients": "all@enron.com",
    "subject": "URGENT: Board meeting rescheduled - respond NOW",
    "body": "This is an urgent matter. Please respond immediately."
  }'
```

---

#### 2. Rails Web App

**Requirements:** Ruby 3.3.4, Bundler

```bash
cd email_priority_system/rails_app

bundle install

# Create and migrate database
bundle exec rails db:create db:migrate

# Start Rails (make sure ML API is running first)
bundle exec rails server
# App is now at http://localhost:3000
```

**Environment variables (optional):**
```bash
export ML_API_URL=http://localhost:5000   # default
export ML_API_TIMEOUT=30                  # seconds
```

---

## ML Pipeline Details

### Dataset Sources

| Dataset | Source | Emails |
|---------|--------|--------|
| Enron Email Dataset | https://www.cs.cmu.edu/~enron/ | ~500k |
| SpamAssassin Public Corpus | https://spamassassin.apache.org/publiccorpus/ | ~6k |

### Priority Labelling (Semi-Supervised)

| Priority | Label | Signals |
|----------|-------|---------|
| Critical | 0 | `urgent`, `asap`, `action required`, C-suite sender, urgent folder |
| High | 1 | `important`, `deadline`, `meeting`, `please respond`, EOD requests |
| Normal | 2 | Standard business email (default) |
| Low | 3 | `fyi`, `newsletter`, `automated`, `noreply`, long forward chains |

### Models Trained

| Model | Features | Notes |
|-------|----------|-------|
| Logistic Regression | TF-IDF (5000 features) | Baseline |
| Random Forest | Full feature matrix | Metadata + TF-IDF |
| XGBoost | Full feature matrix | Best classic model |
| DistilBERT (fine-tuned) | Subject + body text | Best overall |

### Fallback Logic

If `accuracy < 0.75` OR `macro_f1 < 0.65`, the pipeline:
1. Attempts to download a pre-trained model from HuggingFace Hub
2. Falls back to a deterministic rule-based classifier (always available, no download needed)

The `models/evaluation_report.json` records which model is active and whether the fallback was triggered.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a single email |
| `POST` | `/batch_predict` | Classify up to 100 emails |
| `GET`  | `/health` | Health check + model status |
| `GET`  | `/model_info` | Active model name, accuracy, F1 |
| `GET`  | `/models` | All models + performance metrics |

### POST /predict - Request

```json
{
  "sender": "sender@example.com",
  "recipients": "recv@example.com",
  "subject": "Urgent: Contract sign-off needed today",
  "body": "Please review and sign off ASAP...",
  "date": "2025-03-16T09:00:00"
}
```

### POST /predict - Response

```json
{
  "priority": "critical",
  "priority_index": 0,
  "confidence": 0.91,
  "confidence_scores": {
    "critical": 0.91,
    "high": 0.06,
    "normal": 0.02,
    "low": 0.01
  },
  "shap_values": {
    "urgent": 0.42,
    "asap": 0.31,
    "sender_domain_enron": 0.12,
    "subject_length": -0.04
  },
  "model_used": "xgboost",
  "processing_time_ms": 38
}
```

---

## Rails Web App Pages

| Route | Page |
|-------|------|
| `/` or `/dashboard` | Overview: stats cards, priority pie chart, model status, recent history |
| `/classifications/new` | Classify email form |
| `/classifications/:id` | Result: priority badge, confidence chart, SHAP attribution chart |
| `/classifications` | Full history table with priority badges |

---

## Performance Targets (from Proposal)

| Metric | Target |
|--------|--------|
| Accuracy | 85-92% |
| Macro F1 | > 0.82 |
| Fallback threshold | accuracy < 0.75 OR macro_f1 < 0.65 |

---

## Tech Stack

**ML Pipeline:** Python 3.10, scikit-learn, XGBoost, HuggingFace Transformers, DistilBERT, SHAP, Flask, NLTK
**Web App:** Ruby 3.3.4, Rails 7.2, SQLite3, Bootstrap 5.3, Chart.js, HTTParty
**Infrastructure:** Docker, Docker Compose
