"""
Configuration for Email Priority Classification System.
"""
import os
from pathlib import Path

# -- Base paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Dataset URLs --------------------------------------------------------------
ENRON_DATASET_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ENRON_ARCHIVE_NAME = "enron_mail_20150507.tar.gz"
ENRON_EXTRACTED_DIR = RAW_DATA_DIR / "maildir"

SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
SPAMASSASSIN_FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
]

# -- Processed data paths ------------------------------------------------------
PROCESSED_CSV = PROCESSED_DATA_DIR / "processed_emails.csv"
FEATURES_PKL = PROCESSED_DATA_DIR / "features.pkl"
LABEL_ENCODER_PKL = MODELS_DIR / "label_encoder.pkl"
TFIDF_VECTORIZER_PKL = MODELS_DIR / "tfidf_vectorizer.pkl"

# -- Model paths ---------------------------------------------------------------
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
BERT_MODEL_DIR = MODELS_DIR / "distilbert_finetuned"
FALLBACK_MODEL_PATH = MODELS_DIR / "fallback_model.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_model.txt"
TRAINING_RESULTS_JSON = MODELS_DIR / "training_results.json"
EVALUATION_REPORT_JSON = MODELS_DIR / "evaluation_report.json"

# -- Priority labels -----------------------------------------------------------
PRIORITY_LABELS = {
    0: "critical",
    1: "high",
    2: "normal",
    3: "low",
}
PRIORITY_LABEL_NAMES = ["critical", "high", "normal", "low"]
NUM_CLASSES = 4

# -- Keyword lists for semi-supervised labeling --------------------------------
CRITICAL_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "action required",
    "time sensitive", "emergency", "deadline today", "respond now",
    "right away", "top priority", "high priority", "must read",
    "time-sensitive", "urgent action", "immediate attention",
]
HIGH_KEYWORDS = [
    "important", "please respond", "follow up", "by eod", "by end of day",
    "by friday", "meeting", "deadline", "required", "mandatory",
    "please review", "response needed", "action needed", "please confirm",
    "awaiting your response", "please advise", "kindly respond",
    "by tomorrow", "by monday", "request for", "follow-up",
]
LOW_KEYWORDS = [
    "fyi", "newsletter", "unsubscribe", "no reply", "automated",
    "noreply", "notification", "digest", "weekly", "monthly",
    "no-reply", "do not reply", "you are receiving this", "mailing list",
    "this is an automated", "subscription", "promotional", "advertisement",
]

# -- C-suite sender patterns ---------------------------------------------------
CSUITE_TITLES = ["ceo", "cfo", "coo", "cto", "president", "chairman", "vp", "svp", "evp"]

# -- Feature engineering settings ---------------------------------------------
TFIDF_MAX_FEATURES = 5000
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 32
BERT_EMBEDDING_DIM = 768

# -- Training settings ---------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
MAX_EMAILS_TO_PROCESS = 50000   # cap for development; set None for all

# -- Fine-tuning settings ------------------------------------------------------
BERT_FINETUNE_EPOCHS = 3
BERT_FINETUNE_BATCH_SIZE = 16
BERT_FINETUNE_LR = 2e-5
BERT_FINETUNE_MAX_STEPS = 1000  # limit for feasibility

# -- Performance thresholds ----------------------------------------------------
ACCURACY_THRESHOLD = 0.75
MACRO_F1_THRESHOLD = 0.65

# -- Flask API settings --------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# -- HuggingFace fallback ------------------------------------------------------
HF_FALLBACK_REPO = "distilbert-base-uncased"   # placeholder; replaced if community model exists
HF_CACHE_DIR = BASE_DIR / "hf_cache"

# -- Logging -------------------------------------------------------------------
LOG_FILE = BASE_DIR / "pipeline.log"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
