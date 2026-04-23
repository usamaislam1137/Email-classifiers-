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

# -- Keyword lists (semi-supervised + rule fallback) --------------------------
# Expanded with common business, security, compliance, academic, and ops phrasing
# (aligned with typical “high importance” / triage / bulk-mail heuristics).

CRITICAL_KEYWORDS = [
    # Urgency / response
    "urgent", "asap", "a.s.a.p.", "immediately", "right now", "without delay",
    "critical", "crisis", "emergency", "emergent", "code red",
    "action required", "immediate action", "action needed now",
    "time sensitive", "time-sensitive", "time critical",
    "deadline today", "due today", "eod today", "respond now", "respond today",
    "right away", "no later than today", "must read", "read immediately",
    "top priority", "highest priority", "high priority",
    "urgent action", "immediate attention", "needs attention now",
    "escalat", "p0", "p-0", "p1 block", "sev-0", "sev-1", "sev 0", "sev 1", "severity 1",
    # Security / reliability
    "security alert", "security incident", "data breach", "breach detected",
    "account compromised", "unauthorized access", "ransomware", "malware",
    "suspicious activity", "fraud alert", "data leak", "data loss",
    "production down", "prod down", "outage", "system down", "service down",
    "stop work", "all hands", "all-hands",
    # Legal / compliance / account risk
    "legal notice", "compliance deadline", "regulatory", "subpoena", "court order",
    "account suspended", "account locked", "locked out", "final notice", "imminent",
    "payment failed", "transaction declined", "chargeback",
    "do not forward", "eyes only",
    # Academic (imminent / formal)
    "viva", "defense", "examination", "oral exam", "resit", "plagiarism hearing",
]
HIGH_KEYWORDS = [
    # Priority / follow-up
    "important", "important update", "priority", "response requested", "rsvp",
    "please respond", "your reply", "awaiting your reply", "follow up", "follow-up", "circling back",
    "by eod", "eod", "cob", "c.o.b.", "by end of day", "end of day", "close of business", "end of week",
    "by friday", "by monday", "by tuesday", "by wednesday", "by thursday", "by sunday",
    "by 5pm", "by 5:00", "by noon", "this afternoon", "earliest convenience",
    "this week", "next week", "coming week", "shortly",
    "today", "tonight", "tomorrow", "tomorow", "a.m.", "p.m.",
    "3pm", "2pm", "4pm", "5pm", "1pm", "6pm", "7pm", "8pm", "9am", "10am", "11am", "12pm", "1am",
    "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday", "o'clock", "o clock",
    # Meetings & collaboration
    "meeting", "calendar invite", "meeting request", "reschedule", "1:1", "one on one", "one-to-one",
    "standup", "sprint", "milestone", "go-live", "golive", "launch", "rollout", "ship date",
    "board meeting", "stakeholder", "client", "qbr", "review",
    "catch up", "check-in", "check in", "attend", "attendance", "preparation", "prepare",
    # Deadlines & approvals
    "deadline", "due by", "due on", "due date", "cutoff", "cut-off", "timebox",
    "required", "mandatory", "must attend", "must complete", "acknowledg", "read & acknowledge",
    "sign-off", "sign off", "signature required", "e-sign", "docusign",
    "please review", "response needed", "action needed", "action item", "please confirm", "please approve",
    "awaiting your response", "please advise", "kindly respond", "revert at earliest",
    "by tomorrow", "request for", "update", "status update", "status report", "reminder", "agenda", "minutes",
    "dates", "schedule", "scheduling", "project", "roadmap", "scope", "budget", "quarterly",
    # Business / admin / people
    "contract", "nda", "sow", "msa", "invoice", "purchase order", "p.o.", "po number",
    "rfp", "rfq", "rft", "vendor", "procurement", "quotation",
    "interview", "onboarding", "offboarding", "hiring", "headcount", "payroll", "payslip",
    "hr ", "human resources", "background check", "reference check", "offer letter", "relocation",
    "ticket #", "case #", "incident #", "change request", "jira", "servicenow", "azure devops",
    "downtime", "maintenance window", "deployment", "release", "patch",
    # Academic
    "viva", "defense", "thesis", "dissertation", "supervisor", "m.sc", "msc", "ms.c", "phd", "ph.d",
    "module", "credits", "tutor", "lecturer", "coursework", "assignment", "bursar", "registrar",
    "enrollment", "re-enroll", "re-enrol", "extension", "turnitin", "peer review", "submission", "re-submit",
    # Time-bound (scheduling; model may disambiguate)
    "appointment reminder", "clinic", "dental", "surgery", "hr meeting",
]
LOW_KEYWORDS = [
    # FYI / no action
    "fyi", "f.y.i", "for your information", "for info only", "no action required", "no action needed",
    # Opt-out & bulk senders
    "newsletter", "e-newsletter", "bulletin", "unsubscribe", "opt out", "opt-out",
    "manage preferences", "email preferences", "list-unsubscribe", "listserv",
    "no reply", "noreply", "no-reply", "donotreply", "do not reply", "do not respond",
    "mailing list", "bcc mass", "bcc:", "mass email",
    # Automation
    "automated", "automat", "auto-generated", "auto generated", "this is an automated",
    "transactional email", "system generated", "routine notification",
    # Digests & summaries
    "digest", "weekly digest", "monthly wrap", "roundup", "recap", "tldr", "tl;dr",
    "weekly", "monthly", "quarterly newsletter", "year in review",
    # Marketing & promo
    "promotional", "promo code", "promo", "voucher", "coupon", "% off", "per cent off",
    "clearance", "limited time", "flash sale", "black friday", "cyber monday", "sweepstakes",
    "sponsored", "sponsored by", "advertisement", "anzeige", "ad:", "partner offer",
    "view in browser", "view online", "view this email in your browser", "web version",
    "if you're having trouble viewing", "cannot read this email", "add to address book",
    "read our blog", "white paper", "ebook", "infographic", "industry report",
    "webinar on-demand", "on-demand", "recording available", "watch the replay",
    "curated for you", "selected for you", "based on your interests",
    # Social & low-signal
    "social digest", "linkedin weekly", "facebook digest", "twitter digest", "from linkedin", "on linkedin: weekly",
    # Surveys & light touch
    "we value your feedback", "rate your experience", "short survey", "nps survey", "customer survey",
    "press release", "media kit", "pr team", "charity appeal", "donate link",
    # OOO / leave (often filtered)
    "out of office", "out-of-office", "ooo:", "annual leave", "on holiday", "away from email",
    # Misc low-priority markers
    "notification", "noreply@", "no-reply@", "news@", "marketing@", "promo@",
    "this message was sent to", "you are receiving this",
    "disclaimer:", "legal disclaimer", "unsubscribe at the bottom",
]

# -- Far-future planning (de-prioritise vs. immediate deadlines) ----------------
# If these appear and there is no near-term urgency, treat as normal/low planning mail.
DISTANT_HORIZON_PHRASES = [
    "next year",
    "in a year",
    "a year from now",
    "year from now",
    "following year",
    "in 12 months",
    "twelve months from now",
    "not until next year",
    "the following year",
    "in two years",
    "in 18 months",
    "end of next fiscal year",
    "next fiscal year",
    "long-term plan",
    "no rush", "whenever you get a chance", "at your leisure",
    "sometime next year", "tbc next year", "tbd next year",
]
# If any of these are present, we do not apply the distant-horizon downgrade.
NEAR_TERM_URGENCY_PHRASES = [
    "asap",
    "urgent",
    "immediately",
    "right away",
    "today",
    "tonight",
    "tomorrow",
    "tomorow",
    "this week",
    "next week",
    "deadline today",
    "by eod",
    "by end of day",
    "by cob",
    "by close",
    "by noon",
    "this afternoon",
    "by tomorrow",
    "within 24",
    "within 48",
    "within an hour",
    "within 1 hour",
    "in the next hour",
    "before 5pm",
    "before close of business",
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
