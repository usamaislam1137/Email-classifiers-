# Email Priority Classification Dataset

## Overview

This dataset was constructed for the dissertation **"Automated Email Priority Classification Using Machine Learning"** by Usama Islam.

It contains **4,000 real-world-style business emails** labelled across four priority classes using a semi-supervised keyword and sender-heuristic approach — the same methodology described in the dissertation proposal and applied to the Enron corpus.

---

## Files

| File | Description |
|------|-------------|
| `raw/emails_raw.csv` | Raw emails (no labels) — 4,000 records, 9 columns |
| `processed/processed_emails.csv` | Cleaned, feature-enriched, and labelled dataset — 4,000 records, 22 columns |

---

## Class Distribution

| Priority | Label    | Count | Percentage |
|----------|----------|-------|------------|
| 0        | Critical | 600   | 15%        |
| 1        | High     | 1,000 | 25%        |
| 2        | Normal   | 1,800 | 45%        |
| 3        | Low      | 600   | 15%        |
| **Total**|          | **4,000** | **100%** |

---

## Column Reference (`processed_emails.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string | Unique RFC-2822 style message identifier |
| `sender` | string | Sender email address |
| `recipients` | string | Comma-separated recipient addresses |
| `cc` | string | CC recipients (if any) |
| `bcc` | string | BCC recipients (if any) |
| `subject` | string | Email subject line |
| `body` | string | Plain-text email body |
| `date` | string | ISO 8601 datetime |
| `hour_of_day` | int | Hour of send (0–23) |
| `day_of_week` | int | Day of week (0=Mon … 6=Sun) |
| `thread_id` | string | MD5-based thread identifier (from subject) |
| `folder` | string | Mailbox folder (inbox / urgent / newsletter …) |
| `body_length` | int | Character count of body |
| `subject_length` | int | Character count of subject |
| `word_count` | int | Word count of body |
| `recipient_count` | int | Number of recipients |
| `has_cc` | int (0/1) | 1 if CC field is non-empty |
| `has_bcc` | int (0/1) | 1 if BCC field is non-empty |
| `subject_has_re` | int (0/1) | 1 if subject starts with "Re:" |
| `subject_has_fw` | int (0/1) | 1 if subject starts with "Fw:"/"Fwd:" |
| `priority` | int (0–3) | Numeric priority label |
| `priority_label` | string | String label: critical / high / normal / low |

---

## Labelling Methodology

Labels were assigned using the semi-supervised heuristic described in the system design (Chapter 3):

1. **Critical (0)** — Subject/body contains urgency keywords (*urgent, asap, immediately, action required* …) **or** sender holds a C-suite title (CEO, CFO, CTO …)
2. **High (1)** — Subject/body contains importance keywords (*important, please respond, follow up, by EOD* …)
3. **Low (3)** — Subject/body contains low-priority signals (*newsletter, unsubscribe, automated, no-reply* …) **or** sender is `noreply@…`
4. **Normal (2)** — All remaining emails

This mirrors the Enron corpus labelling described in `preprocess.py`.

---

## Train / Test Split

| Split | Records |
|-------|---------|
| Training (80%) | 3,200 |
| Test (20%)     | 800   |

Stratified split (`random_state=42`) ensures class proportions are preserved.
