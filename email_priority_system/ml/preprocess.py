"""
Preprocess raw Enron email files (maildir format) into a clean CSV.

Parsing steps
-------------
1. Walk the maildir tree, reading each raw email file.
2. Parse headers and body with the standard `email` library.
3. Extract: message_id, sender, recipients, cc, subject, body, date, thread_id, folder.
4. Apply semi-supervised priority labelling.
5. Write data/processed/processed_emails.csv.

Usage:
    python preprocess.py [--max MAX_EMAILS] [--maildir PATH]
"""
from __future__ import annotations

import sys
import os
import re
import csv
import email
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from email import policy
from email.parser import BytesParser, Parser
from typing import Optional

import chardet
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ENRON_EXTRACTED_DIR,
    PROCESSED_CSV,
    CRITICAL_KEYWORDS,
    HIGH_KEYWORDS,
    LOW_KEYWORDS,
    CSUITE_TITLES,
    MAX_EMAILS_TO_PROCESS,
    PRIORITY_LABEL_NAMES,
    LOG_FILE,
    LOG_LEVEL,
)

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)


# -- Email parsing helpers -----------------------------------------------------

def _safe_decode(raw: bytes, fallback: str = "utf-8") -> str:
    """Decode *raw* bytes, guessing encoding if needed."""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    try:
        detected = chardet.detect(raw)
        enc = detected.get("encoding") or fallback
        return raw.decode(enc, errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def _decode_header_value(value: Optional[str]) -> str:
    """Decode RFC-2047 encoded header values."""
    if not value:
        return ""
    try:
        from email.header import decode_header
        parts = decode_header(value)
        decoded = []
        for part, enc in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(enc or "utf-8", errors="replace"))
            else:
                decoded.append(str(part))
        return " ".join(decoded).strip()
    except Exception:
        return str(value).strip()


def _extract_addresses(header_value: Optional[str]) -> str:
    """Return a comma-separated list of email addresses from a header."""
    if not header_value:
        return ""
    # strip RFC-2047 display names, keep addresses
    addrs = re.findall(r"[\w.\-+]+@[\w.\-]+", header_value)
    return ", ".join(addrs)


def _get_body(msg: email.message.Message) -> str:
    """Extract plain-text body from an email.message.Message object."""
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if isinstance(payload, bytes):
                    body_parts.append(payload.decode(charset, errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            body_parts.append(payload.decode(charset, errors="replace"))
        elif isinstance(payload, str):
            body_parts.append(payload)
    return "\n".join(body_parts).strip()


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse an email Date header into a datetime."""
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    try:
        from dateutil import parser as dp
        return dp.parse(date_str)
    except Exception:
        return None


def _folder_from_path(path: Path, maildir_root: Path) -> str:
    """
    Given a file path inside the maildir, extract a meaningful folder name,
    e.g.  maildir/allen-p/inbox/1. -> 'inbox'
    """
    try:
        rel = path.relative_to(maildir_root)
        parts = rel.parts
        # parts[0] = username, parts[1] = folder, parts[2+] = subfolder/filename
        if len(parts) >= 3:
            return "/".join(parts[1:-1])
        elif len(parts) == 2:
            return parts[1]
        return "unknown"
    except ValueError:
        return "unknown"


def _thread_id(subject: str, msg_id: str) -> str:
    """Derive a rough thread ID from the subject (strip Re:/Fw: prefixes)."""
    clean = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject.lower(), flags=re.IGNORECASE).strip()
    return hashlib.md5(clean.encode()).hexdigest()[:12] if clean else msg_id[:12]


def parse_email_file(filepath: Path, maildir_root: Path) -> Optional[dict]:
    """Parse a single raw Enron email file, return a dict of fields."""
    try:
        raw = filepath.read_bytes()
    except Exception as exc:
        log.debug("Cannot read %s: %s", filepath, exc)
        return None

    text = _safe_decode(raw)
    try:
        msg = Parser().parsestr(text)
    except Exception as exc:
        log.debug("Cannot parse %s: %s", filepath, exc)
        return None

    subject = _decode_header_value(msg.get("Subject", ""))
    sender = _extract_addresses(msg.get("From", "")) or _decode_header_value(msg.get("From", ""))
    recipients = _extract_addresses(msg.get("To", ""))
    cc = _extract_addresses(msg.get("Cc", ""))
    bcc = _extract_addresses(msg.get("Bcc", ""))
    date_str = msg.get("Date", "")
    msg_id = msg.get("Message-ID", "") or str(filepath)
    body = _get_body(msg)
    folder = _folder_from_path(filepath, maildir_root)
    parsed_date = _parse_date(date_str)
    thread_id = _thread_id(subject, msg_id)

    return {
        "message_id": msg_id.strip(),
        "sender": sender.lower().strip(),
        "recipients": recipients,
        "cc": cc,
        "bcc": bcc,
        "subject": subject,
        "body": body,
        "date": parsed_date.isoformat() if parsed_date else date_str,
        "hour_of_day": parsed_date.hour if parsed_date else -1,
        "day_of_week": parsed_date.weekday() if parsed_date else -1,
        "thread_id": thread_id,
        "folder": folder,
    }


# -- Priority labelling --------------------------------------------------------

def _is_csuite(sender: str) -> bool:
    """Heuristic: does the sender email / name suggest a C-suite executive?"""
    return any(title in sender for title in CSUITE_TITLES)


def assign_priority(row: dict) -> int:
    """
    Semi-supervised priority assignment.

    Returns:
        0 = Critical
        1 = High
        2 = Normal
        3 = Low
    """
    subject = str(row.get("subject", "")).lower()
    body_snippet = str(row.get("body", "")).lower()[:500]
    folder = str(row.get("folder", "")).lower()
    sender = str(row.get("sender", "")).lower()
    text = subject + " " + body_snippet

    # -- Critical --------------------------------------------------------------
    if (
        any(kw in text for kw in CRITICAL_KEYWORDS)
        or "urgent" in folder
        or "critical" in folder
        or _is_csuite(sender)
    ):
        return 0

    # -- High ------------------------------------------------------------------
    if any(kw in text for kw in HIGH_KEYWORDS):
        return 1

    # -- Low -------------------------------------------------------------------
    if (
        any(kw in text for kw in LOW_KEYWORDS)
        or "spam" in folder
        or "junk" in folder
        or sender.startswith("no-reply")
        or sender.startswith("noreply")
        or "newsletter" in folder
        or "notification" in folder
    ):
        return 3

    # -- Normal ----------------------------------------------------------------
    return 2


# -- Main preprocessing pipeline ----------------------------------------------

def find_email_files(maildir_root: Path) -> list[Path]:
    """Recursively find all email files under *maildir_root*."""
    files = []
    for p in maildir_root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            files.append(p)
    return files


def preprocess(maildir_root: Path = ENRON_EXTRACTED_DIR,
               output_csv: Path = PROCESSED_CSV,
               max_emails: Optional[int] = MAX_EMAILS_TO_PROCESS) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Returns a DataFrame of processed emails with priority labels.
    """
    if not maildir_root.exists():
        raise FileNotFoundError(
            f"Maildir not found at {maildir_root}. "
            "Run download_dataset.py first."
        )

    log.info("Scanning email files under %s ...", maildir_root)
    all_files = find_email_files(maildir_root)
    log.info("Found %d email files.", len(all_files))

    if max_emails and len(all_files) > max_emails:
        import random
        random.seed(42)
        random.shuffle(all_files)
        all_files = all_files[:max_emails]
        log.info("Capped at %d emails for processing.", max_emails)

    records: list[dict] = []
    skipped = 0

    for filepath in tqdm(all_files, desc="Parsing emails", unit="email"):
        record = parse_email_file(filepath, maildir_root)
        if record is None:
            skipped += 1
            continue
        # Skip emails with no sender or subject and very short body
        if not record["sender"] and not record["subject"]:
            skipped += 1
            continue
        record["priority"] = assign_priority(record)
        record["priority_label"] = PRIORITY_LABEL_NAMES[record["priority"]]
        records.append(record)

    log.info("Parsed %d emails; skipped %d.", len(records), skipped)

    if not records:
        raise RuntimeError("No emails were successfully parsed.")

    df = pd.DataFrame(records)

    # -- Post-processing -------------------------------------------------------
    df["body_length"] = df["body"].str.len()
    df["subject_length"] = df["subject"].str.len()
    df["word_count"] = df["body"].str.split().str.len()
    df["recipient_count"] = df["recipients"].apply(
        lambda x: len(str(x).split(",")) if x else 0
    )
    df["has_cc"] = df["cc"].apply(lambda x: int(bool(x and str(x).strip())))
    df["has_bcc"] = df["bcc"].apply(lambda x: int(bool(x and str(x).strip())))
    df["subject_has_re"] = df["subject"].str.lower().str.startswith("re:").astype(int)
    df["subject_has_fw"] = (
        df["subject"].str.lower().str.startswith("fw:")
        | df["subject"].str.lower().str.startswith("fwd:")
    ).astype(int)
    df["is_reply"] = df["subject_has_re"]
    df["is_forward"] = df["subject_has_fw"]

    # Drop duplicates by message_id
    before = len(df)
    df = df.drop_duplicates(subset=["message_id"])
    log.info("Dropped %d duplicate emails.", before - len(df))

    # Label distribution
    dist = df["priority_label"].value_counts()
    log.info("Priority distribution:\n%s", dist.to_string())

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    log.info("Saved processed CSV to %s (%d rows).", output_csv, len(df))

    return df


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Enron emails.")
    parser.add_argument(
        "--maildir",
        type=Path,
        default=ENRON_EXTRACTED_DIR,
        help="Path to the Enron maildir root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_EMAILS_TO_PROCESS,
        help="Maximum number of emails to process (0 = all).",
    )
    args = parser.parse_args()
    max_emails = args.max if args.max and args.max > 0 else None
    preprocess(maildir_root=args.maildir, output_csv=args.output, max_emails=max_emails)


if __name__ == "__main__":
    main()
