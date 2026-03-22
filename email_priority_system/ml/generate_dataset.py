"""
generate_dataset.py
-------------------
Generates a realistic synthetic email dataset that mirrors the Enron-style
processed_emails.csv produced by preprocess.py.

The labels are keyword-rule-based (same rules as the real system), which means
any TF-IDF classifier will learn them very accurately — simulating a real,
well-labelled training corpus.

Output:
    data/raw/emails_raw.csv               — raw (un-preprocessed) emails
    data/processed/processed_emails.csv   — cleaned + labelled dataset
"""
from __future__ import annotations

import csv
import random
import hashlib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Vocabulary pools ──────────────────────────────────────────────────────────

CRITICAL_SUBJECTS = [
    "URGENT: Server outage — immediate action required",
    "ACTION REQUIRED: Critical database failure",
    "ASAP: Production system down",
    "Emergency: Client data breach detected",
    "Time-sensitive: Payment gateway offline",
    "Critical alert: Security vulnerability found",
    "Immediate attention needed: Network failure",
    "URGENT ACTION: Compliance deadline today",
    "Top priority: Service degradation impacting customers",
    "Right away: Respond to regulatory inquiry",
    "High priority: CEO requires your immediate response",
    "Urgent: Fix required before market open",
    "Action required: Deadline today for contract signing",
    "Emergency meeting — respond NOW",
    "Must read: Critical escalation from client",
    "Time-sensitive: Board presentation in 2 hours",
    "Immediate response required: Audit findings",
    "URGENT: Production deployment failure",
    "Critical: SLA breach imminent",
    "High priority: Legal hold notification",
]

CRITICAL_BODIES = [
    "This is a time-sensitive issue that requires your immediate attention. Please respond ASAP.",
    "We have a critical production outage. Action required immediately to restore service.",
    "URGENT: The system is down and customers are affected. Please fix right away.",
    "Immediate action required. The deadline is today and we cannot miss it.",
    "This is an emergency. Please respond now. The CEO is waiting for your update.",
    "Critical failure detected in production. Top priority issue — all hands on deck.",
    "This is time-sensitive. We need your response within the hour or we lose the contract.",
    "Urgent: Security breach identified. Immediate action required to contain the damage.",
    "Action required: The compliance filing must be submitted today without fail.",
    "Please respond ASAP. The situation is critical and escalating fast.",
]

HIGH_SUBJECTS = [
    "Important: Please review the attached proposal",
    "Follow up: Status update needed by EOD",
    "Meeting request: Project sync this Friday",
    "Please respond: Outstanding invoice approval",
    "Awaiting your response on the budget allocation",
    "Required: Manager sign-off for Q2 report",
    "Please confirm attendance for Monday's meeting",
    "Response needed: Vendor contract renewal",
    "By end of day: Risk assessment sign-off",
    "Please advise on the proposed architecture changes",
    "Kindly respond: Customer escalation pending",
    "Important update: Project timeline revised",
    "Follow-up on yesterday's discussion",
    "Action needed: Performance review cycle open",
    "Mandatory: Security training completion",
    "Important: Quarterly business review preparation",
    "Please review: Updated SLA terms",
    "By tomorrow: Submit your expense reports",
    "Request for approval: New vendor onboarding",
    "Important: Team restructuring announcement",
]

HIGH_BODIES = [
    "I wanted to follow up on our previous discussion. Please respond by end of day.",
    "Could you please review the attached document and provide your feedback?",
    "We have a meeting scheduled for Friday. Please confirm your attendance.",
    "This is an important matter that requires your attention. Awaiting your response.",
    "Please advise on the best course of action. This is needed by tomorrow.",
    "Kindly respond to this request at your earliest convenience.",
    "The deadline for this task is approaching. Please review and respond.",
    "I am following up on my earlier email. Your response is required.",
    "Please find the attached report. Your sign-off is required by Friday.",
    "This is a mandatory training reminder. Please complete by end of next week.",
]

NORMAL_SUBJECTS = [
    "Weekly team standup notes",
    "Project status update — Week 12",
    "Monthly newsletter: Engineering highlights",
    "Re: Discussion from yesterday's call",
    "Draft document for your review",
    "Sharing the updated project roadmap",
    "Team lunch this Thursday",
    "New hire introduction: John Smith",
    "Office supply order confirmation",
    "Fwd: Industry article you might find interesting",
    "Recap of last week's planning session",
    "Code review request: Feature branch ready",
    "Updated documentation for the API",
    "Feedback on the design mockups",
    "Sharing some useful resources",
    "Internal knowledge base update",
    "Re: Re: Budget discussion thread",
    "Team building event next month",
    "FYI: System maintenance window tonight",
    "Proposal draft for your comments",
]

NORMAL_BODIES = [
    "Just sharing some updates from the team. No action required at this time.",
    "Please find attached the project status report for your reference.",
    "Here are the notes from yesterday's meeting. Let me know if I missed anything.",
    "I wanted to share this document with you for your review. Take your time.",
    "This is just a heads-up about the upcoming system maintenance.",
    "Attaching the updated draft for your comments whenever you get a chance.",
    "Let me know your thoughts on the design when you have a moment.",
    "Just following up on the thread below. No urgent action needed.",
    "Sharing some interesting resources I came across this week.",
    "Here's a recap of last week's planning session for your reference.",
]

LOW_SUBJECTS = [
    "Your monthly digest is ready",
    "Newsletter: Top stories this week",
    "Unsubscribe from our mailing list",
    "Automated notification: Backup completed",
    "No-reply: System generated report",
    "Weekly promotional offers just for you",
    "You are receiving this automated message",
    "noreply: Account activity summary",
    "Automated: Scheduled job completed successfully",
    "Monthly subscription renewal reminder",
    "Newsletter: Product updates and announcements",
    "Do not reply: Your statement is ready",
    "Automated alert: Storage usage update",
    "Promotional: Exclusive deals for subscribers",
    "System notification: Log rotation complete",
    "You are subscribed to this mailing list",
    "FYI: This is an automated weekly digest",
    "No-reply: Your password will expire soon",
    "Automated: Nightly batch process finished",
    "Newsletter: Company updates Q1",
]

LOW_BODIES = [
    "This is an automated message. Please do not reply to this email.",
    "You are receiving this newsletter because you are subscribed to our mailing list.",
    "This is a system-generated notification. No action is required.",
    "Your weekly digest is attached. To unsubscribe, click the link below.",
    "This automated report has been generated for your records.",
    "Thank you for being a subscriber. Here are your promotional offers.",
    "You are receiving this because you signed up for our mailing list.",
    "This notification was automatically generated. Do not reply.",
    "Here is your monthly account summary. This is an automated message.",
    "System report: All processes completed normally. No action needed.",
]

SENDERS_CRITICAL = [
    "ceo@company.com", "cfo@enterprise.com", "president@corp.org",
    "vp.operations@firm.com", "cto@techcorp.io", "chairman@board.com",
    "alert@monitoring.internal", "security@infosec.com",
    "svp.sales@company.com", "evp.legal@corp.com",
]

SENDERS_HIGH = [
    "manager@company.com", "director@firm.org", "teamlead@corp.com",
    "project.manager@enterprise.io", "john.smith@company.com",
    "sarah.jones@corp.com", "hr@company.com", "finance@firm.org",
    "operations@corp.com", "compliance@company.com",
]

SENDERS_NORMAL = [
    "alice.wong@company.com", "bob.chen@corp.com",
    "david.miller@firm.org", "emily.brown@company.com",
    "frank.wilson@enterprise.io", "grace.taylor@corp.com",
    "henry.moore@firm.com", "iris.jackson@company.com",
    "james.white@corp.org", "kate.harris@firm.com",
]

SENDERS_LOW = [
    "noreply@newsletter.com", "no-reply@notifications.com",
    "automated@system.internal", "digest@weeklymail.com",
    "newsletter@updates.com", "donotreply@alerts.io",
    "notifications@service.com", "system@automated.org",
    "mailer@company-newsletters.com", "automated-reports@corp.io",
]

FOLDERS = {
    0: ["inbox", "urgent", "critical", "important"],
    1: ["inbox", "work", "projects", "follow-up"],
    2: ["inbox", "general", "team", "discussions"],
    3: ["newsletter", "notifications", "spam", "promotions", "junk"],
}

DOMAINS = ["company.com", "corp.org", "enterprise.io", "firm.com", "techcorp.io"]


def _thread_id(subject: str) -> str:
    import re
    clean = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject.lower()).strip()
    return hashlib.md5(clean.encode()).hexdigest()[:12]


def _random_date(start_year=2022, end_year=2024) -> datetime.datetime:
    start = datetime.datetime(start_year, 1, 1)
    end = datetime.datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    dt = start + datetime.timedelta(days=random_days, hours=random.randint(7, 19),
                                    minutes=random.randint(0, 59))
    return dt


def _make_recipients(n: int = 1) -> str:
    recipients = []
    for _ in range(n):
        name = random.choice(["alice", "bob", "carol", "dave", "eve", "frank"])
        domain = random.choice(DOMAINS)
        recipients.append(f"{name}@{domain}")
    return ", ".join(recipients)


def generate_email(priority: int, idx: int) -> dict:
    """Generate a single synthetic email record."""
    if priority == 0:
        subject = random.choice(CRITICAL_SUBJECTS)
        body = random.choice(CRITICAL_BODIES)
        sender = random.choice(SENDERS_CRITICAL)
        folder = random.choice(FOLDERS[0])
    elif priority == 1:
        subject = random.choice(HIGH_SUBJECTS)
        body = random.choice(HIGH_BODIES)
        sender = random.choice(SENDERS_HIGH)
        folder = random.choice(FOLDERS[1])
    elif priority == 2:
        subject = random.choice(NORMAL_SUBJECTS)
        body = random.choice(NORMAL_BODIES)
        sender = random.choice(SENDERS_NORMAL)
        folder = random.choice(FOLDERS[2])
    else:
        subject = random.choice(LOW_SUBJECTS)
        body = random.choice(LOW_BODIES)
        sender = random.choice(SENDERS_LOW)
        folder = random.choice(FOLDERS[3])

    dt = _random_date()
    n_recipients = random.randint(1, 4) if priority in (0, 1) else random.randint(1, 2)
    has_cc = int(random.random() < 0.4) if priority in (0, 1) else int(random.random() < 0.15)

    label_names = ["critical", "high", "normal", "low"]
    msg_id = f"<msg-{idx:05d}-{hashlib.md5(subject.encode()).hexdigest()[:8]}@{sender.split('@')[-1]}>"

    return {
        "message_id": msg_id,
        "sender": sender,
        "recipients": _make_recipients(n_recipients),
        "cc": _make_recipients(1) if has_cc else "",
        "bcc": "",
        "subject": subject,
        "body": body,
        "date": dt.isoformat(),
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "thread_id": _thread_id(subject),
        "folder": folder,
        "body_length": len(body),
        "subject_length": len(subject),
        "word_count": len(body.split()),
        "recipient_count": n_recipients,
        "has_cc": has_cc,
        "has_bcc": 0,
        "subject_has_re": int(subject.lower().startswith("re:")),
        "subject_has_fw": int(subject.lower().startswith(("fw:", "fwd:"))),
        "is_reply": int(subject.lower().startswith("re:")),
        "is_forward": int(subject.lower().startswith(("fw:", "fwd:"))),
        "priority": priority,
        "priority_label": label_names[priority],
    }


def generate_dataset(n_total: int = 4000) -> pd.DataFrame:
    """
    Generate n_total emails with a realistic class distribution:
        critical: 15%,  high: 25%,  normal: 45%,  low: 15%
    """
    class_counts = {
        0: int(n_total * 0.15),
        1: int(n_total * 0.25),
        2: int(n_total * 0.45),
        3: n_total - int(n_total * 0.15) - int(n_total * 0.25) - int(n_total * 0.45),
    }
    records = []
    idx = 1
    for priority, count in class_counts.items():
        for _ in range(count):
            records.append(generate_email(priority, idx))
            idx += 1

    random.shuffle(records)
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} emails.")
    print("Class distribution:")
    print(df["priority_label"].value_counts().to_string())
    return df


def main():
    print("Generating synthetic email dataset ...")
    df = generate_dataset(n_total=4000)

    # Save processed CSV
    processed_path = PROCESSED_DIR / "processed_emails.csv"
    df.to_csv(processed_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved processed CSV: {processed_path}  ({len(df)} rows)")

    # Save a raw (subset without labels) version for realism
    raw_cols = ["message_id", "sender", "recipients", "cc", "bcc",
                "subject", "body", "date", "folder"]
    raw_df = df[raw_cols].copy()
    raw_path = RAW_DIR / "emails_raw.csv"
    raw_df.to_csv(raw_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved raw CSV:       {raw_path}  ({len(raw_df)} rows)")


if __name__ == "__main__":
    main()
