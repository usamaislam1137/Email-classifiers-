"""
Build large keyword .txt files under data/keywords/ (thousands of lines per tier).

Run once (or after edits):
    python keyword_corpora.py

Deterministic output (seed fixed) for reproducible diffs.
"""
from __future__ import annotations

import random
from pathlib import Path

TARGET_PER_FILE = 3200
SEED = 42

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "data" / "keywords"


# -----------------------------------------------------------------------------
# Seed pools for combinatorial expansion (lowercase phrases / tokens)
# -----------------------------------------------------------------------------

_CRIT_A = [
    "security", "data", "account", "payment", "legal", "compliance", "network", "production",
    "customer", "internal", "incident", "breach", "access", "credential", "password", "login",
    "session", "fraud", "charge", "invoice", "contract", "settlement", "court", "regulatory",
    "audit", "disciplinary", "safety", "clinical", "patient", "outage", "database", "backup",
    "restore", "failover", "cluster", "api", "sla", "escalation", "customer", "vip", "finance",
]

_CRIT_B = [
    "breach", "incident", "alert", "failure", "suspended", "locked", "compromised", "tampered",
    "expired", "overdue", "violation", "escalated", "p0", "p1", "sev", "severity", "outage",
    "down", "blocked", "denied", "rejected", "declined", "cancelled", "terminated", "void",
]

_CRIT_C = [
    "today", "tonight", "now", "immediately", "asap", "eod", "within the hour",
    "action required", "respond now", "read now", "confirm now",
]

HIGH_A = [
    "please", "kindly", "could you", "when you have a moment", "at your earliest",
    "reminder", "follow-up", "status on", "update on",
]

HIGH_B = [
    "review", "submit", "upload", "sign", "approve", "confirm", "attend", "join", "book",
    "complete", "fill", "return", "circulate", "share", "acknowledge",
]

HIGH_C = [
    "by eod", "by cob", "by friday", "by monday", "tomorrow", "this week", "next week",
    "before noon", "by 5pm", "within 48 hours", "as discussed",
]

NORM_A = [
    "quick", "short", "brief", "minor", "small", "light", "routine", "weekly", "daily",
]

NORM_B = [
    "update", "note", "summary", "recap", "comment", "observation", "thought", "reflection",
]

NORM_C = [
    "on the project", "for the team", "from yesterday", "for awareness", "no rush",
    "when convenient", "fyi-style", "for visibility", "looping everyone",
]

LOW_A = [
    "free", "save", "win", "shop", "deal", "offer", "%", "bonus", "points", "voucher",
    "discount", "clearance", "sale", "bogo", "member", "newsletter",
]

LOW_B = [
    "shipping", "coupon", "flash", "limited", "promo", "seasonal", "holiday", "birthday",
    "anniversary", "reward", "loyalty", "referral",
]

LOW_C = [
    "unsubscribe", "preferences", "view in browser", "click here", "opt out",
    "you are subscribed", "do not reply to this email",
]


def _dedupe_cap(lines: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in lines:
        x = x.strip().lower()
        if len(x) < 2 or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= cap:
            break
    return out


def _gen_critical(rng: random.Random) -> list[str]:
    out: list[str] = []

    # Combinatorial product (sense-ish phrases)
    for a in _CRIT_A:
        for b in _CRIT_B:
            out.append(f"{a} {b}")

    for b in _CRIT_B:
        for c in _CRIT_C:
            out.append(f"{b}: {c}")
            out.append(f"{b} — {c}")

    # Repeated patterns with RNG jitter (spam/malware corpuses often use variants)
    templates = (
        "{} issue reported",
        "{} team alert",
        "attention: {}",
        "{} — immediate review",
        "your {} requires action",
        "security: {} detected",
        "urgent [{}]",
        "priority [{}]",
        "escalated {}",
    )
    pool = list({* _CRIT_A, * _CRIT_B, * {"account", "login", "session", "payment", "credentials"}})
    rng.shuffle(pool)
    for tpl in templates:
        for word in pool:
            out.append(tpl.format(word))

    for i in range(1100):
        out.append(f"sev-queue-{i}: active")
        out.append(f"incident_ticket_c-{8800 + i}")

    return _dedupe_cap(out, TARGET_PER_FILE)


def _gen_high(rng: random.Random) -> list[str]:
    out: list[str] = []
    for a in HIGH_A:
        for b in HIGH_B:
            out.append(f"{a} {b}")
    for b in HIGH_B:
        for c in HIGH_C:
            out.append(f"{b} {c}")
    extras = ("project {}", "timeline {}", "budget {}", "milestone {}", "scope {}", "stakeholders {}")
    topics = ["update", "sync", "review", "check-in", "plan", "risk", "blocker"]
    for e in extras:
        for t in topics:
            out.append(e.replace("{}", t))

    verbs = HIGH_B[:]
    objs = ["document", "report", "slide deck", "sheet", "form", "link", "request", "thread"]
    for v in verbs:
        for o in objs:
            out.append(f"please {v} the {o}")

    depts = [
        "sales", "engineering", "finance", "legal", "people", "hr", "marketing", "operations",
        "product", "design", "research", "labs", "students", "faculty", "admin", "library",
        "it ", "security ", "compliance", "audit", "risk", "treasury", "procurement", "support",
        "success", "professional services", "consulting", "field", "delivery", "implementation",
    ]
    for x in range(80):
        depts.append(f"team_{x}")
        depts.append(f"workstream_{x}")
    kinds = [
        "invite", "checkpoint", "prep", "readthrough", "deck", "slide pack", "memo", "follow-up",
        "thread", "blocker", "dependency", "risk item", "okr", "retro", "demo", "dress rehearsal",
    ]
    for d in depts:
        for k in kinds:
            out.append(f"{d.strip()} {k} — needs input")
            out.append(f"[high] {d.strip()} {k}")

    # Numbered request patterns (helps triage systems that use ticket-like language)
    for i in range(1200):
        out.append(f"request ref #h-{12000 + i} pending")
    return _dedupe_cap(out, TARGET_PER_FILE)


def _gen_normal(rng: random.Random) -> list[str]:
    out: list[str] = []
    for a in NORM_A:
        for b in NORM_B:
            out.append(f"{a} {b}")
            for c in NORM_C:
                out.append(f"{a} {b} {c}")

    fluff = (
        "sharing for transparency",
        "not blocking on this",
        "no decision needed yet",
        "for alignment only",
        "background reading",
        "context only",
        "looping you in — fyi-style",
        "cc colleagues for visibility",
        "will send more detail later",
        "nothing urgent here",
        "carry over to next sprint planning",
        "off-cycle note",
        "offline chat notes",
        "water-cooler follow-up",
    )
    out.extend(fluff)

    # Routine subject-ish templates
    for i in range(500):
        out.append(f"team sync notes batch {i}")
        out.append(f"document version v{i} comments")
        out.append(f"archive reference id-{88000 + i}")
    for i in range(1200):
        out.append(f"routine_update_ref_n-{30000 + i}")
        out.append(f"culture_communications_round_{i}")
    return _dedupe_cap(out, TARGET_PER_FILE)


def _gen_low(rng: random.Random) -> list[str]:
    out: list[str] = []
    for a in LOW_A:
        for b in LOW_B:
            out.append(f"{a} {b}")
        for c in LOW_C:
            out.append(f"{a} — {c}")
    promo = ["spring", "summer", "winter", "fall", "eoy", "q1", "q2", "q3", "q4"]
    for p in promo:
        for n in LOW_B[:20]:
            out.append(f"{p} promo {n}")
    for i in range(1100):
        out.append(f"bulk_campaign_{i}")
        out.append(f"marketing_batch_{i}: opt-out footer")
        out.append(f"news_roundup_sector_{i}")
        out.append(f"retail_spam_pattern_{i}")
    return _dedupe_cap(out, TARGET_PER_FILE)


def write_all() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    gens = (
        ("critical.txt", _gen_critical),
        ("high.txt", _gen_high),
        ("normal.txt", _gen_normal),
        ("low.txt", _gen_low),
    )
    for name, fn in gens:
        path = OUTDIR / name
        lines = sorted(fn(rng))
        hdr = (
            "# Auto-generated by keyword_corpora.py — one phrase per line.\n"
            "# Do not edit by hand unless you know the merge rules.\n\n"
        )
        path.write_text(hdr + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path.name}: {len(lines)} lines")


if __name__ == "__main__":
    write_all()
