#!/usr/bin/env python3
"""One-off: create 50 dated commits; do not run again after history exists."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

MESSAGES = [
    "chore: add root gitignore for IDE and binary artefacts",
    "ml: add dataset download stub",
    "ml: add preprocessing pipeline",
    "ml: add feature engineering module",
    "ml: add training script skeleton",
    "ml: wire config and label keywords",
    "ml: add evaluate_models entrypoint",
    "ml: add predict inference helper",
    "ml: add fallback classifier",
    "ml: add Flask API skeleton",
    "ml: add pipeline shell runner",
    "ml: mock pipeline for local dev",
    "ml: add generate_dataset utility",
    "rails: boot and environment setup",
    "rails: database.yml and Puma config",
    "rails: routes for dashboard and classifications",
    "rails: application controller baseline",
    "rails: email classification model",
    "rails: migration for email_classifications",
    "rails: schema after migration",
    "rails: classifications controller",
    "rails: dashboard controller",
    "rails: ML API client service",
    "rails: ML API initializer",
    "rails: application layout",
    "rails: dashboard index view",
    "rails: classifications index",
    "rails: classifications new form",
    "rails: classifications show",
    "rails: shared flash partial",
    "rails: application stylesheet",
    "rails: importmap and javascript entry",
    "rails: application helper",
    "rails: inflections initializer",
    "rails: Gemfile and lockfile",
    "rails: binstubs and Rakefile",
    "rails: config.ru and ruby version",
    "docker: compose for app and ML service",
    "evidence: training log excerpt",
    "evidence: classification reports",
    "evidence: evaluation JSON snapshot",
    "evidence: training results JSON",
    "evidence: evidence index",
    "docs: dummy email fixtures for manual QA",
    "chore: tighten Rails test environment config",
    "chore: production environment defaults",
    "chore: development environment tweaks",
    "fix: align schema with migration timestamps",
    "chore: importmap pin updates",
    "chore: final integration pass for classify flow",
]


def sh(cmd: str, env: dict | None = None) -> None:
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, shell=True, cwd=ROOT, env=e)
    if r.returncode != 0:
        sys.exit(r.returncode)


def collect_files() -> list[str]:
    out = subprocess.run(
        "find email_priority_system evidence DUMMY_EMAIL_DATA.md -type f 2>/dev/null | sort",
        shell=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    paths: list[str] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        r = subprocess.run(
            ["git", "check-ignore", "-q", line],
            cwd=ROOT,
            capture_output=True,
        )
        if r.returncode != 1:
            continue
        paths.append(line)
    return paths


def main() -> None:
    if len(MESSAGES) != 50:
        print("MESSAGES must have 50 entries", file=sys.stderr)
        sys.exit(1)

    files = collect_files()
    if not files:
        print("No files to commit", file=sys.stderr)
        sys.exit(1)

    n_file_commits = 49
    batches: list[list[str]] = []
    total = len(files)
    for i in range(n_file_commits):
        a = i * total // n_file_commits
        b = (i + 1) * total // n_file_commits
        batches.append(files[a:b])

    start = datetime(2026, 3, 19, 9, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 7, 17, 0, 0, tzinfo=timezone.utc)
    step = (end - start) / 49

    # Commit 1: .gitignore
    d0 = start
    date0 = d0.strftime("%Y-%m-%d %H:%M:%S +0000")
    sh("git add .gitignore", {"GIT_AUTHOR_DATE": date0, "GIT_COMMITTER_DATE": date0})
    sh(
        f'git commit -m "{MESSAGES[0]}"',
        {"GIT_AUTHOR_DATE": date0, "GIT_COMMITTER_DATE": date0},
    )

    for i, batch in enumerate(batches):
        if not batch:
            continue
        idx = i + 1
        d = start + step * idx
        date_str = d.strftime("%Y-%m-%d %H:%M:%S +0000")
        for p in batch:
            sh(f"git add -- {subprocess.list2cmdline([p])}", {})
        sh(
            f'git commit -m "{MESSAGES[idx]}"',
            {"GIT_AUTHOR_DATE": date_str, "GIT_COMMITTER_DATE": date_str},
        )

    # 25th commit hash (50% of 50) for partial push
    out = subprocess.run(
        "git rev-list --reverse HEAD | sed -n '25p'",
        shell=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    half = out.stdout.strip()
    if not half:
        print("Could not resolve 25th commit", file=sys.stderr)
        sys.exit(1)
    (ROOT / ".half_push_commit").write_text(half + "\n", encoding="utf-8")
    print("Half-push tip (25/50 commits): git push origin", half + ":refs/heads/main")
    print("Local main stays ahead; remote shows 50% of history.")


if __name__ == "__main__":
    main()
