"""
Download and extract the Enron Email Dataset and optionally the SpamAssassin corpus.

Usage:
    python download_dataset.py [--enron] [--spamassassin] [--all]
"""
import os
import sys
import argparse
import tarfile
import hashlib
import logging
from pathlib import Path

import requests
from tqdm import tqdm

# Allow running this script directly from the ml/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR,
    ENRON_DATASET_URL,
    ENRON_ARCHIVE_NAME,
    ENRON_EXTRACTED_DIR,
    SPAMASSASSIN_BASE_URL,
    SPAMASSASSIN_FILES,
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


# -- Helpers -------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream-download *url* to *dest*, showing a tqdm progress bar."""
    if dest.exists():
        log.info("Archive already exists at %s - skipping download.", dest)
        return dest

    log.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                desc=dest.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    bar.update(len(chunk))
    except Exception as exc:
        log.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        raise

    log.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _extract_tar(archive: Path, dest_dir: Path, compression: str = "gz") -> None:
    """Extract a tar archive to *dest_dir*."""
    log.info("Extracting %s -> %s", archive, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mode = f"r:{compression}"
    try:
        with tarfile.open(archive, mode) as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc=f"Extracting {archive.name}", unit="file"):
                tf.extract(member, path=dest_dir)
    except Exception as exc:
        log.error("Extraction failed: %s", exc)
        raise
    log.info("Extraction complete -> %s", dest_dir)


# -- Enron ---------------------------------------------------------------------

def download_enron() -> Path:
    """Download and extract the Enron Email Dataset."""
    archive_path = RAW_DATA_DIR / ENRON_ARCHIVE_NAME

    # Check if already extracted
    if ENRON_EXTRACTED_DIR.exists() and any(ENRON_EXTRACTED_DIR.iterdir()):
        log.info(
            "Enron dataset already extracted at %s - skipping.", ENRON_EXTRACTED_DIR
        )
        return ENRON_EXTRACTED_DIR

    # Download
    _download_file(ENRON_DATASET_URL, archive_path)

    # Extract
    _extract_tar(archive_path, RAW_DATA_DIR, compression="gz")

    # Verify extraction
    if not ENRON_EXTRACTED_DIR.exists():
        raise RuntimeError(
            f"Expected extracted directory {ENRON_EXTRACTED_DIR} not found after extraction. "
            "The archive may have a different top-level directory name."
        )

    mailboxes = [p for p in ENRON_EXTRACTED_DIR.iterdir() if p.is_dir()]
    log.info(
        "Enron dataset ready: %d mailboxes found at %s",
        len(mailboxes),
        ENRON_EXTRACTED_DIR,
    )
    return ENRON_EXTRACTED_DIR


# -- SpamAssassin --------------------------------------------------------------

def download_spamassassin() -> Path:
    """Download and extract the SpamAssassin Public Corpus."""
    sa_dir = RAW_DATA_DIR / "spamassassin"
    sa_dir.mkdir(parents=True, exist_ok=True)

    for filename in SPAMASSASSIN_FILES:
        url = f"{SPAMASSASSIN_BASE_URL}/{filename}"
        archive_path = sa_dir / filename

        # Determine extraction sub-dir from filename stem
        stem = filename.replace(".tar.bz2", "")
        extract_dir = sa_dir / stem

        if extract_dir.exists() and any(extract_dir.iterdir()):
            log.info("SpamAssassin subset already extracted: %s", extract_dir)
            continue

        try:
            _download_file(url, archive_path)
            _extract_tar(archive_path, sa_dir, compression="bz2")
        except Exception as exc:
            log.warning("Skipping %s due to error: %s", filename, exc)
            continue

    log.info("SpamAssassin corpus ready at %s", sa_dir)
    return sa_dir


# -- Verification --------------------------------------------------------------

def verify_enron_dataset() -> bool:
    """Quick sanity-check that the Enron dataset looks usable."""
    if not ENRON_EXTRACTED_DIR.exists():
        log.warning("Enron dataset not found at %s", ENRON_EXTRACTED_DIR)
        return False

    mailboxes = [p for p in ENRON_EXTRACTED_DIR.iterdir() if p.is_dir()]
    if not mailboxes:
        log.warning("No mailboxes found inside %s", ENRON_EXTRACTED_DIR)
        return False

    # Sample first mailbox
    sample_box = mailboxes[0]
    email_files = list(sample_box.rglob("*"))
    email_files = [f for f in email_files if f.is_file()]
    log.info(
        "Verification passed: %d mailboxes, sample mailbox '%s' has %d files.",
        len(mailboxes),
        sample_box.name,
        len(email_files),
    )
    return True


# -- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets for the Email Priority Classification System."
    )
    parser.add_argument(
        "--enron", action="store_true", help="Download the Enron Email Dataset."
    )
    parser.add_argument(
        "--spamassassin", action="store_true", help="Download the SpamAssassin corpus."
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets.")
    parser.add_argument(
        "--verify", action="store_true", help="Verify downloaded datasets only."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verify:
        ok = verify_enron_dataset()
        sys.exit(0 if ok else 1)

    if args.all or args.enron or (not args.spamassassin):
        # Default: download Enron if no flag given
        log.info("=== Downloading Enron Email Dataset ===")
        try:
            enron_dir = download_enron()
            verify_enron_dataset()
        except Exception as exc:
            log.error("Failed to download/extract Enron dataset: %s", exc)
            sys.exit(1)

    if args.all or args.spamassassin:
        log.info("=== Downloading SpamAssassin Corpus ===")
        try:
            download_spamassassin()
        except Exception as exc:
            log.warning("SpamAssassin download had errors: %s", exc)

    log.info("=== Dataset download complete ===")


if __name__ == "__main__":
    main()
