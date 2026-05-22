#!/usr/bin/env python3
"""
Remove 'Muhammad Usama Islam | U2911515 |' (and legacy 'ISLAM') prefix from footer
text boxes; keep a single PAGE field per affected paragraph. Does not change body,
headers, or images. One backup: *_before_footer_strip.docx
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document

REPO = Path(__file__).resolve().parents[1]
DOC_PATH = REPO / "Osama_Final_Combined_Dissertation (4).docx"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_P = f"{{{NS}}}p"
W_R = f"{{{NS}}}r"
W_FLD = f"{{{NS}}}fldChar"


def _backup_once(path: Path) -> None:
    bak = path.with_name(path.stem + "_before_footer_strip.docx")
    if not bak.exists():
        shutil.copy2(path, bak)
        print("Backup:", bak)


def _find_page_field_spans(runs: list) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(runs):
        fc = runs[i].find(W_FLD)
        if fc is not None and fc.get(f"{{{NS}}}fldCharType") == "begin":
            start = i
            j = i + 1
            while j < len(runs):
                fc2 = runs[j].find(W_FLD)
                if fc2 is not None and fc2.get(f"{{{NS}}}fldCharType") == "end":
                    spans.append((start, j))
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return spans


def _reduce_paragraph_to_first_page_field(p) -> bool:
    runs = [c for c in p if c.tag == W_R]
    if not runs:
        return False
    spans = _find_page_field_spans(runs)
    if not spans:
        return False
    start, end = spans[0]
    keep = set(runs[start : end + 1])
    removed = 0
    for r in list(runs):
        if r not in keep:
            p.remove(r)
            removed += 1
    return removed > 0 or len(spans) > 1


def _process_footer_root(ft_el) -> int:
    changed = 0
    for txbx in ft_el.iter(f"{{{NS}}}txbxContent"):
        for child in list(txbx):
            if child.tag != W_P:
                continue
            text = "".join((t.text or "") for t in child.iter(f"{{{NS}}}t"))
            if "U2911515" in text or "Usama" in text or "ISLAM" in text or "Islam" in text:
                if _reduce_paragraph_to_first_page_field(child):
                    changed += 1
    return changed


def main() -> int:
    if not DOC_PATH.is_file():
        print("Missing", DOC_PATH, file=sys.stderr)
        return 1
    _backup_once(DOC_PATH)
    doc = Document(str(DOC_PATH))
    seen: set[int] = set()
    total = 0
    for sec in doc.sections:
        root = sec.footer._element
        i = id(root)
        if i in seen:
            continue
        seen.add(i)
        total += _process_footer_root(root)
    doc.save(str(DOC_PATH))
    print(f"Updated footers (paragraphs trimmed): {total} | Saved {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
