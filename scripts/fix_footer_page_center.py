#!/usr/bin/env python3
"""
Center footer page-number text boxes and prefix with 'Page - ' before the PAGE field.
Only touches footer XML. Backup once: *_before_footer_page_center.docx
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn

REPO = Path(__file__).resolve().parents[1]
DOC_PATH = REPO / "Osama_Final_Combined_Dissertation (4).docx"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
W_P = f"{{{NS}}}p"
W_R = f"{{{NS}}}r"
W_FLD = f"{{{NS}}}fldChar"
PAGE_PREFIX = "Page - "
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _backup_once(path: Path) -> None:
    bak = path.with_name(path.stem + "_before_footer_page_center.docx")
    if not bak.exists():
        shutil.copy2(path, bak)
        print("Backup:", bak)


def _ensure_p_center(p) -> None:
    p_pr = p.find(qn("w:pPr"))
    if p_pr is None:
        p_pr = OxmlElement("w:pPr")
        p.insert(0, p_pr)
    jc = p_pr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        p_pr.append(jc)
    jc.set(qn("w:val"), "center")


def _runs_in_order(p) -> list:
    return [c for c in p if c.tag == W_R]


def _first_page_field_begin_run(runs: list):
    for i, r in enumerate(runs):
        fc = r.find(W_FLD)
        if fc is not None and fc.get(f"{{{NS}}}fldCharType") == "begin":
            return i, r
    return None, None


def _text_before_run_index(runs: list, idx: int) -> str:
    s = ""
    for r in runs[:idx]:
        for t in r.iter(f"{{{NS}}}t"):
            s += t.text or ""
    return s


def _remove_runs_before(p, runs: list, before_idx: int) -> None:
    for r in runs[:before_idx]:
        p.remove(r)


def _insert_prefix_run_before(p, before_el, prefix: str) -> None:
    r_pr = before_el.find(qn("w:rPr"))
    new_r = OxmlElement("w:r")
    if r_pr is not None:
        new_r.append(copy.deepcopy(r_pr))
    wt = OxmlElement("w:t")
    wt.set(XML_SPACE, "preserve")
    wt.text = prefix
    new_r.append(wt)
    idx = list(p).index(before_el)
    p.insert(idx, new_r)


def _footer_paragraph_has_page_field(p) -> bool:
    _, br = _first_page_field_begin_run(_runs_in_order(p))
    return br is not None


def _style_footer_page_paragraph(p) -> bool:
    if not _footer_paragraph_has_page_field(p):
        return False
    _ensure_p_center(p)
    runs = _runs_in_order(p)
    idx, begin_run = _first_page_field_begin_run(runs)
    if begin_run is None:
        return False
    before = _text_before_run_index(runs, idx)
    if before == PAGE_PREFIX:
        return True
    if before.strip():
        _remove_runs_before(p, runs, idx)
        runs = _runs_in_order(p)
        _, begin_run = _first_page_field_begin_run(runs)
        if begin_run is None:
            return False
    _insert_prefix_run_before(p, begin_run, PAGE_PREFIX)
    return True


def _center_footer_textbox_anchors(ft_el) -> int:
    """Text boxes ignore paragraph jc; set drawing horizontal align to page center."""
    n = 0
    for anchor in ft_el.iter(f"{{{WP}}}anchor"):
        has_txbx = any(e.tag == f"{{{NS}}}txbxContent" for e in anchor.iter())
        has_page_fld = any(
            e.tag == f"{{{NS}}}fldChar" or e.tag == f"{{{NS}}}instrText"
            for e in anchor.iter()
        )
        if not has_txbx or not has_page_fld:
            continue
        ph = anchor.find(f"{{{WP}}}positionH")
        if ph is None:
            continue
        for c in list(ph):
            ph.remove(c)
        ph.set(f"{{{WP}}}relativeFrom", "page")
        al = parse_xml(
            '<wp:align xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">center</wp:align>'
        )
        ph.append(al)
        n += 1
    return n


def _process_footer_root(ft_el) -> int:
    n = 0
    for txbx in ft_el.iter(f"{{{NS}}}txbxContent"):
        for child in list(txbx):
            if child.tag != W_P:
                continue
            if _footer_paragraph_has_page_field(child):
                if _style_footer_page_paragraph(child):
                    n += 1
    return n


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
        _center_footer_textbox_anchors(root)
    doc.save(str(DOC_PATH))
    print(
        f"Footers updated (Page label + paragraph jc + textbox center): "
        f"{total} txbx paragraphs | {DOC_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
