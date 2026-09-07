"""Buoc 1 cua pipeline ingest.

Doc PDF -> tach text tung trang + suy ra section/so trang in tren giay
+ render anh tung trang de UI preview citation.

Chay:  python -m app.ingest.parse_pdf
Ket qua: data/parsed/pages.jsonl  va  data/pages/page_XXX.jpg
"""
from __future__ import annotations

import json
import re
import sys

import pymupdf as fitz  # PyMuPDF

from app import config

RENDER_DPI = 110
# Moi trang PDF la mot spread 2 trang giay; so trang phai nam o cuoi text: "| 79"
PAGE_RIGHT_RE = re.compile(r"\|\s*(\d{1,3})\s*$")
PAGE_LEFT_RE = re.compile(r"(\d{1,3})\s*\|")
# Bo cac dong lap lai o header/footer khi tim ten section
BOILERPLATE = re.compile(r"b[aá]o\s*c[aá]o\s*th[uư][ơo]ng\s*ni[eê]n", re.IGNORECASE)


def _page_label(text: str, page_index: int) -> str:
    """Tra ve so trang in tren giay, vi du '78-79'. Fallback ve so trang PDF."""
    tail = text.rstrip()
    right = PAGE_RIGHT_RE.search(tail)
    if not right:
        return f"PDF {page_index}"
    rnum = int(right.group(1))
    left = PAGE_LEFT_RE.search(tail)
    lnum = int(left.group(1)) if left else rnum - 1
    if lnum != rnum - 1:
        lnum = rnum - 1
    return f"{lnum}-{rnum}"


def _section_title(page: "fitz.Page") -> str:
    """Doan ten section = cum chu co font lon nhat o 25% tren cua trang.

    Bao cao khong co bookmark nen phai suy ra tu running header.
    """
    top_limit = page.rect.height * 0.25
    best_size, best_text = 0.0, ""
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # chi xet block text
            continue
        for line in block.get("lines", []):
            y = line["bbox"][1]
            if y > top_limit:
                continue
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if len(text) < 4 or text.isdigit() or BOILERPLATE.search(text):
                continue
            size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
            if size > best_size:
                best_size, best_text = size, text
    return best_text.strip(" .,-|")


def main() -> int:
    pdf_path = config.SOURCE_PDF
    if not pdf_path.exists():
        raise SystemExit(f"Khong tim thay file PDF: {pdf_path}")

    config.PAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    last_section = ""
    records = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True) or ""
        section = _section_title(page) or last_section
        last_section = section

        image_file = f"page_{i:03d}.jpg"
        out = config.PAGES_DIR / image_file
        if not out.exists():
            page.get_pixmap(matrix=matrix).save(out, jpg_quality=78)

        records.append(
            {
                "page_index": i,                      # so thu tu trang trong file PDF
                "page_label": _page_label(text, i),   # so trang in tren giay
                "section": section,
                "text": text.strip(),
                "n_images": len(page.get_images(full=True)),
                "n_drawings": len(page.get_drawings()),
                "image_file": image_file,
            }
        )
        if i % 20 == 0:
            print(f"  ...da xu ly {i}/{doc.page_count} trang", flush=True)

    with config.PARSED_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = sum(len(r["text"]) for r in records)
    thin = [r["page_index"] for r in records if len(r["text"]) < 200]
    print(f"\nDa parse {len(records)} trang, {chars:,} ky tu -> {config.PARSED_JSONL}")
    print(f"Anh trang -> {config.PAGES_DIR}")
    if thin:
        print(f"Trang it text (co the la trang anh, nen chay vision pass): {thin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
