from __future__ import annotations

import re
from pathlib import Path

import fitz
from PIL import Image

from .types import PageRecord
from .utils import HarvestError, parse_page_range, sha256_file, write_jsonl


def _printed_page(text: str) -> str | None:
    for pattern in (r"(?m)^\s*[—-]?\s*(\d{1,4})\s*[—-]?\s*$", r"(?m)^\s*(\d{1,4})\s*$"):
        match = re.search(pattern, text[:800])
        if match:
            return match.group(1)
    return None


def _extract_largest_page_image(document: fitz.Document, page: fitz.Page) -> tuple[bytes, str]:
    candidates = []
    for info in page.get_images(full=True):
        xref = info[0]
        width = int(info[2])
        height = int(info[3])
        candidates.append((width * height, xref))
    if not candidates:
        pixmap = page.get_pixmap(dpi=200, colorspace=fitz.csRGB, alpha=False)
        return pixmap.tobytes("png"), "png"
    _, xref = max(candidates)
    image = document.extract_image(xref)
    return image["image"], image.get("ext", "png")


def extract_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    pages: str | None = None,
    document_id: str | None = None,
) -> Path:
    """Extract the largest embedded scan per page and the weak PDF text layer."""
    source = Path(pdf_path).resolve()
    if not source.exists():
        raise HarvestError(f"Input PDF does not exist: {source}")

    target = Path(output_dir).resolve()
    image_dir = target / "pages"
    text_dir = target / "embedded_text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    doc_id = document_id or source.stem
    records: list[PageRecord] = []
    with fitz.open(source) as document:
        selected = parse_page_range(pages, document.page_count)
        for pdf_page in selected:
            page = document[pdf_page - 1]
            text = page.get_text("text") or ""
            text_path = text_dir / f"page-{pdf_page:04d}.txt"
            text_path.write_text(text, encoding="utf-8")

            image_bytes, extension = _extract_largest_page_image(document, page)
            image_path = image_dir / f"page-{pdf_page:04d}.{extension}"
            image_path.write_bytes(image_bytes)
            with Image.open(image_path) as extracted_image:
                width, height = extracted_image.size

            records.append(
                PageRecord(
                    document_id=doc_id,
                    pdf_path=str(source),
                    pdf_page=pdf_page,
                    printed_page=_printed_page(text),
                    image_path=str(image_path),
                    embedded_text_path=str(text_path),
                    width=width,
                    height=height,
                    sha256=sha256_file(image_path),
                )
            )

    manifest = target / "pages.jsonl"
    write_jsonl(manifest, records)
    return manifest
