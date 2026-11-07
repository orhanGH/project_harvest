from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class PageRecord:
    document_id: str
    pdf_path: str
    pdf_page: int
    printed_page: str | None
    image_path: str
    embedded_text_path: str | None
    width: int
    height: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessedPage:
    document_id: str
    pdf_page: int
    source_image: str
    layout_image: str
    ocr_image: str
    binary_image: str
    width: int
    height: int
    crop_box: BBox
    rotation_degrees: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegionRecord:
    region_id: str
    document_id: str
    pdf_page: int
    region_type: str
    region_image: str
    source_image: str
    bbox: BBox
    table_bbox: BBox
    row_index: int | None = None
    column_index: int | None = None
    column_id: str | None = None
    table_number: str | None = None
    page_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRItem:
    text: str
    confidence: float | None = None
    bbox: BBox | None = None
    row_index: int | None = None
    column_index: int | None = None
    block_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRResult:
    region_id: str
    source_model: str
    text: str
    confidence: float | None
    items: list[OCRItem] = field(default_factory=list)
    raw_output_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["items"] = [item.to_dict() for item in self.items]
        return result


def path_str(path: str | Path) -> str:
    return str(Path(path).resolve())
