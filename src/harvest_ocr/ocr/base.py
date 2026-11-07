from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from ..types import OCRResult
from ..utils import read_jsonl, write_jsonl


def load_regions(manifest: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(manifest))


def filter_regions(
    regions: Iterable[dict[str, Any]],
    region_type: str,
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected = [row for row in regions if row.get("region_type") == region_type]
    return selected[:limit] if limit else selected


class BaseOCRAdapter(ABC):
    name: str
    preferred_region_type: str

    @abstractmethod
    def recognize(
        self,
        regions: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> list[OCRResult]:
        raise NotImplementedError

    def run(
        self,
        regions_manifest: str | Path,
        output_dir: str | Path,
        region_type: str | None = None,
        limit: int = 0,
    ) -> Path:
        target = Path(output_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        selected = filter_regions(
            load_regions(regions_manifest),
            region_type or self.preferred_region_type,
            limit,
        )
        results = self.recognize(selected, target)
        output = target / "ocr_results.jsonl"
        write_jsonl(output, results)
        return output
