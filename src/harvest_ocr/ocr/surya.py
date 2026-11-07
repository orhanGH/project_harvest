from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..types import OCRItem, OCRResult
from ..utils import HarvestError, safe_stem
from .base import BaseOCRAdapter
from .html import html_table_items


def _surya_key_data(data: dict[str, Any], stem: str):
    if stem in data:
        return data[stem]
    for key, value in data.items():
        if safe_stem(Path(key).stem) == stem:
            return value
    return []


class SuryaAdapter(BaseOCRAdapter):
    name = "surya"
    preferred_region_type = "table"

    def __init__(self, run_ocr: bool = True, run_table: bool = True, keep_server: bool = False) -> None:
        self.run_ocr = run_ocr
        self.run_table = run_table
        self.keep_server = keep_server

    @staticmethod
    def _execute(command: list[str]) -> None:
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise HarvestError(
                f"Surya failed with exit code {completed.returncode}:\n{completed.stderr[-4000:]}"
            )

    def recognize(self, regions: list[dict[str, Any]], output_dir: str | Path) -> list[OCRResult]:
        ocr_executable = shutil.which("surya_ocr")
        table_executable = shutil.which("surya_table")
        if self.run_ocr and not ocr_executable:
            raise HarvestError("surya_ocr was not found. Install `surya-ocr>=2`.")
        if self.run_table and not table_executable:
            raise HarvestError("surya_table was not found. Install `surya-ocr>=2`.")

        target = Path(output_dir).resolve()
        stage = target / "input"
        ocr_dir = target / "raw_ocr"
        table_dir = target / "raw_table"
        for directory in (stage, ocr_dir, table_dir):
            directory.mkdir(parents=True, exist_ok=True)
        region_by_stem: dict[str, dict[str, Any]] = {}
        for region in regions:
            stem = safe_stem(region["region_id"])
            extension = Path(region["region_image"]).suffix or ".png"
            staged = stage / f"{stem}{extension}"
            if not staged.exists():
                staged.symlink_to(Path(region["region_image"]).resolve())
            region_by_stem[stem] = region

        if self.run_ocr:
            command = [ocr_executable, str(stage), "--output_dir", str(ocr_dir)]
            if self.keep_server:
                command.append("--keep_server")
            self._execute(command)
        if self.run_table:
            command = [
                table_executable,
                str(stage),
                "--output_dir",
                str(table_dir),
                "--skip_table_detection",
            ]
            if self.keep_server:
                command.append("--keep_server")
            self._execute(command)

        ocr_json_path = ocr_dir / "results.json"
        table_json_path = table_dir / "results.json"
        ocr_data = json.loads(ocr_json_path.read_text(encoding="utf-8")) if ocr_json_path.exists() else {}
        table_data = json.loads(table_json_path.read_text(encoding="utf-8")) if table_json_path.exists() else {}

        results: list[OCRResult] = []
        for stem, region in region_by_stem.items():
            items: list[OCRItem] = []
            texts: list[str] = []
            confidences: list[float] = []
            for page in _surya_key_data(ocr_data, stem) or []:
                for block in page.get("blocks", []):
                    html = block.get("html", "")
                    if "<table" in html.lower():
                        items.extend(html_table_items(html))
                    elif html:
                        text = " ".join(html.replace("<br>", " ").split())
                        bbox = block.get("bbox")
                        item = OCRItem(
                            text=text,
                            confidence=block.get("confidence"),
                            bbox=tuple(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
                            block_type=block.get("label"),
                        )
                        items.append(item)
                    if block.get("confidence") is not None:
                        confidences.append(float(block["confidence"]))
            texts.extend(item.text for item in items if item.text)
            mean_confidence = sum(confidences) / len(confidences) if confidences else None
            raw_reference = table_json_path if table_json_path.exists() else ocr_json_path
            results.append(
                OCRResult(
                    region_id=region["region_id"],
                    source_model="surya:2",
                    text="\n".join(texts),
                    confidence=mean_confidence,
                    items=items,
                    raw_output_path=str(raw_reference) if raw_reference.exists() else None,
                    metadata={"table_geometry": _surya_key_data(table_data, stem)},
                )
            )
        return results
