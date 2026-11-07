from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..types import OCRResult
from ..utils import HarvestError, safe_stem
from .base import BaseOCRAdapter
from .html import html_table_items


class ChandraAdapter(BaseOCRAdapter):
    name = "chandra"
    preferred_region_type = "table"

    def __init__(self, method: str = "hf", batch_size: int = 1, max_workers: int = 1) -> None:
        if method not in {"hf", "vllm"}:
            raise HarvestError("Chandra method must be 'hf' or 'vllm'")
        self.method = method
        self.batch_size = batch_size
        self.max_workers = max_workers

    def recognize(self, regions: list[dict[str, Any]], output_dir: str | Path) -> list[OCRResult]:
        executable = shutil.which("chandra")
        if not executable:
            raise HarvestError(
                "Chandra CLI was not found. Install `chandra-ocr[hf]` in a GPU environment."
            )
        target = Path(output_dir).resolve()
        stage = target / "input"
        raw = target / "raw"
        stage.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        region_by_stem: dict[str, dict[str, Any]] = {}
        for region in regions:
            stem = safe_stem(region["region_id"])
            extension = Path(region["region_image"]).suffix or ".png"
            staged = stage / f"{stem}{extension}"
            if not staged.exists():
                staged.symlink_to(Path(region["region_image"]).resolve())
            region_by_stem[stem] = region

        command = [
            executable,
            str(stage),
            str(raw),
            "--method",
            self.method,
            "--batch-size",
            str(self.batch_size),
            "--max-workers",
            str(self.max_workers),
            "--no-images",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise HarvestError(
                f"Chandra failed with exit code {completed.returncode}:\n{completed.stderr[-4000:]}"
            )

        results: list[OCRResult] = []
        for stem, region in region_by_stem.items():
            output_folder = raw / stem
            html_candidates = list(output_folder.glob("*.html")) if output_folder.exists() else []
            markdown_candidates = list(output_folder.glob("*.md")) if output_folder.exists() else []
            if html_candidates:
                html = html_candidates[0].read_text(encoding="utf-8", errors="replace")
                items = html_table_items(html)
                text = "\n".join(item.text for item in items if item.text)
                raw_path = html_candidates[0]
            elif markdown_candidates:
                text = markdown_candidates[0].read_text(encoding="utf-8", errors="replace")
                items = []
                raw_path = markdown_candidates[0]
            else:
                text, items, raw_path = "", [], None
            results.append(
                OCRResult(
                    region_id=region["region_id"],
                    source_model=f"chandra:{self.method}",
                    text=text,
                    confidence=None,
                    items=items,
                    raw_output_path=str(raw_path) if raw_path else None,
                    error=None if raw_path else "No Chandra output file was found",
                )
            )
        return results
