from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import OCRItem, OCRResult
from ..utils import HarvestError, safe_stem, write_json
from .base import BaseOCRAdapter
from .html import collect_html_values, html_table_items


def _result_to_json(result, path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.save_to_json(save_path=str(path))
    except TypeError:
        result.save_to_json(str(path))
    candidates = [path]
    if path.is_dir():
        candidates.extend(sorted(path.glob("*.json")))
    else:
        candidates.extend(sorted(path.parent.glob(f"{path.stem}*.json")))
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                return json.load(handle)
    for attribute in ("json", "res", "_res"):
        value = getattr(result, attribute, None)
        if isinstance(value, (dict, list)):
            return value
    return {"repr": repr(result)}


def _collect_ocr_items(value: Any) -> list[OCRItem]:
    items: list[OCRItem] = []
    if isinstance(value, dict):
        texts = value.get("rec_texts") or value.get("texts")
        boxes = value.get("rec_boxes") or value.get("boxes")
        scores = value.get("rec_scores") or value.get("scores")
        if isinstance(texts, list):
            for index, text in enumerate(texts):
                bbox = None
                if isinstance(boxes, list) and index < len(boxes):
                    box = boxes[index]
                    if isinstance(box, list) and len(box) >= 4:
                        if len(box) == 4 and not isinstance(box[0], list):
                            bbox = tuple(int(round(float(x))) for x in box[:4])
                        elif isinstance(box[0], list):
                            xs = [point[0] for point in box]
                            ys = [point[1] for point in box]
                            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                confidence = None
                if isinstance(scores, list) and index < len(scores):
                    confidence = float(scores[index])
                items.append(OCRItem(str(text), confidence=confidence, bbox=bbox))
        for child in value.values():
            items.extend(_collect_ocr_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_collect_ocr_items(child))
    return items


class PaddleAdapter(BaseOCRAdapter):
    name = "paddle"
    preferred_region_type = "table"

    def __init__(
        self,
        pipeline: str = "table_recognition_v2",
        device: str | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
    ) -> None:
        self.pipeline_name = pipeline
        self.device = device
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping

    def _pipeline(self):
        try:
            from paddleocr import PPStructureV3, TableRecognitionPipelineV2
        except ImportError as exc:
            raise HarvestError(
                "PaddleOCR 3.x is required. Install PaddlePaddle for the Marvin CUDA "
                "version, then `pip install 'paddleocr>=3.3'`."
            ) from exc
        kwargs = {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.pipeline_name == "pp_structure_v3":
            return PPStructureV3(**kwargs)
        if self.pipeline_name == "table_recognition_v2":
            return TableRecognitionPipelineV2(**kwargs)
        raise HarvestError(f"Unsupported Paddle pipeline: {self.pipeline_name}")

    def recognize(self, regions: list[dict[str, Any]], output_dir: str | Path) -> list[OCRResult]:
        pipeline = self._pipeline()
        raw_dir = Path(output_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        results: list[OCRResult] = []
        for region in regions:
            try:
                outputs = list(pipeline.predict(region["region_image"]))
                raw_pages = []
                items: list[OCRItem] = []
                for index, output in enumerate(outputs):
                    raw_path = raw_dir / f"{safe_stem(region['region_id'])}-{index:02d}.json"
                    raw = _result_to_json(output, raw_path)
                    raw_pages.append(raw)
                    html_values = collect_html_values(raw)
                    for html in html_values:
                        items.extend(html_table_items(html))
                    if not html_values:
                        items.extend(_collect_ocr_items(raw))
                consolidated = raw_dir / f"{safe_stem(region['region_id'])}.json"
                write_json(consolidated, raw_pages)
                text = "\n".join(item.text for item in items if item.text)
                confidences = [item.confidence for item in items if item.confidence is not None]
                confidence = sum(confidences) / len(confidences) if confidences else None
                results.append(
                    OCRResult(
                        region_id=region["region_id"],
                        source_model=f"paddle:{self.pipeline_name}",
                        text=text,
                        confidence=confidence,
                        items=items,
                        raw_output_path=str(consolidated),
                    )
                )
            except Exception as exc:
                results.append(
                    OCRResult(
                        region_id=region["region_id"],
                        source_model=f"paddle:{self.pipeline_name}",
                        text="",
                        confidence=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results
