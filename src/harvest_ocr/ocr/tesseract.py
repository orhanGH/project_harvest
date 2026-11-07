from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..types import OCRItem, OCRResult
from ..utils import HarvestError
from .base import BaseOCRAdapter


NUMERIC_WHITELIST = "0123456789.,—-…()*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TesseractAdapter(BaseOCRAdapter):
    name = "tesseract"
    preferred_region_type = "cell"

    def __init__(
        self,
        languages: str = "eng+fra",
        fallback_language: str = "eng",
        scale: float = 2.0,
        country_psm: int = 7,
        numeric_psm: int = 7,
    ) -> None:
        self.languages = languages
        self.fallback_language = fallback_language
        self.scale = scale
        self.country_psm = country_psm
        self.numeric_psm = numeric_psm

    @staticmethod
    def _module():
        try:
            import pytesseract
        except ImportError as exc:
            raise HarvestError(
                "pytesseract is required. Install with `pip install -e '.[tesseract]'` "
                "and ensure the tesseract executable is available."
            ) from exc
        return pytesseract

    def _available_language(self, pytesseract) -> str:
        available = set(pytesseract.get_languages(config=""))
        requested = [lang for lang in self.languages.split("+") if lang in available]
        if requested:
            return "+".join(requested)
        if self.fallback_language in available:
            return self.fallback_language
        if available:
            return sorted(available)[0]
        raise HarvestError("Tesseract has no language data installed")

    def _image(self, path: str) -> Image.Image:
        image = ImageOps.exif_transpose(Image.open(path)).convert("L")
        if self.scale != 1.0:
            width, height = image.size
            image = image.resize(
                (max(1, int(width * self.scale)), max(1, int(height * self.scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def recognize(
        self,
        regions: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> list[OCRResult]:
        del output_dir
        pytesseract = self._module()
        language = self._available_language(pytesseract)
        results: list[OCRResult] = []
        for region in regions:
            column_id = region.get("column_id") or ""
            numeric = column_id and not column_id.startswith("country_")
            psm = self.numeric_psm if numeric else self.country_psm
            config = f"--psm {psm} -c preserve_interword_spaces=1"
            if numeric:
                config += f" -c tessedit_char_whitelist={NUMERIC_WHITELIST}"
            try:
                image = self._image(region["region_image"])
                data = pytesseract.image_to_data(
                    image,
                    lang=language,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                items: list[OCRItem] = []
                confidences: list[float] = []
                for index, raw_text in enumerate(data["text"]):
                    text = str(raw_text).strip()
                    if not text:
                        continue
                    raw_confidence = float(data["conf"][index])
                    confidence = raw_confidence / 100.0 if raw_confidence >= 0 else None
                    if confidence is not None:
                        confidences.append(confidence)
                    scale = self.scale
                    bbox = (
                        int(data["left"][index] / scale),
                        int(data["top"][index] / scale),
                        int((data["left"][index] + data["width"][index]) / scale),
                        int((data["top"][index] + data["height"][index]) / scale),
                    )
                    items.append(OCRItem(text=text, confidence=confidence, bbox=bbox))
                text = " ".join(item.text for item in items)
                mean_confidence = float(np.mean(confidences)) if confidences else None
                results.append(
                    OCRResult(
                        region_id=region["region_id"],
                        source_model=self.name,
                        text=text,
                        confidence=mean_confidence,
                        items=items,
                        metadata={"language": language, "psm": psm, "scale": self.scale},
                    )
                )
            except Exception as exc:  # Keep batch processing and expose the failed region.
                results.append(
                    OCRResult(
                        region_id=region["region_id"],
                        source_model=self.name,
                        text="",
                        confidence=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results
