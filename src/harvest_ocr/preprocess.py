from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import ProcessedPage
from .utils import HarvestError, read_jsonl, write_jsonl


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise HarvestError(
            "OpenCV is required for preprocessing. Install with "
            "`pip install -e '.[vision]'`."
        ) from exc
    return cv2


def detect_paper_box(gray: np.ndarray, padding: int = 12) -> tuple[int, int, int, int]:
    cv2 = _cv2()
    height, width = gray.shape
    threshold = max(25, int(np.percentile(gray, 12)))
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 80), max(9, height // 80)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, width, height)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < width * height * 0.45:
        return (0, 0, width, height)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return (x0, y0, x1, y1)


def estimate_skew(gray: np.ndarray, max_degrees: float = 3.0) -> float:
    cv2 = _cv2()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(80, gray.shape[1] // 7),
        minLineLength=gray.shape[1] * 0.35,
        maxLineGap=gray.shape[1] * 0.03,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) <= max_degrees:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.01:
        return image
    cv2 = _cv2()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    border = int(np.median(image[: min(30, height), :]))
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def normalize_illumination(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    cv2 = _cv2()
    sigma = max(9.0, min(gray.shape) / 35.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=235)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def adaptive_binary(gray: np.ndarray, block_size: int = 41, c: int = 13) -> np.ndarray:
    cv2 = _cv2()
    block_size = max(3, block_size | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def remove_long_rules(gray: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Remove long printed table rules without treating short dashes as lines."""
    cv2 = _cv2()
    height, width = gray.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(45, width // 18), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(45, height // 18))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    rules = cv2.bitwise_or(horizontal, vertical)
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cleaned = gray.copy()
    cleaned[rules > 0] = 255
    return cleaned


def preprocess_manifest(
    pages_manifest: str | Path,
    output_dir: str | Path,
    settings: dict[str, Any] | None = None,
) -> Path:
    cv2 = _cv2()
    settings = settings or {}
    target = Path(output_dir).resolve()
    layout_dir = target / "layout"
    ocr_dir = target / "ocr"
    binary_dir = target / "binary"
    for directory in (layout_dir, ocr_dir, binary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processed: list[ProcessedPage] = []
    for record in read_jsonl(pages_manifest):
        source = Path(record["image_path"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise HarvestError(f"OpenCV could not read {source}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if settings.get("crop_paper", True):
            crop_box = detect_paper_box(gray, int(settings.get("crop_padding", 12)))
            x0, y0, x1, y1 = crop_box
            gray = gray[y0:y1, x0:x1]
        else:
            crop_box = (0, 0, gray.shape[1], gray.shape[0])

        angle = 0.0
        if settings.get("deskew", True):
            angle = estimate_skew(gray, float(settings.get("max_skew_degrees", 3.0)))
            gray = rotate_image(gray, angle)

        layout_gray = gray
        ocr_gray = (
            normalize_illumination(gray, float(settings.get("clahe_clip_limit", 2.0)))
            if settings.get("normalize_illumination", True)
            else gray.copy()
        )
        if settings.get("threshold_method", "otsu") == "adaptive":
            binary = adaptive_binary(
                ocr_gray,
                int(settings.get("threshold_block_size", 41)),
                int(settings.get("threshold_c", 13)),
            )
        else:
            _, binary = cv2.threshold(
                ocr_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        ocr_gray = remove_long_rules(ocr_gray, binary)

        page = int(record["pdf_page"])
        layout_path = layout_dir / f"page-{page:04d}.png"
        ocr_path = ocr_dir / f"page-{page:04d}.png"
        binary_path = binary_dir / f"page-{page:04d}.png"
        cv2.imwrite(str(layout_path), layout_gray)
        cv2.imwrite(str(ocr_path), ocr_gray)
        cv2.imwrite(str(binary_path), binary)
        processed.append(
            ProcessedPage(
                document_id=record["document_id"],
                pdf_page=page,
                source_image=str(source),
                layout_image=str(layout_path),
                ocr_image=str(ocr_path),
                binary_image=str(binary_path),
                width=int(gray.shape[1]),
                height=int(gray.shape[0]),
                crop_box=crop_box,
                rotation_degrees=angle,
            )
        )

    manifest = target / "processed_pages.jsonl"
    write_jsonl(manifest, processed)
    return manifest
