from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import BBox, RegionRecord
from .utils import HarvestError, parse_page_range, read_jsonl, write_json, write_jsonl


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise HarvestError("OpenCV is required for table layout detection.") from exc
    return cv2


@dataclass(slots=True)
class TableGeometry:
    bbox: BBox
    header_bottom: int
    x_boundaries: list[int]
    column_ranges: list[tuple[int, int]]
    row_boundaries: list[tuple[int, int]]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "header_bottom": self.header_bottom,
            "x_boundaries": self.x_boundaries,
            "column_ranges": self.column_ranges,
            "row_boundaries": self.row_boundaries,
            "confidence": self.confidence,
        }


def _cluster_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    clusters: list[list[int]] = [[value] for value in sorted(values)]
    merged: list[list[int]] = []
    for cluster in clusters:
        if merged and cluster[0] - int(np.mean(merged[-1])) <= tolerance:
            merged[-1].extend(cluster)
        else:
            merged.append(cluster)
    return [int(round(float(np.median(cluster)))) for cluster in merged]


def _line_segments(binary: np.ndarray, horizontal: bool) -> list[tuple[int, int, int, int]]:
    cv2 = _cv2()
    height, width = binary.shape
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, width // 12), 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 8)))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def _detect_table_bbox(binary: np.ndarray, min_width_ratio: float) -> tuple[BBox, list[int]]:
    height, width = binary.shape
    segments = _line_segments(binary, horizontal=True)
    long_lines = [(x, y, w, h) for x, y, w, h in segments if w >= width * min_width_ratio]
    if len(long_lines) < 2:
        return (int(width * 0.08), int(height * 0.08), int(width * 0.94), int(height * 0.72)), []
    ys = _cluster_positions([y + h // 2 for _, y, _, h in long_lines], max(2, height // 500))
    usable = [line for line in long_lines if height * 0.04 <= line[1] <= height * 0.9]
    top_line = min(usable, key=lambda value: value[1])
    bottom_line = max(usable, key=lambda value: value[1])
    x0 = max(0, min(line[0] for line in usable))
    x1 = min(width, max(line[0] + line[2] for line in usable))
    y0 = max(0, top_line[1])
    y1 = min(height, bottom_line[1] + bottom_line[3])
    return (x0, y0, x1, y1), ys


def _detect_x_boundaries(
    binary: np.ndarray,
    table_bbox: BBox,
    expected_count: int,
    fallback: list[float],
) -> tuple[list[int], float]:
    x0, y0, x1, y1 = table_bbox
    crop = binary[y0:y1, x0:x1]
    segments = _line_segments(crop, horizontal=False)
    positions = [x + w // 2 for x, _, w, h in segments if h >= crop.shape[0] * 0.18]
    positions = _cluster_positions(positions, max(3, crop.shape[1] // 250))
    absolute = [x0 + value for value in positions if crop.shape[1] * 0.01 < value < crop.shape[1] * 0.99]
    candidates = [x0, *absolute, x1]
    candidates = _cluster_positions(candidates, max(3, crop.shape[1] // 250))
    if len(candidates) == expected_count:
        return candidates, 1.0

    fallback_positions = [x0 + int(round((x1 - x0) * fraction)) for fraction in fallback]
    if len(fallback_positions) != expected_count:
        raise HarvestError(
            f"Expected {expected_count} fallback boundaries but got {len(fallback_positions)}"
        )
    # Snap fallback positions to nearby detected rules without changing column count.
    snap_distance = (x1 - x0) * 0.025
    snapped = []
    for fallback_x in fallback_positions:
        nearby = [candidate for candidate in candidates if abs(candidate - fallback_x) <= snap_distance]
        snapped.append(min(nearby, key=lambda value: abs(value - fallback_x)) if nearby else fallback_x)
    for index in range(1, len(snapped)):
        snapped[index] = max(snapped[index], snapped[index - 1] + 2)
    return snapped, max(0.35, 1.0 - abs(len(candidates) - expected_count) / expected_count)


def _detect_column_ranges(
    binary: np.ndarray,
    table_bbox: BBox,
    fallback_ranges: list[list[float]],
) -> tuple[list[tuple[int, int]], list[int], float]:
    """Build non-contiguous target cell ranges and snap them to printed rules."""
    x0, y0, x1, y1 = table_bbox
    crop = binary[y0:y1, x0:x1]
    segments = _line_segments(crop, horizontal=False)
    detected = [
        x0 + x + w // 2
        for x, _, w, h in segments
        if h >= crop.shape[0] * 0.18
    ]
    detected = _cluster_positions(detected, max(3, crop.shape[1] // 250))
    candidates = _cluster_positions([x0, *detected, x1], max(3, crop.shape[1] // 250))
    snap_distance = (x1 - x0) * 0.035

    def position(fraction: float) -> int:
        fallback = x0 + int(round((x1 - x0) * fraction))
        nearby = [candidate for candidate in candidates if abs(candidate - fallback) <= snap_distance]
        return min(nearby, key=lambda value: abs(value - fallback)) if nearby else fallback

    ranges: list[tuple[int, int]] = []
    snapped_count = 0
    for pair in fallback_ranges:
        if len(pair) != 2 or not 0 <= float(pair[0]) < float(pair[1]) <= 1:
            raise HarvestError(f"Invalid normalized column range: {pair!r}")
        raw_left = x0 + int(round((x1 - x0) * float(pair[0])))
        raw_right = x0 + int(round((x1 - x0) * float(pair[1])))
        left = position(float(pair[0]))
        right = position(float(pair[1]))
        snapped_count += int(left != raw_left) + int(right != raw_right)
        if right <= left:
            right = min(x1, left + 2)
        ranges.append((left, right))
    boundaries = sorted({value for cell_range in ranges for value in cell_range})
    endpoint_count = max(1, len(fallback_ranges) * 2)
    confidence = max(0.45, min(1.0, 0.55 + 0.45 * snapped_count / endpoint_count))
    return ranges, boundaries, confidence


def _header_bottom(table_bbox: BBox, horizontal_ys: list[int]) -> int:
    _, y0, _, y1 = table_bbox
    candidates = [y for y in horizontal_ys if y0 + (y1 - y0) * 0.08 < y < y0 + (y1 - y0) * 0.42]
    if candidates:
        return max(candidates)
    return y0 + int((y1 - y0) * 0.22)


def _detect_rows(
    binary: np.ndarray,
    table_bbox: BBox,
    header_bottom: int,
    min_components: int,
    merge_factor: float,
) -> list[tuple[int, int]]:
    cv2 = _cv2()
    x0, _, x1, y1 = table_bbox
    crop = binary[header_bottom:y1, x0:x1]
    if crop.size == 0:
        return []
    # Remove printed rules before computing horizontal ink density. On the
    # normalized Otsu image, this rejects paper texture without erasing digits.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(35, crop.shape[1] // 12), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(25, crop.shape[0] // 8))
    )
    horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, vertical_kernel)
    text_only = cv2.subtract(crop, cv2.bitwise_or(horizontal, vertical))
    projection = np.count_nonzero(text_only, axis=1)
    # A percentile threshold adapts to both sparse pages and densely printed
    # continuation pages. A fixed fraction of width under-segments dense tables.
    projection_threshold = max(20, int(np.percentile(projection, 60)))
    active = projection > projection_threshold

    bands: list[tuple[int, int]] = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(active)))

    merged: list[tuple[int, int]] = []
    gap_tolerance = max(1, int(2 * merge_factor))
    for band in bands:
        if merged and band[0] - merged[-1][1] <= gap_tolerance:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    filtered_bands = [band for band in merged if 2 <= band[1] - band[0] <= crop.shape[0] * 0.10]
    filtered = [header_bottom + (top + bottom) // 2 for top, bottom in filtered_bands]
    if not filtered:
        return []

    # Superscripts, footnote markers, and touching glyphs can produce two
    # nearby projection bands for one printed row. Merge only centers that
    # are substantially closer than the page's typical row spacing.
    if len(filtered) > 1:
        center_gaps = np.diff(filtered)
        typical_spacing = float(np.median(center_gaps))
        center_merge_distance = max(2, int(round(typical_spacing * 0.65)))
        filtered = _cluster_positions(filtered, center_merge_distance)

    boundaries: list[tuple[int, int]] = []
    for index, center in enumerate(filtered):
        previous = filtered[index - 1] if index else header_bottom
        following = filtered[index + 1] if index + 1 < len(filtered) else y1
        top = max(header_bottom, int(round((previous + center) / 2))) if index else header_bottom
        bottom = min(y1, int(round((center + following) / 2)))
        if bottom - top >= 4:
            boundaries.append((top, bottom))
    return boundaries


def detect_table_geometry(binary: np.ndarray, settings: dict[str, Any]) -> TableGeometry:
    columns = settings["expected_columns"]
    table_bbox, horizontal_ys = _detect_table_bbox(
        binary, float(settings.get("min_table_width_ratio", 0.65))
    )
    configured_ranges = settings.get("column_ranges")
    if configured_ranges:
        if len(configured_ranges) != len(columns):
            raise HarvestError(
                "column_ranges must contain exactly one [left, right] pair per expected column"
            )
        column_ranges, x_boundaries, line_confidence = _detect_column_ranges(
            binary, table_bbox, configured_ranges
        )
    else:
        expected_boundary_count = len(columns) + 1
        x_boundaries, line_confidence = _detect_x_boundaries(
            binary,
            table_bbox,
            expected_boundary_count,
            [float(value) for value in settings["fallback_x_boundaries"]],
        )
        column_ranges = list(zip(x_boundaries[:-1], x_boundaries[1:]))
    header_bottom = _header_bottom(table_bbox, horizontal_ys)
    rows = _detect_rows(
        binary,
        table_bbox,
        header_bottom,
        int(settings.get("min_row_components", 2)),
        float(settings.get("row_merge_factor", 0.85)),
    )
    confidence = min(1.0, line_confidence * 0.7 + min(len(rows) / 8, 1.0) * 0.3)
    return TableGeometry(table_bbox, header_bottom, x_boundaries, column_ranges, rows, confidence)


def _page_settings(settings: dict[str, Any], page_number: int) -> dict[str, Any]:
    merged = {key: value for key, value in settings.items() if key != "page_profiles"}
    for profile in settings.get("page_profiles", []):
        if page_number in parse_page_range(str(profile["pages"])):
            merged.update({key: value for key, value in profile.items() if key != "pages"})
            return merged
    return merged


def _draw_debug(image: np.ndarray, geometry: TableGeometry, output_path: Path) -> None:
    cv2 = _cv2()
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = geometry.bbox
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 255), 3)
    cv2.line(overlay, (x0, geometry.header_bottom), (x1, geometry.header_bottom), (255, 80, 0), 2)
    for x in geometry.x_boundaries:
        cv2.line(overlay, (x, y0), (x, y1), (80, 180, 0), 1)
    for top, bottom in geometry.row_boundaries:
        cv2.rectangle(overlay, (x0, top), (x1, bottom), (180, 0, 180), 1)
    cv2.imwrite(str(output_path), overlay)


def generate_regions(
    processed_manifest: str | Path,
    output_dir: str | Path,
    settings: dict[str, Any],
) -> Path:
    cv2 = _cv2()
    target = Path(output_dir).resolve()
    table_dir = target / "tables"
    row_dir = target / "rows"
    cell_dir = target / "cells"
    debug_dir = target / "debug"
    for directory in (table_dir, row_dir, cell_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    padding = int(settings.get("cell_padding", 4))
    regions: list[RegionRecord] = []
    geometry_index: dict[str, Any] = {}

    for page in read_jsonl(processed_manifest):
        binary = cv2.imread(page["binary_image"], cv2.IMREAD_GRAYSCALE)
        ocr_image = cv2.imread(page["ocr_image"], cv2.IMREAD_GRAYSCALE)
        if binary is None or ocr_image is None:
            raise HarvestError(f"Could not load preprocessed page {page['pdf_page']}")
        page_number = int(page["pdf_page"])
        page_settings = _page_settings(settings, page_number)
        columns = list(page_settings["expected_columns"])
        geometry = detect_table_geometry(binary, page_settings)
        page_key = f"page-{page_number:04d}"
        geometry_index[page_key] = geometry.to_dict()
        x0, y0, x1, y1 = geometry.bbox

        table_path = table_dir / f"{page_key}.png"
        cv2.imwrite(str(table_path), ocr_image[y0:y1, x0:x1])
        regions.append(
            RegionRecord(
                region_id=f"{page_key}-table",
                document_id=page["document_id"],
                pdf_page=page_number,
                region_type="table",
                region_image=str(table_path),
                source_image=page["ocr_image"],
                bbox=geometry.bbox,
                table_bbox=geometry.bbox,
                metadata={"layout_confidence": geometry.confidence},
            )
        )

        for row_index, (top, bottom) in enumerate(geometry.row_boundaries):
            row_id = f"{page_key}-r{row_index:03d}"
            row_path = row_dir / f"{row_id}.png"
            cv2.imwrite(str(row_path), ocr_image[top:bottom, x0:x1])
            regions.append(
                RegionRecord(
                    region_id=row_id,
                    document_id=page["document_id"],
                    pdf_page=page_number,
                    region_type="row",
                    region_image=str(row_path),
                    source_image=page["ocr_image"],
                    bbox=(x0, top, x1, bottom),
                    table_bbox=geometry.bbox,
                    row_index=row_index,
                )
            )
            for column_index, column_id in enumerate(columns):
                left, right = geometry.column_ranges[column_index]
                crop_left = max(x0, left - padding)
                crop_right = min(x1, right + padding)
                crop_top = max(geometry.header_bottom, top - padding)
                crop_bottom = min(y1, bottom + padding)
                cell_id = f"{row_id}-c{column_index:02d}"
                cell_path = cell_dir / f"{cell_id}.png"
                cv2.imwrite(str(cell_path), ocr_image[crop_top:crop_bottom, crop_left:crop_right])
                regions.append(
                    RegionRecord(
                        region_id=cell_id,
                        document_id=page["document_id"],
                        pdf_page=page_number,
                        region_type="cell",
                        region_image=str(cell_path),
                        source_image=page["ocr_image"],
                        bbox=(crop_left, crop_top, crop_right, crop_bottom),
                        table_bbox=geometry.bbox,
                        row_index=row_index,
                        column_index=column_index,
                        column_id=column_id,
                    )
                )

        if page_settings.get("save_debug_overlays", True):
            _draw_debug(ocr_image, geometry, debug_dir / f"{page_key}.png")

    write_json(target / "geometry.json", geometry_index)
    manifest = target / "regions.jsonl"
    write_jsonl(manifest, regions)
    return manifest
