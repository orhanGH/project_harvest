#!/usr/bin/env python3
"""Run Dell Research's EfficientOCR on geometry-defined HARVEST crops.

The adapter intentionally operates on cell or text-line crops. EfficientOCR is a
recognizer, not a table-structure parser; page-to-cell geometry remains a
separate HARVEST stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


PATH_KEYS = ("region_image", "crop_path", "image_path", "path", "image")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a CSV, JSON array/object, or JSONL regions manifest."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("regions", "records", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"Unsupported JSON manifest shape: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def record_image_path(
    record: dict[str, Any], manifest: Path, image_root: Path | None
) -> Path:
    raw_path = next((record.get(key) for key in PATH_KEYS if record.get(key)), None)
    if raw_path is None:
        raise ValueError(f"Region has no image path in any of {PATH_KEYS}: {record}")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (image_root or manifest.parent) / path
    return path.resolve()


def select_records(
    records: Iterable[dict[str, Any]], region_type: str, limit: int
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if region_type == "any" or record.get("region_type", "cell") == region_type
    ]
    return selected[:limit] if limit > 0 else selected


def default_config(model_dir: Path, device: str) -> dict[str, Any]:
    """Configuration for Dell Research's pretrained English ONNX models."""
    shared = {
        "model_backend": "onnx",
        "model_dir": str(model_dir),
        "num_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        # efficient_ocr 0.1.1 has an ONNX provider-selection bug:
        # a non-None value leaves its local `providers` variable undefined.
        # None makes ONNX Runtime discover CPUExecutionProvider automatically.
        "providers": None,
    }
    return {
        "Global": {
            "char_only": True,
            # A HARVEST cell is already a text line. Skipping the page-level
            # line detector prevents neighboring columns from being merged.
            "skip_line_detection": True,
        },
        "Recognizer": {
            "char": {
                **shared,
                "hf_repo_id": "dell-research-harvard/effocr_en/char_recognizer",
                "device": device,
            },
            "word": {
                **shared,
                "hf_repo_id": "dell-research-harvard/effocr_en/word_recognizer",
                "device": device,
            },
        },
        "Localizer": {
            **shared,
            "hf_repo_id": "dell-research-harvard/effocr_en",
            "device": device,
        },
        "Line": {
            **shared,
            "hf_repo_id": "dell-research-harvard/effocr_en",
            "device": device,
        },
    }


def load_config(config_path: Path | None, model_dir: Path, device: str) -> dict[str, Any]:
    if config_path is None:
        return default_config(model_dir, device)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return config


def compact_predictions(preds: Any) -> list[dict[str, Any]]:
    """Keep useful text geometry while excluding image arrays from EffOCR output."""
    if not isinstance(preds, dict):
        return []

    lines: list[dict[str, Any]] = []

    for line_id in sorted(preds, key=lambda value: int(value)):
        line = preds[line_id]
        item: dict[str, Any] = {"line_id": int(line_id)}

        line_bbox = None
        if line.get("bbox") is not None:
            line_bbox = [int(value) for value in line["bbox"]]
            item["bbox"] = line_bbox
            item["bbox_order"] = "y0_x0_y1_x1"

        if line.get("word_preds") is not None:
            item["words"] = [
                "" if value is None else str(value)
                for value in line["word_preds"]
            ]

        character_text = str(line.get("char_preds") or "")
        item["characters"] = character_text

        character_details: list[dict[str, Any]] = []

        for index, character in enumerate(line.get("chars") or []):
            if not isinstance(character, (tuple, list)) or len(character) < 2:
                continue

            raw_bbox = character[1]
            if raw_bbox is None:
                continue

            line_relative_bbox = [int(value) for value in raw_bbox]

            detail: dict[str, Any] = {
                "index": index,
                "prediction": (
                    character_text[index]
                    if index < len(character_text)
                    else None
                ),
                "bbox_yxyx_line": line_relative_bbox,
            }

            if line_bbox is not None and len(line_relative_bbox) == 4:
                line_y0, line_x0, _, _ = line_bbox
                char_y0, char_x0, char_y1, char_x1 = line_relative_bbox

                detail["bbox_yxyx_cell"] = [
                    line_y0 + char_y0,
                    line_x0 + char_x0,
                    line_y0 + char_y1,
                    line_x0 + char_x1,
                ]

            character_details.append(detail)

        item["character_details"] = character_details
        item["character_count_mismatch"] = (
            len(character_text) != len(character_details)
        )

        lines.append(item)

    return lines


def build_engine(config: dict[str, Any]) -> Any:
    try:
        from efficient_ocr import EffOCR
    except ImportError as exc:
        raise SystemExit(
            "efficient_ocr is not installed. Follow the EffOCR installation instructions in the original HARVEST checkout (the INSTALL.md was not supplied in this snapshot)"
        ) from exc
    return EffOCR(config=config)


def run(args: argparse.Namespace) -> Path:
    manifest = Path(args.regions_manifest).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    records = select_records(load_records(manifest), args.region_type, args.limit)
    if not records:
        raise SystemExit(f"No {args.region_type!r} regions found in {manifest}")

    resolved: list[tuple[dict[str, Any], Path]] = []
    for index, record in enumerate(records):
        image_path = record_image_path(record, manifest, image_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing crop image: {image_path}")
        copied = dict(record)
        copied.setdefault("region_id", f"region_{index:06d}")
        resolved.append((copied, image_path))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "effocr_raw.jsonl"

    if args.validate_only:
        print(f"Validated {len(resolved)} regions; no model was loaded.")
        return output_path

    config = load_config(
        Path(args.config).resolve() if args.config else None,
        Path(args.model_dir).resolve(),
        args.device,
    )
    engine = build_engine(config)

    with output_path.open("w", encoding="utf-8") as stream:
        for index, (record, image_path) in enumerate(resolved, start=1):
            # efficient_ocr 0.1.1 rejects homogeneous lists due to an upstream
            # type-check bug. Single-image calls are deliberate and stable.
            try:
                result = engine.infer(str(image_path))[0]
                text = result.text
                lines = compact_predictions(result.preds)
                error = None
            except (ValueError, RuntimeError) as exc:
                # Blank cells or crops without localizer detections can cause
                # efficient_ocr 0.1.1 to concatenate an empty tensor list.
                # Preserve the row and record the failure instead of aborting
                # the complete table.
                text = ""
                lines = []
                error = f"{type(exc).__name__}: {exc}"
                print(
                    f"Skipped {record.get('region_id', image_path.name)}: {error}",
                    file=sys.stderr,
                )

            output = {
                **record,
                "image_path": str(image_path),
                "source_model": "efficient_ocr",
                "text": text,
                "lines": lines,
                "error": error,
            }
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            if index % args.log_every == 0 or index == len(resolved):
                print(f"Processed {index}/{len(resolved)} regions", file=sys.stderr)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-manifest", "--regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--model-dir", default=os.environ.get("HARVEST_EFFOCR_MODELS", "models/effocr"))
    parser.add_argument("--config", help="Optional JSON config replacing the pretrained default")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--region-type", choices=("cell", "row", "any"), default="cell")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
