from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .benchmark import evaluate
from .export import export_product_csvs
from .ingest import extract_pdf_pages
from .layout import generate_regions
from .ocr import ChandraAdapter, PaddleAdapter, SuryaAdapter, TesseractAdapter
from .parsers import AgriculturalTableParser
from .preprocess import preprocess_manifest
from .utils import HarvestError, load_config, resolve_project_path
from .validation import validate_dataset


class HarvestPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        self.project_root = Path(self.config["_project_root"])
        configured_run_dir = os.environ.get(
            "HARVEST_RUN_DIR", str(self.config["project"]["run_dir"])
        )
        self.run_dir = resolve_project_path(self.config, configured_run_dir)

    @property
    def pages_manifest(self) -> Path:
        return self.run_dir / "ingest" / "pages.jsonl"

    @property
    def processed_manifest(self) -> Path:
        return self.run_dir / "preprocess" / "processed_pages.jsonl"

    @property
    def regions_manifest(self) -> Path:
        return self.run_dir / "layout" / "regions.jsonl"

    def extract(self) -> Path:
        configured_pdf = os.environ.get("HARVEST_INPUT_PDF", str(self.config["input"]["pdf"]))
        input_pdf = resolve_project_path(self.config, configured_pdf)
        return extract_pdf_pages(
            input_pdf,
            self.run_dir / "ingest",
            pages=str(self.config["input"].get("pages") or ""),
            document_id=self.config["project"]["name"],
        )

    def preprocess(self) -> Path:
        if not self.pages_manifest.exists():
            raise HarvestError("Missing pages manifest. Run the extract stage first.")
        return preprocess_manifest(
            self.pages_manifest,
            self.run_dir / "preprocess",
            self.config.get("preprocess", {}),
        )

    def layout(self) -> Path:
        if not self.processed_manifest.exists():
            raise HarvestError("Missing processed page manifest. Run preprocess first.")
        return generate_regions(
            self.processed_manifest,
            self.run_dir / "layout",
            self.config["layout"],
        )

    def _adapter(self, model: str):
        settings: dict[str, Any] = self.config.get("ocr", {}).get(model, {})
        if model == "tesseract":
            return TesseractAdapter(**settings)
        if model == "paddle":
            return PaddleAdapter(**settings)
        if model == "chandra":
            return ChandraAdapter(**settings)
        if model == "surya":
            return SuryaAdapter(**settings)
        raise HarvestError(f"Unknown OCR model: {model}")

    def ocr(self, model: str, limit: int = 0, region_type: str | None = None) -> Path:
        if not self.regions_manifest.exists():
            raise HarvestError("Missing regions manifest. Run layout first.")
        adapter = self._adapter(model)
        return adapter.run(
            self.regions_manifest,
            self.run_dir / "ocr" / model,
            region_type=region_type,
            limit=limit,
        )

    def parse(self, model: str) -> Path:
        ocr_path = self.run_dir / "ocr" / model / "ocr_results.jsonl"
        if not ocr_path.exists():
            raise HarvestError(f"Missing OCR output for {model}: {ocr_path}")
        parser_settings = self.config.get("parser", {})
        parser = AgriculturalTableParser(
            columns=list(self.config["layout"]["expected_columns"]),
            table_catalog=parser_settings.get("table_catalog", []),
            keep_empty_rows=bool(parser_settings.get("keep_empty_rows", False)),
        )
        output = self.run_dir / "parsed" / model / "records.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        return parser.parse(self.regions_manifest, ocr_path, output)

    def validate(self, model: str) -> tuple[Path, Path]:
        parsed = self.run_dir / "parsed" / model / "records.jsonl"
        if not parsed.exists():
            raise HarvestError(f"Missing parsed output for {model}: {parsed}")
        output_dir = self.run_dir / "validated" / model
        output_dir.mkdir(parents=True, exist_ok=True)
        return validate_dataset(
            parsed,
            output_dir / "records.jsonl",
            output_dir / "summary.json",
            self.config.get("validation", {}),
        )

    def benchmark(self, model: str, gold_path: str | Path) -> Path:
        prediction = self.run_dir / "validated" / model / "records.jsonl"
        if not prediction.exists():
            raise HarvestError(f"Missing validated output for {model}: {prediction}")
        output = self.run_dir / "benchmarks" / model / "metrics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        return evaluate(gold_path, prediction, output)

    def export(self, model: str) -> Path:
        validated = self.run_dir / "validated" / model / "records.jsonl"
        if not validated.exists():
            raise HarvestError(f"Missing validated output for {model}: {validated}")
        parser_settings = self.config.get("parser", {})
        return export_product_csvs(
            validated,
            self.run_dir / "exports" / model / "1925",
            parser_settings.get("table_catalog", []),
        )

    def run(self, model: str, limit: int = 0) -> dict[str, str]:
        outputs = {
            "pages": str(self.extract()),
            "processed": str(self.preprocess()),
            "regions": str(self.layout()),
            "ocr": str(self.ocr(model, limit=limit)),
            "parsed": str(self.parse(model)),
        }
        validated, summary = self.validate(model)
        outputs["validated"] = str(validated)
        outputs["validation_summary"] = str(summary)
        outputs["product_csvs"] = str(self.export(model))
        return outputs
