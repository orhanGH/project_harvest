from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import HarvestPipeline
from .utils import HarvestError


MODELS = ("tesseract", "paddle", "chandra", "surya")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harvest-ocr",
        description="Historical table OCR, parsing, validation, and benchmarking.",
    )
    parser.add_argument("--config", default="configs/harvest_1926.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("extract", help="Extract original page scans and embedded OCR text")
    subparsers.add_parser("preprocess", help="Crop, deskew, normalize, and binarize pages")
    subparsers.add_parser("layout", help="Detect table geometry and create table/row/cell regions")

    ocr = subparsers.add_parser("ocr", help="Run one OCR adapter")
    ocr.add_argument("--model", choices=MODELS, required=True)
    ocr.add_argument("--region-type", choices=("table", "row", "cell"))
    ocr.add_argument("--limit", type=int, default=0)

    parse = subparsers.add_parser("parse", help="Normalize model output into HARVEST rows")
    parse.add_argument("--model", choices=MODELS, required=True)

    validate = subparsers.add_parser("validate", help="Apply consistency checks")
    validate.add_argument("--model", choices=MODELS, required=True)

    export = subparsers.add_parser("export", help="Write one target-year CSV per product")
    export.add_argument("--model", choices=MODELS, required=True)

    benchmark = subparsers.add_parser("benchmark", help="Compare validated output to gold JSONL")
    benchmark.add_argument("--model", choices=MODELS, required=True)
    benchmark.add_argument("--gold", required=True)

    run = subparsers.add_parser("run", help="Run all stages for one model")
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = HarvestPipeline(args.config)
        if args.command == "extract":
            result = pipeline.extract()
        elif args.command == "preprocess":
            result = pipeline.preprocess()
        elif args.command == "layout":
            result = pipeline.layout()
        elif args.command == "ocr":
            result = pipeline.ocr(args.model, args.limit, args.region_type)
        elif args.command == "parse":
            result = pipeline.parse(args.model)
        elif args.command == "validate":
            result = pipeline.validate(args.model)
        elif args.command == "export":
            result = pipeline.export(args.model)
        elif args.command == "benchmark":
            result = pipeline.benchmark(args.model, Path(args.gold).resolve())
        else:
            result = pipeline.run(args.model, args.limit)
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        elif isinstance(result, tuple):
            print("\n".join(str(item) for item in result))
        else:
            print(result)
        return 0
    except HarvestError as exc:
        print(f"HARVEST error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
