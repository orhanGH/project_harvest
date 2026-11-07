#!/usr/bin/env python3
"""Run geometry-guided Tesseract OCR on HARVEST regions."""

from __future__ import annotations

import argparse

from harvest_ocr.ocr.tesseract import TesseractAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", "--regions-manifest", dest="regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lang", default="eng+fra")
    parser.add_argument("--fallback-lang", default="eng")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--region-type", choices=("cell", "row", "table"), default="cell")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    adapter = TesseractAdapter(
        languages=args.lang,
        fallback_language=args.fallback_lang,
        scale=args.scale,
        country_psm=args.psm,
        numeric_psm=args.psm,
    )
    output = adapter.run(args.regions, args.output_dir, args.region_type, args.limit)
    print(output)


if __name__ == "__main__":
    main()
