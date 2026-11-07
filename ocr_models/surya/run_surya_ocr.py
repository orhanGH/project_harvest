#!/usr/bin/env python3
"""Run Surya OCR 2 and table structure recognition."""

from __future__ import annotations

import argparse

from harvest_ocr.ocr.surya import SuryaAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--regions-manifest", dest="manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-type", choices=("table", "row", "cell"), default="table")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    adapter = SuryaAdapter(
        run_ocr=not args.no_ocr,
        run_table=not args.no_table,
        keep_server=args.keep_server,
    )
    output = adapter.run(args.manifest, args.output_dir, args.region_type, args.limit)
    print(output)


if __name__ == "__main__":
    main()
