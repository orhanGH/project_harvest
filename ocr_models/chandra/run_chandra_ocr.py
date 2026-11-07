#!/usr/bin/env python3
"""Run Chandra OCR 2 on HARVEST table regions."""

from __future__ import annotations

import argparse

from harvest_ocr.ocr.chandra import ChandraAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--regions-manifest", dest="manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--region-type", choices=("table", "row", "cell"), default="table")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    adapter = ChandraAdapter(args.method, args.batch_size, args.max_workers)
    output = adapter.run(args.manifest, args.output_dir, args.region_type, args.limit)
    print(output)


if __name__ == "__main__":
    main()
