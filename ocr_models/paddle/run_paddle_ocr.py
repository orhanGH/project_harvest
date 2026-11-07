#!/usr/bin/env python3
"""Run PaddleOCR 3.x table recognition or PP-StructureV3."""

from __future__ import annotations

import argparse

from harvest_ocr.ocr.paddle import PaddleAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--regions-manifest", dest="manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipeline", choices=("table_recognition_v2", "pp_structure_v3"), default="table_recognition_v2")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--region-type", choices=("table", "row", "cell"), default="table")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--orientation", action="store_true")
    parser.add_argument("--unwarp", action="store_true")
    args = parser.parse_args()

    adapter = PaddleAdapter(
        pipeline=args.pipeline,
        device=args.device or None,
        use_doc_orientation_classify=args.orientation,
        use_doc_unwarping=args.unwarp,
    )
    output = adapter.run(args.manifest, args.output_dir, args.region_type, args.limit)
    print(output)


if __name__ == "__main__":
    main()
