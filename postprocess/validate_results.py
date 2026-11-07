#!/usr/bin/env python3
from __future__ import annotations

import argparse

from harvest_ocr.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--yield-relative-tolerance", type=float, default=0.18)
    parser.add_argument("--yield-absolute-tolerance", type=float, default=0.8)
    args = parser.parse_args()
    validate_dataset(
        args.input,
        args.output,
        args.summary,
        {
            "yield_relative_tolerance": args.yield_relative_tolerance,
            "yield_absolute_tolerance": args.yield_absolute_tolerance,
        },
    )


if __name__ == "__main__":
    main()
