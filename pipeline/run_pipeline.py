#!/usr/bin/env python3
"""Repository-local entry point; equivalent to the installed ``harvest-ocr`` CLI."""

from harvest_ocr.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
