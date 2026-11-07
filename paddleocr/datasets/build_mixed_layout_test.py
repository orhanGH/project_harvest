#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader, PdfWriter


SOURCES = [
    {
        "series": "UNSY",
        "source_id": "UNSY_1948",
        "relative_path": "United_Nations_Statistical_Yearbooks/1948.pdf",
        "pages": [9, 10, 20, 30, 45, 46, 408],
    },
    {
        "series": "UNSY",
        "source_id": "UNSY_1955",
        "relative_path": "United_Nations_Statistical_Yearbooks/1955.pdf",
        "pages": [38, 39, 199, 314, 555, 614, 632],
    },
    {
        "series": "UNSY",
        "source_id": "UNSY_1961",
        "relative_path": "United_Nations_Statistical_Yearbooks/1961.pdf",
        "pages": [129, 228, 408, 457, 460, 558, 679],
    },
    {
        "series": "IIA",
        "source_id": "IIA_1928_1929",
        "relative_path": "IIA_Yearbooks/ac 4 30 7 1928-29.pdf",
        "pages": [7, 8, 10, 19, 74, 79, 217, 254, 292, 296],
    },
    {
        "series": "IIA",
        "source_id": "IIA_1926_1927",
        "relative_path": "IIA_Yearbooks/ac 4 30 7 1926-27.pdf",
        "pages": [10, 13, 16, 114, 195, 225, 243, 302],
    },
    {
        "series": "ISY",
        "source_id": "ISY_1930",
        "relative_path": "International_Statistical_Yearbook/1930",
        "pages": [11, 13, 14, 45, 63, 64, 129, 216, 265],
    },
    {
        "series": "ISY",
        "source_id": "ISY_1941",
        "relative_path": "International_Statistical_Yearbook/1941",
        "pages": [23, 26, 40, 50, 55, 56, 142, 145, 350, 355, 357],
    },
    {
        "series": "WSoA",
        "source_id": "WSoA_1930_VOLUME_V",
        "relative_path": "World_Census_of_Agriculture/1930 VOLUME V.pdf",
        "pages": [46, 53, 58, 153, 194, 218, 222],
    },
    {
        "series": "WSoA",
        "source_id": "WSoA_1930_VOLUME_II",
        "relative_path": "World_Census_of_Agriculture/1930 VOLUME II.pdf",
        "pages": [1, 5, 6, 9, 28, 142, 191, 221],
    },
]


def resolve_pdf(input_root: Path, relative_path: str) -> Path:
    path = input_root / relative_path

    if path.is_file():
        return path.resolve()

    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))

        if len(pdfs) != 1:
            raise RuntimeError(
                f"Expected exactly one PDF in {path}, "
                f"found {len(pdfs)}"
            )

        return pdfs[0].resolve()

    raise FileNotFoundError(f"Source not found: {path}")


def build_mixed_order(records, seed):
    rng = random.Random(seed)

    groups = defaultdict(list)

    for record in records:
        groups[record["series"]].append(record)

    for items in groups.values():
        rng.shuffle(items)

    result = []
    previous_series = None

    while any(groups.values()):
        candidates = [
            series
            for series, items in groups.items()
            if items and series != previous_series
        ]

        if not candidates:
            raise RuntimeError(
                "Could not construct ordering without consecutive "
                "pages from the same series."
            )

        max_remaining = max(len(groups[s]) for s in candidates)

        strongest = [
            s for s in candidates
            if len(groups[s]) == max_remaining
        ]

        chosen_series = rng.choice(strongest)
        result.append(groups[chosen_series].pop())
        previous_series = chosen_series

    if not all(
        a["series"] != b["series"]
        for a, b in zip(result, result[1:])
    ):
        raise RuntimeError("Mixed-order validation failed.")

    return result


def extract_small_source_pdfs(input_root, temp_dir):
    """
    Process one large source PDF at a time.

    Each selected source page is written to its own tiny PDF.
    This prevents all large PdfReader objects from staying in RAM.
    """
    records = []

    print("=== VALIDATING + EXTRACTING SOURCE PAGES ===")

    for source in SOURCES:
        pdf_path = resolve_pdf(
            input_root,
            source["relative_path"],
        )

        print()
        print(f"Opening {source['source_id']}: {pdf_path}")

        with pdf_path.open("rb") as source_fh:
            reader = PdfReader(source_fh)
            page_count = len(reader.pages)

            print(
                f"{source['source_id']:18s} "
                f"{page_count:4d} pages | "
                f"selected={len(source['pages'])}"
            )

            for page_number in source["pages"]:
                if page_number < 1 or page_number > page_count:
                    raise ValueError(
                        f"{source['source_id']}: requested page "
                        f"{page_number}, but PDF contains "
                        f"{page_count} pages"
                    )

                page_file = (
                    temp_dir
                    / f"{source['source_id']}_p{page_number:04d}.pdf"
                )

                writer = PdfWriter()
                writer.add_page(reader.pages[page_number - 1])

                with page_file.open("wb") as out_fh:
                    writer.write(out_fh)

                records.append(
                    {
                        "series": source["series"],
                        "source_id": source["source_id"],
                        "source_pdf": str(pdf_path),
                        "source_page": page_number,
                        "extracted_pdf": str(page_file),
                    }
                )

                del writer

        del reader
        gc.collect()

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261001,
    )

    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "_tmp_selected_pages"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    records = extract_small_source_pdfs(
        input_root,
        temp_dir,
    )

    mixed = build_mixed_order(
        records,
        seed=args.seed,
    )

    print()
    print("=== BUILDING FINAL MIXED PDF ===")

    writer = PdfWriter()

    for mixed_page, record in enumerate(mixed, start=1):
        extracted_path = Path(record["extracted_pdf"])

        with extracted_path.open("rb") as fh:
            reader = PdfReader(fh)
            writer.add_page(reader.pages[0])

        record["mixed_page"] = mixed_page

    pdf_output = output_dir / f"mixed_layout_test_{len(mixed)}p.pdf"

    with pdf_output.open("wb") as fh:
        writer.write(fh)

    manifest_output = (
        output_dir / f"mixed_layout_test_{len(mixed)}p_manifest.csv"
    )

    with manifest_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        fieldnames = [
            "mixed_page",
            "series",
            "source_id",
            "source_pdf",
            "source_page",
        ]

        csv_writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        csv_writer.writeheader()

        for record in mixed:
            csv_writer.writerow(
                {
                    key: record[key]
                    for key in fieldnames
                }
            )

    counts = Counter(
        record["series"]
        for record in mixed
    )

    summary = {
        "dataset": "HARVEST mixed layout validation",
        "seed": args.seed,
        "total_pages": len(mixed),
        "series_counts": dict(counts),
        "no_consecutive_same_series": all(
            a["series"] != b["series"]
            for a, b in zip(mixed, mixed[1:])
        ),
        "output_pdf": str(pdf_output),
        "manifest": str(manifest_output),
    }

    summary_output = (
        output_dir / f"mixed_layout_test_{len(mixed)}p_summary.json"
    )

    summary_output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== MIXED ORDER ===")

    for record in mixed:
        print(
            f"{record['mixed_page']:02d} | "
            f"{record['series']:4s} | "
            f"{record['source_id']:18s} | "
            f"source page {record['source_page']}"
        )

    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Temporary one-page PDFs are no longer needed.
    shutil.rmtree(temp_dir)

    print()
    print("Temporary extraction files removed.")


if __name__ == "__main__":
    main()
