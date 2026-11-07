from __future__ import annotations
import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
from paddleocr import LayoutDetection


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    """
    Convert a human-friendly 1-based page specification such as:
        46
        38,40,46
        38-46
        1,5,10-12

    into zero-based PDF page indices.

    If spec is None, all pages are returned.
    """
    if spec is None:
        return list(range(page_count))

    pages: set[int] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)

            if start > end:
                raise ValueError(f"Invalid page range: {part}")

            for page_number in range(start, end + 1):
                pages.add(page_number - 1)
        else:
            pages.add(int(part) - 1)

    invalid = [idx for idx in pages if idx < 0 or idx >= page_count]
    if invalid:
        human_pages = [idx + 1 for idx in invalid]
        raise ValueError(
            f"Requested page(s) outside PDF range 1-{page_count}: "
            f"{human_pages}"
        )

    return sorted(pages)


def render_page(
    pdf: pdfium.PdfDocument,
    page_index: int,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Render one PDF page to PNG using pypdfium2.
    """
    page = pdf[page_index]

    # PDF coordinates are based on 72 DPI.
    scale = dpi / 72.0

    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    image.save(output_path)

    bitmap.close()
    page.close()


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect table regions in PDF pages using PaddleOCR LayoutDetection."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Input PDF.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for rendered pages and detection results.",
    )

    parser.add_argument(
        "--pages",
        default=None,
        help=(
            "1-based PDF pages to process, e.g. "
            "'46', '38,40,46', or '38-46'. "
            "Default: all pages."
        ),
    )

    parser.add_argument(
        "--model",
        default="PP-DocLayout_plus-L",
        help="PaddleOCR layout detection model.",
    )

    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Inference device, e.g. gpu:0 or cpu.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PDF rendering DPI. Default: 180.",
    )

    parser.add_argument(
        "--table-score-threshold",
        type=float,
        default=0.50,
        help="Minimum confidence for normalized table detections.",
    )

    parser.add_argument(
    "--layout-threshold",
    type=float,
    default=None,
    help=(
        "Confidence threshold applied directly by PaddleOCR "
        "LayoutDetection. Default: model configuration.")
    )
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    if not 0.0 <= args.table_score_threshold <= 1.0:
        raise ValueError("--table-score-threshold must be between 0 and 1")

    rendered_dir = output_dir / "rendered"
    raw_dir = output_dir / "raw"
    visualizations_dir = output_dir / "visualizations"

    rendered_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    tables_jsonl = output_dir / "tables.jsonl"
    manifest_path = output_dir / "manifest.json"

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = len(pdf)
    page_indices = parse_pages(args.pages, page_count)

    print(f"PDF: {pdf_path}")
    print(f"PDF pages: {page_count}")
    print(
        "Processing pages:",
        ", ".join(str(idx + 1) for idx in page_indices),
    )
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"DPI: {args.dpi}")
    print(f"Output: {output_dir}")

    manifest = {
        "source_pdf": str(pdf_path),
        "document_id": pdf_path.stem,
        "pdf_page_count": page_count,
        "processed_page_numbers": [idx + 1 for idx in page_indices],
        "model": args.model,
        "device": args.device,
        "render_dpi": args.dpi,
        "table_score_threshold": args.table_score_threshold,
	"layout_threshold": args.layout_threshold,
    }
    write_json(manifest_path, manifest)

    print("Loading PaddleOCR layout model...")

    model = LayoutDetection(
        model_name=args.model,
        device=args.device,
    )

    total_tables = 0

    with tables_jsonl.open("w", encoding="utf-8") as normalized_output:
        for page_index in page_indices:
            page_number = page_index + 1

            print()
            print(f"=== PDF page {page_number} / index {page_index} ===")

            rendered_path = (
                rendered_dir / f"page_{page_number:04d}.png"
            )

            render_page(
                pdf=pdf,
                page_index=page_index,
                output_path=rendered_path,
                dpi=args.dpi,
            )

            results = model.predict(
                str(rendered_path),
                batch_size=1,
                layout_nms=True,
		threshold={
            # Selected HARVEST 9 classes: threshold = 0.50
            0: 0.50,   # paragraph_title
            1: 1.00,   # image
            2: 0.50,   # text
            3: 0.50,   # number
            4: 1.00,   # abstract
            5: 0.50,   # content
            6: 0.50,   # figure_title
            7: 1.00,   # formula
            8: 0.38,   # table
            9: 1.00,   # reference
            10: 0.50,  # doc_title
            11: 0.50,  # footnote
            12: 0.50,  # header
            13: 1.00,  # algorithm
            14: 1.00,  # footer
            15: 1.00,  # seal
            16: 1.00,  # chart
            17: 1.00,  # formula_number
            18: 1.00,  # aside_text
            19: 1.00,  # reference_content
        },
            )

            for result in results:
                raw_result = result.json

                raw_path = (
                    raw_dir / f"page_{page_number:04d}.json"
                )
                write_json(raw_path, raw_result)

                result.save_to_img(
                    save_path=str(visualizations_dir)
                )

                payload = raw_result.get("res", raw_result)
                boxes = payload.get("boxes", [])

                tables = []

                for box in boxes:
                    if str(box.get("label", "")).lower() != "table":
                        continue

                    score = float(box["score"])

                    if score < args.table_score_threshold:
                        continue

                    coordinate = [
                        float(value)
                        for value in box["coordinate"]
                    ]

                    tables.append(
                        {
                            "confidence": score,
                            "bbox": coordinate,
                            "class_id": int(box["cls_id"]),
                        }
                    )

                # Deterministic ordering: top-to-bottom, then left-to-right.
                tables.sort(
                    key=lambda table: (
                        table["bbox"][1],
                        table["bbox"][0],
                    )
                )

                normalized_tables = []

                for table_index, table in enumerate(tables, start=1):
                    table_id = (
                        f"{pdf_path.stem}"
                        f"_p{page_number:04d}"
                        f"_t{table_index:03d}"
                    )

                    normalized_tables.append(
                        {
                            "table_id": table_id,
                            **table,
                        }
                    )

                record = {
                    "document_id": pdf_path.stem,
                    "source_pdf": str(pdf_path),
                    "page_index": page_index,
                    "page_number": page_number,
                    "render_dpi": args.dpi,
                    "rendered_image": str(rendered_path),
                    "detector": args.model,
                    "tables": normalized_tables,
                }

                normalized_output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_tables += len(normalized_tables)

                print(
                    f"Detected {len(normalized_tables)} table(s) "
                    f"above threshold "
                    f"{args.table_score_threshold:.2f}"
                )

                for table in normalized_tables:
                    print(
                        f"  {table['table_id']} "
                        f"score={table['confidence']:.4f} "
                        f"bbox={table['bbox']}"
                    )

    pdf.close()

    print()
    print("=== Finished ===")
    print(f"Pages processed: {len(page_indices)}")
    print(f"Tables detected: {total_tables}")
    print(f"Normalized results: {tables_jsonl}")
    print(f"Raw PaddleOCR results: {raw_dir}")
    print(f"Visualizations: {visualizations_dir}")


if __name__ == "__main__":
    main()
