import argparse
import json
import time
from pathlib import Path

import pymupdf
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output-dir", required=True)

    # Human-readable PDF page numbers, starting from 1
    parser.add_argument("--start-page", type=int, default=20)
    parser.add_argument("--end-page", type=int, default=45)

    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_name = "microsoft/table-transformer-detection"

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=True
    )

    print("Loading model...")
    model = TableTransformerForObjectDetection.from_pretrained(
        model_name,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    print("Model ready.")

    doc = pymupdf.open(args.pdf)

    print("PDF pages:", len(doc))
    print(
        f"Testing PDF pages "
        f"{args.start_page}–{args.end_page}"
    )

    scale = args.dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)

    summary = []

    total_start = time.time()

    for pdf_page_number in range(
        args.start_page,
        args.end_page + 1
    ):
        page_index = pdf_page_number - 1

        if page_index >= len(doc):
            break

        print()
        print("=" * 70)
        print(
            f"PDF page {pdf_page_number} "
            f"(index {page_index})"
        )

        page = doc[page_index]

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        start = time.time()

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]]
        )

        result = processor.post_process_object_detection(
            outputs,
            threshold=args.threshold,
            target_sizes=target_sizes
        )[0]

        elapsed = time.time() - start

        detections = []

        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"]
        ):
            score = float(score.cpu())
            label_id = int(label.cpu())

            box = [
                round(float(v), 1)
                for v in box.cpu().tolist()
            ]

            label_name = model.config.id2label[label_id]

            detections.append({
                "label": label_name,
                "confidence": round(score, 4),
                "bbox": box
            })

        print(
            "Detected:",
            len(detections),
            "table region(s)"
        )

        for detection in detections:
            print(
                " ",
                detection["label"],
                "confidence=",
                detection["confidence"],
                "bbox=",
                detection["bbox"]
            )

        #
        # Save visual results only when a table was detected.
        #
        if detections:
            original_path = (
                output_dir /
                f"page_{pdf_page_number:03d}_original.png"
            )

            image.save(original_path)

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=10
                )

                label = (
                    f"{detection['label']} "
                    f"{detection['confidence']:.3f}"
                )

                draw.rectangle(
                    [x1, y1, x1 + 360, y1 + 45],
                    fill="white"
                )

                draw.text(
                    (x1 + 8, y1 + 8),
                    label,
                    fill="red"
                )

            detection_path = (
                output_dir /
                f"page_{pdf_page_number:03d}_detected.png"
            )

            annotated.save(detection_path)

        summary.append({
            "pdf_page": pdf_page_number,
            "page_index": page_index,
            "runtime_seconds": round(elapsed, 3),
            "detections": detections
        })

    total_elapsed = time.time() - total_start

    summary_path = output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            {
                "pdf": args.pdf,
                "start_page": args.start_page,
                "end_page": args.end_page,
                "threshold": args.threshold,
                "total_runtime_seconds":
                    round(total_elapsed, 2),
                "pages": summary
            },
            indent=2
        ),
        encoding="utf-8"
    )

    detected_pages = [
        x for x in summary
        if x["detections"]
    ]

    print()
    print("=" * 70)
    print("FINISHED")
    print("Pages tested:", len(summary))
    print(
        "Pages with detected tables:",
        len(detected_pages)
    )
    print(
        "Total runtime:",
        round(total_elapsed, 2),
        "seconds"
    )
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
