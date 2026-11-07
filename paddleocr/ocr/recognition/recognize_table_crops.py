#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from paddleocr import TableRecognitionPipelineV2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR TableRecognitionPipelineV2 "
            "on HARVEST table-region crops."
        )
    )

    parser.add_argument(
        "--crops-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--device",
        default="gpu:0",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_crops = sorted(
        args.crops_root.glob(
            "page_*/region_*_table.png"
        )
    )

    print()
    print("==============================================")
    print("HARVEST TABLE RECOGNITION")
    print("==============================================")
    print()
    print("Crops root:")
    print(args.crops_root)
    print()
    print("Output:")
    print(args.output_dir)
    print()
    print(
        "Table crops found:",
        len(table_crops),
    )
    print()

    if not table_crops:
        raise RuntimeError(
            "No table crops were found."
        )

    print(
        "Initializing TableRecognitionPipelineV2..."
    )
    print()
    print(
        "HARVEST already performed layout detection."
    )
    print(
        "Table pipeline layout detection: DISABLED"
    )
    print(
        "Text recognition model: PP-OCRv6_medium_rec"
    )
    print()

    pipeline = TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
        text_recognition_model_name=
            "PP-OCRv6_medium_rec",
        device=args.device,
    )

    print("Pipeline initialized.")
    print()

    summary = []

    for index, crop_path in enumerate(
        table_crops,
        start=1,
    ):
        page_name = crop_path.parent.name
        table_name = crop_path.stem

        table_output = (
            args.output_dir
            / page_name
            / table_name
        )

        table_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 70)
        print(
            f"[{index}/{len(table_crops)}]"
        )
        print(crop_path)
        print("=" * 70)

        outputs = list(
            pipeline.predict(
                str(crop_path),
                use_layout_detection=False,
                use_ocr_model=True,
            )
        )

        print(
            "Result objects:",
            len(outputs),
        )

        result_records = []

        for result_index, result in enumerate(
            outputs,
            start=1,
        ):
            result_dir = (
                table_output
                / f"result_{result_index:02d}"
            )

            result_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print()
            print(
                f"Result {result_index}"
            )

            result.print()

            result.save_to_json(
                save_path=str(
                    result_dir
                )
            )

            result.save_to_html(
                save_path=str(
                    result_dir
                )
            )

            result.save_to_xlsx(
                save_path=str(
                    result_dir
                )
            )

            result.save_to_img(
                save_path=str(
                    result_dir
                )
            )

            result_records.append(
                {
                    "result_index":
                        result_index,

                    "output_dir":
                        str(result_dir),
                }
            )

        summary.append(
            {
                "table_crop":
                    str(crop_path),

                "page":
                    page_name,

                "table":
                    table_name,

                "result_count":
                    len(outputs),

                "results":
                    result_records,
            }
        )

    summary_path = (
        args.output_dir
        / "table_recognition_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("==============================================")
    print("FINISHED")
    print("==============================================")
    print()
    print(
        "Tables processed:",
        len(table_crops),
    )
    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
