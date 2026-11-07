from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "ocr_models"
    / "effocr"
    / "run_effocr_ocr.py"
)
SPEC = importlib.util.spec_from_file_location("harvest_effocr_adapter", SCRIPT)
assert SPEC and SPEC.loader
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class EffOCRAdapterTests(unittest.TestCase):
    def test_jsonl_manifest_and_relative_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "cell.png"
            crop.touch()
            manifest = root / "regions.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "region_id": "r1",
                        "region_type": "cell",
                        "field": "area_1925",
                        "crop_path": "cell.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = ADAPTER.load_records(manifest)
            self.assertEqual(len(records), 1)
            self.assertEqual(
                ADAPTER.record_image_path(records[0], manifest, None), crop.resolve()
            )

    def test_region_filter_and_limit(self) -> None:
        records = [
            {"region_type": "cell", "region_id": "a"},
            {"region_type": "row", "region_id": "b"},
            {"region_type": "cell", "region_id": "c"},
        ]
        selected = ADAPTER.select_records(records, "cell", 1)
        self.assertEqual([record["region_id"] for record in selected], ["a"])

    def test_compact_predictions_drops_crop_arrays(self) -> None:
        predictions = {
            0: {
                "bbox": (1, 2, 3, 4),
                "word_preds": ["12,345"],
                "words": [(object(), (0, 0, 1, 1))],
            }
        }
        compact = ADAPTER.compact_predictions(predictions)
        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0]["line_id"], 0)
        self.assertEqual(compact[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(compact[0]["words"], ["12,345"])
        # The current adapter also retains character-level provenance metadata.
        self.assertIn("character_details", compact[0])


if __name__ == "__main__":
    unittest.main()
