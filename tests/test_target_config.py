from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "harvest_1926.yaml"
pytestmark = pytest.mark.skipif(
    not CONFIG_PATH.exists(),
    reason="The original configs/harvest_1926.yaml was not included in the exported code snapshot",
)

from harvest_ocr.utils import load_config, parse_page_range


def test_1926_config_covers_tables_7_through_40():
    config_path = CONFIG_PATH
    config = load_config(config_path)
    catalog = config["parser"]["table_catalog"]
    assert [int(item["table_number"]) for item in catalog] == list(range(7, 41))
    assert parse_page_range(config["input"]["pages"]) == list(range(44, 84))
    assert config["parser"]["target_year"] == 1925


def test_standard_table_ranges_select_only_the_1925_subcolumns():
    config_path = CONFIG_PATH
    layout = load_config(config_path)["layout"]

    assert layout["expected_columns"] == [
        "country_fr",
        "area_1925",
        "production_1925",
        "yield_1925",
        "country_en",
    ]
    assert layout["column_ranges"] == [
        [0.0, 0.185],
        [0.32, 0.405],
        [0.575, 0.66],
        [0.795, 0.865],
        [0.85, 1.0],
    ]
