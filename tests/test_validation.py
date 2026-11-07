from harvest_ocr.validation import validate_record


def test_yield_relation_accepts_consistent_row():
    record = {"country_en": "Example", "area_1925": 30, "production_1925": 774, "yield_1925": 26.1}
    flags = validate_record(record, {})
    assert "yield_mismatch_1925" not in flags


def test_yield_relation_flags_shifted_value():
    record = {"country_en": "Example", "area_1925": 30, "production_1925": 774, "yield_1925": 2.0}
    flags = validate_record(record, {})
    assert "yield_mismatch_1925" in flags
