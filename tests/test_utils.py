from harvest_ocr.utils import parse_page_range


def test_parse_page_range():
    assert parse_page_range("1,3-5,5") == [1, 3, 4, 5]
