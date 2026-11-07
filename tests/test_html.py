from harvest_ocr.ocr.html import html_table_items


def test_html_table_cells_have_positions():
    items = html_table_items("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    assert [(item.text, item.row_index, item.column_index) for item in items] == [
        ("A", 0, 0), ("B", 0, 1), ("1", 1, 0), ("2", 1, 1)
    ]
