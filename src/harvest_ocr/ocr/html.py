from __future__ import annotations

from html.parser import HTMLParser

from ..types import OCRItem


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.current: list[str] = []
        self.row = -1
        self.col = 0
        self.items: list[OCRItem] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            self.col = 0
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.current = []
        elif tag == "br" and self.in_cell:
            self.current.append(" ")

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self.in_cell:
            text = " ".join("".join(self.current).split())
            self.items.append(
                OCRItem(
                    text=text,
                    row_index=max(self.row, 0),
                    column_index=self.col,
                    block_type="table_cell",
                )
            )
            self.col += 1
            self.in_cell = False


def html_table_items(html: str) -> list[OCRItem]:
    parser = _TableHTMLParser()
    parser.feed(html or "")
    return parser.items


def collect_html_values(value) -> list[str]:
    """Recursively collect HTML table strings from model-native JSON."""
    found: list[str] = []
    if isinstance(value, str) and "<table" in value.lower():
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(collect_html_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_html_values(item))
    return found
