from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


class HarvestError(RuntimeError):
    """Actionable pipeline error."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def parse_page_range(spec: str | None, page_count: int | None = None) -> list[int]:
    """Parse one-based page ranges such as ``44-59,65,67``."""
    if not spec:
        if page_count is None:
            return []
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise HarvestError(f"Invalid page range component: {part!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            raise HarvestError(f"Invalid page range component: {part!r}")
        pages.update(range(start, end + 1))
    if page_count is not None:
        invalid = [page for page in pages if page > page_count]
        if invalid:
            raise HarvestError(
                f"Requested page {min(invalid)} but the PDF has only {page_count} pages"
            )
    return sorted(pages)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarvestError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def require_import(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise HarvestError(
            f"Missing optional dependency {module_name!r}. Install with: {install_hint}"
        ) from exc


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"
