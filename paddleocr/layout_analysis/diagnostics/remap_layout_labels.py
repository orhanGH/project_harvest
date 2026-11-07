#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def parse_mapping(values):
    mapping = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid mapping '{value}'. "
                f"Expected OLD=NEW."
            )

        old, new = value.split(
            "=",
            1,
        )

        old = old.strip()
        new = new.strip()

        if not old or not new:
            raise ValueError(
                f"Invalid mapping '{value}'."
            )

        mapping[old] = new

    return mapping


def remap_label(label, mapping):
    label = str(label)

    return mapping.get(
        label,
        label,
    )


def process_labelme(
    data,
    mapping,
):
    result = copy.deepcopy(
        data
    )

    before = Counter()
    after = Counter()

    for shape in result.get(
        "shapes",
        [],
    ):
        label = str(
            shape.get(
                "label",
                "",
            )
        )

        before[label] += 1

        new_label = remap_label(
            label,
            mapping,
        )

        shape["label"] = (
            new_label
        )

        after[new_label] += 1

    return result, before, after


def process_paddle(
    data,
    mapping,
):
    result = copy.deepcopy(
        data
    )

    if (
        "res" in result
        and isinstance(
            result["res"],
            dict,
        )
    ):
        payload = result["res"]
    else:
        payload = result

    before = Counter()
    after = Counter()

    for box in payload.get(
        "boxes",
        [],
    ):
        label = str(
            box.get(
                "label",
                "",
            )
        )

        before[label] += 1

        new_label = remap_label(
            label,
            mapping,
        )

        box["label"] = (
            new_label
        )

        after[new_label] += 1

    return result, before, after


def detect_format(data):
    if "shapes" in data:
        return "labelme"

    if (
        "boxes" in data
        or (
            "res" in data
            and isinstance(
                data["res"],
                dict,
            )
            and "boxes"
                in data["res"]
        )
    ):
        return "paddle"

    raise RuntimeError(
        "Could not determine JSON format."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help=(
            "Label mapping OLD=NEW. "
            "May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    mapping = parse_mapping(
        args.map
    )

    if not mapping:
        raise RuntimeError(
            "No mappings supplied."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        args.input_dir.glob(
            "page_*.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No page_*.json files in "
            f"{args.input_dir}"
        )

    total_before = Counter()
    total_after = Counter()

    formats = Counter()

    for path in files:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        file_format = detect_format(
            data
        )

        formats[
            file_format
        ] += 1

        if file_format == "labelme":
            (
                result,
                before,
                after,
            ) = process_labelme(
                data,
                mapping,
            )

        elif file_format == "paddle":
            (
                result,
                before,
                after,
            ) = process_paddle(
                data,
                mapping,
            )

        else:
            raise RuntimeError(
                file_format
            )

        total_before.update(
            before
        )

        total_after.update(
            after
        )

        output_path = (
            args.output_dir
            / path.name
        )

        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print(
        "======================================"
    )
    print(
        "HARVEST LABEL REMAPPING"
    )
    print(
        "======================================"
    )

    print()
    print(
        "Input:",
        args.input_dir,
    )

    print(
        "Output:",
        args.output_dir,
    )

    print()
    print(
        "Mappings:"
    )

    for old, new in mapping.items():
        print(
            f"  {old} -> {new}"
        )

    print()
    print(
        "Files:",
        len(files),
    )

    print(
        "Formats:",
        dict(formats),
    )

    print()
    print(
        "BEFORE"
    )

    for label, count in sorted(
        total_before.items()
    ):
        print(
            f"  {label:25s} "
            f"{count:4d}"
        )

    print()
    print(
        "AFTER"
    )

    for label, count in sorted(
        total_after.items()
    ):
        print(
            f"  {label:25s} "
            f"{count:4d}"
        )


if __name__ == "__main__":
    main()
