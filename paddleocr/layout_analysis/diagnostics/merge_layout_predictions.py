#!/usr/bin/env python3

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path


MERGE_CLASSES = {
    "text",
    "content",
    "header",
    "figure_title",
    "paragraph_title",
}


SETTINGS = {
    "conservative": {
        "x_overlap": 0.65,
        "vertical_gap_factor": 0.55,
        "y_overlap": 0.70,
        "horizontal_gap_factor": 0.75,
    },
    "moderate": {
        "x_overlap": 0.50,
        "vertical_gap_factor": 0.90,
        "y_overlap": 0.55,
        "horizontal_gap_factor": 1.20,
    },
    "aggressive": {
        "x_overlap": 0.35,
        "vertical_gap_factor": 1.40,
        "y_overlap": 0.40,
        "horizontal_gap_factor": 1.80,
    },
}


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def width(box):
    return max(
        0.0,
        box[2] - box[0],
    )


def height(box):
    return max(
        0.0,
        box[3] - box[1],
    )


def x_overlap_ratio(a, b):
    overlap = max(
        0.0,
        min(a[2], b[2])
        - max(a[0], b[0]),
    )

    denominator = min(
        width(a),
        width(b),
    )

    if denominator <= 0:
        return 0.0

    return overlap / denominator


def y_overlap_ratio(a, b):
    overlap = max(
        0.0,
        min(a[3], b[3])
        - max(a[1], b[1]),
    )

    denominator = min(
        height(a),
        height(b),
    )

    if denominator <= 0:
        return 0.0

    return overlap / denominator


def vertical_gap(a, b):
    if a[3] < b[1]:
        return b[1] - a[3]

    if b[3] < a[1]:
        return a[1] - b[3]

    return 0.0


def horizontal_gap(a, b):
    if a[2] < b[0]:
        return b[0] - a[2]

    if b[2] < a[0]:
        return a[0] - b[2]

    return 0.0


def should_merge(a, b, settings):
    """
    Two boxes merge if they are either:

    1. vertically stacked with substantial x overlap, or
    2. horizontally adjacent with substantial y overlap.

    All gap limits are relative to box dimensions so that
    this works across different page resolutions.
    """

    box_a = a["coordinate"]
    box_b = b["coordinate"]

    ha = height(box_a)
    hb = height(box_b)

    if ha <= 0 or hb <= 0:
        return False

    max_height = max(
        ha,
        hb,
    )

    # ----------------------------------------
    # Vertical stacking
    # ----------------------------------------

    x_overlap = x_overlap_ratio(
        box_a,
        box_b,
    )

    v_gap = vertical_gap(
        box_a,
        box_b,
    )

    vertical_match = (
        x_overlap
        >= settings["x_overlap"]
        and
        v_gap
        <= (
            settings[
                "vertical_gap_factor"
            ]
            * max_height
        )
    )

    # ----------------------------------------
    # Horizontal adjacency
    # ----------------------------------------

    y_overlap = y_overlap_ratio(
        box_a,
        box_b,
    )

    h_gap = horizontal_gap(
        box_a,
        box_b,
    )

    horizontal_match = (
        y_overlap
        >= settings["y_overlap"]
        and
        h_gap
        <= (
            settings[
                "horizontal_gap_factor"
            ]
            * max_height
        )
    )

    return (
        vertical_match
        or horizontal_match
    )


class UnionFind:
    def __init__(self, n):
        self.parent = list(
            range(n)
        )

        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def merge_component(boxes):
    """
    Construct one enclosing bounding box for a connected
    component.

    Metadata is copied from the highest-confidence member.
    """

    representative = max(
        boxes,
        key=lambda box: float(
            box.get(
                "score",
                0.0,
            )
        ),
    )

    result = copy.deepcopy(
        representative
    )

    coordinates = [
        box["coordinate"]
        for box in boxes
    ]

    merged_coordinate = [
        min(
            box[0]
            for box in coordinates
        ),
        min(
            box[1]
            for box in coordinates
        ),
        max(
            box[2]
            for box in coordinates
        ),
        max(
            box[3]
            for box in coordinates
        ),
    ]

    result["coordinate"] = (
        merged_coordinate
    )

    result["score"] = max(
        float(
            box.get(
                "score",
                0.0,
            )
        )
        for box in boxes
    )

    result[
        "merged_from_count"
    ] = len(boxes)

    result[
        "merged_source_scores"
    ] = [
        float(
            box.get(
                "score",
                0.0,
            )
        )
        for box in boxes
    ]

    return result


def merge_label_boxes(
    boxes,
    settings,
):
    if len(boxes) <= 1:
        return [
            copy.deepcopy(box)
            for box in boxes
        ]

    uf = UnionFind(
        len(boxes)
    )

    for i in range(len(boxes)):
        for j in range(
            i + 1,
            len(boxes),
        ):
            if should_merge(
                boxes[i],
                boxes[j],
                settings,
            ):
                uf.union(
                    i,
                    j,
                )

    components = defaultdict(list)

    for index, box in enumerate(
        boxes
    ):
        root = uf.find(
            index
        )

        components[
            root
        ].append(box)

    merged = []

    for component in (
        components.values()
    ):
        merged.append(
            merge_component(
                component
            )
        )

    return merged


def process_page(
    data,
    settings,
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

    boxes = payload.get(
        "boxes",
        [],
    )

    by_label = defaultdict(list)
    untouched = []

    for box in boxes:
        label = normalize_label(
            box.get(
                "label",
                "",
            )
        )

        coordinate = box.get(
            "coordinate"
        )

        if (
            coordinate is None
            or len(coordinate) != 4
        ):
            untouched.append(
                copy.deepcopy(box)
            )
            continue

        prepared = copy.deepcopy(
            box
        )

        prepared[
            "coordinate"
        ] = [
            float(value)
            for value in coordinate
        ]

        if label in MERGE_CLASSES:
            by_label[
                label
            ].append(
                prepared
            )
        else:
            untouched.append(
                prepared
            )

    merged_boxes = []

    for label in sorted(
        by_label
    ):
        merged_boxes.extend(
            merge_label_boxes(
                by_label[label],
                settings,
            )
        )

    final_boxes = (
        untouched
        + merged_boxes
    )

    final_boxes.sort(
        key=lambda box: (
            float(
                box[
                    "coordinate"
                ][1]
            ),
            float(
                box[
                    "coordinate"
                ][0]
            ),
        )
    )

    payload["boxes"] = (
        final_boxes
    )

    return (
        result,
        len(boxes),
        len(final_boxes),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--strength",
        choices=sorted(
            SETTINGS
        ),
        required=True,
    )

    args = parser.parse_args()

    settings = SETTINGS[
        args.strength
    ]

    raw_out = (
        args.output_dir
        / "raw"
    )

    raw_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        args.pred_dir.glob(
            "page_*.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No page JSON files in "
            f"{args.pred_dir}"
        )

    total_before = 0
    total_after = 0

    print()
    print(
        "=========================================="
    )
    print(
        "HARVEST LAYOUT BOX MERGING"
    )
    print(
        "=========================================="
    )

    print(
        "Strength:",
        args.strength,
    )

    print(
        "Merge classes:",
        ", ".join(
            sorted(
                MERGE_CLASSES
            )
        ),
    )

    print()
    print(
        "Settings:"
    )

    for key, value in (
        settings.items()
    ):
        print(
            f"  {key:25s} "
            f"{value}"
        )

    print()

    for path in files:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        (
            result,
            before,
            after,
        ) = process_page(
            data,
            settings,
        )

        total_before += (
            before
        )

        total_after += (
            after
        )

        output_path = (
            raw_out
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

        print(
            f"{path.stem}: "
            f"{before:3d} -> "
            f"{after:3d} "
            f"({before-after:+3d})"
        )

    print()
    print(
        "=========================================="
    )

    print(
        "Pages:",
        len(files),
    )

    print(
        "Boxes before:",
        total_before,
    )

    print(
        "Boxes after:",
        total_after,
    )

    print(
        "Boxes merged away:",
        total_before
        - total_after,
    )

    print(
        "Reduction:",
        (
            100.0
            * (
                total_before
                - total_after
            )
            / total_before
            if total_before
            else 0.0
        ),
        "%",
    )

    print(
        "Output:",
        raw_out,
    )


if __name__ == "__main__":
    main()
