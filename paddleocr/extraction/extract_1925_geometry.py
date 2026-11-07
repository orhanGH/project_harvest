#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


TARGET_NAMES = [
    "area_1925",
    "production_1925",
    "yield_1925",
]


NUMBER_RE = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?"
)


def box_center(box):
    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def normalize_space(value):
    return " ".join(
        str(value).split()
    )


def parse_numeric(raw):
    if raw is None:
        return None, None

    raw = normalize_space(raw)

    match = NUMBER_RE.search(raw)

    if not match:
        return None, raw

    number_text = match.group(0)

    cleaned = number_text.replace(
        ",",
        "",
    )

    try:
        if "." in cleaned:
            value = float(cleaned)
        else:
            value = int(cleaned)
    except ValueError:
        value = None

    qualifier = (
        raw[:match.start()]
        + raw[match.end():]
    ).strip()

    qualifier = normalize_space(
        qualifier
    )

    if not qualifier:
        qualifier = None

    return value, qualifier


def load_ocr(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload = data.get(
        "res",
        data,
    )

    ocr = payload.get(
        "overall_ocr_res",
        {},
    )

    texts = ocr.get(
        "rec_texts",
        [],
    )

    boxes = ocr.get(
        "rec_boxes",
        [],
    )

    scores = ocr.get(
        "rec_scores",
        [],
    )

    if len(texts) != len(boxes):
        raise RuntimeError(
            "rec_texts and rec_boxes "
            "have different lengths."
        )

    records = []

    for index, (
        text,
        box,
    ) in enumerate(
        zip(
            texts,
            boxes,
        )
    ):
        text = normalize_space(
            text
        )

        box = [
            float(value)
            for value in box
        ]

        cx, cy = box_center(
            box
        )

        score = (
            float(scores[index])
            if index < len(scores)
            else None
        )

        records.append(
            {
                "index": index,
                "text": text,
                "bbox": box,
                "cx": cx,
                "cy": cy,
                "score": score,
            }
        )

    return records


def find_exact(
    records,
    target,
):
    target = target.casefold()

    return [
        record
        for record in records
        if record["text"].casefold()
        == target
    ]


def find_header_anchors(
    records,
):
    candidates = [
        record
        for record in find_exact(
            records,
            "1925",
        )
        if record["cy"] < 170
    ]

    if len(candidates) < 3:
        raise RuntimeError(
            "Could not locate the three "
            "1925 column headers."
        )

    candidates = sorted(
        candidates,
        key=lambda item:
            item["cx"],
    )

    if len(candidates) > 3:
        candidates = candidates[:3]

    anchors = {}

    for name, item in zip(
        TARGET_NAMES,
        candidates,
    ):
        anchors[name] = item

    return anchors


def find_country_columns(
    records,
):
    pays = find_exact(
        records,
        "PAYS",
    )

    country = find_exact(
        records,
        "COUNTRY",
    )

    if not pays:
        raise RuntimeError(
            "PAYS header not found."
        )

    if not country:
        raise RuntimeError(
            "COUNTRY header not found."
        )

    pays_header = min(
        pays,
        key=lambda item:
            item["cy"],
    )

    country_header = min(
        country,
        key=lambda item:
            item["cy"],
    )

    left_max = (
        pays_header["cx"]
        + 120.0
    )

    right_min = (
        country_header["cx"]
        - 70.0
    )

    return (
        left_max,
        right_min,
        pays_header,
        country_header,
    )


def assign_target_tokens(
    records,
    anchors,
    x_tolerance,
):
    header_bottom = max(
        item["bbox"][3]
        for item
        in anchors.values()
    )

    body_min_y = (
        header_bottom
        + 35.0
    )

    target_tokens = []

    for record in records:
        if record["cy"] <= body_min_y:
            continue

        nearest_name = None
        nearest_distance = None

        for name, anchor in (
            anchors.items()
        ):
            distance = abs(
                record["cx"]
                - anchor["cx"]
            )

            if (
                nearest_distance is None
                or distance
                < nearest_distance
            ):
                nearest_name = name
                nearest_distance = (
                    distance
                )

        if (
            nearest_distance
            is not None
            and nearest_distance
            <= x_tolerance
        ):
            item = dict(record)

            item[
                "target_column"
            ] = nearest_name

            item[
                "x_distance"
            ] = nearest_distance

            target_tokens.append(
                item
            )

    return target_tokens


def cluster_rows(
    target_tokens,
    row_tolerance,
):
    tokens = sorted(
        target_tokens,
        key=lambda item:
            item["cy"],
    )

    clusters = []

    for token in tokens:
        if not clusters:
            clusters.append(
                {
                    "tokens": [
                        token
                    ],
                    "cy": token["cy"],
                }
            )

            continue

        cluster = clusters[-1]

        if (
            abs(
                token["cy"]
                - cluster["cy"]
            )
            <= row_tolerance
        ):
            cluster[
                "tokens"
            ].append(
                token
            )

            cluster["cy"] = sum(
                item["cy"]
                for item
                in cluster["tokens"]
            ) / len(
                cluster["tokens"]
            )

        else:
            clusters.append(
                {
                    "tokens": [
                        token
                    ],
                    "cy": token["cy"],
                }
            )

    return clusters


def collect_label(
    records,
    row_y,
    *,
    left_max=None,
    right_min=None,
    tolerance=18.0,
):
    selected = []

    for record in records:
        if (
            abs(
                record["cy"]
                - row_y
            )
            > tolerance
        ):
            continue

        x1, _, x2, _ = (
            record["bbox"]
        )

        if (
            left_max is not None
            and x2 <= left_max
        ):
            selected.append(
                record
            )

        elif (
            right_min is not None
            and x1 >= right_min
        ):
            selected.append(
                record
            )

    selected = sorted(
        selected,
        key=lambda item: (
            item["cy"],
            item["cx"],
        ),
    )

    texts = []

    for item in selected:
        text = item["text"]

        if text not in texts:
            texts.append(
                text
            )

    return normalize_space(
        " ".join(texts)
    )


def combine_column_tokens(
    tokens,
    column_name,
):
    items = [
        item
        for item in tokens
        if item[
            "target_column"
        ] == column_name
    ]

    items = sorted(
        items,
        key=lambda item: (
            item["cx"],
            item["cy"],
        ),
    )

    text = normalize_space(
        " ".join(
            item["text"]
            for item in items
        )
    )

    scores = [
        item["score"]
        for item in items
        if item["score"]
        is not None
    ]

    mean_score = (
        sum(scores) / len(scores)
        if scores
        else None
    )

    return (
        text or None,
        mean_score,
        items,
    )


def classify_row(
    country_fr,
    country_en,
):
    values = [
        value
        for value in (
            country_fr,
            country_en,
        )
        if value
    ]

    joined = " ".join(
        values
    ).strip()

    if not joined:
        return "unknown"

    letters = [
        character
        for character in joined
        if character.isalpha()
    ]

    if not letters:
        return "unknown"

    uppercase = sum(
        character.isupper()
        for character in letters
    )

    ratio = (
        uppercase
        / len(letters)
    )

    if ratio >= 0.75:
        return "aggregate"

    return "country"


def build_rows(
    records,
    clusters,
    left_max,
    right_min,
    label_tolerance,
):
    rows = []

    for cluster in clusters:
        tokens = cluster[
            "tokens"
        ]

        target_columns = {
            token[
                "target_column"
            ]
            for token in tokens
        }

        country_fr = collect_label(
            records,
            cluster["cy"],
            left_max=left_max,
            tolerance=label_tolerance,
        )

        country_en = collect_label(
            records,
            cluster["cy"],
            right_min=right_min,
            tolerance=label_tolerance,
        )

        if (
            not country_fr
            and not country_en
        ):
            continue

        if not target_columns:
            continue

        row = {
            "row_y":
                cluster["cy"],

            "row_type":
                classify_row(
                    country_fr,
                    country_en,
                ),

            "country_fr_raw":
                country_fr or None,

            "country_en_raw":
                country_en or None,
        }

        for column_name in (
            TARGET_NAMES
        ):
            (
                raw,
                mean_score,
                source_tokens,
            ) = combine_column_tokens(
                tokens,
                column_name,
            )

            value, qualifier = (
                parse_numeric(
                    raw
                )
            )

            row[
                f"{column_name}_raw"
            ] = raw

            row[
                f"{column_name}_value"
            ] = value

            row[
                f"{column_name}_qualifier"
            ] = qualifier

            row[
                f"{column_name}_ocr_confidence"
            ] = mean_score

            row[
                f"{column_name}_source_tokens"
            ] = source_tokens

        area_value = row[
            "area_1925_value"
        ]

        production_value = row[
            "production_1925_value"
        ]

        yield_value = row[
            "yield_1925_value"
        ]

        row[
            "area_1925_ha"
        ] = (
            area_value * 1000
            if isinstance(
                area_value,
                (int, float),
            )
            else None
        )

        row[
            "production_1925_quintals"
        ] = (
            production_value * 1000
            if isinstance(
                production_value,
                (int, float),
            )
            else None
        )

        row[
            "yield_1925_quintals_per_ha"
        ] = yield_value

        rows.append(
            row
        )

    return rows


def write_csv(
    path,
    rows,
):
    fields = [
        "row_type",
        "country_fr_raw",
        "country_en_raw",
        "area_1925_raw",
        "area_1925_value",
        "area_1925_qualifier",
        "area_1925_ha",
        "production_1925_raw",
        "production_1925_value",
        "production_1925_qualifier",
        "production_1925_quintals",
        "yield_1925_raw",
        "yield_1925_value",
        "yield_1925_qualifier",
        "yield_1925_quintals_per_ha",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field:
                        row.get(field)
                    for field
                    in fields
                }
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--x-tolerance",
        type=float,
        default=65.0,
    )

    parser.add_argument(
        "--row-tolerance",
        type=float,
        default=9.0,
    )

    parser.add_argument(
        "--label-tolerance",
        type=float,
        default=18.0,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_ocr(
        args.input_json
    )

    anchors = (
        find_header_anchors(
            records
        )
    )

    (
        left_max,
        right_min,
        pays_header,
        country_header,
    ) = find_country_columns(
        records
    )

    print()
    print(
        "=== 1925 HEADER ANCHORS ==="
    )

    for name in TARGET_NAMES:
        item = anchors[name]

        print(
            f"{name:20s} "
            f"x={item['cx']:.1f} "
            f"y={item['cy']:.1f} "
            f"text={item['text']}"
        )

    print()
    print(
        "French country column max x:",
        f"{left_max:.1f}",
    )

    print(
        "English country column min x:",
        f"{right_min:.1f}",
    )

    target_tokens = (
        assign_target_tokens(
            records,
            anchors,
            args.x_tolerance,
        )
    )

    clusters = cluster_rows(
        target_tokens,
        args.row_tolerance,
    )

    rows = build_rows(
        records,
        clusters,
        left_max,
        right_min,
        args.label_tolerance,
    )

    print()
    print(
        "Candidate rows:",
        len(rows),
    )

    print()
    print(
        f"{'French':28s} "
        f"{'English':28s} "
        f"{'Area':>10s} "
        f"{'Production':>12s} "
        f"{'Yield':>8s}"
    )

    print("-" * 94)

    for row in rows:
        print(
            f"{(row['country_fr_raw'] or '')[:28]:28s} "
            f"{(row['country_en_raw'] or '')[:28]:28s} "
            f"{str(row['area_1925_raw'] or ''):>10s} "
            f"{str(row['production_1925_raw'] or ''):>12s} "
            f"{str(row['yield_1925_raw'] or ''):>8s}"
        )

    json_path = (
        args.output_dir
        / "target_1925_geometry.json"
    )

    csv_path = (
        args.output_dir
        / "target_1925_geometry.csv"
    )

    json_path.write_text(
        json.dumps(
            {
                "source_json":
                    str(
                        args.input_json
                    ),

                "units": {
                    "area_1925_raw":
                        "thousand hectares",

                    "area_1925_ha":
                        "hectares",

                    "production_1925_raw":
                        "thousand quintals",

                    "production_1925_quintals":
                        "quintals",

                    "yield_1925":
                        "quintals per hectare",
                },

                "anchors": {
                    name: {
                        "bbox":
                            anchors[
                                name
                            ][
                                "bbox"
                            ],

                        "x_center":
                            anchors[
                                name
                            ][
                                "cx"
                            ],
                    }
                    for name
                    in TARGET_NAMES
                },

                "rows":
                    rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        csv_path,
        rows,
    )

    print()
    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()
