#!/usr/bin/env python3

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path


METRICS = [
    "area",
    "production",
    "yield",
]


SUPERSCRIPT_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹",
    "0123456789",
)


REGION_ALIASES = {
    "afrique": "AFRICA",
    "africa": "AFRICA",

    "amerique du nord": "NORTH AMERICA",
    "north america": "NORTH AMERICA",

    "amerique du sud": "SOUTH AMERICA",
    "south america": "SOUTH AMERICA",

    "asie": "ASIA",
    "asia": "ASIA",

    "europe": "EUROPE",

    "oceanie": "OCEANIA",
    "oceania": "OCEANIA",
}


WORLD_ALIASES = {
    "monde entier",
    "world",
}


MAIN_PERIOD_RE = re.compile(
    r"^\d{4}(?:-\d{4})?$"
)

NUMBER_RE = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?"
)


def normalize_space(value):
    return " ".join(
        str(value).split()
    )


def normalize_name(value):
    value = normalize_space(
        value
    )

    value = value.translate(
        SUPERSCRIPT_TRANSLATION
    )

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    value = value.casefold()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
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
        bbox = [
            float(value)
            for value in box
        ]

        cx, cy = box_center(
            bbox
        )

        score = (
            float(scores[index])
            if index < len(scores)
            else None
        )

        records.append(
            {
                "index": index,
                "text": normalize_space(
                    text
                ),
                "bbox": bbox,
                "cx": cx,
                "cy": cy,
                "score": score,
            }
        )

    return records


def infer_columns(
    records,
    target_period,
):
    candidates = []

    for record in records:
        if record["cy"] >= 150:
            continue

        text = record[
            "text"
        ].strip()

        if MAIN_PERIOD_RE.fullmatch(
            text
        ):
            candidates.append(
                record
            )

    candidates = sorted(
        candidates,
        key=lambda item:
            item["cx"],
    )

    if len(candidates) != 9:
        print()
        print(
            "Main period-header "
            "candidates:"
        )

        for item in candidates:
            print(
                item["text"],
                f"x={item['cx']:.1f}",
                f"y={item['cy']:.1f}",
            )

        raise RuntimeError(
            "Expected exactly 9 main "
            "period headers, found "
            f"{len(candidates)}."
        )

    columns = []

    for index, record in enumerate(
        candidates
    ):
        metric_index = (
            index // 3
        )

        period_index = (
            index % 3
        )

        columns.append(
            {
                "index": index,
                "metric":
                    METRICS[
                        metric_index
                    ],
                "period_index":
                    period_index,
                "period":
                    record["text"],
                "cx":
                    record["cx"],
                "header_record":
                    record,
            }
        )

    target_columns = {}

    for column in columns:
        if (
            column["period"]
            == target_period
        ):
            target_columns[
                column["metric"]
            ] = column["index"]

    missing_metrics = [
        metric
        for metric in METRICS
        if metric
        not in target_columns
    ]

    if missing_metrics:
        raise RuntimeError(
            "Could not find target "
            f"period {target_period!r} "
            "for metrics: "
            + ", ".join(
                missing_metrics
            )
        )

    centers = [
        column["cx"]
        for column in columns
    ]

    internal_boundaries = [
        (
            centers[index]
            + centers[index + 1]
        ) / 2.0
        for index
        in range(
            len(centers) - 1
        )
    ]

    left_width = (
        centers[1]
        - centers[0]
    )

    right_width = (
        centers[-1]
        - centers[-2]
    )

    numeric_left = (
        centers[0]
        - left_width / 2.0
    )

    numeric_right = (
        centers[-1]
        + right_width / 2.0
    )

    boundaries = [
        numeric_left,
        *internal_boundaries,
        numeric_right,
    ]

    return (
        columns,
        target_columns,
        boundaries,
        numeric_left,
        numeric_right,
    )


def infer_body_start(
    records,
    numeric_left,
    numeric_right,
):
    header_records = [
        record
        for record in records
        if (
            numeric_left
            <= record["cx"]
            <= numeric_right
            and record["cy"] < 160
        )
    ]

    if not header_records:
        return 160.0

    bottom = max(
        record["bbox"][3]
        for record
        in header_records
    )

    return bottom + 12.0


def assign_column(
    x,
    boundaries,
):
    if (
        x < boundaries[0]
        or x > boundaries[-1]
    ):
        return None

    for index in range(
        len(boundaries) - 1
    ):
        if (
            boundaries[index]
            <= x
            < boundaries[index + 1]
        ):
            return index

    if x == boundaries[-1]:
        return (
            len(boundaries)
            - 2
        )

    return None


def cluster_numeric_rows(
    records,
    boundaries,
    body_start,
    row_tolerance,
):
    numeric_tokens = []

    for record in records:
        if record["cy"] <= body_start:
            continue

        column_index = (
            assign_column(
                record["cx"],
                boundaries,
            )
        )

        if column_index is None:
            continue

        token = dict(
            record
        )

        token[
            "column_index"
        ] = column_index

        numeric_tokens.append(
            token
        )

    numeric_tokens.sort(
        key=lambda item:
            item["cy"]
    )

    clusters = []

    for token in numeric_tokens:
        if not clusters:
            clusters.append(
                {
                    "tokens": [
                        token
                    ],
                    "row_y":
                        token["cy"],
                }
            )

            continue

        current = clusters[-1]

        if (
            abs(
                token["cy"]
                - current["row_y"]
            )
            <= row_tolerance
        ):
            current[
                "tokens"
            ].append(
                token
            )

            current[
                "row_y"
            ] = statistics.median(
                item["cy"]
                for item
                in current["tokens"]
            )

        else:
            clusters.append(
                {
                    "tokens": [
                        token
                    ],
                    "row_y":
                        token["cy"],
                }
            )

    return clusters


def build_row_bands(
    clusters,
    body_start,
):
    clusters = sorted(
        clusters,
        key=lambda item:
            item["row_y"],
    )

    bands = []

    for index, cluster in enumerate(
        clusters
    ):
        row_y = cluster[
            "row_y"
        ]

        if index == 0:
            top = body_start
        else:
            top = (
                clusters[
                    index - 1
                ]["row_y"]
                + row_y
            ) / 2.0

        if (
            index
            == len(clusters) - 1
        ):
            bottom = (
                row_y + 30.0
            )
        else:
            bottom = (
                row_y
                + clusters[
                    index + 1
                ]["row_y"]
            ) / 2.0

        bands.append(
            {
                "cluster":
                    cluster,
                "top":
                    top,
                "bottom":
                    bottom,
            }
        )

    return bands


def collect_label_tokens(
    records,
    top,
    bottom,
    *,
    numeric_left,
    numeric_right,
):
    left = []
    right = []

    for record in records:
        if not (
            top
            <= record["cy"]
            < bottom
        ):
            continue

        if (
            record["cx"]
            < numeric_left
        ):
            left.append(
                record
            )

        elif (
            record["cx"]
            > numeric_right
        ):
            right.append(
                record
            )

    def join_tokens(tokens):
        tokens = sorted(
            tokens,
            key=lambda item: (
                item["cy"],
                item["cx"],
            )
        )

        return normalize_space(
            " ".join(
                item["text"]
                for item in tokens
            )
        )

    return (
        join_tokens(left),
        join_tokens(right),
    )


def extract_entity_marker(
    text,
):
    text = normalize_space(
        text
    )

    if not text:
        return "", None

    match = re.search(
        r"\s+([⁰¹²³⁴⁵⁶⁷⁸⁹]+|\d+)$",
        text,
    )

    if not match:
        return text, None

    marker = match.group(
        1
    ).translate(
        SUPERSCRIPT_TRANSLATION
    )

    base = text[
        :match.start()
    ].strip()

    return base, marker


def classify_entity(
    entity_fr,
    entity_en,
):
    fr = normalize_name(
        entity_fr
    )

    en = normalize_name(
        entity_en
    )

    for name in (
        fr,
        en,
    ):
        if name in WORLD_ALIASES:
            return (
                "world_total",
                None,
            )

    for name in (
        fr,
        en,
    ):
        if name in REGION_ALIASES:
            return (
                "region_total",
                REGION_ALIASES[
                    name
                ],
            )

    return (
        "country",
        None,
    )


def normalize_missing_marker(
    text,
):
    stripped = (
        text.replace(
            " ",
            ""
        )
    )

    if (
        "..." in stripped
        or "···" in stripped
        or stripped
        in {
            "..",
            "··",
        }
    ):
        return "..."

    return None


def parse_cell(
    tokens,
):
    if not tokens:
        return {
            "raw": None,
            "value": None,
            "qualifier": None,
            "ocr_confidence": None,
            "source_tokens": [],
        }

    tokens = sorted(
        tokens,
        key=lambda item:
            item["cx"]
    )

    text = normalize_space(
        " ".join(
            item["text"]
            for item
            in tokens
        )
    )

    missing = (
        normalize_missing_marker(
            text
        )
    )

    matches = list(
        NUMBER_RE.finditer(
            text
        )
    )

    scores = [
        item["score"]
        for item in tokens
        if item["score"]
        is not None
    ]

    mean_score = (
        sum(scores)
        / len(scores)
        if scores
        else None
    )

    if not matches:
        qualifier = (
            missing
            if missing
            else (
                text
                if text
                else None
            )
        )

        return {
            "raw": None,
            "value": None,
            "qualifier":
                qualifier,
            "ocr_confidence":
                mean_score,
            "source_tokens":
                tokens,
        }

    match = matches[-1]

    raw_number = match.group(
        0
    )

    cleaned_number = (
        raw_number.replace(
            ",",
            ""
        )
    )

    try:
        if "." in cleaned_number:
            value = float(
                cleaned_number
            )
        else:
            value = int(
                cleaned_number
            )
    except ValueError:
        value = None

    before = text[
        :match.start()
    ].strip()

    after = text[
        match.end():
    ].strip()

    qualifier_parts = [
        part
        for part in (
            before,
            after,
        )
        if part
    ]

    qualifier = normalize_space(
        " ".join(
            qualifier_parts
        )
    )

    if missing:
        qualifier = missing

    if not qualifier:
        qualifier = None

    return {
        "raw":
            raw_number,
        "value":
            value,
        "qualifier":
            qualifier,
        "ocr_confidence":
            mean_score,
        "source_tokens":
            tokens,
    }


def build_rows(
    records,
    row_bands,
    target_columns,
    numeric_left,
    numeric_right,
):
    output = []

    current_region = None

    for band in row_bands:
        cluster = band[
            "cluster"
        ]

        entity_fr, entity_en = (
            collect_label_tokens(
                records,
                band["top"],
                band["bottom"],
                numeric_left=
                    numeric_left,
                numeric_right=
                    numeric_right,
            )
        )

        if (
            not entity_fr
            and not entity_en
        ):
            continue

        (
            entity_fr,
            marker_fr,
        ) = extract_entity_marker(
            entity_fr
        )

        (
            entity_en,
            marker_en,
        ) = extract_entity_marker(
            entity_en
        )

        entity_qualifier = (
            marker_fr
            or marker_en
        )

        (
            row_type,
            canonical_region,
        ) = classify_entity(
            entity_fr,
            entity_en,
        )

        if row_type == "region_total":
            current_region = (
                canonical_region
            )

            parent_region = None

        elif row_type == "world_total":
            parent_region = None

        else:
            parent_region = (
                current_region
            )

        row = {
            "row_y":
                cluster[
                    "row_y"
                ],

            "row_type":
                row_type,

            "country_fr_raw":
                entity_fr
                or None,

            "country_en_raw":
                entity_en
                or None,

            "parent_region":
                parent_region,

            "entity_qualifier":
                entity_qualifier,
        }

        for metric in METRICS:
            column_index = (
                target_columns[
                    metric
                ]
            )

            cell_tokens = [
                token
                for token
                in cluster["tokens"]
                if token[
                    "column_index"
                ] == column_index
            ]

            parsed = parse_cell(
                cell_tokens
            )

            prefix = (
                f"{metric}_1925"
            )

            row[
                f"{prefix}_raw"
            ] = parsed[
                "raw"
            ]

            row[
                f"{prefix}_value"
            ] = parsed[
                "value"
            ]

            row[
                f"{prefix}_qualifier"
            ] = parsed[
                "qualifier"
            ]

            row[
                f"{prefix}_ocr_confidence"
            ] = parsed[
                "ocr_confidence"
            ]

            row[
                f"{prefix}_source_tokens"
            ] = parsed[
                "source_tokens"
            ]

        output.append(
            row
        )

    return output


def write_csv(
    path,
    rows,
):
    fields = [
        "row_type",
        "country_fr_raw",
        "country_en_raw",
        "parent_region",
        "entity_qualifier",
        "area_1925_raw",
        "production_1925_raw",
        "yield_1925_raw",
        "area_1925_qualifier",
        "production_1925_qualifier",
        "yield_1925_qualifier",
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
                        row.get(
                            field
                        )
                    for field in fields
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse ISY historical "
            "statistical tables from "
            "PaddleOCR geometry."
        )
    )

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
        "--target-period",
        default="1925",
    )

    parser.add_argument(
        "--row-tolerance",
        type=float,
        default=6.0,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_ocr(
        args.input_json
    )

    (
        columns,
        target_columns,
        boundaries,
        numeric_left,
        numeric_right,
    ) = infer_columns(
        records,
        args.target_period,
    )

    body_start = infer_body_start(
        records,
        numeric_left,
        numeric_right,
    )

    clusters = cluster_numeric_rows(
        records,
        boundaries,
        body_start,
        args.row_tolerance,
    )

    bands = build_row_bands(
        clusters,
        body_start,
    )

    rows = build_rows(
        records,
        bands,
        target_columns,
        numeric_left,
        numeric_right,
    )

    print()
    print(
        "=============================================="
    )
    print(
        "ISY TABLE PARSER"
    )
    print(
        "=============================================="
    )
    print()

    print(
        "Target period:",
        args.target_period,
    )

    print(
        "Numeric zone:",
        f"{numeric_left:.1f}",
        "→",
        f"{numeric_right:.1f}",
    )

    print(
        "Body starts at y:",
        f"{body_start:.1f}",
    )

    print()
    print(
        "=== INFERRED COLUMNS ==="
    )

    for column in columns:
        marker = (
            " <-- TARGET"
            if (
                column["index"]
                in target_columns.values()
            )
            else ""
        )

        print(
            f"{column['index']:2d}  "
            f"{column['metric']:11s} "
            f"{column['period']:10s} "
            f"x={column['cx']:7.1f}"
            f"{marker}"
        )

    print()
    print(
        "Numeric row clusters:",
        len(clusters),
    )

    print(
        "Logical rows retained:",
        len(rows),
    )

    print()
    print(
        f"{'Type':13s} "
        f"{'French':28s} "
        f"{'English':26s} "
        f"{'Area':>9s} "
        f"{'Production':>12s} "
        f"{'Yield':>8s}"
    )

    print("-" * 105)

    for row in rows:
        print(
            f"{row['row_type'][:13]:13s} "
            f"{str(row['country_fr_raw'] or '')[:28]:28s} "
            f"{str(row['country_en_raw'] or '')[:26]:26s} "
            f"{str(row['area_1925_raw'] or ''):>9s} "
            f"{str(row['production_1925_raw'] or ''):>12s} "
            f"{str(row['yield_1925_raw'] or ''):>8s}"
        )

    csv_path = (
        args.output_dir
        / "isy_target_period.csv"
    )

    json_path = (
        args.output_dir
        / "isy_target_period.json"
    )

    write_csv(
        csv_path,
        rows,
    )

    json_path.write_text(
        json.dumps(
            {
                "source_json":
                    str(
                        args.input_json
                    ),

                "target_period":
                    args.target_period,

                "numeric_zone": {
                    "left":
                        numeric_left,
                    "right":
                        numeric_right,
                },

                "columns":
                    columns,

                "target_columns":
                    target_columns,

                "rows":
                    rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "CSV:",
        csv_path,
    )

    print(
        "JSON:",
        json_path,
    )


if __name__ == "__main__":
    main()
