
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyspark import SparkContext

from .document_loader import process_dataset


def build_positional_index(
    records: list[dict[str, Any]],
    spark_context: SparkContext | None = None
) -> dict[str, dict[str, list[int]]]:

    if not records:
        return {}

    sc = spark_context or SparkContext.getOrCreate()

    records_rdd = sc.parallelize(records)

    term_doc_positions = records_rdd.map(
        lambda record: (
            (record["term"], record["doc_id"]),
            record["position"]
        )
    )

    grouped_positions = term_doc_positions.groupByKey()

    sorted_positions = grouped_positions.mapValues(
        lambda positions: sorted(positions)
    )

    collected_data = sorted_positions.collect()

    positional_index = {}

    for (term, doc_id), positions in collected_data:
        if term not in positional_index:
            positional_index[term] = {}

        positional_index[term][doc_id] = positions

    positional_index = {
        term: dict(sorted(documents.items()))
        for term, documents in sorted(positional_index.items())
    }

    return positional_index


def save_positional_index(
    positional_index: dict[str, dict[str, list[int]]],
    output_path: Path
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            positional_index,
            output_file,
            indent=4,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build a positional inverted index using Apache Spark."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/positional_index/positional_index.json")
    )

    args = parser.parse_args()

    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("ERROR")

    try:
        records = process_dataset(args.dataset_dir)

        positional_index = build_positional_index(
            records,
            spark_context
        )

        save_positional_index(
            positional_index,
            args.output
        )

        print("Positional Index created successfully!")
        print(f"Documents loaded: {len(set(r['doc_id'] for r in records))}")
        print(f"Total tokens: {len(records)}")
        print(f"Unique terms: {len(positional_index)}")
        print(f"Output file: {args.output.resolve()}")

    finally:
        spark_context.stop()


if __name__ == "__main__":
    main()
