from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .term_frequency import load_positional_index


# def get_document_ids(positional_index: dict) -> list[str]:
#     document_ids = set()
#     for documents in positional_index.values():
#         document_ids.update(documents.keys())
#     return sorted(document_ids)


def count_documents(dataset_path: Path) -> int:
    return sum(1 for file in dataset_path.glob("*.txt") if file.is_file())


def calculate_df(positional_index: dict) -> dict[str, int]:
    df = {}
    for term, documents in positional_index.items():
        df[term] = len(documents)
    return df


def calculate_idf(df: dict[str, int], total_documents: int) -> dict[str, float]:
    idf = {}
    for term, frequency in df.items():
        idf[term] = np.log10(total_documents / frequency)
    return idf


def generate_df_idf_table(df: dict[str, int], idf: dict[str, float]) -> pd.DataFrame:

    data = []

    for term in df:
        data.append({"term": term, "df": df[term], "idf": idf[term]})

    return pd.DataFrame(data, columns=["term", "df", "idf"])


def save_df_idf_table(df_idf_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_idf_table.to_csv(output_dir / "df_idf_table.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Document Frequency (DF) and Inverse Document Frequency (IDF)."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/positional_index/positional_index.json"),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("output/df_idf"))

    parser.add_argument("--dataset", type=Path, default=Path("dataset"))

    args = parser.parse_args()

    # Load positional index
    positional_index = load_positional_index(args.input)

    # Count all documents, including empty documents
    total_documents = count_documents(args.dataset)

    # Calculate DF and IDF
    df = calculate_df(positional_index)
    idf = calculate_idf(df, total_documents)

    # Generate DF-IDF table
    df_idf_table = generate_df_idf_table(df, idf)

    # Save DF-IDF table as CSV file
    save_df_idf_table(df_idf_table, args.output_dir)

    print("DF/IDF table created successfully!")


if __name__ == "__main__":
    main()
