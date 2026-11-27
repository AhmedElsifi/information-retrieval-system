from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .tfidf import calculate_vector_lengths


def load_tfidf_matrix(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, index_col=0)


def load_vector_lengths(input_path: Path) -> pd.Series:
    lengths = pd.read_csv(input_path, index_col=0)
    return lengths["vector_length"]


def normalize_matrix(tfidf_matrix: pd.DataFrame, vector_lengths: pd.Series = None) -> pd.DataFrame:
    if vector_lengths is None:
        vector_lengths = calculate_vector_lengths(tfidf_matrix)

    normalized_matrix = tfidf_matrix.copy()

    for doc_id in tfidf_matrix.columns:
        length = vector_lengths[doc_id]
        if length > 0:
            normalized_matrix[doc_id] = tfidf_matrix[doc_id] / length
        else:
            normalized_matrix[doc_id] = 0.0

    return normalized_matrix


def verify_normalized_lengths(normalized_matrix: pd.DataFrame) -> pd.Series:
    lengths = calculate_vector_lengths(normalized_matrix)

    for doc_id, length in lengths.items():
        if not (np.isclose(length, 1.0) or np.isclose(length, 0.0)):
            raise ValueError(f"Document {doc_id} has length {length}, expected 1")

    return lengths


def save_normalized_matrix(normalized_matrix: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_matrix.to_csv(output_dir / "normalized_tfidf_matrix.csv")


def save_normalized_lengths(lengths: pd.Series, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths.to_csv(output_dir / "normalized_vector_lengths.csv", header=["vector_length"])


def main():
    parser = argparse.ArgumentParser(description="Normalize TF-IDF document vectors.")
    parser.add_argument("--input", type=Path, default=Path("output/tfidf_matrix/tfidf_matrix.csv"))
    parser.add_argument("--lengths", type=Path, default=Path("output/tfidf_matrix/vector_lengths.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/normalized_matrix"))
    args = parser.parse_args()

    tfidf_matrix = load_tfidf_matrix(args.input)
    vector_lengths = load_vector_lengths(args.lengths)

    normalized_matrix = normalize_matrix(tfidf_matrix, vector_lengths)
    normalized_lengths = verify_normalized_lengths(normalized_matrix)

    save_normalized_matrix(normalized_matrix, args.output_dir)
    save_normalized_lengths(normalized_lengths, args.output_dir)

    print("Normalized TF-IDF matrix created successfully!")


if __name__ == "__main__":
    main()
