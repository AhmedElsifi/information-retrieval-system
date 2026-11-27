from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Load Weighted TF matrix and IDF table.
def load_inputs(weighted_tf_path: Path, df_idf_path: Path):
    weighted_tf = pd.read_csv(weighted_tf_path, index_col=0)
    df_idf = pd.read_csv(df_idf_path)

    return weighted_tf, df_idf


# Calculate TF-IDF weights.
def calculate_tfidf(
    weighted_tf: pd.DataFrame,
    idf_table: pd.DataFrame
) -> pd.DataFrame:

    idf = idf_table.set_index("term")["idf"]

    tfidf_matrix = weighted_tf.mul(idf, axis=0)

    return tfidf_matrix.fillna(0)


# Save TF-IDF matrix.
def save_tfidf_matrix(
    tfidf_matrix: pd.DataFrame,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tfidf_matrix.to_csv(output_dir / "tfidf_matrix.csv")


# Calculate the Euclidean length of each document vector.
def calculate_vector_lengths(tfidf_matrix: pd.DataFrame) -> pd.Series:
    vector_lengths = np.sqrt((tfidf_matrix ** 2).sum(axis=0))
    return vector_lengths


# Save document vector lengths as a CSV file.
def save_vector_lengths(vector_lengths: pd.Series, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_lengths.to_csv(output_dir / "vector_lengths.csv", header=["vector_length"])



# Main function.
def main():

    parser = argparse.ArgumentParser(
        description="Calculate TF-IDF matrix and document vector lengths."
    )

    parser.add_argument(
        "--weighted-tf",
        type=Path,
        default=Path("output/tf_matrix/weighted_tf.csv")
    )

    parser.add_argument(
        "--idf",
        type=Path,
        default=Path("output/df_idf/df_idf_table.csv")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tfidf_matrix")
    )

    args = parser.parse_args()

    # Load Weighted TF and IDF
    weighted_tf, idf_table = load_inputs(
        args.weighted_tf,
        args.idf
    )

    # Calculate TF-IDF matrix
    tfidf_matrix = calculate_tfidf(weighted_tf, idf_table)

    # Calculate document vector lengths
    vector_lengths = calculate_vector_lengths(tfidf_matrix)

    # Save results
    save_tfidf_matrix(tfidf_matrix, args.output_dir)
    save_vector_lengths(vector_lengths, args.output_dir)

    print("TF-IDF matrix and vector lengths created successfully!")


if __name__ == "__main__":
    main()