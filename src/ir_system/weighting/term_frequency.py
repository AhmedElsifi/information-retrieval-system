from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


# Load the positional index from a JSON file.
def load_positional_index(input_path: Path) -> dict:
    if not input_path.exists():
        print("Error: Positional index file not found.")
    with open(input_path, "r", encoding="utf-8") as file:
        positional_index = json.load(file)
    return positional_index


# TF = number of occurrences of a term in a document.
def calculate_raw_tf(positional_index: dict, document_ids: list[str]) -> pd.DataFrame:
    tf_data = {}
    
    for term, documents in positional_index.items():
        tf_data[term] = {}
        for doc_id in document_ids:
            positions = documents.get(doc_id, [])
            tf_data[term][doc_id] = len(positions)
            
    tf_matrix = pd.DataFrame.from_dict(tf_data, orient="index")
    return tf_matrix


# TF-weight = 1 + log10(TF), if TF > 0
# TF-weight = 0, otherwise.
def calculate_weighted_tf(tf_matrix: pd.DataFrame) -> pd.DataFrame:
    weighted_tf_matrix = tf_matrix.copy()
    weighted_tf_matrix = weighted_tf_matrix.map(
        lambda tf: 1 + np.log10(tf) if tf > 0 else 0
    )
    return weighted_tf_matrix


# Save TF matrices as CSV files.
def save_tf_matrices(raw_tf_matrix, weighted_tf_matrix, output_dir):
    raw_tf_matrix.to_csv(output_dir / "raw_tf.csv")
    weighted_tf_matrix.to_csv(output_dir / "weighted_tf.csv")
    
    
def main():

    parser = argparse.ArgumentParser(
        description="Calculate Raw TF and Weighted TF."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/positional_index/positional_index.json")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tf_matrix")
    )

    args = parser.parse_args()

    # Load positional index
    positional_index = load_positional_index(args.input)

    # Get all document IDs
    document_ids = sorted({
        doc_id
        for documents in positional_index.values()
        for doc_id in documents
    })

    # Calculate Raw TF
    raw_tf_matrix = calculate_raw_tf(
        positional_index, document_ids
    )

    # Calculate Weighted TF
    weighted_tf_matrix = calculate_weighted_tf(raw_tf_matrix)

    # Create output folder
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save TF matrices
    save_tf_matrices(
        raw_tf_matrix,
        weighted_tf_matrix,
        args.output_dir
    )

    print("TF matrices created successfully!")
    
if __name__ == "__main__":
    main()