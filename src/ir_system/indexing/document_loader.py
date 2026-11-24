from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def natural_sort_key(path: Path) -> list[Any]:
    """Sort d2.txt before d10.txt while remaining filename-independent."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def process_document(file_path: Path) -> list[dict[str, Any]]:
    """Return token records for one document, with positions starting at 1."""
    text = file_path.read_text(encoding="utf-8-sig")
    tokens = text.split()  # whitespace tokenization exactly as required

    return [
        {
            "doc_id": file_path.stem,
            "term": token.lower(),
            "position": position,
        }
        for position, token in enumerate(tokens, start=1)
    ]


def process_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    """Dynamically process all .txt files directly inside dataset_dir."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {dataset_dir}")

    document_files = sorted(dataset_dir.glob("*.txt"), key=natural_sort_key)
    if not document_files:
        raise FileNotFoundError(f"No .txt files found in: {dataset_dir}")

    all_records: list[dict[str, Any]] = []
    for file_path in document_files:
        all_records.extend(process_document(file_path))

    return all_records


# def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
#     """Save records as JSON Lines: one token record per line."""
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with output_path.open("w", encoding="utf-8") as output_file:
#         for record in records:
#             output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Load text documents, lowercase whitespace tokens, and assign positions."
#     )
#     parser.add_argument(
#         "--dataset-dir",
#         type=Path,
#         default=Path("dataset"),
#         help="Folder containing the .txt documents (default: ./dataset)",
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=Path("processed_tokens.jsonl"),
#         help="Output JSONL file (default: ./processed_tokens.jsonl)",
#     )
#     args = parser.parse_args()

#     records = process_dataset(args.dataset_dir)
#     save_jsonl(records, args.output)

#     document_ids = sorted({record["doc_id"] for record in records}, key=lambda x: natural_sort_key(Path(x)))
#     print("Document Processing completed successfully.")
#     print(f"Documents loaded: {len(document_ids)}")
#     print(f"Total tokens: {len(records)}")
#     print(f"Unique terms: {len({record['term'] for record in records})}")
#     print(f"Output file: {args.output.resolve()}")
#     print("Documents:", ", ".join(document_ids))


# if __name__ == "__main__":
#     main()
