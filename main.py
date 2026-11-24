from pathlib import Path
from src.ir_system.indexing.document_loader import process_dataset

records = process_dataset(Path("dataset"))

print(records[:10])
print(f"Total tokens: {len(records)}")