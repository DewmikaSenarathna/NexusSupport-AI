"""
preprocessing.py

"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Basic text normalization: lowercase, collapse whitespace, strip odd chars."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces/newlines
    text = re.sub(r"[^a-z0-9\s.,!?'-]", "", text)  # drop stray symbols/emoji
    return text


def load_and_clean_tickets(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "category"])
    df["text_clean"] = df["text"].apply(clean_text)
    df = df.drop_duplicates(subset="text_clean")
    df = df[df["text_clean"].str.len() > 3].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, label_col: str = "category",
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42):
    """
    Stratified split into train/val/test. Stratification matters here because
    our categories are not perfectly balanced (see generate_sample_data.py) —
    a random split could accidentally starve the test set of a rare class.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[label_col], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Splits a document into overlapping chunks for embedding/retrieval.

    `overlap` prevents an answer from being awkwardly cut in half between
    two chunks - the tail of one chunk repeats at the head of the next,
    so context isn't lost at chunk boundaries.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_knowledge_base(kb_dir: str) -> list[dict]:
    """
    Reads every .txt file in the knowledge base folder and chunks it.
    Returns a list of {"source": filename, "chunk_id": int, "text": str}
    ready to be embedded by the RAG pipeline.
    """
    kb_dir = Path(kb_dir)
    records = []
    for file_path in sorted(kb_dir.glob("*.txt")):
        content = file_path.read_text()
        for i, chunk in enumerate(chunk_text(content)):
            records.append({
                "source": file_path.name,
                "chunk_id": i,
                "text": chunk,
            })
    return records


if __name__ == "__main__":
    df = load_and_clean_tickets("data/support_tickets.csv")
    train, val, test = split_dataset(df)
    print(f"Total clean rows: {len(df)}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Class balance (train):\n{train['category'].value_counts()}")

    kb_records = load_knowledge_base("data/knowledge_base")
    print(f"\nKnowledge base chunks: {len(kb_records)}")
    print(kb_records[0])
