"""
classical_ml.py

"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .preprocessing import load_and_clean_tickets, split_dataset
except ImportError:  # pragma: no cover - supports script execution
    from preprocessing import load_and_clean_tickets, split_dataset


def build_pipeline() -> Pipeline:
    """
    A scikit-learn Pipeline chains preprocessing + model into one object,
    so at inference time you call pipeline.predict(raw_text) directly -
    no risk of forgetting a preprocessing step in production.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),   # unigrams + bigrams capture phrases like "log in"
            min_df=1,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # compensates for our imbalanced classes
            random_state=42,
        )),
    ])


def train_and_evaluate(csv_path: str, model_out_path: str):
    df = load_and_clean_tickets(csv_path)
    train, val, test = split_dataset(df)

    pipeline = build_pipeline()
    pipeline.fit(train["text_clean"], train["category"])

    print("=== Validation performance ===")
    val_preds = pipeline.predict(val["text_clean"])
    print(classification_report(val["category"], val_preds, zero_division=0))

    print("=== Test performance ===")
    test_preds = pipeline.predict(test["text_clean"])
    print(classification_report(test["category"], test_preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(test["category"], test_preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    Path(model_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out_path)
    print(f"\nSaved trained pipeline to {model_out_path}")
    return pipeline


def load_classifier(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def classify_ticket(pipeline: Pipeline, text: str) -> dict:
    """Used by the agent and the FastAPI endpoint at inference time."""
    from preprocessing import clean_text
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    confidence = float(max(proba))
    return {
        "category": pred,
        "confidence": round(confidence, 3),
        "all_scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    pipeline = train_and_evaluate(
        csv_path="data/support_tickets.csv",
        model_out_path="models/ticket_classifier.joblib",
    )
    sample = "I can't log in, the app just shows a server error"
    print("\nSample prediction:", classify_ticket(pipeline, sample))
