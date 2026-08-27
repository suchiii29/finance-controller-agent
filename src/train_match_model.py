"""Offline training entry point for the residual reconciliation matcher."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.ml_matcher import (
    FEATURE_NAMES,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    MLReconciliationMatcher,
    build_candidate_pairs_dataset,
)

RANDOM_SEED = 42
MODEL_VERSION = "match-model-v1"
THRESHOLD = 0.90


def train_and_save(
    data_dir: Path = Path(__file__).parent.parent / "data",
    artifact_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict:
    """Train from ground truth offline, validate, and persist model plus metadata."""
    train_df, test_df = build_candidate_pairs_dataset(data_dir)
    matcher = MLReconciliationMatcher(data_dir=data_dir, ml_threshold=THRESHOLD)
    training_accuracy = matcher.train_model(train_df)

    predictions = matcher.model_pipeline.predict(test_df[FEATURE_NAMES])
    validation = {
        "accuracy": float(accuracy_score(test_df["label"], predictions)),
        "precision": float(precision_score(test_df["label"], predictions, zero_division=0)),
        "recall": float(recall_score(test_df["label"], predictions, zero_division=0)),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matcher.model_pipeline, artifact_path)
    metadata = {
        "model_type": "sklearn.pipeline.Pipeline(LogisticRegression)",
        "feature_names": FEATURE_NAMES,
        "random_seed": RANDOM_SEED,
        "threshold": THRESHOLD,
        "validation_metrics": validation,
        "training_accuracy": training_accuracy,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(test_df)),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_and_save()
    print(f"Saved model artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Saved model metadata: {MODEL_METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
