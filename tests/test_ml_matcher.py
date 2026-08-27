"""
test_ml_matcher.py
==================
Unit tests for ML-assisted candidate pair scoring & Reconciliation engine.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import joblib
import pandas as pd
import numpy as np

from src.ml_matcher import (
    extract_pair_features,
    MLReconciliationMatcher,
    build_candidate_pairs_dataset,
    evaluate_thresholds,
    FEATURE_NAMES,
    ModelArtifactError,
)
from src.matcher import SourceMatch, ReconciliationMatcher


class TestMLMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).parent.parent / "data"

    def test_feature_extraction(self):
        anchor = {
            "amount": 1000.0,
            "date": pd.Timestamp("2026-01-15"),
            "_cp_norm": "razorpay software pvt ltd",
            "_ref_norm": "PAY-1001",
            "tax_line": "GST-18",
            "description": "Payment for software license",
        }
        cand = {
            "amount": 1000.0,
            "date": pd.Timestamp("2026-01-16"),
            "_cp_norm": "razorpay software",
            "_ref_norm": "PAY-1001",
            "tax_line": "GST-18",
            "description": "Payment for software",
        }

        feats = extract_pair_features(anchor, cand, candidate_count=2)
        self.assertEqual(feats["amt_diff_abs"], 0.0)
        self.assertEqual(feats["date_diff_days"], 1.0)
        self.assertEqual(feats["exact_ref_agree"], 1.0)
        self.assertGreaterEqual(feats["cp_sim"], 0.8)
        self.assertEqual(feats["candidate_count"], 2.0)

    def test_no_ground_truth_leakage_in_inference(self):
        """Verify that MLReconciliationMatcher does NOT access ground_truth.csv during inference."""
        matcher = MLReconciliationMatcher(data_dir=self.data_dir, ml_threshold=0.90)
        matcher.load_sources()
        matcher.load_model_artifact()

        original_read_csv = pd.read_csv
        def source_only_read_csv(path, *args, **kwargs):
            if "ground_truth.csv" in str(path):
                raise AssertionError("runtime inference attempted to read ground_truth.csv")
            return original_read_csv(path, *args, **kwargs)

        with patch("pandas.read_csv", side_effect=source_only_read_csv):
            res = matcher.reconcile()

        # Verify ground truth file was not accessed by matcher methods
        self.assertIsNotNone(res)
        self.assertEqual(res.total_processed, 99)

    def test_missing_corrupt_and_incompatible_artifacts_fail_explicitly(self):
        matcher = MLReconciliationMatcher(data_dir=self.data_dir)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.joblib"
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"feature_names": FEATURE_NAMES}), encoding="utf-8")
            with self.assertRaises(ModelArtifactError):
                matcher.load_model_artifact(missing, metadata)

            corrupt = root / "corrupt.joblib"
            corrupt.write_text("not a joblib artifact", encoding="utf-8")
            with self.assertRaises(ModelArtifactError):
                matcher.load_model_artifact(corrupt, metadata)

            incompatible = root / "incompatible.joblib"
            joblib.dump({"not": "a pipeline"}, incompatible)
            with self.assertRaises(ModelArtifactError):
                matcher.load_model_artifact(incompatible, metadata)

    def test_reconcile_without_artifact_falls_back_without_training(self):
        matcher = MLReconciliationMatcher(data_dir=self.data_dir)
        matcher.load_sources()
        with patch.object(matcher, "train_model", side_effect=AssertionError("runtime training attempted")):
            result = matcher.reconcile()
        self.assertEqual(result.total_processed, 99)

    def test_ml_ambiguity_safety(self):
        """Verify close ML probabilities trigger ML_AMBIGUOUS -> EXCEPTION/NEEDS_REVIEW."""
        matcher = MLReconciliationMatcher(data_dir=self.data_dir, ml_threshold=0.80)
        matcher.load_sources()
        matcher.train_model()

        anchor = {
            "amount": 500.0,
            "date": pd.Timestamp("2026-01-10"),
            "_cp_norm": "acme Corp",
            "_ref_norm": "INV-9999",
            "tax_line": "",
            "description": "Services",
        }
        cand1 = {
            "record_id": "BNK-001",
            "amount": 500.0,
            "date": pd.Timestamp("2026-01-11"),
            "_cp_norm": "acme Corp",
            "_ref_norm": "INV-9999",
            "tax_line": "",
            "description": "Services",
        }
        cand2 = {
            "record_id": "BNK-002",
            "amount": 500.0,
            "date": pd.Timestamp("2026-01-11"),
            "_cp_norm": "acme Corp",
            "_ref_norm": "INV-9999",
            "tax_line": "",
            "description": "Services",
        }

        # Mock weight_and_score to return close probabilities (0.92 vs 0.90)
        matcher.weight_and_score = lambda l, c, cnt: (0.92 if c["record_id"] == "BNK-001" else 0.90, {})

        match = matcher._match_source_ml(
            ledger_rec=anchor,
            candidates=[cand1, cand2],
            claimed=set(),
            source_name="bank",
            poisoned_refs=set(),
        )

        self.assertTrue(match.is_ambiguous)
        self.assertIn("ML_AMBIGUOUS", match.reason)
        self.assertIsNone(match.record_id)

    def test_threshold_evaluation_runs(self):
        results = evaluate_thresholds(thresholds=[0.70, 0.90], data_dir=self.data_dir)
        self.assertEqual(len(results), 2)
        self.assertIn("precision", results[0])
        self.assertIn("recall", results[0])


if __name__ == "__main__":
    unittest.main()
