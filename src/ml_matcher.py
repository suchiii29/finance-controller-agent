"""
ml_matcher.py
=============
Machine Learning Residual Candidate Matching Engine for Finance Controller.

Architecture:
  SOURCE RECORDS
  → normalization
  → deterministic matching (Tier 1 & Tier 2)
  → residual candidate generation
  → ML match scoring (Tier 3)
  → confidence & ambiguity check
  → safe match OR review OR unresolved

Ground truth is used ONLY for offline model training / validation datasets.
GROUND TRUTH IS NEVER ACCESSED DURING INFERENCE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.matcher import (
    ReconciliationMatcher,
    ReconciliationResult,
    ReconciliationDecision,
    SourceMatch,
    _to_norm_records,
    _resolve_col,
    _load_csv,
    _ID_CANDIDATES,
    _DATE_CANDIDATES,
    _build_poisoned_refs,
    _str_sim,
    _date_diff,
    _norm_name,
    _norm_ref,
    _determine_status,
    _match_source,
    DATA_DIR,
)

# Feature vector column names
FEATURE_NAMES = [
    "amt_diff_abs",
    "amt_diff_rel",
    "date_diff_days",
    "cp_sim",
    "ref_sim",
    "desc_sim",
    "tax_line_match",
    "ref_present",
    "exact_ref_agree",
    "amt_in_tol_t1",
    "amt_in_tol_t2",
    "date_in_tol_t1",
    "date_in_tol_t2",
    "candidate_count",
]


def extract_pair_features(
    anchor: dict,
    candidate: dict,
    candidate_count: int = 1,
) -> dict[str, float]:
    """
    Extract interpretable matching features for an anchor-candidate pair.
    
    NO canonical_id, NO ground-truth IDs, NO labels used.
    """
    l_amt = anchor["amount"]
    c_amt = candidate["amount"]
    amt_diff_abs = abs(l_amt - c_amt)
    amt_diff_rel = amt_diff_abs / max(abs(l_amt), 1.0)
    date_diff_days = float(_date_diff(anchor["date"], candidate["date"]))

    cp_sim = _str_sim(anchor["_cp_norm"], candidate["_cp_norm"])
    ref_sim = _str_sim(anchor["_ref_norm"], candidate["_ref_norm"])
    desc_sim = _str_sim(
        _norm_name(anchor["description"]),
        _norm_name(candidate["description"]),
    )

    # Tax line consistency
    a_tax = anchor.get("tax_line", "").strip()
    c_tax = candidate.get("tax_line", "").strip()
    if a_tax and c_tax:
        tax_line_match = 1.0 if a_tax == c_tax else 0.0
    elif not a_tax and not c_tax:
        tax_line_match = 0.5
    else:
        tax_line_match = 0.25

    ref_present = 1.0 if (anchor["_ref_norm"] and candidate["_ref_norm"]) else 0.0
    exact_ref_agree = (
        1.0
        if (anchor["_ref_norm"] and anchor["_ref_norm"] == candidate["_ref_norm"])
        else 0.0
    )

    amt_in_tol_t1 = 1.0 if amt_diff_abs <= 0.01 else 0.0
    amt_in_tol_t2 = 1.0 if amt_diff_abs <= 1.00 else 0.0
    date_in_tol_t1 = 1.0 if date_diff_days <= 1.0 else 0.0
    date_in_tol_t2 = 1.0 if date_diff_days <= 4.0 else 0.0

    return {
        "amt_diff_abs": amt_diff_abs,
        "amt_diff_rel": amt_diff_rel,
        "date_diff_days": date_diff_days,
        "cp_sim": cp_sim,
        "ref_sim": ref_sim,
        "desc_sim": desc_sim,
        "tax_line_match": tax_line_match,
        "ref_present": ref_present,
        "exact_ref_agree": exact_ref_agree,
        "amt_in_tol_t1": amt_in_tol_t1,
        "amt_in_tol_t2": amt_in_tol_t2,
        "date_in_tol_t1": date_in_tol_t1,
        "date_in_tol_t2": date_in_tol_t2,
        "candidate_count": float(candidate_count),
    }


def build_candidate_pairs_dataset(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a supervised dataset of candidate pairs from source CSVs and ground_truth.csv.
    Used ONLY for offline training & validation of the ML model.
    
    Returns (train_df, test_df) partitioned at the anchor/transaction level.
    """
    matcher = ReconciliationMatcher(data_dir=data_dir)
    matcher.load_sources()

    gt_df = pd.read_csv(data_dir / "ground_truth.csv").fillna("")
    gt_by_ledger: dict[str, dict[str, str]] = {}
    for _, row in gt_df.iterrows():
        lid = str(row["ledger_record_id"]).strip()
        if lid:
            gt_by_ledger[lid] = {
                "bank": str(row["bank_record_id"]).strip(),
                "invoice": str(row["invoice_record_id"]).strip(),
                "settlement": str(row["settlement_record_id"]).strip(),
            }

    # Leakage-safe split on ledger record IDs
    ledger_ids = [r["record_id"] for r in matcher._ledger_records]
    train_lids, test_lids = train_test_split(
        ledger_ids, test_size=0.3, random_state=42
    )
    train_lid_set = set(train_lids)

    rows = []
    target_sources = [
        ("bank", matcher._bank_records),
        ("invoice", matcher._invoice_records),
        ("settlement", matcher._settlement_records),
    ]

    for ledger_rec in matcher._ledger_records:
        lid = ledger_rec["record_id"]
        is_train = lid in train_lid_set
        gt_targets = gt_by_ledger.get(lid, {})

        for source_name, target_recs in target_sources:
            # Generate broad candidate set (e.g. date diff <= 30 days)
            cands = []
            for c in target_recs:
                d_diff = _date_diff(ledger_rec["date"], c["date"])
                a_diff = abs(ledger_rec["amount"] - c["amount"])
                ref_match = (
                    ledger_rec["_ref_norm"]
                    and ledger_rec["_ref_norm"] == c["_ref_norm"]
                )
                if d_diff <= 30 or ref_match or a_diff <= 10.0:
                    cands.append(c)

            cand_cnt = len(cands)
            gt_correct_id = gt_targets.get(source_name, "")

            for c in cands:
                feats = extract_pair_features(ledger_rec, c, candidate_count=cand_cnt)
                label = 1 if (gt_correct_id and c["record_id"] == gt_correct_id) else 0
                feats["label"] = label
                feats["ledger_id"] = lid
                feats["source_name"] = source_name
                feats["candidate_id"] = c["record_id"]
                feats["is_train"] = is_train
                rows.append(feats)

    full_df = pd.DataFrame(rows)
    train_df = full_df[full_df["is_train"]].copy()
    test_df = full_df[~full_df["is_train"]].copy()
    return train_df, test_df


class MLReconciliationMatcher(ReconciliationMatcher):
    """
    ML-assisted reconciliation matcher.
    
    Order of evaluation:
      1. Tier 1: Deterministic Exact / High Confidence match
      2. Tier 2: Deterministic Strong Fuzzy match
      3. Tier 3: ML Scorer for residual / ambiguous candidates
    
    GROUND TRUTH IS NEVER ACCESSED AT INFERENCE TIME.
    """

    def __init__(self, data_dir: Path = DATA_DIR, ml_threshold: float = 0.90):
        super().__init__(data_dir=data_dir)
        self.ml_threshold = ml_threshold
        self.model_pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.ml_stats = {
            "scored_pairs": 0,
            "ml_matches": 0,
            "ml_ambiguous": 0,
            "ml_unresolved": 0,
        }

    def train_model(self, train_df: Optional[pd.DataFrame] = None) -> float:
        """Train lightweight supervised model (LogisticRegression Pipeline)."""
        if train_df is None:
            train_df, _ = build_candidate_pairs_dataset(self.data_dir)

        X_train = train_df[FEATURE_NAMES]
        y_train = train_df["label"]

        self.model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)),
        ])

        self.model_pipeline.fit(X_train, y_train)
        self.is_fitted = True

        acc = self.model_pipeline.score(X_train, y_train)
        return float(acc)

    def _score_candidate_ml(
        self,
        anchor: dict,
        cand: dict,
        cand_count: int,
    ) -> tuple[float, dict[str, float]]:
        """Score candidate pair using ML model. Returns (probability, feature_dict)."""
        feats = extract_pair_features(anchor, cand, candidate_count=cand_count)
        X = pd.DataFrame([feats])[FEATURE_NAMES]
        prob = float(self.model_pipeline.predict_proba(X)[0, 1])
        return prob, feats

    def _match_source_ml(
        self,
        ledger_rec: dict,
        candidates: list[dict],
        claimed: set,
        source_name: str,
        poisoned_refs: set,
        corroborated: bool = False,
    ) -> SourceMatch:
        """
        Match source using Tier 1 & Tier 2 deterministic first, then ML Tier 3 for residual candidates.
        """
        # First attempt deterministic Tier 1 & Tier 2
        det_match = _match_source(
            ledger_rec, candidates, claimed, source_name, poisoned_refs, corroborated
        )

        # If Tier 1 or Tier 2 found a clean match, use it!
        if det_match.tier in (1, 2) and det_match.record_id is not None and not det_match.is_ambiguous:
            return det_match

        # Otherwise, run ML Tier 3 on unclaimed residual candidates
        if not self.is_fitted or self.model_pipeline is None:
            self.train_model()

        unclaimed_cands = [c for c in candidates if c["record_id"] not in claimed]
        if not unclaimed_cands:
            return SourceMatch(
                source=source_name, record_id=None, tier=0,
                confidence=0.0, reason="NO_CANDIDATE_FOUND",
            )

        feat_list = [
            extract_pair_features(ledger_rec, c, candidate_count=len(unclaimed_cands))
            for c in unclaimed_cands
        ]
        self.ml_stats["scored_pairs"] += len(unclaimed_cands)
        X_batch = pd.DataFrame(feat_list)[FEATURE_NAMES]
        probs = self.model_pipeline.predict_proba(X_batch)[:, 1]

        scored_cands = [
            (float(prob), cand, feats)
            for prob, cand, feats in zip(probs, unclaimed_cands, feat_list)
        ]
        scored_cands.sort(key=lambda x: -x[0])
        top_prob, top_cand, top_feats = scored_cands[0]

        # Safety Check 1: Amount mismatch safety
        # In financial ops, an amount discrepancy > ₹1.00 cannot be automatically
        # reconciled without human approval. Flag as low confidence / review.
        if top_feats["amt_diff_abs"] > 1.00:
            self.ml_stats["ml_unresolved"] += 1
            return SourceMatch(
                source=source_name, record_id=None, tier=0,
                confidence=top_prob,
                reason=(
                    f"ML_AMOUNT_MISMATCH (amt_diff=₹{top_feats['amt_diff_abs']:.2f} > 1.00) "
                    f"→ NO_MATCH (requires exception review) top_cand={top_cand['record_id']}"
                ),
            )

        # Safety Check 2: Ambiguity / close probability safety check
        if len(scored_cands) > 1:
            second_prob = scored_cands[1][0]
            if top_prob >= self.ml_threshold and (top_prob - second_prob) < 0.05:
                self.ml_stats["ml_ambiguous"] += 1
                ids = [sc[1]["record_id"] for sc in scored_cands[:2]]
                return SourceMatch(
                    source=source_name, record_id=None, tier=3,
                    confidence=top_prob, is_ambiguous=True,
                    reason=(
                        f"ML_AMBIGUOUS: top prob={top_prob:.3f} vs 2nd prob={second_prob:.3f} "
                        f"(gap < 0.05) → NEEDS_REVIEW candidates={ids}"
                    ),
                )

        if top_prob >= self.ml_threshold:
            self.ml_stats["ml_matches"] += 1
            reason = (
                f"ML_MATCH prob={top_prob:.3f} (thresh={self.ml_threshold}) | "
                f"amt_diff=₹{top_feats['amt_diff_abs']:.2f}, date_diff={int(top_feats['date_diff_days'])}d, "
                f"cp_sim={top_feats['cp_sim']:.2f}, ref_sim={top_feats['ref_sim']:.2f}"
            )
            return SourceMatch(
                source=source_name,
                record_id=top_cand["record_id"],
                tier=3,
                confidence=top_prob,
                reason=reason,
            )
        else:
            self.ml_stats["ml_unresolved"] += 1
            return SourceMatch(
                source=source_name, record_id=None, tier=0,
                confidence=top_prob,
                reason=(
                    f"ML_LOW_CONFIDENCE (prob={top_prob:.3f} < {self.ml_threshold}) "
                    f"→ NO_MATCH top_cand={top_cand['record_id']}"
                ),
            )

    def weight_and_score(
        self, ledger_rec: dict, cand: dict, cand_cnt: int
    ) -> tuple[float, dict[str, float]]:
        return self._score_candidate_ml(ledger_rec, cand, cand_cnt)

def _determine_status_ml(
    bank_m: SourceMatch,
    inv_m: SourceMatch,
    stl_m: SourceMatch,
) -> tuple[str, float, int, str, str]:
    """Determine operational status for ML-assisted matcher decisions."""
    matches = [bank_m, inv_m, stl_m]

    # Any explicit ambiguity → EXCEPTION
    if any(m.is_ambiguous for m in matches):
        ambig = [m for m in matches if m.is_ambiguous]
        reason = "; ".join(m.reason for m in ambig)
        return (
            "EXCEPTION", 0.0, min(m.tier for m in ambig if m.tier > 0) if any(m.tier > 0 for m in ambig) else 3,
            f"Ambiguous ML candidates detected: {reason}",
            "Investigate duplicate references or near-identical candidates.",
        )

    real_matches = [m for m in matches if m.record_id is not None and not m.is_ambiguous]
    n_matched = len(real_matches)

    if n_matched == 0:
        return (
            "UNRESOLVED", 0.0, 0,
            "No corresponding record found in any of the three target sources.",
            "Check for missing source records or data import failures.",
        )

    avg_conf = sum(m.confidence for m in real_matches) / n_matched
    min_tier = min(m.tier for m in real_matches)

    if n_matched == 3:
        reason = " | ".join(
            f"{m.source}={m.record_id}(T{m.tier},c={m.confidence:.2f})"
            for m in matches
        )
        return (
            "MATCHED", avg_conf, min_tier,
            f"All three targets matched (ML-Assisted): {reason}",
            "No action required.",
        )

    missing = [m.source for m in matches if m.record_id is None]
    found = " | ".join(
        f"{m.source}={m.record_id}(T{m.tier},c={m.confidence:.2f})"
        for m in matches if m.record_id
    )
    return (
        "PARTIAL", avg_conf, min_tier,
        f"Partial match — missing: {missing}. Found: {found}",
        f"Investigate missing records from: {missing}",
    )


    def reconcile(self) -> ReconciliationResult:
        if not self.is_fitted:
            self.train_model()

        if not self._ledger_records:
            self.load_sources()

        t0 = time.perf_counter()

        poisoned_bank = _build_poisoned_refs(self._bank_records)
        poisoned_inv = _build_poisoned_refs(self._invoice_records)
        poisoned_stl = _build_poisoned_refs(self._settlement_records)

        claimed_bank = set()
        claimed_inv = set()
        claimed_stl = set()

        decisions: list[ReconciliationDecision] = []
        exceptions: list[ReconciliationDecision] = []
        status_counts: dict[str, int] = {
            "MATCHED": 0, "PARTIAL": 0, "EXCEPTION": 0, "UNRESOLVED": 0,
        }

        for ledger_rec in self._ledger_records:
            bank_m = self._match_source_ml(
                ledger_rec, self._bank_records, claimed_bank, "bank",
                poisoned_bank, corroborated=False,
            )
            inv_m = self._match_source_ml(
                ledger_rec, self._invoice_records, claimed_inv, "invoice",
                poisoned_inv, corroborated=False,
            )

            both_tier1 = (
                bank_m.tier == 1 and bank_m.record_id is not None and
                inv_m.tier == 1 and inv_m.record_id is not None
            )
            stl_m = self._match_source_ml(
                ledger_rec, self._settlement_records, claimed_stl, "settlement",
                poisoned_stl, corroborated=both_tier1,
            )

            status, conf, tier, reason, action = _determine_status_ml(bank_m, inv_m, stl_m)

            if bank_m.record_id and not bank_m.is_ambiguous:
                claimed_bank.add(bank_m.record_id)
            if inv_m.record_id and not inv_m.is_ambiguous:
                claimed_inv.add(inv_m.record_id)
            if stl_m.record_id and not stl_m.is_ambiguous:
                claimed_stl.add(stl_m.record_id)

            dec = ReconciliationDecision(
                ledger_id=ledger_rec["record_id"],
                bank_match=bank_m,
                invoice_match=inv_m,
                settlement_match=stl_m,
                status=status,
                confidence=conf,
                tier=tier,
                reason=reason,
                recommended_action=action,
            )
            decisions.append(dec)
            status_counts[status] += 1
            if status in ("EXCEPTION", "UNRESOLVED"):
                exceptions.append(dec)

        elapsed = time.perf_counter() - t0
        n = len(decisions)

        return ReconciliationResult(
            total_processed=n,
            decisions=decisions,
            exceptions=exceptions,
            elapsed_seconds=elapsed,
            throughput_per_second=n / elapsed if elapsed > 0 else 0.0,
            status_counts=status_counts,
        )


def evaluate_thresholds(
    thresholds: list[float] = [0.70, 0.80, 0.90, 0.95],
    data_dir: Path = DATA_DIR,
) -> list[dict[str, Any]]:
    """Evaluate precision vs recall trade-off across different ML confidence thresholds."""
    from src.evaluate import evaluate

    train_df, _ = build_candidate_pairs_dataset(data_dir)

    results = []
    for th in thresholds:
        matcher = MLReconciliationMatcher(data_dir=data_dir, ml_threshold=th)
        matcher.load_sources()
        matcher.train_model(train_df)
        res = matcher.reconcile()
        ev = evaluate(res)

        results.append({
            "threshold": th,
            "matched": res.status_counts.get("MATCHED", 0),
            "partial": res.status_counts.get("PARTIAL", 0),
            "exception": res.status_counts.get("EXCEPTION", 0),
            "correct_full": ev.correct_full_matches,
            "incorrect_matches": ev.incorrect_full_matches,
            "precision": ev.match_precision,
            "recall": ev.match_recall,
            "coverage": ev.operational_coverage,
        })
    return results
