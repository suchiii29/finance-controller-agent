"""
evaluate.py
===========
Evaluation of ReconciliationMatcher output against ground_truth.csv.

THIS IS THE ONLY MODULE ALLOWED TO LOAD ground_truth.csv.

Metrics produced
----------------
- Total ledger records evaluated
- Correctly matched / incorrectly matched
- True positives / false positives / false negatives / true negatives
- Match precision, recall, F1
- Operational match coverage
- Exception identification accuracy
- Per-exception-type breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.matcher import ReconciliationResult, ReconciliationDecision

GROUND_TRUTH_PATH = Path(__file__).parent.parent / "data" / "ground_truth.csv"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExceptionEntry:
    ledger_id:        str
    matcher_status:   str
    gt_status:        str
    exception_type:   str
    exception_source: str
    bank_id_match:    bool
    inv_id_match:     bool
    stl_id_match:     bool
    reason:           str


@dataclass
class EvaluationResult:
    total_evaluated:        int
    ledger_missing_in_gt:   int          # canonical txns where ledger was the missing source

    # Decision-level counts (based on ledger-record match status)
    true_positives:         int          # GT matched → matcher matched (correct IDs)
    false_positives:        int          # GT unresolved/wrong → matcher matched
    false_negatives:        int          # GT matched → matcher unresolved/exception
    true_negatives:         int          # GT unresolved → matcher unresolved/exception

    # Sub-counts
    correctly_matched:      int          # TP with all present IDs correct
    incorrectly_matched:    int          # FP
    correctly_unresolved:   int          # TN

    # Classic metrics
    precision:              float
    recall:                 float
    f1:                     float
    operational_coverage:   float        # (MATCHED + PARTIAL) / total

    # Exception handling
    exception_id_accuracy:  float        # fraction of GT-unresolved correctly flagged
    exception_breakdown:    dict[str, dict[str, int]]  # type → {correct, missed, over_flagged}

    # Detail lists
    exception_entries:      list[ExceptionEntry]
    status_counts:          dict[str, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gt() -> pd.DataFrame:
    df = pd.read_csv(GROUND_TRUTH_PATH).fillna("")
    return df


def _ids_correct(dec: ReconciliationDecision, gt_row: pd.Series) -> tuple[bool, bool, bool]:
    """
    For each of the three target sources, check whether the matcher's assigned ID
    matches what the ground truth says the correct ID is.

    Returns (bank_ok, inv_ok, stl_ok).
    A source is 'ok' if both GT and matcher agree (both empty OR same ID).
    """
    def check(matcher_id: Optional[str], gt_id: str) -> bool:
        m = matcher_id or ""
        g = str(gt_id).strip()
        return m == g

    bank_ok = check(dec.bank_match.record_id,       gt_row["bank_record_id"])
    inv_ok  = check(dec.invoice_match.record_id,    gt_row["invoice_record_id"])
    stl_ok  = check(dec.settlement_match.record_id, gt_row["settlement_record_id"])
    return bank_ok, inv_ok, stl_ok


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(result: ReconciliationResult) -> EvaluationResult:
    """
    Compare ReconciliationMatcher output to ground_truth.csv.

    Parameters
    ----------
    result : ReconciliationResult
        Output from ReconciliationMatcher.reconcile().

    Returns
    -------
    EvaluationResult
    """
    gt = _load_gt()

    # Index GT by ledger_record_id for O(1) lookup
    gt_by_ledger: dict[str, pd.Series] = {}
    for _, row in gt.iterrows():
        lid = str(row["ledger_record_id"]).strip()
        if lid:
            gt_by_ledger[lid] = row

    # Canonical txns where the ledger record itself is absent
    ledger_missing_in_gt = (gt["ledger_record_id"] == "").sum()

    # ── Per-decision evaluation ──────────────────────────────────────────────
    tp = fp = fn = tn = 0
    correctly_matched = incorrectly_matched = correctly_unresolved = 0

    exception_entries: list[ExceptionEntry] = []

    # exception_type → {correct, missed, over_flagged}
    exc_breakdown: dict[str, dict[str, int]] = {}

    for dec in result.decisions:
        gt_row = gt_by_ledger.get(dec.ledger_id)
        if gt_row is None:
            # No GT row for this ledger record — skip (shouldn't happen)
            continue

        gt_status   = str(gt_row["match_status"]).strip()   # "matched" / "unresolved"
        exc_type    = str(gt_row["exception_type"]).strip()  # "none" / specific type
        exc_source  = str(gt_row["exception_source"]).strip()

        gt_is_match = (gt_status == "matched")
        mat_is_match = dec.status in ("MATCHED", "PARTIAL")

        # ── TP / FP / FN / TN ─────────────────────────────────────────────
        if gt_is_match and mat_is_match:
            bank_ok, inv_ok, stl_ok = _ids_correct(dec, gt_row)
            # TP requires at least the IDs the matcher claims to be correct
            # We do NOT penalise for sources absent in both GT and matcher
            id_ok = bank_ok and inv_ok and stl_ok
            if id_ok:
                tp += 1
                correctly_matched += 1
            else:
                fp += 1
                incorrectly_matched += 1

        elif gt_is_match and not mat_is_match:
            fn += 1
            bank_ok = inv_ok = stl_ok = False

        elif not gt_is_match and mat_is_match:
            fp += 1
            incorrectly_matched += 1
            bank_ok, inv_ok, stl_ok = _ids_correct(dec, gt_row)

        else:   # GT unresolved, matcher unresolved/exception
            tn += 1
            correctly_unresolved += 1
            bank_ok = inv_ok = stl_ok = True  # both agree: nothing to match

        # ── Exception breakdown ────────────────────────────────────────────
        if exc_type != "none":
            if exc_type not in exc_breakdown:
                exc_breakdown[exc_type] = {"correct": 0, "missed": 0, "over_flagged": 0}
            # "correct" here: GT expects unresolved/exception, matcher also flagged it
            if not gt_is_match and not mat_is_match:
                exc_breakdown[exc_type]["correct"] += 1
            elif not gt_is_match and mat_is_match:
                exc_breakdown[exc_type]["over_flagged"] += 1  # should have flagged
            elif gt_is_match and not mat_is_match:
                exc_breakdown[exc_type]["missed"] += 1        # false-flagged as exception

        # ── Exception entries list (for all non-clean cases) ──────────────
        needs_entry = (
            dec.status in ("EXCEPTION", "UNRESOLVED")
            or not gt_is_match
            or (mat_is_match and not (bank_ok and inv_ok and stl_ok))
        )
        if needs_entry:
            exception_entries.append(ExceptionEntry(
                ledger_id=dec.ledger_id,
                matcher_status=dec.status,
                gt_status=gt_status,
                exception_type=exc_type,
                exception_source=exc_source,
                bank_id_match=bank_ok,
                inv_id_match=inv_ok,
                stl_id_match=stl_ok,
                reason=dec.reason,
            ))

    # ── Aggregate metrics ────────────────────────────────────────────────────
    total = len(result.decisions)
    matched_count = result.status_counts.get("MATCHED", 0) + result.status_counts.get("PARTIAL", 0)
    operational_coverage = matched_count / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    gt_unresolved_count = sum(1 for r in gt_by_ledger.values() if r["match_status"] == "unresolved")
    exception_id_accuracy = tn / gt_unresolved_count if gt_unresolved_count else 0.0

    return EvaluationResult(
        total_evaluated=total,
        ledger_missing_in_gt=ledger_missing_in_gt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        correctly_matched=correctly_matched,
        incorrectly_matched=incorrectly_matched,
        correctly_unresolved=correctly_unresolved,
        precision=precision,
        recall=recall,
        f1=f1,
        operational_coverage=operational_coverage,
        exception_id_accuracy=exception_id_accuracy,
        exception_breakdown=exc_breakdown,
        exception_entries=exception_entries,
        status_counts=result.status_counts,
    )
