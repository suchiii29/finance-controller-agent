"""
evaluate.py
===========
Evaluation of ReconciliationMatcher output against ground_truth.csv.

THIS IS THE ONLY MODULE ALLOWED TO LOAD ground_truth.csv.

Evaluation philosophy
---------------------
Reconciliation is NOT a standard binary classification problem.
The four matcher statuses are operational workflow outcomes:

  MATCHED    – all sources confidently reconciled
  PARTIAL    – some sources found; one or more absent or unclaimable
  EXCEPTION  – ambiguity or detectable discrepancy
  UNRESOLVED – no reliable correspondence found

These statuses are evaluated against ground-truth expectations:

  correct_full_matches
      Matcher=MATCHED, GT=matched, all claimed IDs correct.
      The gold standard — everything recovered correctly.

  correct_partial_detections
      Matcher=PARTIAL, GT=matched (planted exception on one source),
      AND all claimed source IDs are correct.
      The matcher correctly refused to claim a mismatched/unavailable source;
      the sources it DID claim are the right ones.

  correctly_escalated
      Matcher=PARTIAL/EXCEPTION/UNRESOLVED, GT=unresolved (source genuinely absent).
      The matcher correctly identified the incomplete state and did not
      over-claim a non-existent source.

  incorrect_full_matches   (genuine false positive)
      Matcher=MATCHED or PARTIAL, but at least one claimed source ID
      does NOT match the ground-truth ID.
      This is the only real error category.

  missed_resolvable   (false negative)
      Matcher=EXCEPTION or UNRESOLVED, but GT=matched.
      A recoverable transaction was not found.

  incorrectly_auto_resolved
      Matcher=MATCHED, GT=unresolved.
      The matcher claimed a full match when the GT says it should be partial.

Precision:  correct_full_matches / (correct_full_matches + incorrect_full_matches)
Recall:     (correct_full_matches + correct_partial_detections) / gt_matched_evaluable

Missing ledger anchor (CAN-0090):
  One canonical transaction has no ledger record and is therefore
  ineligible to enter the ledger-anchored matching workflow.
  It is counted separately and NOT included in precision/recall denominators.
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
    bank_id_match:    bool   # claimed bank ID == GT bank ID (or both absent)
    inv_id_match:     bool
    stl_id_match:     bool
    classification:   str    # 'correct_full_match' | 'correct_partial' | 'correctly_escalated' | 'incorrect_match' | 'missed'
    reason:           str


@dataclass
class EvaluationResult:
    # ── Canonical transaction universe ──────────────────────────────────────
    total_canonical:           int   # 100 (all GT rows)
    gt_matched_count:          int   # GT rows with match_status="matched"
    gt_unresolved_total:       int   # GT rows with match_status="unresolved"
    ledger_missing_in_gt:      int   # Canonical txns with no ledger record (excluded from eval)
    gt_unresolved_evaluable:   int   # GT-unresolved with a ledger anchor (can be evaluated)

    # ── Operational status counts (matcher output) ──────────────────────────
    total_evaluated:           int   # Ledger anchors evaluated (= 99)
    status_counts:             dict[str, int]

    # ── Correctness classification ───────────────────────────────────────────
    correct_full_matches:      int   # GT=matched, MATCHED, all IDs correct
    correct_partial_detections: int  # GT=matched, PARTIAL, claimed IDs correct
    correctly_escalated:       int   # GT=unresolved, PARTIAL/EXCEPTION/UNRESOLVED, no wrong claims
    incorrect_full_matches:    int   # Any claimed ID does NOT match GT  (genuine FP)
    missed_resolvable:         int   # GT=matched, but EXCEPTION/UNRESOLVED           (genuine FN)
    incorrectly_auto_resolved: int   # GT=unresolved, but MATCHED with wrong claim

    # ── Precision / Recall / F1 (reconciliation-specific definitions) ────────
    match_precision:   float   # correct_full_matches / (correct + incorrect)
    match_recall:      float   # (correct_full + correct_partial) / gt_matched_evaluable
    match_f1:          float

    # ── Coverage ────────────────────────────────────────────────────────────
    operational_coverage: float  # (MATCHED + PARTIAL) / total_evaluated

    # ── Exception detection ──────────────────────────────────────────────────
    exception_detection_rate: float  # correctly_escalated / gt_unresolved_evaluable

    # ── Per-type breakdown ───────────────────────────────────────────────────
    exception_breakdown: dict[str, dict[str, int]]
    #  type → {gt_count, correctly_handled, correctly_escalated, missed, incorrectly_resolved}

    # ── Detail list ─────────────────────────────────────────────────────────
    exception_entries: list[ExceptionEntry]

    # ── Backward compat aliases (deprecated — use new names) ─────────────────
    @property
    def true_positives(self) -> int:
        return self.correct_full_matches

    @property
    def false_positives(self) -> int:
        return self.incorrect_full_matches

    @property
    def false_negatives(self) -> int:
        return self.missed_resolvable

    @property
    def true_negatives(self) -> int:
        return self.correctly_escalated

    @property
    def precision(self) -> float:
        return self.match_precision

    @property
    def recall(self) -> float:
        return self.match_recall

    @property
    def f1(self) -> float:
        return self.match_f1

    @property
    def exception_id_accuracy(self) -> float:
        return self.exception_detection_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gt() -> pd.DataFrame:
    df = pd.read_csv(GROUND_TRUTH_PATH).fillna("")
    return df


def _check_claimed_ids(
    dec: ReconciliationDecision,
    gt_row: pd.Series,
) -> tuple[bool, bool, bool, bool]:
    """
    For each of the three target sources, check whether the matcher's claimed ID
    is consistent with ground truth.

    Returns (bank_ok, inv_ok, stl_ok, any_incorrect_claim).

    A source is "ok" when:
      - Matcher claimed an ID AND it matches the GT ID  → ok
      - Matcher claimed None (did not claim) → ok (absence of claim is never wrong)
      - Both matcher and GT have the same empty value   → ok

    A source is NOT ok only when:
      - Matcher claimed a specific ID AND GT says a DIFFERENT specific ID
      - Matcher claimed a specific ID AND GT says no record exists for that source

    We never penalise the matcher for NOT claiming a source it could not resolve.
    """
    def check(matcher_id: Optional[str], gt_id: str) -> bool:
        m = (matcher_id or "").strip()
        g = str(gt_id).strip()
        if not m:
            # Matcher did not claim this source → not an incorrect claim
            return True
        # Matcher DID claim something → verify it matches GT
        return m == g

    bank_ok = check(dec.bank_match.record_id,       gt_row["bank_record_id"])
    inv_ok  = check(dec.invoice_match.record_id,    gt_row["invoice_record_id"])
    stl_ok  = check(dec.settlement_match.record_id, gt_row["settlement_record_id"])
    any_incorrect = not (bank_ok and inv_ok and stl_ok)
    return bank_ok, inv_ok, stl_ok, any_incorrect


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(result: ReconciliationResult) -> EvaluationResult:
    """
    Compare ReconciliationMatcher output to ground_truth.csv.

    Uses reconciliation-specific metric definitions — see module docstring.
    """
    gt = _load_gt()

    # ── Index GT by ledger_record_id ─────────────────────────────────────────
    gt_by_ledger: dict[str, pd.Series] = {}
    for _, row in gt.iterrows():
        lid = str(row["ledger_record_id"]).strip()
        if lid:
            gt_by_ledger[lid] = row

    # ── Canonical universe counts ────────────────────────────────────────────
    total_canonical   = len(gt)
    gt_matched_count  = int((gt["match_status"] == "matched").sum())
    gt_unresolved_all = int((gt["match_status"] == "unresolved").sum())

    # Canonical txns where the ledger anchor itself is absent (CAN-0090)
    ledger_missing_in_gt = int(
        ((gt["match_status"] == "unresolved") & (gt["ledger_record_id"] == "")).sum()
    )
    gt_unresolved_evaluable = gt_unresolved_all - ledger_missing_in_gt

    # ── Exception type → initialise breakdown ────────────────────────────────
    exc_breakdown: dict[str, dict[str, int]] = {}

    def _init_exc(etype: str) -> None:
        if etype not in exc_breakdown:
            exc_breakdown[etype] = {
                "gt_count": 0,
                "correctly_handled": 0,  # GT=matched, matcher=MATCHED or correct PARTIAL
                "correctly_escalated": 0, # GT=unresolved, matcher=PARTIAL/EXCEPTION
                "missed": 0,              # GT=matched, matcher=UNRESOLVED/EXCEPTION (FN)
                "incorrectly_resolved": 0,# Matcher claimed wrong IDs
            }

    # Pre-populate all planted exception types
    for _, row in gt.iterrows():
        etype = str(row["exception_type"]).strip()
        if etype and etype != "none":
            _init_exc(etype)
            exc_breakdown[etype]["gt_count"] += 1

    # ── Per-decision evaluation ───────────────────────────────────────────────
    correct_full_matches       = 0
    correct_partial_detections = 0
    correctly_escalated        = 0
    incorrect_full_matches     = 0
    missed_resolvable          = 0
    incorrectly_auto_resolved  = 0

    exception_entries: list[ExceptionEntry] = []

    for dec in result.decisions:
        gt_row = gt_by_ledger.get(dec.ledger_id)
        if gt_row is None:
            # Should not occur for a well-formed dataset;
            # the one missing-ledger canonical (CAN-0090) has no ledger record
            # so it never appears in result.decisions.
            continue

        gt_status   = str(gt_row["match_status"]).strip()
        exc_type    = str(gt_row["exception_type"]).strip()
        exc_source  = str(gt_row["exception_source"]).strip()

        gt_is_matched    = (gt_status == "matched")
        gt_is_unresolved = (gt_status == "unresolved")

        mat_fully_matched = (dec.status == "MATCHED")
        mat_partial       = (dec.status == "PARTIAL")
        mat_exception     = (dec.status == "EXCEPTION")
        mat_unresolved    = (dec.status == "UNRESOLVED")

        bank_ok, inv_ok, stl_ok, any_incorrect = _check_claimed_ids(dec, gt_row)

        # ── Classify the decision ────────────────────────────────────────────
        if gt_is_matched:
            if (mat_fully_matched or mat_partial) and not any_incorrect:
                if mat_fully_matched:
                    correct_full_matches += 1
                    classification = "correct_full_match"
                else:
                    # PARTIAL but all claimed IDs are correct — matcher correctly
                    # refused to claim the problematic source
                    correct_partial_detections += 1
                    classification = "correct_partial_detection"
            elif (mat_fully_matched or mat_partial) and any_incorrect:
                # Matcher claimed a source with the WRONG record ID
                incorrect_full_matches += 1
                classification = "incorrect_match"
            elif mat_exception or mat_unresolved:
                # GT says recoverable, matcher gave up entirely
                missed_resolvable += 1
                classification = "missed_resolvable"
            else:
                classification = "unknown"

        elif gt_is_unresolved:
            if mat_fully_matched and not any_incorrect:
                # Shouldn't happen (MATCHED with all IDs when GT has a missing source)
                # but handle defensively
                incorrectly_auto_resolved += 1
                classification = "incorrectly_auto_resolved"
            elif any_incorrect:
                # Claimed a wrong ID on an unresolvable case
                incorrectly_auto_resolved += 1
                classification = "incorrectly_auto_resolved"
            else:
                # PARTIAL / EXCEPTION / UNRESOLVED with no wrong claims
                correctly_escalated += 1
                classification = "correctly_escalated"
        else:
            classification = "unknown"

        # ── Update exception type breakdown ──────────────────────────────────
        if exc_type and exc_type != "none" and exc_type in exc_breakdown:
            if classification == "correct_full_match":
                exc_breakdown[exc_type]["correctly_handled"] += 1
            elif classification == "correct_partial_detection":
                exc_breakdown[exc_type]["correctly_handled"] += 1
            elif classification == "correctly_escalated":
                exc_breakdown[exc_type]["correctly_escalated"] += 1
            elif classification == "missed_resolvable":
                exc_breakdown[exc_type]["missed"] += 1
            elif classification in ("incorrect_match", "incorrectly_auto_resolved"):
                exc_breakdown[exc_type]["incorrectly_resolved"] += 1

        # ── Evaluation entries (all evaluated cases) ───────────────────────────
        exception_entries.append(ExceptionEntry(
            ledger_id=dec.ledger_id,
            matcher_status=dec.status,
            gt_status=gt_status,
            exception_type=exc_type,
            exception_source=exc_source,
            bank_id_match=bank_ok,
            inv_id_match=inv_ok,
            stl_id_match=stl_ok,
            classification=classification,
            reason=dec.reason,
        ))

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total_evaluated  = len(result.decisions)
    matched_cnt      = result.status_counts.get("MATCHED", 0)
    partial_cnt      = result.status_counts.get("PARTIAL", 0)
    op_coverage      = (matched_cnt + partial_cnt) / total_evaluated if total_evaluated else 0.0

    # Precision: among decisions where a full/claimed match was made,
    # what fraction had all IDs correct?
    claimed_total = correct_full_matches + incorrect_full_matches
    match_precision = correct_full_matches / claimed_total if claimed_total else 1.0

    # Recall: among GT-matched evaluable transactions,
    # what fraction did the matcher produce a valid outcome for?
    # (correct PARTIAL still counts — matcher found the right records it could)
    # Denominator = GT-matched rows that have a ledger anchor (all 93 of them)
    gt_matched_evaluable = gt_matched_count  # all GT-matched rows have ledger anchors
    correct_recoveries   = correct_full_matches + correct_partial_detections
    match_recall = correct_recoveries / gt_matched_evaluable if gt_matched_evaluable else 0.0

    match_f1 = (
        2 * match_precision * match_recall / (match_precision + match_recall)
        if (match_precision + match_recall) else 0.0
    )

    exception_detection_rate = (
        correctly_escalated / gt_unresolved_evaluable
        if gt_unresolved_evaluable else 0.0
    )

    return EvaluationResult(
        total_canonical=total_canonical,
        gt_matched_count=gt_matched_count,
        gt_unresolved_total=gt_unresolved_all,
        ledger_missing_in_gt=ledger_missing_in_gt,
        gt_unresolved_evaluable=gt_unresolved_evaluable,
        total_evaluated=total_evaluated,
        status_counts=result.status_counts,
        correct_full_matches=correct_full_matches,
        correct_partial_detections=correct_partial_detections,
        correctly_escalated=correctly_escalated,
        incorrect_full_matches=incorrect_full_matches,
        missed_resolvable=missed_resolvable,
        incorrectly_auto_resolved=incorrectly_auto_resolved,
        match_precision=match_precision,
        match_recall=match_recall,
        match_f1=match_f1,
        operational_coverage=op_coverage,
        exception_detection_rate=exception_detection_rate,
        exception_breakdown=exc_breakdown,
        exception_entries=exception_entries,
    )
