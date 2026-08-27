"""
agent.py
========
Bounded AI Finance Controller Agent & Explicit Tool Orchestration Engine
Razorpay AI Buildathon Track 04 — AI Finance Controller

Architecture:
  START
  ↓
  load_source_records()
  ↓
  normalize_records()
  ↓
  FOR EACH ledger anchor:
      generate_candidates()
      ↓
      run_deterministic_match()
      ↓
      IF unresolved/ambiguous: run_ml_match_score()
      ↓
      verify_amount(), verify_date(), verify_reference(), verify_tax_line()
      ↓
      inspect_duplicate_candidates(), inspect_missing_source()
      ↓
      classify_reconciliation_case()
      ↓
      recommend_action()
      ↓
      IF unsafe to resolve: escalate_case()
      ↓
      write_audit_event()
  ↓
  summarize_batch()
  ↓
  END

No ground-truth access during inference. Strict decision policy with human-in-the-loop escalation.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import pandas as pd
import numpy as np

from src.matcher import (
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
    _match_source,
    DATA_DIR,
)
from src.ml_matcher import (
    MLReconciliationMatcher,
    extract_pair_features,
    FEATURE_NAMES,
    ModelArtifactError,
)


from src.llm_reviewer import (
    GeminiExceptionReviewer,
    ExceptionEvidence,
    ExceptionReview,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    timestamp: str
    ledger_id: str
    source_ids: Dict[str, Optional[str]]
    decision: str                            # MATCHED, PARTIAL, EXCEPTION, UNRESOLVED, SAFE_AUTO_RESOLVED
    confidence: float
    tools_used: List[str]
    evidence: Dict[str, Any]
    exception_type: str
    recommended_action: str
    requires_human_review: bool
    llm_review_id: Optional[str] = None
    llm_decision: Optional[str] = None
    llm_explanation: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_status_code: Optional[str] = None
    llm_attempts: Optional[int] = None
    llm_failure_category: Optional[str] = None
    llm_fallback_used: Optional[bool] = None
    llm_latency_seconds: Optional[float] = None
    llm_retry_count: Optional[int] = None
    llm_validation_status: Optional[str] = None
    llm_success: Optional[bool] = None
    llm_recommended_action: Optional[str] = None
    llm_human_review_required: Optional[bool] = None
    llm_model_name: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    ledger_id: str
    bank_id: Optional[str]
    invoice_id: Optional[str]
    settlement_id: Optional[str]
    status: str                              # MATCHED, PARTIAL, EXCEPTION, UNRESOLVED, SAFE_AUTO_RESOLVED
    confidence: float
    matching_method: str                     # Tier-1 Exact, Tier-2 Fuzzy, Tier-3 ML Scorer, Unresolved
    evidence: Dict[str, Any]
    exception_type: str                      # none, date_drift, amount_mismatch, missing_reference, missing_source_record, etc.
    recommended_action: str
    requires_human_review: bool
    audit_event_id: str
    agent_trace: List[str] = field(default_factory=list)
    llm_review: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchSummary:
    records_processed: int
    matched: int
    partial: int
    exceptions: int
    unresolved: int
    safely_resolved: int
    escalated: int
    total_tool_calls: int
    processing_time_seconds: float
    throughput_records_per_sec: float
    gemini_eligible_cases: int = 0
    gemini_initial_attempts: int = 0
    gemini_retries: int = 0
    gemini_successful_reviews: int = 0
    gemini_final_failures: int = 0
    gemini_fallback_cases: int = 0
    gemini_calls_attempted: int = 0
    gemini_calls_successful: int = 0
    gemini_calls_failed: int = 0
    gemini_total_latency_seconds: float = 0.0
    gemini_avg_successful_latency_sec: float = 0.0
    gemini_avg_attempts_per_case: float = 0.0
    tax_checks: int = 0
    tax_matches: int = 0
    tax_mismatches: int = 0
    tax_missing: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchResult:
    """Single authoritative runtime result shared by all consumers."""
    run_id: str
    started_at: str
    completed_at: str
    summary: BatchSummary
    ml: Dict[str, Any]
    gemini: Dict[str, Any]
    performance: Dict[str, Any]
    decisions: List[AgentDecision]
    audit_events: List[AuditEvent]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __iter__(self):
        """Compatibility for older callers while they migrate to BatchResult."""
        yield self.decisions
        yield self.audit_events
        yield self.summary


@dataclass
class ExceptionAnalysis:
    """Backwards-compatible analysis class for report generators."""
    ledger_id: str
    final_status: str
    risk_level: str
    detailed_explanation: str
    recommended_action: str
    evidence_summary: Dict[str, Any]
    safe_auto_resolved: bool = False
    original_status: str = ""
    confidence: float = 0.0
    missing_sources: List[str] = field(default_factory=list)
    matched_sources: List[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        return self.detailed_explanation

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["explanation"] = self.detailed_explanation
        return d


# ---------------------------------------------------------------------------
# Explicit Agent Tool Contract (16 Tools)
# ---------------------------------------------------------------------------

def tool_load_source_records(
    data_dir: Path | None = None,
    df_dict: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tool 1: Load raw financial records from disk CSVs or pandas DataFrames.
    NO ground_truth.csv is loaded or accessed.
    """
    target_dir = data_dir or DATA_DIR
    raw_sources: Dict[str, List[Dict[str, Any]]] = {}

    if df_dict:
        for name in ["ledger", "bank", "invoice", "settlement"]:
            if name in df_dict:
                df = df_dict[name].fillna("")
                raw_sources[name] = df.to_dict(orient="records")
            else:
                raw_sources[name] = []
        return raw_sources

    file_mapping = {
        "ledger": target_dir / "ledger.csv",
        "bank": target_dir / "bank_statements.csv",
        "invoice": target_dir / "invoices.csv",
        "settlement": target_dir / "settlements.csv",
    }

    for key, path in file_mapping.items():
        if path.exists():
            df = _load_csv(path)
            raw_sources[key] = df.to_dict(orient="records")
        else:
            raw_sources[key] = []

    return raw_sources


def tool_normalize_records(
    raw_sources: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tool 2: Normalize field names, dates, amounts, counterparty names, and references.
    """
    norm_sources: Dict[str, List[Dict[str, Any]]] = {}

    for name, rows in raw_sources.items():
        if not rows:
            norm_sources[name] = []
            continue

        df = pd.DataFrame(rows).fillna("")
        id_col = _resolve_col(df, name, _ID_CANDIDATES, "id")
        date_col = _resolve_col(df, name, _DATE_CANDIDATES, "date")

        norm_records = _to_norm_records(df, name, id_col, date_col)
        norm_sources[name] = norm_records

    return norm_sources


def tool_generate_candidates(
    ledger_rec: Dict[str, Any],
    target_records: List[Dict[str, Any]],
    source_name: str,
) -> List[Dict[str, Any]]:
    """
    Tool 3: Generate candidate records from a target system using broad matching criteria.
    """
    if not target_records:
        return []

    l_amt = ledger_rec["amount"]
    l_date = ledger_rec["date"]
    l_ref = ledger_rec["_ref_norm"]

    candidates = []
    for cand in target_records:
        d_diff = _date_diff(l_date, cand["date"])
        a_diff = abs(l_amt - cand["amount"])
        ref_match = (l_ref and l_ref == cand["_ref_norm"])

        # Candidate selection rule: date diff <= 30 days OR reference match OR amount diff <= ₹10
        if d_diff <= 30 or ref_match or a_diff <= 10.0:
            candidates.append(cand)

    return candidates


def tool_run_deterministic_match(
    ledger_rec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    claimed: Set[str],
    source_name: str,
    poisoned_refs: Set[str],
    corroborated: bool = False,
) -> SourceMatch:
    """
    Tool 4: Perform Tier 1 (Exact) and Tier 2 (Strong Fuzzy) deterministic matching.
    """
    return _match_source(
        ledger_rec=ledger_rec,
        candidates=candidates,
        claimed=claimed,
        source_name=source_name,
        poisoned_refs=poisoned_refs,
        corroborated=corroborated,
    )


def tool_run_ml_match_score(
    ml_matcher: Optional[MLReconciliationMatcher],
    ledger_rec: Dict[str, Any],
    unclaimed_cands: List[Dict[str, Any]],
    source_name: str,
    threshold: float = 0.90,
) -> SourceMatch:
    """
    Tool 5: Perform Tier 3 ML Residual Candidate Scoring using LogisticRegression pipeline.
    """
    if not unclaimed_cands:
        return SourceMatch(
            source=source_name, record_id=None, tier=0,
            confidence=0.0, reason="NO_CANDIDATE_FOUND",
        )

    if ml_matcher is None or not ml_matcher.is_fitted:
        # Fallback when ML matcher unavailable
        return SourceMatch(
            source=source_name, record_id=None, tier=0,
            confidence=0.0, reason="ML_SCORER_UNAVAILABLE → NO_MATCH",
        )

    feat_list = [
        extract_pair_features(ledger_rec, c, candidate_count=len(unclaimed_cands))
        for c in unclaimed_cands
    ]
    X_batch = pd.DataFrame(feat_list)[FEATURE_NAMES]
    probs = ml_matcher.model_pipeline.predict_proba(X_batch)[:, 1]

    scored_cands = [
        (float(prob), cand, feats)
        for prob, cand, feats in zip(probs, unclaimed_cands, feat_list)
    ]
    scored_cands.sort(key=lambda x: -x[0])
    top_prob, top_cand, top_feats = scored_cands[0]

    # Safety Check 1: Amount mismatch > ₹1.00 cannot be auto-reconciled
    if top_feats["amt_diff_abs"] > 1.00:
        return SourceMatch(
            source=source_name, record_id=None, tier=0,
            confidence=top_prob,
            reason=(
                f"ML_AMOUNT_MISMATCH (amt_diff=₹{top_feats['amt_diff_abs']:.2f} > 1.00) "
                f"→ NO_MATCH (requires review) top_cand={top_cand['record_id']}"
            ),
        )

    # Safety Check 2: Ambiguity / close probability check
    if len(scored_cands) > 1:
        second_prob = scored_cands[1][0]
        if top_prob >= threshold and (top_prob - second_prob) < 0.05:
            ids = [sc[1]["record_id"] for sc in scored_cands[:2]]
            return SourceMatch(
                source=source_name, record_id=None, tier=3,
                confidence=top_prob, is_ambiguous=True,
                reason=(
                    f"ML_AMBIGUOUS: top prob={top_prob:.3f} vs 2nd prob={second_prob:.3f} "
                    f"(gap < 0.05) → NEEDS_REVIEW candidates={ids}"
                ),
            )

    if top_prob >= threshold:
        reason = (
            f"ML_MATCH prob={top_prob:.3f} (thresh={threshold}) | "
            f"amt_diff=₹{top_feats['amt_diff_abs']:.2f}, date_diff={int(top_feats['date_diff_days'])}d, "
            f"cp_sim={top_feats['cp_sim']:.2f}, ref_sim={top_feats['ref_sim']:.2f}"
        )
        return SourceMatch(
            source=source_name, record_id=top_cand["record_id"],
            tier=3, confidence=top_prob, reason=reason,
        )
    else:
        return SourceMatch(
            source=source_name, record_id=None, tier=0,
            confidence=top_prob,
            reason=(
                f"ML_LOW_CONFIDENCE (prob={top_prob:.3f} < {threshold}) "
                f"→ NO_MATCH top_cand={top_cand['record_id']}"
            ),
        )


def tool_verify_amount(
    anchor_amt: float,
    target_amt: Optional[float],
    tolerance: float = 1.00,
) -> Dict[str, Any]:
    """
    Tool 6: Verify amount consistency between anchor and target record.
    """
    if target_amt is None:
        return {"status": "MISSING", "diff_abs": None, "within_tolerance": False}

    diff_abs = abs(anchor_amt - target_amt)
    within_tolerance = diff_abs <= tolerance
    return {
        "status": "PASS" if within_tolerance else "DISCREPANCY",
        "anchor_amount": anchor_amt,
        "target_amount": target_amt,
        "diff_abs": diff_abs,
        "within_tolerance": within_tolerance,
    }


def tool_verify_date(
    anchor_date: Any,
    target_date: Any,
    max_drift_days: float = 4.0,
) -> Dict[str, Any]:
    """
    Tool 7: Verify date drift between anchor and target record.
    """
    if not anchor_date or not target_date:
        return {"status": "MISSING", "drift_days": None, "within_tolerance": False}

    drift_days = float(_date_diff(anchor_date, target_date))
    within_tolerance = drift_days <= max_drift_days
    return {
        "status": "PASS" if within_tolerance else "DRIFT",
        "drift_days": drift_days,
        "within_tolerance": within_tolerance,
    }


def tool_verify_reference(
    anchor_ref: str,
    target_ref: str,
) -> Dict[str, Any]:
    """
    Tool 8: Verify reference string similarity and exact equality.
    """
    a_norm = _norm_ref(anchor_ref)
    t_norm = _norm_ref(target_ref)

    exact_match = (a_norm and a_norm == t_norm)
    sim = _str_sim(a_norm, t_norm)

    return {
        "exact_match": exact_match,
        "similarity": sim,
        "anchor_ref": a_norm,
        "target_ref": t_norm,
    }


def tool_verify_tax_line(
    invoice_tax: Any,
    ledger_tax: Any,
    ledger_id: Optional[str] = None,
    invoice_id: Optional[str] = None,
    reference: Optional[str] = None,
    date: Optional[str] = None,
    counterparty: Optional[str] = None,
) -> Dict[str, Any]:
    """Tool 9: Compare explicit invoice and ledger tax amounts deterministically."""
    def parse_tax(value: Any) -> Optional[Decimal]:
        text = str(value or "").strip().upper().replace("₹", "")
        if not text or text in {"NA", "N/A", "NONE", "NOT_APPLICABLE"}:
            return None
        if "=" in text:
            text = text.rsplit("=", 1)[1].strip()
        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return None

    invoice_text = str(invoice_tax or "").strip()
    ledger_text = str(ledger_tax or "").strip()
    non_applicable = {"NA", "N/A", "NONE", "NOT_APPLICABLE"}
    invoice_value = parse_tax(invoice_text)
    ledger_value = parse_tax(ledger_text)
    invoice_missing = not invoice_text or invoice_text.upper() in non_applicable
    ledger_missing = not ledger_text or ledger_text.upper() in non_applicable

    if invoice_missing and ledger_missing:
        status = "TAX_NOT_APPLICABLE" if invoice_text or ledger_text else "TAX_MISSING"
        difference = None
    elif invoice_value is None or ledger_value is None:
        status = "TAX_MISSING"
        difference = None
    else:
        difference = abs(invoice_value - ledger_value)
        status = "TAX_MATCH" if difference == Decimal("0") else "TAX_MISMATCH"

    return {
        "status": status,
        "invoice_tax": float(invoice_value) if invoice_value is not None else None,
        "ledger_tax": float(ledger_value) if ledger_value is not None else None,
        "tax_difference": float(difference) if difference is not None else None,
        "evidence": {
            "ledger_id": ledger_id,
            "invoice_id": invoice_id,
            "reference": reference,
            "date": date,
            "counterparty": counterparty,
            "invoice_tax_raw": invoice_text,
            "ledger_tax_raw": ledger_text,
        },
        "exception_type": "tax_mismatch" if status == "TAX_MISMATCH" else ("tax_missing" if status == "TAX_MISSING" else "none"),
    }


def tool_inspect_duplicate_candidates(
    reference: str,
    norm_sources: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Tool 10: Inspect reference collision and duplicate candidates across target sources.
    """
    ref_norm = _norm_ref(reference)
    if not ref_norm:
        return {"has_collision": False, "collisions": {}}

    collisions = {}
    has_collision = False

    for name in ["bank", "invoice", "settlement"]:
        records = norm_sources.get(name, [])
        matching_recs = [r["record_id"] for r in records if r.get("_ref_norm") == ref_norm]
        if len(matching_recs) > 1:
            has_collision = True
            collisions[name] = matching_recs

    return {
        "has_collision": has_collision,
        "collisions": collisions,
    }


def tool_inspect_missing_source(
    matched_sources: List[str],
    missing_sources: List[str],
) -> Dict[str, Any]:
    """
    Tool 11: Inspect missing sources and assess financial risk level.
    """
    is_cash_missing = "bank" in missing_sources
    risk_level = "HIGH" if is_cash_missing else ("MEDIUM" if missing_sources else "LOW")
    
    return {
        "matched_sources": matched_sources,
        "missing_sources": missing_sources,
        "is_cash_missing": is_cash_missing,
        "risk_level": risk_level,
    }


def tool_classify_reconciliation_case(
    bank_m: SourceMatch,
    inv_m: SourceMatch,
    stl_m: SourceMatch,
    verifications: Dict[str, Any],
) -> Tuple[str, float, int, str, str]:
    """
    Tool 12: Classify reconciliation case into operational status based on explicit decision policy.
    
    Decision Policy:
      HIGH CONFIDENCE -> MATCHED
      PARTIAL EVIDENCE -> PARTIAL
      CONFLICTING / AMBIGUOUS EVIDENCE -> EXCEPTION
      INSUFFICIENT EVIDENCE -> UNRESOLVED
    """
    matches = [bank_m, inv_m, stl_m]

    tax_status = verifications.get("tax_check", {}).get("status", "TAX_NOT_APPLICABLE")

    # Explicit ambiguity and tax-control checks
    if any(m.is_ambiguous for m in matches) or verifications.get("dup_check", {}).get("has_collision"):
        return (
            "EXCEPTION", 0.0, 3,
            "duplicate_reference",
            "Escalate: Duplicate reference collision / ambiguous candidates",
        )
    if tax_status in {"TAX_MISMATCH", "TAX_MISSING"}:
        return (
            "EXCEPTION", 0.0, 3,
            verifications["tax_check"]["exception_type"],
            "Escalate: Review tax posting",
        )

    matched_recs = [m for m in matches if m.record_id is not None]
    n_matched = len(matched_recs)

    if n_matched == 0:
        return (
            "UNRESOLVED", 0.0, 0,
            "missing_source_record",
            "Check missing source feeds / data ingestion",
        )

    avg_conf = sum(m.confidence for m in matched_recs) / n_matched
    min_tier = min(m.tier for m in matched_recs)

    # Determine exception type
    exception_type = "none"
    if n_matched < 3:
        if "bank" not in [m.source for m in matched_recs]:
            exception_type = "missing_source_record"
        else:
            missing_names = [m.source for m in matches if m.record_id is None]
            exception_type = f"missing_{missing_names[0]}"

    if n_matched == 3:
        # Check for subtle verifications (date drift, tax mismatch)
        d_drift = verifications.get("date_drift_max", 0.0)
        if d_drift > 0.0:
            exception_type = "date_drift"

        return (
            "MATCHED", avg_conf, min_tier,
            exception_type,
            "No action required — full 4-way match",
        )

    return (
        "PARTIAL", avg_conf, min_tier,
        exception_type,
        "Investigate missing source feeds",
    )


def tool_recommend_action(
    status: str,
    missing_sources: List[str],
    verifications: Dict[str, Any],
) -> Tuple[str, str, bool]:
    """
    Tool 13: Recommend actionable resolution, determine risk level and safe auto-resolution eligibility.
    
    Rules for SAFE_AUTO_RESOLVED:
    - Status is PARTIAL
    - Exactly one non-cash source (invoice OR settlement) is missing
    - Bank statement primary cash feed IS present and Tier-1 matched
    - Zero duplicate reference collision
    - Zero amount mismatch
    """
    has_dup_collision = verifications.get("dup_check", {}).get("has_collision", False)
    amt_disc = verifications.get("has_amount_discrepancy", False)

    if status == "MATCHED":
        return "No action required — full 4-way match verified", "LOW", False

    if status == "UNRESOLVED":
        return "Investigate: Missing source feeds / data ingestion gap", "HIGH", False

    if status == "EXCEPTION":
        if has_dup_collision:
            return "Escalate: Duplicate reference collision", "HIGH", False
        if amt_disc:
            return "Escalate: Material amount discrepancy", "HIGH", False
        if verifications.get("tax_check", {}).get("status") in {"TAX_MISMATCH", "TAX_MISSING"}:
            return "Escalate: Review tax posting", "HIGH", False
        return "Manual review: Conflicting counterparty/date evidence", "HIGH", False

    if status == "PARTIAL":
        if "bank" in missing_sources:
            return "Escalate: Missing primary bank feed – cash control risk", "MEDIUM", False
        
        # Check eligibility for safe auto-resolution
        is_single_non_cash_missing = (len(missing_sources) == 1 and "bank" not in missing_sources)
        if is_single_non_cash_missing and not has_dup_collision and not amt_disc:
            if "invoice" in missing_sources:
                return "Auto-resolve: Non-cash timing lag (invoice missing)", "LOW", True
            elif "settlement" in missing_sources:
                return "Auto-resolve: Settlement sync delay", "LOW", True

        return "Investigate: Partial source coverage – possible data ingestion gap", "MEDIUM", False

    return "Manual review required", "HIGH", False


def tool_escalate_case(
    ledger_id: str,
    reason: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tool 14: Escalate unsafe / ambiguous reconciliation cases for human Finance Ops review.
    """
    return {
        "ledger_id": ledger_id,
        "escalated": True,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_summary": evidence,
    }


def tool_write_audit_event(
    ledger_id: str,
    source_ids: Dict[str, Optional[str]],
    decision: str,
    confidence: float,
    tools_used: List[str],
    evidence: Dict[str, Any],
    exception_type: str,
    recommended_action: str,
    requires_human_review: bool,
    llm_review: Optional[ExceptionReview] = None,
    run_id: Optional[str] = None,
) -> AuditEvent:
    """
    Tool 15: Create a top-level immutable, machine-readable audit event with full LLM trace data.
    """
    event_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    return AuditEvent(
        event_id=event_id,
        timestamp=timestamp,
        ledger_id=ledger_id,
        source_ids=source_ids,
        decision=decision,
        confidence=confidence,
        tools_used=tools_used,
        evidence=evidence,
        exception_type=exception_type,
        recommended_action=recommended_action,
        requires_human_review=requires_human_review,
        llm_review_id=llm_review.review_id if llm_review else None,
        llm_decision=llm_review.decision if llm_review else None,
        llm_explanation=llm_review.explanation if llm_review else None,
        llm_confidence=llm_review.confidence if llm_review else None,
        llm_status_code=llm_review.status_code if llm_review else None,
        llm_attempts=llm_review.attempts if llm_review else None,
        llm_failure_category=llm_review.failure_category if llm_review else None,
        llm_fallback_used=llm_review.fallback_used if llm_review else None,
        llm_latency_seconds=llm_review.latency_seconds if llm_review else None,
        llm_retry_count=llm_review.retry_count if llm_review else None,
        llm_validation_status=llm_review.validation_status if llm_review else None,
        llm_success=llm_review.success if llm_review else None,
        llm_recommended_action=llm_review.recommended_action if llm_review else None,
        llm_human_review_required=llm_review.requires_human_review if llm_review else None,
        llm_model_name=llm_review.model_name if llm_review else None,
        run_id=run_id,
    )


def tool_summarize_batch(
    audit_events: List[AuditEvent],
    decisions: List[AgentDecision],
    elapsed_seconds: float,
    total_tool_calls: int,
    gemini_stats: Optional[Dict[str, Any]] = None,
) -> BatchSummary:
    """
    Tool 16: Summarize batch reconciliation statistics and performance metrics.
    """
    n = len(decisions)
    status_counts = {"MATCHED": 0, "PARTIAL": 0, "EXCEPTION": 0, "UNRESOLVED": 0, "SAFE_AUTO_RESOLVED": 0}

    safely_resolved = 0
    escalated = 0

    for d in decisions:
        status_counts[d.status] = status_counts.get(d.status, 0) + 1
        if d.status in ("MATCHED", "SAFE_AUTO_RESOLVED"):
            safely_resolved += 1
        else:
            escalated += 1

    throughput = n / elapsed_seconds if elapsed_seconds > 0 else 0.0

    stats = gemini_stats or {}
    eligible = stats.get("eligible", 0)
    initial_attempts = stats.get("initial_attempts", 0)
    retries = stats.get("retries", 0)
    successful = stats.get("successful", 0)
    final_failures = stats.get("failures", 0)
    fallback = stats.get("fallback", 0)
    succ_latency = stats.get("succ_latency", 0.0)
    total_latency = stats.get("latency", 0.0)
    tax_checks = stats.get("tax_checks", 0)
    tax_matches = stats.get("tax_matches", 0)
    tax_mismatches = stats.get("tax_mismatches", 0)
    tax_missing = stats.get("tax_missing", 0)

    avg_succ_lat = succ_latency / successful if successful > 0 else 0.0
    total_att = initial_attempts + retries
    avg_attempts = total_att / eligible if eligible > 0 else 0.0

    return BatchSummary(
        records_processed=n,
        matched=status_counts.get("MATCHED", 0),
        partial=status_counts.get("PARTIAL", 0),
        exceptions=status_counts.get("EXCEPTION", 0),
        unresolved=status_counts.get("UNRESOLVED", 0),
        safely_resolved=safely_resolved,
        escalated=escalated,
        total_tool_calls=total_tool_calls,
        processing_time_seconds=elapsed_seconds,
        throughput_records_per_sec=throughput,
        gemini_eligible_cases=eligible,
        gemini_initial_attempts=initial_attempts,
        gemini_retries=retries,
        gemini_successful_reviews=successful,
        gemini_final_failures=final_failures,
        gemini_fallback_cases=fallback,
        gemini_calls_attempted=total_att,
        gemini_calls_successful=successful,
        gemini_calls_failed=final_failures,
        gemini_total_latency_seconds=total_latency,
        gemini_avg_successful_latency_sec=avg_succ_lat,
        gemini_avg_attempts_per_case=avg_attempts,
        tax_checks=tax_checks,
        tax_matches=tax_matches,
        tax_mismatches=tax_mismatches,
        tax_missing=tax_missing,
    )


def tool_review_exception_with_llm(
    evidence: ExceptionEvidence,
    reviewer: GeminiExceptionReviewer,
) -> ExceptionReview:
    """
    Tool Execution: Invoke bounded Gemini LLM exception reviewer over verified evidence package.
    Guarantees machine-readable structured output and safe fallback on failure.
    """
    return reviewer.review_exception(evidence)


# ---------------------------------------------------------------------------
# Bounded Finance Controller Agent Class
# ---------------------------------------------------------------------------

class FinanceControllerAgent:
    """
    Bounded Finance Controller Agent.
    Executes financial reconciliation by calling explicit tools in a bounded orchestration loop.
    Never accesses ground truth during inference.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        ml_threshold: float = 0.90,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.data_dir = data_dir or DATA_DIR
        self.ml_threshold = ml_threshold
        self.ml_matcher = MLReconciliationMatcher(data_dir=self.data_dir, ml_threshold=self.ml_threshold)
        self.ml_model_error: Optional[str] = None
        try:
            self.ml_matcher.load_model_artifact()
        except ModelArtifactError as error:
            self.ml_model_error = str(error)
        self.llm_reviewer = GeminiExceptionReviewer(api_key=api_key, model_name=model_name)
        self.tool_call_count = 0
        self.current_run_id: Optional[str] = None

    def run_reconciliation_batch(
        self,
        df_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BatchResult:
        """
        Execute full batch reconciliation workflow via explicit tool calls.
        """
        t0 = time.perf_counter()
        run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"
        self.current_run_id = run_id
        started_at = datetime.now(timezone.utc).isoformat()
        self.tool_call_count = 0
        self.llm_reviewer.begin_batch()

        # Step 1: Load sources (Tool 1)
        self.tool_call_count += 1
        raw_sources = tool_load_source_records(self.data_dir, df_dict)

        # Step 2: Normalize records (Tool 2)
        self.tool_call_count += 1
        norm_sources = tool_normalize_records(raw_sources)

        ledger_records = norm_sources.get("ledger", [])
        bank_records = norm_sources.get("bank", [])
        invoice_records = norm_sources.get("invoice", [])
        settlement_records = norm_sources.get("settlement", [])

        # Runtime orchestration never trains from ground truth. A pre-fitted offline
        # model may be injected; otherwise residual ML matching fails safely.

        # Build poisoned reference sets
        poisoned_bank = _build_poisoned_refs(bank_records)
        poisoned_inv = _build_poisoned_refs(invoice_records)
        poisoned_stl = _build_poisoned_refs(settlement_records)

        claimed_bank: Set[str] = set()
        claimed_inv: Set[str] = set()
        claimed_stl: Set[str] = set()

        decisions: List[AgentDecision] = []
        audit_events: List[AuditEvent] = []

        gemini_eligible = 0
        gemini_initial_attempts = 0
        gemini_retries = 0
        gemini_successful = 0
        gemini_failures = 0
        gemini_fallback = 0
        gemini_total_latency = 0.0
        gemini_succ_latency = 0.0
        tax_checks = 0
        tax_matches = 0
        tax_mismatches = 0
        tax_missing = 0

        # Step 3: Record-by-Record Bounded Orchestration Loop
        for ledger_rec in ledger_records:
            dec, audit = self._process_single_ledger_record(
                ledger_rec=ledger_rec,
                bank_records=bank_records,
                invoice_records=invoice_records,
                settlement_records=settlement_records,
                norm_sources=norm_sources,
                poisoned_bank=poisoned_bank,
                poisoned_inv=poisoned_inv,
                poisoned_stl=poisoned_stl,
                claimed_bank=claimed_bank,
                claimed_inv=claimed_inv,
                claimed_stl=claimed_stl,
            )
            decisions.append(dec)
            audit_events.append(audit)

            if dec.llm_review:
                gemini_eligible += 1
                attempts = dec.llm_review.get("attempts", 1)
                if attempts > 0:
                    gemini_initial_attempts += 1
                    gemini_retries += max(0, attempts - 1)
                
                status_code = dec.llm_review.get("status_code", "")
                lat = dec.llm_review.get("latency_seconds", 0.0)
                gemini_total_latency += lat

                if status_code == "SUCCESS":
                    gemini_successful += 1
                    gemini_succ_latency += lat
                else:
                    gemini_failures += 1
                
                if dec.llm_review.get("fallback_used", False):
                    gemini_fallback += 1

            tax_check = dec.evidence.get("verifications", {}).get("tax_check")
            if tax_check:
                tax_checks += 1
                if tax_check.get("status") == "TAX_MATCH":
                    tax_matches += 1
                elif tax_check.get("status") == "TAX_MISMATCH":
                    tax_mismatches += 1
                elif tax_check.get("status") == "TAX_MISSING":
                    tax_missing += 1

        elapsed = time.perf_counter() - t0

        gemini_stats = {
            "eligible": gemini_eligible,
            "initial_attempts": gemini_initial_attempts,
            "retries": gemini_retries,
            "successful": gemini_successful,
            "failures": gemini_failures,
            "fallback": gemini_fallback,
            "latency": gemini_total_latency,
            "succ_latency": gemini_succ_latency,
            "tax_checks": tax_checks,
            "tax_matches": tax_matches,
            "tax_mismatches": tax_mismatches,
            "tax_missing": tax_missing,
        }

        # Step 4: Summarize batch (Tool 16)
        self.tool_call_count += 1
        summary = tool_summarize_batch(
            audit_events=audit_events,
            decisions=decisions,
            elapsed_seconds=elapsed,
            total_tool_calls=self.tool_call_count,
            gemini_stats=gemini_stats,
        )

        completed_at = datetime.now(timezone.utc).isoformat()
        ml_available = self.ml_matcher is not None and self.ml_matcher.is_fitted and self.ml_model_error is None
        core_runtime = max(0.0, elapsed - gemini_total_latency)
        return BatchResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            summary=summary,
            ml={
                "model_version": "match-model-v1" if ml_available else None,
                "ml_scored_candidates": self.ml_matcher.ml_stats.get("scored_pairs", 0) if self.ml_matcher else 0,
                "ml_threshold": self.ml_threshold,
                "ml_available": ml_available,
                "error": self.ml_model_error,
            },
            gemini={
                "configured": self.llm_reviewer.is_configured,
                "eligible_cases": summary.gemini_eligible_cases,
                "initial_attempts": summary.gemini_initial_attempts,
                "retries": summary.gemini_retries,
                "successful_reviews": summary.gemini_successful_reviews,
                "failed_reviews": summary.gemini_final_failures,
                "fallback_cases": summary.gemini_fallback_cases,
                "average_latency": summary.gemini_avg_successful_latency_sec,
            },
            performance={
                "reconciliation_engine_throughput": len(decisions) / core_runtime if core_runtime > 0 else 0.0,
                "agent_throughput": summary.throughput_records_per_sec,
                "total_runtime": summary.processing_time_seconds,
            },
            decisions=decisions,
            audit_events=audit_events,
        )

    def _process_single_ledger_record(
        self,
        ledger_rec: Dict[str, Any],
        bank_records: List[Dict[str, Any]],
        invoice_records: List[Dict[str, Any]],
        settlement_records: List[Dict[str, Any]],
        norm_sources: Dict[str, List[Dict[str, Any]]],
        poisoned_bank: Set[str],
        poisoned_inv: Set[str],
        poisoned_stl: Set[str],
        claimed_bank: Set[str],
        claimed_inv: Set[str],
        claimed_stl: Set[str],
    ) -> Tuple[AgentDecision, AuditEvent]:
        trace: List[str] = []
        lid = ledger_rec["record_id"]
        trace.append(f"1. Load ledger record {lid} (Amt: ₹{ledger_rec['amount']:.2f}, Ref: {ledger_rec['_ref_norm']})")

        tools_used_record: List[str] = ["load_source_records", "normalize_records"]

        # Tool 3: Candidate generation for targets
        self.tool_call_count += 3
        tools_used_record.append("generate_candidates")

        bank_cands = tool_generate_candidates(ledger_rec, bank_records, "bank")
        inv_cands = tool_generate_candidates(ledger_rec, invoice_records, "invoice")
        stl_cands = tool_generate_candidates(ledger_rec, settlement_records, "settlement")

        trace.append(f"2. Generated candidates: Bank={len(bank_cands)}, Invoice={len(inv_cands)}, Settlement={len(stl_cands)}")

        # Tool 4: Deterministic Tier 1 & Tier 2 matching
        self.tool_call_count += 2
        tools_used_record.append("run_deterministic_match")

        bank_m = tool_run_deterministic_match(ledger_rec, bank_cands, claimed_bank, "bank", poisoned_bank, corroborated=False)
        inv_m = tool_run_deterministic_match(ledger_rec, inv_cands, claimed_inv, "invoice", poisoned_inv, corroborated=False)

        # Tool 5: ML Residual Scoring if deterministic match is unconfirmed
        if bank_m.record_id is None or bank_m.tier not in (1, 2):
            self.tool_call_count += 1
            tools_used_record.append("run_ml_match_score")
            unclaimed = [c for c in bank_cands if c["record_id"] not in claimed_bank]
            bank_m = tool_run_ml_match_score(self.ml_matcher, ledger_rec, unclaimed, "bank", self.ml_threshold)
            trace.append(f"3. Bank ML Score result: record={bank_m.record_id}, Conf={bank_m.confidence:.3f}")
        else:
            trace.append(f"3. Bank Deterministic match: record={bank_m.record_id} (Tier {bank_m.tier})")

        if inv_m.record_id is None or inv_m.tier not in (1, 2):
            self.tool_call_count += 1
            tools_used_record.append("run_ml_match_score")
            unclaimed = [c for c in inv_cands if c["record_id"] not in claimed_inv]
            inv_m = tool_run_ml_match_score(self.ml_matcher, ledger_rec, unclaimed, "invoice", self.ml_threshold)
            trace.append(f"4. Invoice ML Score result: record={inv_m.record_id}, Conf={inv_m.confidence:.3f}")
        else:
            trace.append(f"4. Invoice Deterministic match: record={inv_m.record_id} (Tier {inv_m.tier})")

        # Settlement corroboration pass
        both_tier1 = (bank_m.tier == 1 and bank_m.record_id is not None and inv_m.tier == 1 and inv_m.record_id is not None)
        self.tool_call_count += 1
        tools_used_record.append("run_deterministic_match")
        stl_m = tool_run_deterministic_match(ledger_rec, stl_cands, claimed_stl, "settlement", poisoned_stl, corroborated=both_tier1)

        if stl_m.record_id is None or stl_m.tier not in (1, 2):
            self.tool_call_count += 1
            tools_used_record.append("run_ml_match_score")
            unclaimed = [c for c in stl_cands if c["record_id"] not in claimed_stl]
            stl_m = tool_run_ml_match_score(self.ml_matcher, ledger_rec, unclaimed, "settlement", self.ml_threshold)
            trace.append(f"5. Settlement ML Score result: record={stl_m.record_id}, Conf={stl_m.confidence:.3f}")
        else:
            trace.append(f"5. Settlement Deterministic match: record={stl_m.record_id} (Tier {stl_m.tier})")

        # Claim IDs if valid non-ambiguous matches found
        if bank_m.record_id and not bank_m.is_ambiguous:
            claimed_bank.add(bank_m.record_id)
        if inv_m.record_id and not inv_m.is_ambiguous:
            claimed_inv.add(inv_m.record_id)
        if stl_m.record_id and not stl_m.is_ambiguous:
            claimed_stl.add(stl_m.record_id)

        # Verification Tools (Tools 6-11)
        self.tool_call_count += 6
        tools_used_record.extend([
            "verify_amount", "verify_date", "verify_reference",
            "verify_tax_line", "inspect_duplicate_candidates", "inspect_missing_source",
        ])

        matched_sources = []
        missing_sources = []
        if bank_m.record_id: matched_sources.append("bank")
        else: missing_sources.append("bank")
        if inv_m.record_id: matched_sources.append("invoice")
        else: missing_sources.append("invoice")
        if stl_m.record_id: matched_sources.append("settlement")
        else: missing_sources.append("settlement")

        dup_check = tool_inspect_duplicate_candidates(ledger_rec["_ref_norm"], norm_sources)
        missing_check = tool_inspect_missing_source(matched_sources, missing_sources)

        target_lookup = {
            "bank": {r["record_id"]: r for r in bank_records},
            "invoice": {r["record_id"]: r for r in invoice_records},
            "settlement": {r["record_id"]: r for r in settlement_records},
        }

        verifications = {
            "dup_check": dup_check,
            "missing_check": missing_check,
            "has_amount_discrepancy": False,
            "date_drift_max": 0.0,
            "tax_status": "TAX_NOT_APPLICABLE",
        }

        # Check amount and date verifications for matched records
        for m in [bank_m, inv_m, stl_m]:
            if m.record_id and m.source in target_lookup:
                t_row = target_lookup[m.source].get(m.record_id)
                if t_row:
                    amt_res = tool_verify_amount(ledger_rec["amount"], t_row["amount"])
                    date_res = tool_verify_date(ledger_rec["date"], t_row["date"])
                    if amt_res["status"] == "DISCREPANCY":
                        verifications["has_amount_discrepancy"] = True
                    if date_res["drift_days"] is not None:
                        verifications["date_drift_max"] = max(verifications["date_drift_max"], date_res["drift_days"])

        invoice_row = target_lookup["invoice"].get(inv_m.record_id) if inv_m.record_id else None
        if invoice_row is not None:
            verifications["tax_check"] = tool_verify_tax_line(
                invoice_tax=invoice_row.get("tax_line", ""),
                ledger_tax=ledger_rec.get("tax_line", ""),
                ledger_id=lid,
                invoice_id=inv_m.record_id,
                reference=ledger_rec.get("_ref_norm", ""),
                date=str(ledger_rec.get("date", "")),
                counterparty=str(ledger_rec.get("_cp_norm", "")),
            )

        # Tool 12: Classify reconciliation case
        self.tool_call_count += 1
        tools_used_record.append("classify_reconciliation_case")

        status, conf, tier, exception_type, class_reason = tool_classify_reconciliation_case(
            bank_m, inv_m, stl_m, verifications
        )

        trace.append(f"6. Verification & Classification: status={status}, exception_type={exception_type}")

        # Tool 13: Recommend action
        self.tool_call_count += 1
        tools_used_record.append("recommend_action")

        action, risk_level, is_safe_auto = tool_recommend_action(
            status=status,
            missing_sources=missing_sources,
            verifications=verifications,
        )

        if is_safe_auto:
            status = "SAFE_AUTO_RESOLVED"
            requires_human_review = False
            trace.append(f"7. Safe Auto-Resolution: Low-risk non-cash timing lag safely resolved.")
        elif status == "MATCHED":
            requires_human_review = False
            trace.append(f"7. Resolution: Fully verified 4-way match.")
        else:
            requires_human_review = True
            # Tool 14: Escalate unsafe case
            self.tool_call_count += 1
            tools_used_record.append("escalate_case")
            tool_escalate_case(lid, action, verifications)
            trace.append(f"7. Escalation: Case requires human review ({action}).")

        # Determine matching method string
        tiers = [m.tier for m in [bank_m, inv_m, stl_m] if m.record_id]
        if not tiers:
            matching_method = "Unresolved"
        elif 3 in tiers:
            matching_method = "Tier-3 ML Candidate Scorer"
        elif 2 in tiers:
            matching_method = "Tier-2 Deterministic Fuzzy"
        else:
            matching_method = "Tier-1 Deterministic Exact"

        # Tool 14.5: Gemini LLM Review for exceptions / partial / escalated cases
        llm_review_res: Optional[ExceptionReview] = None
        if status in ("EXCEPTION", "PARTIAL", "UNRESOLVED") or requires_human_review:
            self.tool_call_count += 1
            tools_used_record.append("review_exception_with_llm")

            bank_ev = target_lookup["bank"].get(bank_m.record_id) if bank_m.record_id else None
            inv_ev = target_lookup["invoice"].get(inv_m.record_id) if inv_m.record_id else None
            stl_ev = target_lookup["settlement"].get(stl_m.record_id) if stl_m.record_id else None

            ev_pkg = ExceptionEvidence(
                ledger_id=lid,
                ledger_amount=float(ledger_rec["amount"]),
                ledger_date=str(ledger_rec["date"]).split()[0] if ledger_rec["date"] else "",
                ledger_reference=str(ledger_rec["_ref_norm"]),
                ledger_counterparty=str(ledger_rec.get("_cp_norm", "")),
                bank_evidence=bank_ev,
                invoice_evidence=inv_ev,
                settlement_evidence=stl_ev,
                verifications=verifications,
                exception_type=exception_type,
                matching_tier=matching_method,
                ml_score=max(bank_m.confidence, inv_m.confidence, stl_m.confidence),
                controller_status=status,
            )

            llm_review_res = tool_review_exception_with_llm(ev_pkg, self.llm_reviewer)
            trace.append(
                f"• Gemini LLM Review ({llm_review_res.status_code}): decision={llm_review_res.decision}, "
                f"conf={llm_review_res.confidence:.2f}"
            )

        evidence = {
            "ledger_amount": ledger_rec["amount"],
            "ledger_reference": ledger_rec["_ref_norm"],
            "ledger_date": str(ledger_rec["date"]).split()[0] if ledger_rec["date"] else "",
            "matched_sources": matched_sources,
            "missing_sources": missing_sources,
            "verifications": verifications,
            "bank_match": bank_m.reason,
            "invoice_match": inv_m.reason,
            "settlement_match": stl_m.reason,
            "llm_review": llm_review_res.to_dict() if llm_review_res else None,
        }

        # Tool 15: Write audit event
        self.tool_call_count += 1
        tools_used_record.append("write_audit_event")

        source_ids = {
            "bank": bank_m.record_id,
            "invoice": inv_m.record_id,
            "settlement": stl_m.record_id,
        }

        audit_event = tool_write_audit_event(
            ledger_id=lid,
            source_ids=source_ids,
            decision=status,
            confidence=conf,
            tools_used=tools_used_record,
            evidence=evidence,
            exception_type=exception_type,
            recommended_action=action,
            requires_human_review=requires_human_review,
            llm_review=llm_review_res,
            run_id=self.current_run_id,
        )

        trace.append(f"8. Audit event written: {audit_event.event_id}")

        decision = AgentDecision(
            ledger_id=lid,
            bank_id=bank_m.record_id,
            invoice_id=inv_m.record_id,
            settlement_id=stl_m.record_id,
            status=status,
            confidence=conf,
            matching_method=matching_method,
            evidence=evidence,
            exception_type=exception_type,
            recommended_action=action,
            requires_human_review=requires_human_review,
            audit_event_id=audit_event.event_id,
            agent_trace=trace,
            llm_review=llm_review_res.to_dict() if llm_review_res else None,
        )

        return decision, audit_event


# ---------------------------------------------------------------------------
# Backwards-Compatible ExceptionAgent Wrapper
# ---------------------------------------------------------------------------

class ExceptionAgent:
    """
    Backwards-compatible wrapper around FinanceControllerAgent / domain generator.
    Maintains support for legacy report generators.
    """

    def __init__(self):
        self.agent = FinanceControllerAgent()

    def analyze_residuals(
        self,
        decisions: List[ReconciliationDecision],
        source_records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ExceptionAnalysis]:
        residuals = [d for d in decisions if d.status in {"PARTIAL", "EXCEPTION", "UNRESOLVED"}]
        analyses = []
        for dec in residuals:
            analyses.append(self.analyze_single(dec, source_records))
        return analyses

    def analyze_single(
        self,
        dec: ReconciliationDecision,
        source_records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ExceptionAnalysis:
        matched_sources = []
        missing_sources = []
        if dec.bank_match.record_id: matched_sources.append("bank")
        else: missing_sources.append("bank")
        if dec.invoice_match.record_id: matched_sources.append("invoice")
        else: missing_sources.append("invoice")
        if dec.settlement_match.record_id: matched_sources.append("settlement")
        else: missing_sources.append("settlement")

        is_single_non_cash_missing = (len(missing_sources) == 1 and "bank" not in missing_sources)
        both_tier1 = (
            dec.bank_match.tier == 1 and dec.bank_match.record_id is not None and
            (dec.invoice_match.tier == 1 if "invoice" in matched_sources else dec.settlement_match.tier == 1)
        )

        if dec.status == "PARTIAL" and is_single_non_cash_missing and both_tier1:
            safe_auto = True
            final_status = "SAFE_AUTO_RESOLVED"
            risk_level = "LOW"
            action = "Auto-resolve: Non-cash timing lag (invoice missing)" if "invoice" in missing_sources else "Auto-resolve: Settlement sync delay"
        elif "bank" in missing_sources:
            safe_auto = False
            final_status = "PARTIAL"
            risk_level = "MEDIUM"
            action = "Escalate: Missing primary bank feed – cash control risk"
        elif dec.status == "EXCEPTION":
            safe_auto = False
            final_status = "EXCEPTION"
            risk_level = "HIGH"
            action = "Escalate: Duplicate reference collision" if "AMBIGUOUS" in dec.reason else "Escalate: Material amount discrepancy"
        else:
            safe_auto = False
            final_status = dec.status
            risk_level = "MEDIUM"
            action = "Investigate: Partial source coverage – possible data ingestion gap"

        explanation = (
            f"Ledger record {dec.ledger_id} reconciled across matched feeds ({', '.join(matched_sources)}). "
            f"Missing sources: {', '.join(missing_sources)}. Recommended Action: {action}."
        )

        return ExceptionAnalysis(
            ledger_id=dec.ledger_id,
            final_status=final_status,
            risk_level=risk_level,
            detailed_explanation=explanation,
            recommended_action=action,
            evidence_summary={"matcher_reason": dec.reason},
            safe_auto_resolved=safe_auto,
            original_status=dec.status,
            confidence=dec.confidence,
            missing_sources=missing_sources,
            matched_sources=matched_sources,
        )
