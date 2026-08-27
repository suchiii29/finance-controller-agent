"""Render runtime reports from the authoritative BatchResult."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from src.agent import BatchResult

REPORT_DIR = Path(__file__).parent.parent / "reports"
DATA_DIR = Path(__file__).parent.parent / "data"


def generate_final_report(batch_result: BatchResult, evaluation_result: Optional[object] = None) -> Tuple[Path, Path]:
    """Generate report and exception export without rerunning reconciliation."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "exceptions.json"
    md_path = Path(__file__).parent.parent / "report.md"
    summary = batch_result.summary

    records = []
    for decision in batch_result.decisions:
        if not decision.requires_human_review and decision.status not in {"EXCEPTION", "UNRESOLVED"}:
            continue
        tax = decision.evidence.get("verifications", {}).get("tax_check", {})
        llm = decision.llm_review or {}
        records.append({
            "run_id": batch_result.run_id,
            "ledger_id": decision.ledger_id,
            "reconciliation_status": decision.status,
            "controller_decision": decision.status,
            "exception_type": decision.exception_type,
            "source_ids": {"bank": decision.bank_id, "invoice": decision.invoice_id, "settlement": decision.settlement_id},
            "confidence": decision.confidence,
            "recommendation": decision.recommended_action,
            "requires_human_review": decision.requires_human_review,
            "tax_status": tax.get("status", "NOT_CHECKED"),
            "gemini_status": llm.get("status_code", "NOT_ATTEMPTED"),
            "audit_event_id": decision.audit_event_id,
        })
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    evaluation_section = ""
    if evaluation_result is not None:
        evaluation_section = f"""
## Offline Evaluation

Evaluation against ground truth is separate from the runtime controller result.

| Metric | Value |
| :--- | ---: |
| Correct full matches | {evaluation_result.correct_full_matches} |
| Correct partial detections | {evaluation_result.correct_partial_detections} |
| Correctly escalated | {evaluation_result.correctly_escalated} |
| Match precision | {evaluation_result.match_precision * 100:.1f}% |
| Match recall | {evaluation_result.match_recall * 100:.1f}% |
"""

    md_path.write_text(f"""# Finance Controller Runtime Report

**Run ID:** `{batch_result.run_id}`
**Started:** {batch_result.started_at}
**Completed:** {batch_result.completed_at}

## Controller Summary

| Metric | Value |
| :--- | ---: |
| Records processed | {summary.records_processed} |
| Matched | {summary.matched} |
| Partial | {summary.partial} |
| Exceptions | {summary.exceptions} |
| Unresolved | {summary.unresolved} |
| Safe auto-resolved | {summary.safely_resolved - summary.matched} |
| Human review | {summary.escalated} |

## Tax Verification

| Metric | Value |
| :--- | ---: |
| Tax checks | {summary.tax_checks} |
| Tax matches | {summary.tax_matches} |
| Tax mismatches | {summary.tax_mismatches} |
| Tax missing | {summary.tax_missing} |

## Gemini

| Metric | Value |
| :--- | ---: |
| Configured | {batch_result.gemini['configured']} |
| Eligible cases | {batch_result.gemini['eligible_cases']} |
| Initial attempts | {batch_result.gemini['initial_attempts']} |
| Retries | {batch_result.gemini['retries']} |
| Successful reviews | {batch_result.gemini['successful_reviews']} |
| Failed reviews | {batch_result.gemini['failed_reviews']} |
| Fallback cases | {batch_result.gemini['fallback_cases']} |

## Performance

| Metric | Value |
| :--- | ---: |
| Agent throughput | {batch_result.performance['agent_throughput']:.2f} records/sec |
| Total runtime | {batch_result.performance['total_runtime']:.3f} sec |

## Audit

- Audit events: {len(batch_result.audit_events)}
- Exception export records: {len(records)}
- Every audit event carries run ID `{batch_result.run_id}`.
{evaluation_section}""".rstrip() + "\n", encoding="utf-8")
    return json_path, md_path
