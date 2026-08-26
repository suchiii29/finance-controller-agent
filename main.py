"""
main.py
=======
Finance Controller Agent — End-to-End Pipeline
==============================================

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from src.matcher import ReconciliationMatcher
from src.evaluate import evaluate
from src.agent import ExceptionAgent
from src.report import generate_final_report


def main() -> None:
    print("=" * 64)
    print("  RAZORPAY AI BUILDATHON — AI FINANCE CONTROLLER PIPELINE")
    print("=" * 64)

    # 1. Load Sources & Run Deterministic Reconciliation
    print("\n[1/4] Loading source files and normalising records...")
    matcher = ReconciliationMatcher()
    matcher.load_sources()

    print("\n[2/4] Running deterministic multi-source reconciliation...")
    result = matcher.reconcile()

    # Extract source records dictionary for Exception Agent
    source_records = {
        "ledger": {r["record_id"]: r for r in matcher._ledger_records},
        "bank": {r["record_id"]: r for r in matcher._bank_records},
        "invoice": {r["record_id"]: r for r in matcher._invoice_records},
        "settlement": {r["record_id"]: r for r in matcher._settlement_records},
    }

    # 2. Pass Residuals + Source Records to Exception Agent
    print("\n[3/4] Running Exception Agent on residual cases...")
    agent = ExceptionAgent()
    analyses = agent.analyze_residuals(result.decisions, source_records)

    # 3. Evaluate Ground Truth Metrics
    ev = evaluate(result)

    # 4. Generate Final Reports & Exports
    print("\n[4/4] Generating final reports and exporting exception JSON...")
    json_path, md_path = generate_final_report(result, ev, analyses)

    # 5. Print Executive Summary
    total = result.total_processed
    det_matched = result.status_counts.get("MATCHED", 0)
    agent_auto = sum(1 for a in analyses if a.safe_auto_resolved)
    escalated = len(analyses) - agent_auto
    effective_resolved = det_matched + agent_auto
    op_coverage = (effective_resolved / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 64)
    print("  EXECUTIVE SUMMARY (30-SECOND VIEW)")
    print("=" * 64)
    print(f"  • Total Anchor Records Processed     : {total}")
    print(f"  • Fully Matched (Deterministic Rules) : {det_matched}")
    print(f"  • Safely Auto-Resolved (AI Agent)    : {agent_auto}")
    print(f"  • Still Escalated (Finance Ops)      : {escalated}")
    print(f"  • Total Effective Automation Rate    : {op_coverage:.1f}% ({effective_resolved}/{total})")
    print(f"  • Processing Throughput              : {result.throughput_per_second:.0f} rec/sec")
    print(f"  • Reports Generated                  : {json_path.name}, {md_path.name}")
    print("  ─" * 32)
    print("  FINANCIAL SAFETY & HONESTY STATEMENT:")
    print("  The deterministic matcher prefers conservative high-confidence matches.")
    print(f"  The AI Agent safely auto-resolved {agent_auto} low-risk timing lag cases")
    print(f"  with 100% Tier-1 corroboration. The remaining {escalated} medium-risk cases")
    print("  (missing primary bank feeds) are strictly escalated to Finance Ops.")
    print("=" * 64)

    # 6. Print 5 Detailed Example Exceptions with Varied Recommended Actions
    print("\n" + "=" * 64)
    print("  DETAILED EXCEPTION ANALYSIS EXAMPLES (5 SAMPLE CASES)")
    print("=" * 64)

    # Pick 2 auto-resolved (1 missing invoice, 1 missing settlement) and 3 bank-missing escalated cases
    missing_inv = [a for a in analyses if a.safe_auto_resolved and "invoice" in a.missing_sources][:1]
    missing_stl = [a for a in analyses if a.safe_auto_resolved and "settlement" in a.missing_sources][:1]
    bank_escalated = [a for a in analyses if not a.safe_auto_resolved and "bank" in a.missing_sources]

    sample_examples = missing_inv + missing_stl + bank_escalated

    for i, analysis in enumerate(sample_examples, 1):
        status_tag = "SAFE_AUTO_RESOLVED" if analysis.safe_auto_resolved else f"ESCALATED ({analysis.final_status})"
        print(f"\n[{i}] Ledger ID: {analysis.ledger_id} | Decision: {status_tag} | Risk Level: {analysis.risk_level}")
        print(f"    Action Assigned     : {analysis.recommended_action}")
        print(f"    Safe Auto Resolved  : {analysis.safe_auto_resolved}")
        print(f"    Missing Sources     : {', '.join(analysis.missing_sources)}")
        print(f"    Matched Sources     : {', '.join(analysis.matched_sources)}")
        print(f"    Detailed Explanation:\n    {analysis.detailed_explanation}")
        print(f"    Evidence Summary    : Amount = ₹{analysis.evidence_summary['ledger']['amount']}, Ref = {analysis.evidence_summary['ledger']['reference']}, CP = {analysis.evidence_summary['ledger']['counterparty']}")
        print("  " + "─" * 60)

    print("\n" + "=" * 64)
    print("  Pipeline complete.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
