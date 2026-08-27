"""
main.py
=======
Finance Controller Agent — End-to-End Bounded Agent Orchestration & Evaluation
==============================================================================

Usage:
    python main.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.matcher import ReconciliationMatcher
from src.ml_matcher import MLReconciliationMatcher, evaluate_thresholds
from src.evaluate import evaluate
from src.agent import FinanceControllerAgent, ExceptionAgent
from src.report import generate_final_report

LINE = "=" * 70
DASH = "─" * 70


def _section(title: str) -> None:
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def main() -> None:
    print(LINE)
    print("  RAZORPAY AI BUILDATHON — BOUNDED FINANCE CONTROLLER AGENT")
    print(LINE)

    # ── 1. Run Bounded Finance Controller Agent Batch Execution ───────────────
    print("\n[1/5] Executing Bounded Finance Controller Agent Batch Workflow...")
    agent = FinanceControllerAgent(ml_threshold=0.90)
    
    mode_str = "GEMINI LIVE MODE" if agent.llm_reviewer.is_configured else "GEMINI FALLBACK MODE"
    print(f"  [STATUS] Operating in: {mode_str} (Model: {agent.llm_reviewer.model_name})")

    agent_decisions, audit_events, batch_summary = agent.run_reconciliation_batch()

    # ── 2. Run Deterministic Baseline & ML Evaluation ─────────────────────────
    print("\n[2/5] Running Deterministic Baseline & Evaluation Benchmarks...")
    base_matcher = ReconciliationMatcher()
    base_matcher.load_sources()
    base_result = base_matcher.reconcile()
    base_ev = evaluate(base_result)

    ml_matcher = MLReconciliationMatcher(ml_threshold=0.90)
    ml_matcher.load_sources()
    ml_matcher.train_model()
    ml_result = ml_matcher.reconcile()
    ml_ev = evaluate(ml_result)

    source_records = {
        "ledger":     {r["record_id"]: r for r in base_matcher._ledger_records},
        "bank":       {r["record_id"]: r for r in base_matcher._bank_records},
        "invoice":    {r["record_id"]: r for r in base_matcher._invoice_records},
        "settlement": {r["record_id"]: r for r in base_matcher._settlement_records},
    }

    # ── 3. Exception Agent Analysis (for legacy reporting) ────────────────────
    print("\n[3/5] Generating Exception Register & Audit Analysis...")
    legacy_agent = ExceptionAgent()
    analyses = legacy_agent.analyze_residuals(ml_result.decisions, source_records)

    # ── 4. Generate Reports ───────────────────────────────────────────────────
    print("\n[4/5] Exporting exceptions.json and generating report.md...")
    json_path, md_path = generate_final_report(ml_result, ml_ev, analyses)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — AGENT BATCH SUMMARY & RELIABILITY METRICS
    # ══════════════════════════════════════════════════════════════════════
    _section("1. FINANCE CONTROLLER AGENT — BATCH SUMMARY & RELIABILITY METRICS")
    print(f"  {'Metric':<42} | {'Value':<30}")
    print(f"  {'─'*42}-+-{'─'*30}")
    print(f"  {'Execution Mode':<42} | {mode_str:<30}")
    print(f"  {'Gemini Model Configured':<42} | {agent.llm_reviewer.model_name:<30}")
    print(f"  {'Total Records Processed':<42} | {batch_summary.records_processed:<30}")
    print(f"  {'Matched (Full 4-Way)':<42} | {batch_summary.matched:<30}")
    print(f"  {'Safely Auto-Resolved (Timing Lag)':<42} | {batch_summary.safely_resolved - batch_summary.matched:<30}")
    print(f"  {'Partial Coverage':<42} | {batch_summary.partial:<30}")
    print(f"  {'Exceptions / Ambiguous':<42} | {batch_summary.exceptions:<30}")
    print(f"  {'Unresolved':<42} | {batch_summary.unresolved:<30}")
    print(f"  {'Escalated for Human Review':<42} | {batch_summary.escalated:<30}")
    print(f"  {'Total Gemini-Eligible Cases':<42} | {batch_summary.gemini_eligible_cases:<30}")
    print(f"  {'Gemini Initial Attempts':<42} | {batch_summary.gemini_initial_attempts:<30}")
    print(f"  {'Gemini Retries Performed':<42} | {batch_summary.gemini_retries:<30}")
    print(f"  {'Successful Final Gemini Reviews':<42} | {batch_summary.gemini_successful_reviews:<30}")
    print(f"  {'Final Gemini Failures':<42} | {batch_summary.gemini_final_failures:<30}")
    print(f"  {'Safe Fallback Cases Used':<42} | {batch_summary.gemini_fallback_cases:<30}")
    print(f"  {'Average Successful Gemini Latency':<42} | {batch_summary.gemini_avg_successful_latency_sec:.3f} sec")
    print(f"  {'Average Attempts Per Case':<42} | {batch_summary.gemini_avg_attempts_per_case:.2f}")
    print(f"  {'Gemini Total Latency':<42} | {batch_summary.gemini_total_latency_seconds:.3f} sec")
    print(f"  {'Reconciliation Engine Throughput':<42} | {ml_result.throughput_per_second:.0f} rec/sec")
    print(f"  {'Agent Orchestration Throughput':<42} | {batch_summary.throughput_records_per_sec:.0f} rec/sec")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — REPRESENTATIVE AGENT EXECUTION TRACES (5 CASES)
    # ══════════════════════════════════════════════════════════════════════
    _section("2. REPRESENTATIVE AGENT EXECUTION TRACES (TOOL-CALL FLOW)")

    # Select 5 representative cases: Safe Match, Safe Auto-Resolve, Partial Missing Bank, Duplicate Exception, Amount/Tax Drift
    selected_lids = ["LED-0001", "LED-0049", "LED-0026", "LED-0034", "LED-0061"]
    dec_dict = {d.ledger_id: d for d in agent_decisions}

    for lid in selected_lids:
        if lid in dec_dict:
            d = dec_dict[lid]
            print(f"\n  [AGENT TRACE] Ledger Anchor Record: {d.ledger_id}")
            print(f"    Status               : {d.status}")
            print(f"    Matching Method      : {d.matching_method}")
            print(f"    Confidence           : {d.confidence:.3f}")
            print(f"    Recommended Action   : {d.recommended_action}")
            print(f"    Requires Human Review: {d.requires_human_review}")
            print(f"    Audit Event ID       : {d.audit_event_id}")
            print(f"    Execution Steps      :")
            for step in d.agent_trace:
                print(f"      • {step}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — GEMINI LLM EXCEPTION REASONING & DISCREPANCY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    _section("3. GEMINI LLM EXCEPTION REASONING & DISCREPANCY ANALYSIS")

    if agent.llm_reviewer.is_configured:
        print("  *** GEMINI LIVE MODE ACTIVE ***")
        print(f"  API Key configured. Model: {agent.llm_reviewer.model_name}")
        print("  Executing targeted live reasoning test on selected difficult exception cases:\n")
    else:
        print("  *** GEMINI FALLBACK MODE ACTIVE ***")
        print("  GEMINI_API_KEY is not configured in environment.")
        print("  Deterministic & ML Controller decisions retained with safe escalation policy.\n")

    exception_cases = [d for d in agent_decisions if d.llm_review is not None]
    print(f"  Total Gemini-Eligible Cases Analyzed: {len(exception_cases)}\n")

    for d in exception_cases:
        llm = d.llm_review or {}
        ev = d.evidence
        status_code = llm.get("status_code", "N/A")
        val_status = "VALID" if status_code == "SUCCESS" else ("INVALID" if status_code in ("PARSE_ERROR", "UNSAFE_DECISION_REJECTED") else "N/A")
        cat = llm.get("failure_category") or ("NONE" if status_code == "SUCCESS" else "UNKNOWN")

        print(f"  ┌─ [PER-CASE GEMINI OBSERVABILITY RECORD | {d.ledger_id}]")
        print(f"  │  Ledger ID            : {d.ledger_id}")
        print(f"  │  Exception Type       : {d.exception_type}")
        print(f"  │  Attempts Made        : {llm.get('attempts', 1)}")
        print(f"  │  Final API Status     : {status_code}")
        print(f"  │  Failure Category     : {cat}")
        print(f"  │  Gemini Decision      : {llm.get('decision', 'N/A')}")
        print(f"  │  Validation Status    : {val_status}")
        print(f"  │  LLM Confidence       : {llm.get('confidence', 0.0):.2f}")
        print(f"  │  Latency              : {llm.get('latency_seconds', 0.0):.3f} sec")
        print(f"  │  Controller Decision  : {d.status} (Authoritative)")
        print(f"  │  Human Review Required: {llm.get('requires_human_review', True)}")
        print(f"  │  Fallback Used        : {llm.get('fallback_used', False)}")
        print(f"  │  Explanation          : {llm.get('explanation', 'N/A')}")
        print(f"  └─────────────────────────────────────────────────────────────\n")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — DETERMINISTIC BASELINE VS ML-ASSISTED MATCHING
    # ══════════════════════════════════════════════════════════════════════
    _section("4. DETERMINISTIC BASELINE VS ML-ASSISTED MATCHING")
    print(f"\n  {'Metric':<35} | {'Baseline (Deterministic)':<24} | {'ML-Assisted (Thresh=0.90)':<24}")
    print(f"  {'─'*35}-+-{'─'*24}-+-{'─'*24}")
    print(f"  {'Ledger Anchors Evaluated':<35} | {base_result.total_processed:<24} | {ml_result.total_processed:<24}")
    print(f"  {'MATCHED (All 3 Targets)':<35} | {base_result.status_counts.get('MATCHED',0):<24} | {ml_result.status_counts.get('MATCHED',0):<24}")
    print(f"  {'PARTIAL (1-2 Targets)':<35} | {base_result.status_counts.get('PARTIAL',0):<24} | {ml_result.status_counts.get('PARTIAL',0):<24}")
    print(f"  {'EXCEPTION / Ambiguous':<35} | {base_result.status_counts.get('EXCEPTION',0):<24} | {ml_result.status_counts.get('EXCEPTION',0):<24}")
    print(f"  {'UNRESOLVED':<35} | {base_result.status_counts.get('UNRESOLVED',0):<24} | {ml_result.status_counts.get('UNRESOLVED',0):<24}")
    print(f"  {'Operational Coverage':<35} | {base_ev.operational_coverage*100:>23.1f}% | {ml_ev.operational_coverage*100:>23.1f}%")
    print(f"  {'Correct Full Matches':<35} | {base_ev.correct_full_matches:<24} | {ml_ev.correct_full_matches:<24}")
    print(f"  {'Correct Partial Detections':<35} | {base_ev.correct_partial_detections:<24} | {ml_ev.correct_partial_detections:<24}")
    print(f"  {'Correctly Escalated':<35} | {base_ev.correctly_escalated:<24} | {ml_ev.correctly_escalated:<24}")
    print(f"  {'Incorrect Auto Matches (FP)':<35} | {base_ev.incorrect_full_matches:<24} | {ml_ev.incorrect_full_matches:<24}")
    print(f"  {'Missed Resolvable (FN)':<35} | {base_ev.missed_resolvable:<24} | {ml_ev.missed_resolvable:<24}")
    print(f"  {'Match Precision':<35} | {base_ev.match_precision*100:>23.1f}% | {ml_ev.match_precision*100:>23.1f}%")
    print(f"  {'Match Recall':<35} | {base_ev.match_recall*100:>23.1f}% | {ml_ev.match_recall*100:>23.1f}%")
    print(f"  {'Match F1 Score':<35} | {base_ev.match_f1*100:>23.1f}% | {ml_ev.match_f1*100:>23.1f}%")
    print(f"  {'Throughput (records/sec)':<35} | {base_result.throughput_per_second:>24.0f} | {ml_result.throughput_per_second:>24.0f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — FINANCIAL SAFETY ASSESSMENT
    # ══════════════════════════════════════════════════════════════════════
    _section("5. FINANCIAL SAFETY ASSESSMENT")
    print("  ✓ Bounded Orchestration: Agent operates via explicit tool calls with stopping conditions.")
    print("  ✓ Zero Ground-Truth Leakage: Ground truth is completely inaccessible during inference.")
    print("  ✓ Strict Escalation Policy: Missing bank statement cash feeds and duplicate references are escalated.")
    print("  ✓ Safe Auto-Resolution: Only single non-cash timing lags with 100% Tier-1 proof are auto-resolved.")
    print("  ✓ Controller Authority: Gemini explanation NEVER overrides authoritative financial decisions.")
    print("  ✓ Structured Output & Validation: Gemini outputs validated via Pydantic schema and safety filters.")
    print("  ✓ Complete Audit Trail: Immutable AuditEvent generated for 100% of decisions.")

    print(f"\n{LINE}")
    print("  Pipeline complete.")
    print(LINE + "\n")


if __name__ == "__main__":
    main()
