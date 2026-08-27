"""
main.py
=======
Finance Controller Agent — End-to-End Pipeline
==============================================

Usage:
    python main.py
"""

from __future__ import annotations

from src.matcher import ReconciliationMatcher
from src.evaluate import evaluate
from src.agent import ExceptionAgent
from src.report import generate_final_report

LINE = "=" * 64
DASH = "─" * 64


def _section(title: str) -> None:
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def main() -> None:
    print(LINE)
    print("  RAZORPAY AI BUILDATHON — AI FINANCE CONTROLLER PIPELINE")
    print(LINE)

    # ── 1. Load & Reconcile ────────────────────────────────────────────────
    print("\n[1/4] Loading source files and normalising records...")
    matcher = ReconciliationMatcher()
    matcher.load_sources()

    print("\n[2/4] Running deterministic multi-source reconciliation...")
    result = matcher.reconcile()

    source_records = {
        "ledger":     {r["record_id"]: r for r in matcher._ledger_records},
        "bank":       {r["record_id"]: r for r in matcher._bank_records},
        "invoice":    {r["record_id"]: r for r in matcher._invoice_records},
        "settlement": {r["record_id"]: r for r in matcher._settlement_records},
    }

    # ── 2. Exception Agent ────────────────────────────────────────────────
    print("\n[3/4] Running Exception Agent on residual cases...")
    agent = ExceptionAgent()
    analyses = agent.analyze_residuals(result.decisions, source_records)

    # ── 3. Evaluate ───────────────────────────────────────────────────────
    ev = evaluate(result)

    # ── 4. Reports ────────────────────────────────────────────────────────
    print("\n[4/4] Generating final reports and exporting exception JSON...")
    json_path, md_path = generate_final_report(result, ev, analyses)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — BASELINE RECONCILIATION RESULTS (Operational)
    # ══════════════════════════════════════════════════════════════════════
    _section("BASELINE RECONCILIATION RESULTS")
    total = result.total_processed
    print(f"  Ledger anchor records processed : {total}")
    print(f"  ─")
    print(f"  Operational workflow outcomes:")
    print(f"    MATCHED   (all 3 sources found) : {result.status_counts.get('MATCHED', 0)}")
    print(f"    PARTIAL   (some sources absent) : {result.status_counts.get('PARTIAL', 0)}")
    print(f"    EXCEPTION (ambiguity/discrepancy): {result.status_counts.get('EXCEPTION', 0)}")
    print(f"    UNRESOLVED (no match found)      : {result.status_counts.get('UNRESOLVED', 0)}")
    print(f"  ─")
    print(f"  Operational coverage (MATCHED+PARTIAL / total) : {ev.operational_coverage * 100:.1f}%")

    # ── Agent layer
    det_matched  = result.status_counts.get("MATCHED", 0)
    agent_auto   = sum(1 for a in analyses if a.safe_auto_resolved)
    escalated    = len(analyses) - agent_auto
    eff_resolved = det_matched + agent_auto
    op_rate      = eff_resolved / total * 100 if total else 0.0
    print(f"  ─")
    print(f"  After Exception Agent:")
    print(f"    Deterministic MATCHED            : {det_matched}")
    print(f"    Agent safely auto-resolved       : {agent_auto}")
    print(f"    Still escalated to Finance Ops   : {escalated}")
    print(f"    Total effective automation rate  : {op_rate:.1f}%  ({eff_resolved}/{total})")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — EVALUATION AGAINST GROUND TRUTH
    # ══════════════════════════════════════════════════════════════════════
    _section("EVALUATION AGAINST GROUND TRUTH")
    print(f"  Ground-truth universe:")
    print(f"    Canonical transactions (total)  : {ev.total_canonical}")
    print(f"    GT status=matched               : {ev.gt_matched_count}")
    print(f"    GT status=unresolved (total)    : {ev.gt_unresolved_total}")
    print(f"    GT unresolved, evaluable        : {ev.gt_unresolved_evaluable}")
    print()
    print(f"  Correctness classification:")
    print(f"    Correct full matches            : {ev.correct_full_matches}")
    print(f"      (GT=matched, MATCHED, all IDs correct)")
    print(f"    Correct partial detections      : {ev.correct_partial_detections}")
    print(f"      (GT=matched, PARTIAL, claimed IDs correct — planted exception on missing source)")
    print(f"    Correctly escalated             : {ev.correctly_escalated}")
    print(f"      (GT=unresolved, PARTIAL/EXCEPTION/UNRESOLVED, no wrong claims)")
    print(f"    Incorrect automatic matches     : {ev.incorrect_full_matches}")
    print(f"      (Claimed source ID not supported by ground truth — genuine FP)")
    print(f"    Missed resolvable transactions  : {ev.missed_resolvable}")
    print(f"      (GT=matched but matcher returned EXCEPTION/UNRESOLVED — genuine FN)")
    print(f"    Incorrectly auto-resolved       : {ev.incorrectly_auto_resolved}")
    print(f"      (GT=unresolved but matcher claimed MATCHED with wrong IDs)")
    print()
    print(f"  Precision  = correct_full_matches / (correct + incorrect)")
    print(f"             = {ev.correct_full_matches} / {ev.correct_full_matches + ev.incorrect_full_matches}")
    print(f"             = {ev.match_precision * 100:.1f}%")
    print()
    print(f"  Recall     = (correct_full + correct_partial) / gt_matched_evaluable")
    print(f"             = ({ev.correct_full_matches} + {ev.correct_partial_detections}) / {ev.gt_matched_count}")
    print(f"             = {ev.match_recall * 100:.1f}%")
    print()
    print(f"  F1         = {ev.match_f1 * 100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — EXCEPTION HANDLING SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    _section("EXCEPTION HANDLING SUMMARY")
    print(f"  Exception detection rate (GT-unresolved correctly escalated):")
    print(f"    {ev.correctly_escalated} / {ev.gt_unresolved_evaluable} = {ev.exception_detection_rate * 100:.1f}%")
    print()
    print(f"  {'Exception Type':<28}  {'GT':>4}  {'Correct':>8}  {'Escalated':>9}  {'Missed':>6}  {'IncorrectRes':>12}")
    print(f"  {'─'*28}  {'─'*4}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*12}")
    for etype, counts in ev.exception_breakdown.items():
        handled = counts.get("correctly_handled", 0)
        escal   = counts.get("correctly_escalated", 0)
        missed  = counts.get("missed", 0)
        inc     = counts.get("incorrectly_resolved", 0)
        gt_cnt  = counts.get("gt_count", 0)
        print(f"  {etype:<28}  {gt_cnt:>4}  {handled:>8}  {escal:>9}  {missed:>6}  {inc:>12}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — MISSING LEDGER ANCHOR
    # ══════════════════════════════════════════════════════════════════════
    _section("MISSING LEDGER ANCHOR REPORT")
    print(f"  Canonical transactions with no ledger record : {ev.ledger_missing_in_gt}")
    print()
    print(f"  Detail:")
    print(f"    CAN-0090  match_status=unresolved  exception=missing_source_record(ledger)")
    print(f"    This transaction has no ledger entry and therefore cannot enter the")
    print(f"    ledger-anchored reconciliation workflow.  It is excluded from all")
    print(f"    precision/recall/coverage denominators and reported here separately.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — MANUALLY INSPECTED EXAMPLE CASES
    # ══════════════════════════════════════════════════════════════════════
    _section("MANUALLY INSPECTED EXAMPLE CASES")
    PROBE = {"LED-0004", "LED-0005", "LED-0026", "LED-0029",
             "LED-0035", "LED-0045", "LED-0049", "LED-0061"}

    for entry in ev.exception_entries:
        if entry.ledger_id in PROBE:
            print(f"  {entry.ledger_id}")
            print(f"    Matcher status   : {entry.matcher_status}")
            print(f"    GT status        : {entry.gt_status}")
            print(f"    Exception type   : {entry.exception_type} ({entry.exception_source})")
            print(f"    Bank ID ok?      : {entry.bank_id_match}")
            print(f"    Invoice ID ok?   : {entry.inv_id_match}")
            print(f"    Settlement ID ok?: {entry.stl_id_match}")
            print(f"    Classification   : {entry.classification}")
            print()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 — THROUGHPUT & REPORTS
    # ══════════════════════════════════════════════════════════════════════
    _section("THROUGHPUT & OUTPUT")
    print(f"  Processing throughput : {result.throughput_per_second:.0f} records/sec")
    print(f"  Elapsed time          : {result.elapsed_seconds * 1000:.1f} ms")
    print(f"  Reports generated     : {json_path.name}, {md_path.name}")

    print(f"\n{LINE}")
    print("  Pipeline complete.")
    print(LINE + "\n")


if __name__ == "__main__":
    main()
