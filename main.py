"""
main.py
=======
Finance Controller Agent — reconciliation pipeline entry point.

Usage:
    python main.py
"""

from __future__ import annotations

from src.matcher  import ReconciliationMatcher
from src.evaluate import evaluate


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 62
SEP2 = "─" * 62


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _print_decision(dec, label: str) -> None:
    bank_id = dec.bank_match.record_id or "(none)"
    inv_id  = dec.invoice_match.record_id or "(none)"
    stl_id  = dec.settlement_match.record_id or "(none)"
    print(f"  [{label}]  Ledger: {dec.ledger_id}")
    print(f"    Status     : {dec.status}  (conf={dec.confidence:.2f}, tier={dec.tier})")
    print(f"    Bank       : {bank_id}")
    print(f"    Invoice    : {inv_id}")
    print(f"    Settlement : {stl_id}")
    print(f"    Reason     : {dec.reason[:100]}")
    print(f"    Action     : {dec.recommended_action}")
    print()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  FINANCE CONTROLLER AGENT — Reconciliation Pipeline")
    print(SEP)

    # ── Step 1 & 2: Load and normalise ──────────────────────────────────────
    print("\n[1/4] Loading source files and normalising records...")
    matcher = ReconciliationMatcher()
    matcher.load_sources()

    n_led = len(matcher._ledger_records)
    n_bnk = len(matcher._bank_records)
    n_inv = len(matcher._invoice_records)
    n_stl = len(matcher._settlement_records)
    print(f"      Ledger records   : {n_led}")
    print(f"      Bank records     : {n_bnk}")
    print(f"      Invoice records  : {n_inv}")
    print(f"      Settlement records: {n_stl}")

    # ── Step 3: Reconcile ────────────────────────────────────────────────────
    print("\n[2/4] Running deterministic reconciliation...")
    result = matcher.reconcile()

    print(f"      Processed        : {result.total_processed}")
    print(f"      Elapsed          : {result.elapsed_seconds:.3f}s")
    print(f"      Throughput       : {result.throughput_per_second:.0f} rec/s")

    sc = result.status_counts
    print(f"\n      MATCHED          : {sc['MATCHED']}")
    print(f"      PARTIAL          : {sc['PARTIAL']}")
    print(f"      EXCEPTION        : {sc['EXCEPTION']}")
    print(f"      UNRESOLVED       : {sc['UNRESOLVED']}")

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    print("\n[3/4] Evaluating against ground truth...")
    ev = evaluate(result)

    print("\n" + SEP)
    print("  EVALUATION SUMMARY")
    print(SEP)
    print(f"  Records evaluated           : {ev.total_evaluated}")
    print(f"  (Canonical with no ledger)  : {ev.ledger_missing_in_gt}")
    print()
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {SEP2}")
    print(f"  {'MATCHED':<35} {sc['MATCHED']:>10}")
    print(f"  {'PARTIAL':<35} {sc['PARTIAL']:>10}")
    print(f"  {'EXCEPTION':<35} {sc['EXCEPTION']:>10}")
    print(f"  {'UNRESOLVED':<35} {sc['UNRESOLVED']:>10}")
    print(f"  {SEP2}")
    print(f"  {'Operational coverage':<35} {_pct(ev.operational_coverage):>10}")
    print(f"  {SEP2}")
    print(f"  {'True positives (correct matches)':<35} {ev.true_positives:>10}")
    print(f"  {'False positives (wrong matches)':<35} {ev.false_positives:>10}")
    print(f"  {'False negatives (missed matches)':<35} {ev.false_negatives:>10}")
    print(f"  {'True negatives (correct flags)':<35} {ev.true_negatives:>10}")
    print(f"  {SEP2}")
    print(f"  {'Match precision':<35} {_pct(ev.precision):>10}")
    print(f"  {'Match recall':<35} {_pct(ev.recall):>10}")
    print(f"  {'Match F1':<35} {_pct(ev.f1):>10}")
    print(f"  {'Exception identification accuracy':<35} {_pct(ev.exception_id_accuracy):>10}")
    print(f"  {SEP2}")
    print(f"  {'Correctly matched':<35} {ev.correctly_matched:>10}")
    print(f"  {'Incorrectly matched':<35} {ev.incorrectly_matched:>10}")
    print(f"  {'Correctly flagged unresolved':<35} {ev.correctly_unresolved:>10}")

    # Exception breakdown by type
    print(f"\n  Exception breakdown by planted type:")
    print(f"  {'Type':<30} {'Correct':>8} {'Missed':>8} {'Over-flagged':>13}")
    print(f"  {'─'*60}")
    for etype, counts in sorted(ev.exception_breakdown.items()):
        print(
            f"  {etype:<30} {counts['correct']:>8} "
            f"{counts['missed']:>8} {counts['over_flagged']:>13}"
        )

    # ── Step 5: Representative examples ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  REPRESENTATIVE EXAMPLES")
    print(SEP)

    decisions = result.decisions

    # 1. Successful Tier-1 exact match
    exact = next(
        (d for d in decisions
         if d.status == "MATCHED"
         and d.tier == 1
         and d.bank_match.tier == 1
         and d.invoice_match.tier == 1
         and d.settlement_match.tier == 1),
        None,
    )
    if exact:
        _print_decision(exact, "EXACT MATCH (Tier-1, all sources)")

    # 2. Strong fuzzy match
    fuzzy = next(
        (d for d in decisions
         if d.status == "MATCHED"
         and d.tier == 2),
        None,
    )
    if fuzzy:
        _print_decision(fuzzy, "FUZZY MATCH (Tier-2)")

    # 3. Partial (one source missing — e.g. missing_source_record)
    partial = next(
        (d for d in decisions if d.status == "PARTIAL"),
        None,
    )
    if partial:
        _print_decision(partial, "PARTIAL (missing source)")

    # 4. Exception — ambiguity or discrepancy
    exception = next(
        (d for d in decisions if d.status == "EXCEPTION"),
        None,
    )
    if exception:
        _print_decision(exception, "EXCEPTION")

    # 5. Unresolved
    unresolved = next(
        (d for d in decisions if d.status == "UNRESOLVED"),
        None,
    )
    if unresolved:
        _print_decision(unresolved, "UNRESOLVED")

    # 6. Show a few exception entries from evaluator
    print(f"{SEP2}")
    print("  Top exception entries (evaluator view):")
    print(f"{SEP2}")
    shown = 0
    for ee in ev.exception_entries[:8]:
        print(
            f"  {ee.ledger_id}  matcher={ee.matcher_status:<11} "
            f"gt={ee.gt_status:<10} exc_type={ee.exception_type}"
        )
        shown += 1
    if shown == 0:
        print("  (none)")

    print(f"\n{SEP}")
    print("  Pipeline complete.")
    print(SEP)


if __name__ == "__main__":
    main()
