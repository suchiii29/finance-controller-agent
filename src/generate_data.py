"""
generate_data.py
================
Synthetic financial data generator for the AI Finance Controller project.

Produces four source CSVs (bank_statements, ledger, invoices, settlements)
and one ground-truth mapping CSV, all traceable to 100 canonical transactions.

~20% of canonical transactions have deliberately planted exceptions.

Usage:
    python -m src.generate_data
"""

from __future__ import annotations

import os
import random
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED: int = 42
N_CANONICAL: int = 100
EXCEPTION_FRACTION: float = 0.20   # ~20 exceptions out of 100

DATA_DIR: Path = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Exception types and their approximate weights (must sum to 1.0)
# ---------------------------------------------------------------------------

EXCEPTION_TYPES = [
    "amount_mismatch",          # ± small amount difference
    "date_drift",               # 1–3 day shift in one source
    "missing_reference",        # ref/invoice ID stripped from one source
    "duplicate_reference",      # same reference used twice in one source
    "missing_tax_line",         # tax_line absent in ledger or invoice
    "fuzzy_counterparty",       # name variation (Ltd vs Limited, etc.)
    "missing_source_record",    # entire record absent from one source
]

EXCEPTION_WEIGHTS = [0.20, 0.20, 0.15, 0.10, 0.15, 0.10, 0.10]

# Counterparty name fuzzing pairs
FUZZY_NAME_PAIRS: list[tuple[str, str]] = [
    ("Limited", "Ltd"),
    ("Private Limited", "Pvt Ltd"),
    ("Corporation", "Corp"),
    ("Technologies", "Tech"),
    ("Solutions", "Soln"),
    ("Incorporated", "Inc"),
    ("& Sons", "and Sons"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fake = Faker("en_IN")
fake.seed_instance(SEED)
rng = random.Random(SEED)


def _pick_exception_types(n: int) -> list[str]:
    """Return `n` exception type strings drawn with replacement by weight."""
    return rng.choices(EXCEPTION_TYPES, weights=EXCEPTION_WEIGHTS, k=n)


def _fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _drift_date(d: date, max_days: int = 3) -> date:
    """Shift a date by 1–max_days in either direction."""
    delta = rng.randint(1, max_days) * rng.choice([-1, 1])
    return d + timedelta(days=delta)


def _amount_mismatch(amount: float) -> float:
    """Return a slightly different amount (±0.5 to 50)."""
    delta = round(rng.uniform(0.5, 50.0), 2)
    return round(amount + rng.choice([-1, 1]) * delta, 2)


def _fuzz_name(name: str) -> str:
    """Apply one random fuzzy substitution if possible."""
    for full, abbr in FUZZY_NAME_PAIRS:
        if full in name:
            return name.replace(full, abbr, 1)
        if abbr in name:
            return name.replace(abbr, full, 1)
    # fallback: append " & Co"
    return name + " & Co"


def _make_counterparty() -> str:
    """Generate a realistic Indian company name."""
    return fake.company()


def _make_description(counterparty: str, amount: float) -> str:
    purposes = [
        "Payment for services rendered",
        "Invoice settlement",
        "Monthly retainer",
        "Software subscription",
        "Consulting fee",
        "Vendor payment",
        "Professional services",
        "Annual license fee",
        "Maintenance contract",
        "Project milestone payment",
    ]
    return f"{rng.choice(purposes)} - {counterparty}"


def _make_reference() -> str:
    """Generate a REF-XXXXXXXX style reference string."""
    return f"REF-{rng.randint(10000000, 99999999)}"


def _make_tax_line(amount: float) -> str:
    """GST at 18% formatted as a string."""
    gst = round(amount * 0.18, 2)
    return f"GST@18%={gst}"


# ---------------------------------------------------------------------------
# Canonical transaction generation
# ---------------------------------------------------------------------------

def build_canonical(n: int) -> pd.DataFrame:
    """
    Build `n` canonical ground-truth transactions.

    Each row is the single authoritative record of a financial event.

    Returns
    -------
    pd.DataFrame
        Columns: canonical_id, date, amount, description, counterparty,
                 tax_line, reference
    """
    start_date = date(2024, 1, 1)
    end_date   = date(2024, 6, 30)
    date_range = (end_date - start_date).days

    rows: list[dict] = []
    for i in range(1, n + 1):
        d          = start_date + timedelta(days=rng.randint(0, date_range))
        amount     = round(rng.uniform(500.0, 500_000.0), 2)
        counterparty = _make_counterparty()
        desc       = _make_description(counterparty, amount)
        reference  = _make_reference()
        tax_line   = _make_tax_line(amount)

        rows.append({
            "canonical_id": f"CAN-{i:04d}",
            "date":          _fmt_date(d),
            "amount":        amount,
            "description":   desc,
            "counterparty":  counterparty,
            "tax_line":      tax_line,
            "reference":     reference,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exception assignment
# ---------------------------------------------------------------------------

def assign_exceptions(canonical: pd.DataFrame) -> dict[str, dict]:
    """
    Choose ~20% of canonical transactions to receive a planted exception.

    Returns a dict keyed by canonical_id mapping to exception metadata:
        {
            "exception_type": str,
            "affected_source": str,  # which source gets the corrupted record
        }
    """
    n_exceptions = math.ceil(len(canonical) * EXCEPTION_FRACTION)
    chosen_ids   = rng.sample(list(canonical["canonical_id"]), k=n_exceptions)
    types        = _pick_exception_types(n_exceptions)

    sources = ["bank", "ledger", "invoice", "settlement"]
    exceptions: dict[str, dict] = {}
    for cid, etype in zip(chosen_ids, types):
        # missing_source_record always removes from one source; others corrupt
        affected = rng.choice(sources)
        exceptions[cid] = {
            "exception_type":   etype,
            "affected_source":  affected,
        }
    return exceptions


# ---------------------------------------------------------------------------
# Source record builders
# ---------------------------------------------------------------------------

def build_bank_statements(
    canonical: pd.DataFrame,
    exceptions: dict[str, dict],
) -> pd.DataFrame:
    """
    Build bank_statements.csv from canonical transactions.

    Introduces:
        - amount_mismatch  → corrupted amount
        - date_drift       → shifted value_date
        - missing_reference → ref stripped
        - duplicate_reference → ref repeated from previous row
        - fuzzy_counterparty → name variant
        - missing_source_record → row entirely absent
    (missing_tax_line is not planted here; bank statements rarely carry tax)
    """
    rows: list[dict] = []
    prev_ref: Optional[str] = None

    for _, row in canonical.iterrows():
        cid   = row["canonical_id"]
        ex    = exceptions.get(cid)
        etype = ex["exception_type"]   if ex else None
        asrc  = ex["affected_source"]  if ex else None

        if etype == "missing_source_record" and asrc == "bank":
            continue  # deliberately skip

        txn_date    = date.fromisoformat(row["date"])
        amount      = float(row["amount"])
        counterparty = row["counterparty"]
        reference   = row["reference"]

        if etype == "amount_mismatch" and asrc == "bank":
            amount = _amount_mismatch(amount)
        if etype == "date_drift" and asrc == "bank":
            txn_date = _drift_date(txn_date)
        if etype == "missing_reference" and asrc == "bank":
            reference = ""
        if etype == "duplicate_reference" and asrc == "bank":
            reference = prev_ref if prev_ref else reference
        if etype == "fuzzy_counterparty" and asrc == "bank":
            counterparty = _fuzz_name(counterparty)

        bank_id = f"BNK-{cid[4:]}"   # e.g. BNK-0001
        rows.append({
            "bank_txn_id":   bank_id,
            "canonical_id":  cid,           # retained only for GT linking
            "value_date":    _fmt_date(txn_date),
            "amount":        amount,
            "description":   row["description"],
            "counterparty":  counterparty,
            "reference":     reference,
            "bank_channel":  rng.choice(["NEFT", "RTGS", "IMPS", "UPI"]),
        })
        prev_ref = reference

    df = pd.DataFrame(rows)
    # Drop canonical_id column – sources must NOT directly reference it
    return df.drop(columns=["canonical_id"])


def build_ledger(
    canonical: pd.DataFrame,
    exceptions: dict[str, dict],
) -> pd.DataFrame:
    """
    Build ledger.csv from canonical transactions.

    Ledger uses accounting-style fields and may carry different date
    (posting date vs transaction date).
    """
    rows: list[dict] = []
    prev_ref: Optional[str] = None

    for _, row in canonical.iterrows():
        cid   = row["canonical_id"]
        ex    = exceptions.get(cid)
        etype = ex["exception_type"]   if ex else None
        asrc  = ex["affected_source"]  if ex else None

        if etype == "missing_source_record" and asrc == "ledger":
            continue

        txn_date     = date.fromisoformat(row["date"])
        posting_date = txn_date + timedelta(days=rng.randint(0, 2))  # natural lag
        amount       = float(row["amount"])
        counterparty = row["counterparty"]
        reference    = row["reference"]
        tax_line     = row["tax_line"]

        if etype == "amount_mismatch" and asrc == "ledger":
            amount = _amount_mismatch(amount)
        if etype == "date_drift" and asrc == "ledger":
            posting_date = _drift_date(posting_date)
        if etype == "missing_reference" and asrc == "ledger":
            reference = ""
        if etype == "duplicate_reference" and asrc == "ledger":
            reference = prev_ref if prev_ref else reference
        if etype == "missing_tax_line" and asrc == "ledger":
            tax_line = ""
        if etype == "fuzzy_counterparty" and asrc == "ledger":
            counterparty = _fuzz_name(counterparty)

        ledger_id = f"LED-{cid[4:]}"
        rows.append({
            "ledger_entry_id": ledger_id,
            "canonical_id":    cid,
            "posting_date":    _fmt_date(posting_date),
            "txn_date":        _fmt_date(txn_date),
            "debit_credit":    rng.choice(["DR", "CR"]),
            "amount":          amount,
            "counterparty":    counterparty,
            "tax_line":        tax_line,
            "reference":       reference,
            "cost_center":     rng.choice(["CC-SALES", "CC-OPS", "CC-TECH", "CC-ADMIN"]),
        })
        prev_ref = reference

    df = pd.DataFrame(rows)
    return df.drop(columns=["canonical_id"])


def build_invoices(
    canonical: pd.DataFrame,
    exceptions: dict[str, dict],
) -> pd.DataFrame:
    """
    Build invoices.csv from canonical transactions.

    Invoices carry invoice numbers, due dates, and tax details.
    """
    rows: list[dict] = []
    prev_ref: Optional[str] = None

    for _, row in canonical.iterrows():
        cid   = row["canonical_id"]
        ex    = exceptions.get(cid)
        etype = ex["exception_type"]   if ex else None
        asrc  = ex["affected_source"]  if ex else None

        if etype == "missing_source_record" and asrc == "invoice":
            continue

        invoice_date = date.fromisoformat(row["date"])
        due_date     = invoice_date + timedelta(days=rng.randint(7, 30))
        amount       = float(row["amount"])
        counterparty = row["counterparty"]
        reference    = row["reference"]
        tax_line     = row["tax_line"]

        if etype == "amount_mismatch" and asrc == "invoice":
            amount = _amount_mismatch(amount)
        if etype == "date_drift" and asrc == "invoice":
            invoice_date = _drift_date(invoice_date)
        if etype == "missing_reference" and asrc == "invoice":
            reference = ""
        if etype == "duplicate_reference" and asrc == "invoice":
            reference = prev_ref if prev_ref else reference
        if etype == "missing_tax_line" and asrc == "invoice":
            tax_line = ""
        if etype == "fuzzy_counterparty" and asrc == "invoice":
            counterparty = _fuzz_name(counterparty)

        inv_id = f"INV-{cid[4:]}"
        rows.append({
            "invoice_id":      inv_id,
            "canonical_id":    cid,
            "invoice_date":    _fmt_date(invoice_date),
            "due_date":        _fmt_date(due_date),
            "amount":          amount,
            "counterparty":    counterparty,
            "tax_line":        tax_line,
            "reference":       reference,
            "invoice_status":  rng.choice(["PAID", "PENDING", "OVERDUE"]),
        })
        prev_ref = reference

    df = pd.DataFrame(rows)
    return df.drop(columns=["canonical_id"])


def build_settlements(
    canonical: pd.DataFrame,
    exceptions: dict[str, dict],
) -> pd.DataFrame:
    """
    Build settlements.csv from canonical transactions.

    Settlement records represent payment gateway / clearing house data.
    They carry settlement IDs and UTR numbers.
    """
    rows: list[dict] = []
    prev_ref: Optional[str] = None

    for _, row in canonical.iterrows():
        cid   = row["canonical_id"]
        ex    = exceptions.get(cid)
        etype = ex["exception_type"]   if ex else None
        asrc  = ex["affected_source"]  if ex else None

        if etype == "missing_source_record" and asrc == "settlement":
            continue

        settlement_date = date.fromisoformat(row["date"]) + timedelta(days=rng.randint(0, 1))
        amount          = float(row["amount"])
        counterparty    = row["counterparty"]
        reference       = row["reference"]

        if etype == "amount_mismatch" and asrc == "settlement":
            amount = _amount_mismatch(amount)
        if etype == "date_drift" and asrc == "settlement":
            settlement_date = _drift_date(settlement_date)
        if etype == "missing_reference" and asrc == "settlement":
            reference = ""
        if etype == "duplicate_reference" and asrc == "settlement":
            reference = prev_ref if prev_ref else reference
        if etype == "fuzzy_counterparty" and asrc == "settlement":
            counterparty = _fuzz_name(counterparty)

        stl_id = f"STL-{cid[4:]}"
        utr    = f"UTR{rng.randint(100000000000, 999999999999)}"
        rows.append({
            "settlement_id":   stl_id,
            "canonical_id":    cid,
            "settlement_date": _fmt_date(settlement_date),
            "amount":          amount,
            "counterparty":    counterparty,
            "reference":       reference,
            "utr_number":      utr,
            "payment_mode":    rng.choice(["NEFT", "RTGS", "IMPS", "Cheque", "UPI"]),
            "status":          rng.choice(["SETTLED", "PENDING", "FAILED"]),
        })
        prev_ref = reference

    df = pd.DataFrame(rows)
    return df.drop(columns=["canonical_id"])


# ---------------------------------------------------------------------------
# Ground-truth table builder
# ---------------------------------------------------------------------------

def build_ground_truth(
    canonical: pd.DataFrame,
    exceptions: dict[str, dict],
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invoices: pd.DataFrame,
    settlements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ground_truth.csv linking every canonical transaction to its
    corresponding source records, flagging exceptions and unresolved cases.

    This file is for EVALUATION ONLY and must not be used during inference.
    """
    # Build ID presence sets from each source (using the deterministic IDs)
    bank_ids       = set(bank["bank_txn_id"])
    ledger_ids     = set(ledger["ledger_entry_id"])
    invoice_ids    = set(invoices["invoice_id"])
    settlement_ids = set(settlements["settlement_id"])

    rows: list[dict] = []
    for _, row in canonical.iterrows():
        cid  = row["canonical_id"]
        idx  = cid[4:]   # e.g. "0001"
        ex   = exceptions.get(cid)
        etype = ex["exception_type"]  if ex else None
        asrc  = ex["affected_source"] if ex else None

        bank_id  = f"BNK-{idx}" if f"BNK-{idx}" in bank_ids  else ""
        led_id   = f"LED-{idx}" if f"LED-{idx}" in ledger_ids else ""
        inv_id   = f"INV-{idx}" if f"INV-{idx}" in invoice_ids else ""
        stl_id   = f"STL-{idx}" if f"STL-{idx}" in settlement_ids else ""

        # A record is "unresolved" if a source record is missing entirely
        # or if there's an unrecoverable exception (duplicate ref makes
        # match ambiguous; amount mismatch beyond tolerance = unresolved)
        missing_count = sum([bank_id == "", led_id == "", inv_id == "", stl_id == ""])
        is_unresolved = (
            missing_count > 0
            or etype in ("missing_source_record", "duplicate_reference")
        )

        match_status = "unresolved" if is_unresolved else "matched"

        rows.append({
            "canonical_id":        cid,
            "canonical_date":      row["date"],
            "canonical_amount":    row["amount"],
            "canonical_reference": row["reference"],
            "bank_record_id":      bank_id,
            "ledger_record_id":    led_id,
            "invoice_record_id":   inv_id,
            "settlement_record_id":stl_id,
            "match_status":        match_status,
            "exception_type":      etype if etype else "none",
            "exception_source":    asrc  if asrc  else "none",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    canonical: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invoices: pd.DataFrame,
    settlements: pd.DataFrame,
    ground_truth: pd.DataFrame,
    exceptions: dict[str, dict],
) -> None:
    """
    Run post-generation validation checks.

    Raises AssertionError on any failure.
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 1. All required files exist
    required_files = [
        DATA_DIR / "bank_statements.csv",
        DATA_DIR / "ledger.csv",
        DATA_DIR / "invoices.csv",
        DATA_DIR / "settlements.csv",
        DATA_DIR / "ground_truth.csv",
    ]
    for f in required_files:
        assert f.exists(), f"Missing file: {f}"
    print("[PASS] All 5 required CSV files exist")

    # 2. Required columns exist
    bank_required   = {"bank_txn_id", "value_date", "amount", "counterparty", "reference"}
    ledger_required = {"ledger_entry_id", "posting_date", "amount", "counterparty", "reference", "tax_line"}
    inv_required    = {"invoice_id", "invoice_date", "amount", "counterparty", "reference", "tax_line"}
    stl_required    = {"settlement_id", "settlement_date", "amount", "counterparty", "reference"}
    gt_required     = {
        "canonical_id", "canonical_amount", "bank_record_id", "ledger_record_id",
        "invoice_record_id", "settlement_record_id", "match_status", "exception_type",
    }

    assert bank_required.issubset(bank.columns),       f"Bank missing cols: {bank_required - set(bank.columns)}"
    assert ledger_required.issubset(ledger.columns),   f"Ledger missing cols: {ledger_required - set(ledger.columns)}"
    assert inv_required.issubset(invoices.columns),    f"Invoice missing cols: {inv_required - set(invoices.columns)}"
    assert stl_required.issubset(settlements.columns), f"Settlement missing cols: {stl_required - set(settlements.columns)}"
    assert gt_required.issubset(ground_truth.columns), f"GT missing cols: {gt_required - set(ground_truth.columns)}"
    print("[PASS] All required columns present in every source")

    # 3. No unexpected nulls in required identifiers
    assert bank["bank_txn_id"].notna().all(),           "Null bank_txn_id found"
    assert ledger["ledger_entry_id"].notna().all(),     "Null ledger_entry_id found"
    assert invoices["invoice_id"].notna().all(),        "Null invoice_id found"
    assert settlements["settlement_id"].notna().all(),  "Null settlement_id found"
    assert ground_truth["canonical_id"].notna().all(),  "Null canonical_id in GT"
    print("[PASS] No unexpected nulls in required identifier columns")

    # 4. Dates are valid
    for df, col, name in [
        (bank, "value_date", "bank"),
        (ledger, "posting_date", "ledger"),
        (invoices, "invoice_date", "invoices"),
        (settlements, "settlement_date", "settlements"),
    ]:
        pd.to_datetime(df[col], format="%Y-%m-%d")   # raises on bad dates
    print("[PASS] Dates are valid ISO-8601 in all sources")

    # 5. Amounts are numeric and non-negative
    for df, name in [(bank, "bank"), (ledger, "ledger"), (invoices, "invoices"), (settlements, "settlements")]:
        assert pd.to_numeric(df["amount"], errors="coerce").notna().all(), f"Non-numeric amount in {name}"
        assert (pd.to_numeric(df["amount"]) >= 0).all(), f"Negative amount in {name}"
    print("[PASS] All amounts are numeric and non-negative")

    # 6. Canonical transaction count
    assert len(canonical) == N_CANONICAL, f"Expected {N_CANONICAL} canonical rows, got {len(canonical)}"
    assert len(ground_truth) == N_CANONICAL, f"GT row count mismatch: {len(ground_truth)}"
    print(f"[PASS] Canonical transaction count = {N_CANONICAL}")

    # 7. Planted exception counts match design
    n_exceptions_actual = sum(1 for v in ground_truth["exception_type"] if v != "none")
    assert n_exceptions_actual == len(exceptions), (
        f"GT exception count {n_exceptions_actual} != assigned {len(exceptions)}"
    )
    print(f"[PASS] Exception count matches design ({n_exceptions_actual} exceptions)")

    # 8. Ground-truth mappings are internally consistent
    # Every non-empty bank_record_id must exist in bank source
    bank_ids = set(bank["bank_txn_id"])
    led_ids  = set(ledger["ledger_entry_id"])
    inv_ids  = set(invoices["invoice_id"])
    stl_ids  = set(settlements["settlement_id"])
    for _, gt_row in ground_truth.iterrows():
        if gt_row["bank_record_id"]:
            assert gt_row["bank_record_id"] in bank_ids, f"GT references non-existent bank record {gt_row['bank_record_id']}"
        if gt_row["ledger_record_id"]:
            assert gt_row["ledger_record_id"] in led_ids, f"GT references non-existent ledger record {gt_row['ledger_record_id']}"
        if gt_row["invoice_record_id"]:
            assert gt_row["invoice_record_id"] in inv_ids, f"GT references non-existent invoice record {gt_row['invoice_record_id']}"
        if gt_row["settlement_record_id"]:
            assert gt_row["settlement_record_id"] in stl_ids, f"GT references non-existent settlement record {gt_row['settlement_record_id']}"
    print("[PASS] Ground-truth mappings are internally consistent")

    # 9. No accidental duplicate canonical IDs
    assert ground_truth["canonical_id"].nunique() == len(ground_truth), "Duplicate canonical_ids in GT"
    print("[PASS] No duplicate canonical IDs in ground truth")

    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    canonical: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invoices: pd.DataFrame,
    settlements: pd.DataFrame,
    ground_truth: pd.DataFrame,
    exceptions: dict[str, dict],
) -> None:
    """Print a human-readable generation summary to stdout."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    print(f"\n  Random seed (fixed)      : {SEED}")
    print(f"  Canonical transactions   : {len(canonical)}")
    print(f"\n  Rows per source:")
    print(f"    bank_statements.csv    : {len(bank)}")
    print(f"    ledger.csv             : {len(ledger)}")
    print(f"    invoices.csv           : {len(invoices)}")
    print(f"    settlements.csv        : {len(settlements)}")
    print(f"    ground_truth.csv       : {len(ground_truth)}")

    n_exc = len(exceptions)
    print(f"\n  Planted exceptions total : {n_exc}")

    # Count by type
    from collections import Counter
    type_counter = Counter(v["exception_type"] for v in exceptions.values())
    for etype, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"    {etype:<30}: {cnt}")

    # Missing record count (missing_source_record exception type)
    n_missing = sum(1 for v in exceptions.values() if v["exception_type"] == "missing_source_record")
    print(f"\n  Canonical IDs with missing source record : {n_missing}")

    # Unresolved in GT
    n_unresolved = (ground_truth["match_status"] == "unresolved").sum()
    print(f"  Unresolved in ground truth               : {n_unresolved}")
    print(f"  Matched in ground truth                  : {len(ground_truth) - n_unresolved}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end data generation pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating canonical transactions...")
    canonical = build_canonical(N_CANONICAL)

    print("Assigning exceptions...")
    exceptions = assign_exceptions(canonical)

    print("Building source tables...")
    bank        = build_bank_statements(canonical, exceptions)
    ledger      = build_ledger(canonical, exceptions)
    invoices    = build_invoices(canonical, exceptions)
    settlements = build_settlements(canonical, exceptions)
    ground_truth = build_ground_truth(canonical, exceptions, bank, ledger, invoices, settlements)

    print("Saving CSV files...")
    bank.to_csv(DATA_DIR / "bank_statements.csv", index=False)
    ledger.to_csv(DATA_DIR / "ledger.csv", index=False)
    invoices.to_csv(DATA_DIR / "invoices.csv", index=False)
    settlements.to_csv(DATA_DIR / "settlements.csv", index=False)
    ground_truth.to_csv(DATA_DIR / "ground_truth.csv", index=False)

    print_summary(canonical, bank, ledger, invoices, settlements, ground_truth, exceptions)
    validate(canonical, bank, ledger, invoices, settlements, ground_truth, exceptions)


if __name__ == "__main__":
    main()
