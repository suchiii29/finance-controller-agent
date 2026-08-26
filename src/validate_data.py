"""
validate_data.py
================
Deep validation of the generated synthetic financial dataset.

Checks:
  1.  Source file existence
  2.  Record counts
  3.  Schema validation (required columns + types)
  4.  Ground-truth integrity
  5.  Planted exception coverage
  6.  Data quality (nulls, dupes, bad dates, bad amounts)
  7.  Reproducibility (run generator twice; compare outputs)

Usage:
    python -m src.validate_data
"""

from __future__ import annotations

import hashlib
import io
import sys
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

issues: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str, why: str, fix: str) -> None:
    full = f"{FAIL} {msg}\n        WHY : {why}\n        FIX : {fix}"
    print(f"  {full}")
    issues.append(msg)


def warn(msg: str) -> None:
    print(f"  {WARN} {msg}")


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    files = {
        "bank":         DATA_DIR / "bank_statements.csv",
        "ledger":       DATA_DIR / "ledger.csv",
        "invoices":     DATA_DIR / "invoices.csv",
        "settlements":  DATA_DIR / "settlements.csv",
        "ground_truth": DATA_DIR / "ground_truth.csv",
    }
    dfs = {}
    for key, path in files.items():
        dfs[key] = pd.read_csv(path)
    return dfs


# ---------------------------------------------------------------------------
# Check 1 – File existence
# ---------------------------------------------------------------------------

def check_files() -> bool:
    section("CHECK 1 – Source file existence")
    required = [
        "bank_statements.csv", "ledger.csv", "invoices.csv",
        "settlements.csv", "ground_truth.csv",
    ]
    all_ok = True
    for fname in required:
        p = DATA_DIR / fname
        size = p.stat().st_size if p.exists() else 0
        if p.exists():
            ok(f"{fname}  ({size:,} bytes)")
        else:
            fail(f"{fname} MISSING",
                 "Required source file not found on disk.",
                 "Re-run: python -m src.generate_data")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Check 2 – Record counts
# ---------------------------------------------------------------------------

def check_counts(dfs: dict[str, pd.DataFrame]) -> None:
    section("CHECK 2 – Record counts")
    b, l, inv, s, gt = (
        dfs["bank"], dfs["ledger"], dfs["invoices"],
        dfs["settlements"], dfs["ground_truth"],
    )
    print(f"    bank_statements.csv  : {len(b):>4} rows")
    print(f"    ledger.csv           : {len(l):>4} rows")
    print(f"    invoices.csv         : {len(inv):>4} rows")
    print(f"    settlements.csv      : {len(s):>4} rows")
    print(f"    ground_truth.csv     : {len(gt):>4} rows (canonical)")

    matched   = (gt["match_status"] == "matched").sum()
    unresolved = (gt["match_status"] == "unresolved").sum()
    print(f"\n    GT matched           : {matched}")
    print(f"    GT unresolved        : {unresolved}")

    # Completeness: rows with all 4 source records present
    complete = (
        (gt["bank_record_id"] != "") &
        (gt["ledger_record_id"] != "") &
        (gt["invoice_record_id"] != "") &
        (gt["settlement_record_id"] != "")
    ).sum()
    partial = len(gt) - complete
    print(f"\n    Complete (all 4 sources present) : {complete}")
    print(f"    Partial  (≥1 source missing)     : {partial}")
    ok("Record counts printed")


# ---------------------------------------------------------------------------
# Check 3 – Schema validation
# ---------------------------------------------------------------------------

COMMON_REQUIRED = {"amount", "counterparty", "reference"}

SCHEMA = {
    "bank": {
        "required": {"bank_txn_id", "value_date", "amount", "counterparty", "reference"},
        "source_specific": {"bank_channel"},
        "date_col": "value_date",
        "id_col": "bank_txn_id",
    },
    "ledger": {
        "required": {"ledger_entry_id", "posting_date", "txn_date", "amount",
                     "counterparty", "reference", "tax_line"},
        "source_specific": {"debit_credit", "cost_center"},
        "date_col": "posting_date",
        "id_col": "ledger_entry_id",
    },
    "invoices": {
        "required": {"invoice_id", "invoice_date", "due_date", "amount",
                     "counterparty", "reference", "tax_line"},
        "source_specific": {"invoice_status"},
        "date_col": "invoice_date",
        "id_col": "invoice_id",
    },
    "settlements": {
        "required": {"settlement_id", "settlement_date", "amount",
                     "counterparty", "reference"},
        "source_specific": {"utr_number", "payment_mode", "status"},
        "date_col": "settlement_date",
        "id_col": "settlement_id",
    },
}


def check_schema(dfs: dict[str, pd.DataFrame]) -> None:
    section("CHECK 3 – Schema validation")
    for src, meta in SCHEMA.items():
        df = dfs[src]
        cols = set(df.columns)
        missing = meta["required"] - cols
        if missing:
            fail(f"{src}: missing required columns {missing}",
                 "The normalisation layer depends on these columns.",
                 f"Add missing columns to build_{src}() in generate_data.py")
        else:
            ok(f"{src}: all required columns present → {sorted(meta['required'])}")

        extra = cols - meta["required"] - meta.get("source_specific", set())
        if extra:
            warn(f"{src}: unexpected extra columns {extra}")

        # Type checks
        numeric_ok = pd.to_numeric(df["amount"], errors="coerce").notna().all()
        if not numeric_ok:
            fail(f"{src}: non-numeric values in 'amount'",
                 "Downstream pandas arithmetic will raise or produce NaN.",
                 "Audit _amount_mismatch(); ensure no string is written.")
        else:
            ok(f"{src}: 'amount' column is fully numeric")

        # Date parseable
        try:
            pd.to_datetime(df[meta["date_col"]], format="%Y-%m-%d")
            ok(f"{src}: '{meta['date_col']}' parses as ISO-8601")
        except Exception as e:
            fail(f"{src}: bad date in '{meta['date_col']}': {e}",
                 "Invalid dates will crash the normalisation layer.",
                 "Ensure _drift_date() stays within calendar bounds.")

    # Ground-truth schema
    gt_required = {
        "canonical_id", "canonical_date", "canonical_amount",
        "canonical_reference", "bank_record_id", "ledger_record_id",
        "invoice_record_id", "settlement_record_id",
        "match_status", "exception_type", "exception_source",
    }
    gt_cols = set(dfs["ground_truth"].columns)
    missing_gt = gt_required - gt_cols
    if missing_gt:
        fail(f"ground_truth: missing columns {missing_gt}",
             "Evaluation scripts expect these columns.",
             "Add them in build_ground_truth().")
    else:
        ok(f"ground_truth: all required columns present")


# ---------------------------------------------------------------------------
# Check 4 – Ground-truth integrity
# ---------------------------------------------------------------------------

def check_ground_truth(dfs: dict[str, pd.DataFrame]) -> None:
    section("CHECK 4 – Ground-truth integrity")
    gt  = dfs["ground_truth"]
    b   = dfs["bank"]
    l   = dfs["ledger"]
    inv = dfs["invoices"]
    s   = dfs["settlements"]

    # Unique canonical IDs
    dupes = gt["canonical_id"].duplicated().sum()
    if dupes:
        fail(f"{dupes} duplicate canonical_ids in ground_truth",
             "Matcher evaluation will double-count those transactions.",
             "Ensure build_ground_truth() iterates canonical exactly once.")
    else:
        ok("All canonical_ids are unique in ground_truth")

    # Forward references valid
    bank_ids = set(b["bank_txn_id"])
    led_ids  = set(l["ledger_entry_id"])
    inv_ids  = set(inv["invoice_id"])
    stl_ids  = set(s["settlement_id"])

    bad_bank = gt[gt["bank_record_id"].ne("") & ~gt["bank_record_id"].isin(bank_ids)]
    bad_led  = gt[gt["ledger_record_id"].ne("") & ~gt["ledger_record_id"].isin(led_ids)]
    bad_inv  = gt[gt["invoice_record_id"].ne("") & ~gt["invoice_record_id"].isin(inv_ids)]
    bad_stl  = gt[gt["settlement_record_id"].ne("") & ~gt["settlement_record_id"].isin(stl_ids)]

    for label, bad in [("bank", bad_bank), ("ledger", bad_led),
                       ("invoice", bad_inv), ("settlement", bad_stl)]:
        if len(bad):
            fail(f"GT references {len(bad)} non-existent {label} record(s): {list(bad[f'{label}_record_id' if label!='ledger' else 'ledger_record_id'][:3])}",
                 "Evaluation would silently treat these as matched when they aren't.",
                 "Re-generate data with python -m src.generate_data")
        else:
            ok(f"GT → {label}: all foreign keys resolve correctly")

    # No source record mapped to >1 canonical (unless designed)
    for id_col, src_df, label in [
        ("bank_record_id",       gt, "bank"),
        ("ledger_record_id",     gt, "ledger"),
        ("invoice_record_id",    gt, "invoices"),
        ("settlement_record_id", gt, "settlements"),
    ]:
        non_empty = gt[gt[id_col].ne("")]
        multi = non_empty[non_empty[id_col].duplicated(keep=False)]
        if len(multi):
            fail(f"{len(multi)} {label} IDs appear in >1 GT row: {list(multi[id_col].unique()[:3])}",
                 "A source record cannot belong to two canonical transactions.",
                 "Audit duplicate_reference exception logic in build_ground_truth().")
        else:
            ok(f"No {label} source record is mapped to multiple canonical IDs")

    # Exception types represented in GT
    exc_in_gt = set(gt[gt["exception_type"] != "none"]["exception_type"].unique())
    expected_types = {
        "amount_mismatch", "date_drift", "missing_reference",
        "duplicate_reference", "missing_tax_line",
        "fuzzy_counterparty", "missing_source_record",
    }
    missing_types = expected_types - exc_in_gt
    if missing_types:
        warn(f"Exception types designed but absent from GT: {missing_types}")
    else:
        ok(f"All 7 designed exception types are present in ground_truth")


# ---------------------------------------------------------------------------
# Check 5 – Planted exception counts
# ---------------------------------------------------------------------------

def check_exceptions(dfs: dict[str, pd.DataFrame]) -> None:
    section("CHECK 5 – Planted exception counts")
    gt = dfs["ground_truth"]
    exc = gt[gt["exception_type"] != "none"]
    counter = Counter(exc["exception_type"])

    print(f"\n    Total planted exceptions : {len(exc)}")
    print(f"    {'Exception type':<35} {'Count':>5}")
    print(f"    {'─'*42}")
    expected_types = [
        "amount_mismatch", "date_drift", "missing_reference",
        "duplicate_reference", "missing_tax_line",
        "fuzzy_counterparty", "missing_source_record",
    ]
    for etype in expected_types:
        cnt = counter.get(etype, 0)
        flag = "  ← ZERO" if cnt == 0 else ""
        print(f"    {etype:<35} {cnt:>5}{flag}")

    # Any unrecognised types
    unknown = set(counter.keys()) - set(expected_types)
    if unknown:
        warn(f"Unrecognised exception types in GT: {unknown}")

    total = len(exc)
    if total != 20:
        fail(f"Expected 20 planted exceptions, found {total}",
             "Design spec requires ~20% of 100 = 20 exceptions.",
             "Check EXCEPTION_FRACTION in generate_data.py")
    else:
        ok(f"Total exception count = 20 ✓")


# ---------------------------------------------------------------------------
# Check 6 – Data quality
# ---------------------------------------------------------------------------

def check_quality(dfs: dict[str, pd.DataFrame]) -> None:
    section("CHECK 6 – Data quality")

    sources = {
        "bank":        (dfs["bank"],        "value_date",      "bank_txn_id",      "reference"),
        "ledger":      (dfs["ledger"],       "posting_date",    "ledger_entry_id",  "reference"),
        "invoices":    (dfs["invoices"],     "invoice_date",    "invoice_id",       "reference"),
        "settlements": (dfs["settlements"],  "settlement_date", "settlement_id",    "reference"),
    }

    for src, (df, date_col, id_col, ref_col) in sources.items():

        # Null values in any column
        null_counts = df.isnull().sum()
        nulls = null_counts[null_counts > 0]
        if len(nulls):
            fail(f"{src}: null values in {dict(nulls)}",
                 "Nulls in non-exception rows indicate generator bugs.",
                 "Ensure all row dicts are fully populated before append.")
        else:
            ok(f"{src}: no null values in any column")

        # Duplicate rows (exact)
        n_dupes = df.duplicated().sum()
        if n_dupes:
            fail(f"{src}: {n_dupes} fully duplicate rows",
                 "Exact row duplicates corrupt reconciliation counts.",
                 "Add deduplication or audit the generator loop.")
        else:
            ok(f"{src}: no fully duplicate rows")

        # Duplicate primary IDs
        id_dupes = df[id_col].duplicated().sum()
        if id_dupes:
            fail(f"{src}: {id_dupes} duplicate {id_col} values",
                 "Primary key must be unique within each source.",
                 "Ensure ID generation (BNK-XXXX) is 1-to-1 with canonical_id.")
        else:
            ok(f"{src}: {id_col} is unique (no duplicate primary keys)")

        # Duplicate references (warn — some are intentional)
        ref_dupes = df[df[ref_col].ne("")][ref_col].duplicated().sum()
        if ref_dupes:
            warn(f"{src}: {ref_dupes} duplicate reference value(s) — expected if duplicate_reference exception was planted here")
        else:
            ok(f"{src}: no duplicate reference values")

        # Invalid dates
        try:
            dates = pd.to_datetime(df[date_col], format="%Y-%m-%d")
            in_range = dates.between("2024-01-01", "2024-09-30")  # generous window
            out = (~in_range).sum()
            if out:
                warn(f"{src}: {out} date(s) outside expected 2024 window in '{date_col}'")
            else:
                ok(f"{src}: all dates within expected 2024 window")
        except Exception as e:
            fail(f"{src}: unparseable dates in '{date_col}': {e}",
                 "Downstream normalisation will crash.",
                 "Constrain _drift_date() to stay within ISO-8601 calendar.")

        # Negative or zero amounts
        amounts = pd.to_numeric(df["amount"], errors="coerce")
        negatives = (amounts < 0).sum()
        zeros     = (amounts == 0).sum()
        if negatives:
            fail(f"{src}: {negatives} negative amount(s)",
                 "Financial amounts must be ≥ 0 in this dataset.",
                 "Audit _amount_mismatch(); clamp to max(0, result).")
        else:
            ok(f"{src}: no negative amounts")
        if zeros:
            warn(f"{src}: {zeros} zero-value amount(s) — may be intentional")

        # Impossible amounts (> 10 Cr is suspicious for this dataset)
        huge = (amounts > 10_000_000).sum()
        if huge:
            warn(f"{src}: {huge} amount(s) > ₹1 Cr — verify this is within design range")


# ---------------------------------------------------------------------------
# Check 7 – Reproducibility
# ---------------------------------------------------------------------------

def check_reproducibility() -> None:
    section("CHECK 7 – Reproducibility (run generator twice)")

    def csv_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    files = [
        "bank_statements.csv", "ledger.csv", "invoices.csv",
        "settlements.csv", "ground_truth.csv",
    ]

    # Capture hashes before second run
    hashes_before = {f: csv_hash(DATA_DIR / f) for f in files}

    # Run generator again
    result = subprocess.run(
        [sys.executable, "-m", "src.generate_data"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        fail("Generator exited with non-zero status on second run",
             result.stderr[:500],
             "Fix errors reported above.")
        return

    hashes_after = {f: csv_hash(DATA_DIR / f) for f in files}

    all_match = True
    for f in files:
        if hashes_before[f] == hashes_after[f]:
            ok(f"{f}: identical MD5 across two runs ✓")
        else:
            fail(f"{f}: MD5 changed between runs — NOT reproducible",
                 "Random state is not fully controlled by the fixed seed.",
                 "Ensure all RNG calls (including Faker) use seeded instances.")
            all_match = False

    if all_match:
        ok("Dataset is fully reproducible with SEED=42")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  FINANCE CONTROLLER AGENT — DATA VALIDATION REPORT")
    print("=" * 60)

    # Check files first; abort early if missing
    if not check_files():
        print("\n[ABORT] Critical files missing. Run: python -m src.generate_data")
        sys.exit(1)

    dfs = load_all()

    check_counts(dfs)
    check_schema(dfs)
    check_ground_truth(dfs)
    check_exceptions(dfs)
    check_quality(dfs)
    check_reproducibility()

    # Final verdict
    print("\n" + "=" * 60)
    if issues:
        print(f"  RESULT: ✗ VALIDATION FAILED  ({len(issues)} issue(s))")
        print("=" * 60)
        print("\n  Issues requiring action:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print("\n  Dataset is NOT ready for the matching stage.")
        sys.exit(1)
    else:
        print("  RESULT: ✓ ALL CHECKS PASSED")
        print("=" * 60)
        print("\n  Dataset is READY for the matching stage.")
    print()


if __name__ == "__main__":
    main()
