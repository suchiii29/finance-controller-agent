"""
matcher.py
==========
Deterministic multi-source reconciliation engine.

Anchor: ledger.csv
Targets: bank_statements.csv, invoices.csv, settlements.csv

ground_truth.csv is NEVER loaded or accessed here.

Matching tiers
--------------
Tier 1 – Exact / High Confidence
    reference exact match (non-empty, normalised) AND
    |amount_diff| <= 0.01 AND date_diff <= 1 day

Tier 2 – Strong Fuzzy
    |amount_diff| <= 1.00 AND date_diff <= 4 days AND
    counterparty_similarity >= 0.85

Tier 3 – Weak / Residual  →  EXCEPTION / UNRESOLVED (never auto-reconciled)

Statuses
--------
MATCHED    – all three targets found with sufficient evidence
PARTIAL    – some targets matched, one or more absent
EXCEPTION  – ambiguity or detectable discrepancy
UNRESOLVED – no reliable correspondence found
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Optional rapidfuzz
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib as _difflib
    _HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

T1_AMOUNT_TOL   = 0.01   # Tier-1 amount tolerance (absolute)
T1_DATE_TOL     = 1      # Tier-1 date tolerance (days)

T2_AMOUNT_TOL   = 1.00   # Tier-2 amount tolerance (absolute)
T2_DATE_TOL     = 4      # Tier-2 date tolerance — covers natural lag + planted drift
T2_CP_SIM_MIN   = 0.85   # Tier-2 counterparty similarity floor

CONFIDENCE_MIN  = 0.80   # Minimum confidence to count as a real match

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceMatch:
    source: str
    record_id: Optional[str]
    tier: int           # 0 = no match
    confidence: float
    reason: str
    is_ambiguous: bool = False


@dataclass
class ReconciliationDecision:
    ledger_id:          str
    bank_match:         SourceMatch
    invoice_match:      SourceMatch
    settlement_match:   SourceMatch
    status:             str      # MATCHED / PARTIAL / EXCEPTION / UNRESOLVED
    confidence:         float
    tier:               int
    reason:             str
    recommended_action: str


@dataclass
class ReconciliationResult:
    total_processed:      int
    decisions:            list[ReconciliationDecision]
    exceptions:           list[ReconciliationDecision]
    elapsed_seconds:      float
    throughput_per_second: float
    status_counts:        dict[str, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_ref(ref) -> str:
    if ref is None or (isinstance(ref, float)):
        return ""
    s = str(ref).strip()
    return s.upper() if s else ""


def _norm_name(name) -> str:
    if name is None or (isinstance(name, float)):
        return ""
    s = re.sub(r"[^\w\s]", "", str(name).lower().strip())
    return re.sub(r"\s+", " ", s)


def _str_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _rfuzz.token_set_ratio(a, b) / 100.0
    return _difflib.SequenceMatcher(None, a, b).ratio()


def _date_diff(d1, d2) -> int:
    """Return |d1 - d2| in days. Both are pd.Timestamp or datetime."""
    return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Treat blank cells as empty string, not NaN
    return df.fillna("")


def _to_norm_records(df: pd.DataFrame, source: str, id_col: str, date_col: str) -> list[dict]:
    """Convert a source DataFrame to a list of normalised dicts."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "source":       source,
            "record_id":    str(row[id_col]),
            "date":         pd.Timestamp(row[date_col]),
            "amount":       float(row["amount"]),
            "counterparty": str(row.get("counterparty", "")),
            "reference":    str(row.get("reference", "")),
            "tax_line":     str(row.get("tax_line", "")),
            "description":  str(row.get("description", "")),
            # pre-computed normalised fields
            "_ref_norm":    _norm_ref(row.get("reference", "")),
            "_cp_norm":     _norm_name(row.get("counterparty", "")),
        })
    return records


# ---------------------------------------------------------------------------
# Per-source matching
# ---------------------------------------------------------------------------

def _score_tier2(ledger: dict, cand: dict, amount_diff: float, date_diff_days: int) -> tuple[float, str]:
    """Return (confidence, reason) for a Tier-2 candidate."""
    cp_sim  = _str_sim(ledger["_cp_norm"], cand["_cp_norm"])
    ref_sim = _str_sim(ledger["_ref_norm"], cand["_ref_norm"])
    desc_sim = _str_sim(_norm_name(ledger["description"]), _norm_name(cand["description"]))

    amt_score  = max(0.0, 1.0 - amount_diff / T2_AMOUNT_TOL)
    date_score = max(0.0, 1.0 - date_diff_days / T2_DATE_TOL)

    confidence = (
        0.35 * cp_sim +
        0.25 * amt_score +
        0.20 * date_score +
        0.15 * ref_sim +
        0.05 * desc_sim
    )

    reason = (
        f"cp_sim={cp_sim:.2f} amt_diff={amount_diff:.2f} "
        f"date_diff={date_diff_days}d ref_sim={ref_sim:.2f} "
        f"desc_sim={desc_sim:.2f} → conf={confidence:.2f}"
    )
    return confidence, reason


def _match_source(
    ledger_rec: dict,
    candidates: list[dict],
    claimed: set,
    source_name: str,
) -> SourceMatch:
    """
    Attempt to match a single ledger record against all records in one source.
    Returns the best SourceMatch found (or a no-match with tier=0).
    """
    l_ref    = ledger_rec["_ref_norm"]
    l_amount = ledger_rec["amount"]
    l_date   = ledger_rec["date"]

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    tier1_hits = []
    for c in candidates:
        if c["record_id"] in claimed:
            continue
        ref_match  = (l_ref and c["_ref_norm"] and l_ref == c["_ref_norm"])
        amt_ok     = abs(l_amount - c["amount"]) <= T1_AMOUNT_TOL
        date_ok    = _date_diff(l_date, c["date"]) <= T1_DATE_TOL
        if ref_match and amt_ok and date_ok:
            tier1_hits.append(c)

    if len(tier1_hits) > 1:
        ids = [h["record_id"] for h in tier1_hits]
        return SourceMatch(
            source=source_name, record_id=None, tier=1,
            confidence=0.0, is_ambiguous=True,
            reason=f"AMBIGUOUS: {len(tier1_hits)} Tier-1 hits on same reference → {ids}",
        )

    if len(tier1_hits) == 1:
        c = tier1_hits[0]
        amt_diff  = abs(l_amount - c["amount"])
        date_days = _date_diff(l_date, c["date"])
        return SourceMatch(
            source=source_name, record_id=c["record_id"], tier=1,
            confidence=1.0,
            reason=f"EXACT ref={c['_ref_norm']} amt_diff={amt_diff:.4f} date_diff={date_days}d",
        )

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    tier2_hits: list[tuple[float, str, dict]] = []
    for c in candidates:
        if c["record_id"] in claimed:
            continue
        amt_diff  = abs(l_amount - c["amount"])
        date_days = _date_diff(l_date, c["date"])
        cp_sim    = _str_sim(ledger_rec["_cp_norm"], c["_cp_norm"])

        if amt_diff <= T2_AMOUNT_TOL and date_days <= T2_DATE_TOL and cp_sim >= T2_CP_SIM_MIN:
            conf, reason = _score_tier2(ledger_rec, c, amt_diff, date_days)
            tier2_hits.append((conf, reason, c))

    if tier2_hits:
        tier2_hits.sort(key=lambda x: -x[0])
        best_conf, best_reason, best_c = tier2_hits[0]

        if len(tier2_hits) > 1 and abs(tier2_hits[0][0] - tier2_hits[1][0]) < 0.05:
            ids = [h[2]["record_id"] for h in tier2_hits]
            return SourceMatch(
                source=source_name, record_id=None, tier=2,
                confidence=best_conf, is_ambiguous=True,
                reason=f"AMBIGUOUS: {len(tier2_hits)} Tier-2 candidates with similar scores → {ids}",
            )

        if best_conf >= CONFIDENCE_MIN:
            return SourceMatch(
                source=source_name, record_id=best_c["record_id"], tier=2,
                confidence=best_conf, reason=f"FUZZY {best_reason}",
            )
        else:
            return SourceMatch(
                source=source_name, record_id=None, tier=3,
                confidence=best_conf,
                reason=f"WEAK_MATCH (conf={best_conf:.2f} < {CONFIDENCE_MIN}) → NEEDS_REVIEW {best_reason}",
            )

    # ── No match ────────────────────────────────────────────────────────────
    return SourceMatch(
        source=source_name, record_id=None, tier=0,
        confidence=0.0, reason="NO_CANDIDATE_FOUND",
    )


# ---------------------------------------------------------------------------
# Overall status determination
# ---------------------------------------------------------------------------

def _determine_status(
    bank_m: SourceMatch,
    inv_m:  SourceMatch,
    stl_m:  SourceMatch,
) -> tuple[str, float, int, str, str]:
    """Return (status, confidence, tier, reason, recommended_action)."""
    matches = [bank_m, inv_m, stl_m]

    # Any ambiguity → EXCEPTION
    if any(m.is_ambiguous for m in matches):
        ambig = [m for m in matches if m.is_ambiguous]
        reason = "; ".join(m.reason for m in ambig)
        return (
            "EXCEPTION", 0.0, min(m.tier for m in ambig),
            f"Ambiguous candidates detected: {reason}",
            "Investigate duplicate references or near-identical records.",
        )

    # Any Tier-3 weak hit (explicit discrepancy) → EXCEPTION
    if any(m.tier == 3 for m in matches):
        weak = [m for m in matches if m.tier == 3]
        reason = "; ".join(f"{m.source}: {m.reason}" for m in weak)
        return (
            "EXCEPTION", max(m.confidence for m in weak), 3,
            f"Low-confidence residual candidate: {reason}",
            "Flag for manual review — do not auto-reconcile.",
        )

    real_matches = [m for m in matches if m.record_id is not None and m.tier > 0]
    n_matched = len(real_matches)

    if n_matched == 0:
        return (
            "UNRESOLVED", 0.0, 0,
            "No corresponding record found in any of the three target sources.",
            "Check for missing source records or data import failures.",
        )

    avg_conf = sum(m.confidence for m in real_matches) / n_matched
    min_tier = min(m.tier for m in real_matches)

    if n_matched == 3:
        reason = " | ".join(
            f"{m.source}={m.record_id}(T{m.tier},c={m.confidence:.2f})"
            for m in matches
        )
        return (
            "MATCHED", avg_conf, min_tier,
            f"All three targets matched: {reason}",
            "No action required.",
        )

    missing = [m.source for m in matches if m.record_id is None]
    found   = " | ".join(
        f"{m.source}={m.record_id}(T{m.tier},c={m.confidence:.2f})"
        for m in matches if m.record_id
    )
    return (
        "PARTIAL", avg_conf, min_tier,
        f"Partial match — missing: {missing}. Found: {found}",
        f"Investigate missing records from: {missing}",
    )


# ---------------------------------------------------------------------------
# Main matcher class
# ---------------------------------------------------------------------------

class ReconciliationMatcher:
    """
    Deterministic reconciliation engine.
    Anchor: ledger.csv. Targets: bank, invoices, settlements.
    Never reads ground_truth.csv.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self._ledger_records:     list[dict] = []
        self._bank_records:       list[dict] = []
        self._invoice_records:    list[dict] = []
        self._settlement_records: list[dict] = []

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_sources(self) -> None:
        bank = _load_csv(self.data_dir / "bank_statements.csv")
        led  = _load_csv(self.data_dir / "ledger.csv")
        inv  = _load_csv(self.data_dir / "invoices.csv")
        stl  = _load_csv(self.data_dir / "settlements.csv")

        self._ledger_records     = _to_norm_records(led, "ledger",      "ledger_entry_id",  "txn_date")
        self._bank_records       = _to_norm_records(bank, "bank",       "bank_txn_id",      "value_date")
        self._invoice_records    = _to_norm_records(inv,  "invoice",    "invoice_id",       "invoice_date")
        self._settlement_records = _to_norm_records(stl,  "settlement", "settlement_id",    "settlement_date")

    # ── Reconciliation ───────────────────────────────────────────────────────

    def reconcile(self) -> ReconciliationResult:
        if not self._ledger_records:
            self.load_sources()

        t0 = time.perf_counter()

        # Claimed-ID sets prevent unsafe many-to-one assignment
        claimed_bank = set()
        claimed_inv  = set()
        claimed_stl  = set()

        decisions:  list[ReconciliationDecision] = []
        exceptions: list[ReconciliationDecision] = []
        status_counts: dict[str, int] = {
            "MATCHED": 0, "PARTIAL": 0, "EXCEPTION": 0, "UNRESOLVED": 0,
        }

        for ledger_rec in self._ledger_records:
            bank_m = _match_source(ledger_rec, self._bank_records,       claimed_bank, "bank")
            inv_m  = _match_source(ledger_rec, self._invoice_records,    claimed_inv,  "invoice")
            stl_m  = _match_source(ledger_rec, self._settlement_records, claimed_stl,  "settlement")

            status, conf, tier, reason, action = _determine_status(bank_m, inv_m, stl_m)

            # Claim matched IDs to prevent reuse
            if bank_m.record_id and not bank_m.is_ambiguous:
                claimed_bank.add(bank_m.record_id)
            if inv_m.record_id and not inv_m.is_ambiguous:
                claimed_inv.add(inv_m.record_id)
            if stl_m.record_id and not stl_m.is_ambiguous:
                claimed_stl.add(stl_m.record_id)

            dec = ReconciliationDecision(
                ledger_id=ledger_rec["record_id"],
                bank_match=bank_m,
                invoice_match=inv_m,
                settlement_match=stl_m,
                status=status,
                confidence=conf,
                tier=tier,
                reason=reason,
                recommended_action=action,
            )
            decisions.append(dec)
            status_counts[status] += 1
            if status in ("EXCEPTION", "UNRESOLVED"):
                exceptions.append(dec)

        elapsed = time.perf_counter() - t0
        n = len(decisions)

        return ReconciliationResult(
            total_processed=n,
            decisions=decisions,
            exceptions=exceptions,
            elapsed_seconds=elapsed,
            throughput_per_second=n / elapsed if elapsed > 0 else 0.0,
            status_counts=status_counts,
        )
