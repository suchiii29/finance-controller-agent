"""
agent.py
========
Senior AI Exception Agent for financial reconciliation residual cases.

Receives residual decisions from ReconciliationMatcher (PARTIAL, EXCEPTION, UNRESOLVED)
along with actual source record fields (ledger, bank, invoice, settlement).

Uses an LLM (Gemini or OpenAI) when API keys are present (or an enhanced domain-specific
field-referencing expert generator as fallback) to produce detailed, highly-differentiated,
professional financial explanations and precise recommended actions.

Decision Policy & Safety Rules:
- Proposes SAFE_AUTO_RESOLVE only for LOW-risk cases where exactly one non-cash source
  (invoice or settlement) is missing, and the remaining 2 sources (bank + other) match Tier-1
  with 100% confidence, zero duplicate reference collision, and exact amount alignment.
- Always escalates cases missing primary bank statement feeds ("Escalate: Missing primary bank feed – cash control risk").
- Always escalates duplicate reference collisions or amount discrepancies.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.matcher import ReconciliationDecision

# ---------------------------------------------------------------------------
# LLM Integration Helpers
# ---------------------------------------------------------------------------

_LLM_CLIENT_TYPE: Optional[str] = None
_GENAI_MODEL = None
_OPENAI_CLIENT = None

def _init_llm():
    global _LLM_CLIENT_TYPE, _GENAI_MODEL, _OPENAI_CLIENT
    if _LLM_CLIENT_TYPE is not None:
        return

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            _GENAI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
            _LLM_CLIENT_TYPE = "gemini"
            print("[INFO] ExceptionAgent: Using Gemini LLM for explanation generation.")
            return
        except Exception as e:
            print(f"[WARN] ExceptionAgent: Gemini init failed ({e}). Checking OpenAI...")

    if openai_key:
        try:
            import openai
            _OPENAI_CLIENT = openai.OpenAI(api_key=openai_key)
            _LLM_CLIENT_TYPE = "openai"
            print("[INFO] ExceptionAgent: Using OpenAI LLM for explanation generation.")
            return
        except Exception as e:
            print(f"[WARN] ExceptionAgent: OpenAI init failed ({e}).")

    _LLM_CLIENT_TYPE = "none"
    print("[INFO] ExceptionAgent: No LLM API key detected (GEMINI_API_KEY/OPENAI_API_KEY). Using domain-specific field-referencing generator.")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExceptionAnalysis:
    ledger_id: str
    final_status: str                # SAFE_AUTO_RESOLVED / PARTIAL / EXCEPTION / UNRESOLVED
    risk_level: str                  # LOW / MEDIUM / HIGH
    detailed_explanation: str
    recommended_action: str
    evidence_summary: Dict[str, Any]
    safe_auto_resolved: bool = False
    original_status: str = ""
    confidence: float = 0.0
    missing_sources: List[str] = None
    matched_sources: List[str] = None

    @property
    def explanation(self) -> str:
        """Alias for compatibility with report generators."""
        return self.detailed_explanation

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["explanation"] = self.detailed_explanation
        return d


class ExceptionAgent:
    """
    Senior Exception Agent for financial reconciliation residual cases.
    Analyzes missing or conflicting source records, applies strict safety guardrails,
    and assigns highly-differentiated actions and explanations.
    """

    ALLOWED_ACTIONS = [
        "Auto-resolve: Non-cash timing lag (invoice missing)",
        "Auto-resolve: Settlement sync delay",
        "Escalate: Missing primary bank feed – cash control risk",
        "Escalate: Duplicate reference collision",
        "Escalate: Material amount discrepancy",
        "Manual review: Conflicting counterparty evidence",
        "Investigate: Partial source coverage – possible data ingestion gap",
    ]

    def __init__(self):
        _init_llm()

    def analyze_residuals(
        self,
        decisions: List[ReconciliationDecision],
        source_records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ExceptionAnalysis]:
        """
        Analyze only residual decisions (PARTIAL, EXCEPTION, UNRESOLVED).
        """
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
        # 1. Extract source rows
        ledger_row = {}
        bank_row = None
        invoice_row = None
        settlement_row = None

        if source_records:
            ledger_row = source_records.get("ledger", {}).get(dec.ledger_id, {})
            if dec.bank_match.record_id:
                bank_row = source_records.get("bank", {}).get(dec.bank_match.record_id)
            if dec.invoice_match.record_id:
                invoice_row = source_records.get("invoice", {}).get(dec.invoice_match.record_id)
            if dec.settlement_match.record_id:
                settlement_row = source_records.get("settlement", {}).get(dec.settlement_match.record_id)

        # 2. Identify matched and missing systems
        matched_sources = []
        missing_sources = []

        if dec.bank_match.record_id:
            matched_sources.append("bank")
        else:
            missing_sources.append("bank")

        if dec.invoice_match.record_id:
            matched_sources.append("invoice")
        else:
            missing_sources.append("invoice")

        if dec.settlement_match.record_id:
            matched_sources.append("settlement")
        else:
            missing_sources.append("settlement")

        # 3. Build Evidence Summary
        ledger_amount = ledger_row.get("amount", dec.confidence)
        ledger_date = str(ledger_row.get("date", "")).split()[0] if ledger_row.get("date") else ""
        ledger_ref = str(ledger_row.get("reference", "")).strip().upper()
        ledger_cp = ledger_row.get("counterparty", "")

        evidence_summary = {
            "ledger": {
                "id": dec.ledger_id,
                "amount": ledger_amount,
                "date": ledger_date,
                "reference": ledger_ref,
                "counterparty": ledger_cp,
            },
            "matched_sources": matched_sources,
            "missing_sources": missing_sources,
            "matched_records": {
                "bank": {
                    "id": dec.bank_match.record_id,
                    "amount": bank_row.get("amount") if bank_row else None,
                    "date": str(bank_row.get("date", "")).split()[0] if bank_row else None,
                    "reference": bank_row.get("reference") if bank_row else None,
                } if bank_row else None,
                "invoice": {
                    "id": dec.invoice_match.record_id,
                    "amount": invoice_row.get("amount") if invoice_row else None,
                    "date": str(invoice_row.get("date", "")).split()[0] if invoice_row else None,
                    "reference": invoice_row.get("reference") if invoice_row else None,
                } if invoice_row else None,
                "settlement": {
                    "id": dec.settlement_match.record_id,
                    "amount": settlement_row.get("amount") if settlement_row else None,
                    "date": str(settlement_row.get("date", "")).split()[0] if settlement_row else None,
                    "reference": settlement_row.get("reference") if settlement_row else None,
                } if settlement_row else None,
            },
            "matcher_reason": dec.reason,
        }

        # 4. Check for duplicate reference collision across target sources
        has_dup_ref_collision = False
        if ledger_ref and source_records:
            for sys_key in ["bank", "invoice", "settlement"]:
                recs = source_records.get(sys_key, {}).values()
                ref_count = sum(1 for r in recs if str(r.get("reference", "")).strip().upper() == ledger_ref)
                if ref_count > 1:
                    has_dup_ref_collision = True
                    break

        # 5. Evaluate Safety Policy for Safe Auto-Resolution & Action Mapping
        tier1_matches = [
            m for m in [dec.bank_match, dec.invoice_match, dec.settlement_match]
            if m.tier == 1 and m.record_id is not None
        ]
        is_single_non_cash_missing = (len(missing_sources) == 1 and "bank" not in missing_sources)
        both_matched_tier1 = (len(tier1_matches) == 2)

        safe_auto_resolved = False
        final_status = dec.status
        risk_level = "MEDIUM"
        action = "Investigate: Partial source coverage – possible data ingestion gap"

        if dec.status == "UNRESOLVED":
            risk_level = "HIGH"
            action = "Investigate: Partial source coverage – possible data ingestion gap"
        elif dec.status == "EXCEPTION":
            risk_level = "HIGH"
            if "AMBIGUOUS" in dec.reason or "duplicate" in dec.reason.lower() or has_dup_ref_collision:
                action = "Escalate: Duplicate reference collision"
            elif "WEAK_MATCH" in dec.reason or "amount" in dec.reason.lower():
                action = "Escalate: Material amount discrepancy"
            elif "counterparty" in dec.reason.lower():
                action = "Manual review: Conflicting counterparty evidence"
            else:
                action = "Investigate: Partial source coverage – possible data ingestion gap"
        elif dec.status == "PARTIAL":
            if "bank" in missing_sources:
                # Primary bank feed missing -> ALWAYS ESCALATE
                risk_level = "MEDIUM"
                safe_auto_resolved = False
                final_status = "PARTIAL"
                action = "Escalate: Missing primary bank feed – cash control risk"
            elif is_single_non_cash_missing and both_matched_tier1 and not has_dup_ref_collision:
                # Meets ALL strict safety guardrails for SAFE_AUTO_RESOLVE
                risk_level = "LOW"
                safe_auto_resolved = True
                final_status = "SAFE_AUTO_RESOLVED"
                if "invoice" in missing_sources:
                    action = "Auto-resolve: Non-cash timing lag (invoice missing)"
                elif "settlement" in missing_sources:
                    action = "Auto-resolve: Settlement sync delay"
                else:
                    action = "Auto-resolve: Non-cash timing lag (invoice missing)"
            else:
                risk_level = "HIGH" if len(missing_sources) >= 2 else "MEDIUM"
                safe_auto_resolved = False
                action = "Investigate: Partial source coverage – possible data ingestion gap"

        # 6. Generate Differentiated Field-Referencing Detailed Explanation
        explanation = self._generate_explanation(
            dec=dec,
            ledger_row=ledger_row,
            bank_row=bank_row,
            invoice_row=invoice_row,
            settlement_row=settlement_row,
            matched_sources=matched_sources,
            missing_sources=missing_sources,
            risk_level=risk_level,
            action=action,
            safe_auto_resolved=safe_auto_resolved,
        )

        return ExceptionAnalysis(
            ledger_id=dec.ledger_id,
            final_status=final_status,
            risk_level=risk_level,
            detailed_explanation=explanation,
            recommended_action=action,
            evidence_summary=evidence_summary,
            safe_auto_resolved=safe_auto_resolved,
            original_status=dec.status,
            confidence=dec.confidence,
            missing_sources=missing_sources,
            matched_sources=matched_sources,
        )

    def _generate_explanation(
        self,
        dec: ReconciliationDecision,
        ledger_row: dict,
        bank_row: Optional[dict],
        invoice_row: Optional[dict],
        settlement_row: Optional[dict],
        matched_sources: list[str],
        missing_sources: list[str],
        risk_level: str,
        action: str,
        safe_auto_resolved: bool,
    ) -> str:
        """
        Generate a professional, field-specific financial explanation.
        Uses LLM if available; otherwise uses structured domain generator.
        """
        if _LLM_CLIENT_TYPE in {"gemini", "openai"}:
            try:
                return self._call_llm_explanation(
                    dec, ledger_row, bank_row, invoice_row, settlement_row,
                    matched_sources, missing_sources, risk_level, action, safe_auto_resolved
                )
            except Exception as e:
                print(f"[WARN] ExceptionAgent: LLM call failed ({e}). Falling back to domain generator.")

        return self._build_domain_explanation(
            dec, ledger_row, bank_row, invoice_row, settlement_row,
            matched_sources, missing_sources, risk_level, action, safe_auto_resolved
        )

    def _call_llm_explanation(
        self,
        dec: ReconciliationDecision,
        ledger_row: dict,
        bank_row: Optional[dict],
        invoice_row: Optional[dict],
        settlement_row: Optional[dict],
        matched_sources: list[str],
        missing_sources: list[str],
        risk_level: str,
        action: str,
        safe_auto_resolved: bool,
    ) -> str:
        prompt = f"""You are a Senior AI Financial Controller reviewing a 4-way financial reconciliation exception.

LEDGER TRANSACTION:
- ID: {dec.ledger_id}
- Date: {ledger_row.get('date', 'N/A')}
- Amount: ₹{ledger_row.get('amount', 'N/A')}
- Reference: {ledger_row.get('reference', 'N/A')}
- Counterparty: {ledger_row.get('counterparty', 'N/A')}

MATCHED SOURCE RECORDS:
- Bank: {f"ID: {dec.bank_match.record_id}, Amount: ₹{bank_row.get('amount')}, Date: {bank_row.get('date')}, Ref: {bank_row.get('reference')}" if bank_row else "MISSING"}
- Invoice: {f"ID: {dec.invoice_match.record_id}, Amount: ₹{invoice_row.get('amount')}, Date: {invoice_row.get('date')}, Ref: {invoice_row.get('reference')}" if invoice_row else "MISSING"}
- Settlement: {f"ID: {dec.settlement_match.record_id}, Amount: ₹{settlement_row.get('amount')}, Date: {settlement_row.get('date')}, Ref: {settlement_row.get('reference')}" if settlement_row else "MISSING"}

DECISION:
- Original Matcher Status: {dec.status}
- Safe Auto Resolved: {safe_auto_resolved}
- Risk Level: {risk_level}
- Action Recommended: {action}
- Missing Systems: {", ".join(missing_sources)}

Task:
Write a 3-5 sentence concise, executive financial narrative.
Requirements:
1. Refer explicitly to exact field values (e.g. ₹amount, dates, reference strings, counterparty names).
2. Clearly state which sources corroborate (e.g. Bank BNK-0014 and Settlement STL-0014 match Tier-1) and which systems are absent or conflicting.
3. If safe_auto_resolved is True, explicitly state why it was safely auto-resolved as a low-risk timing lag.
4. If safe_auto_resolved is False, clearly explain why it must remain escalated (e.g. missing primary bank statement cash feed, duplicate reference collision, or amount discrepancy).
5. Use professional accounting and finance terminology.
6. Do not include markdown formatting or quotes.
"""
        if _LLM_CLIENT_TYPE == "gemini":
            response = _GENAI_MODEL.generate_content(prompt)
            return response.text.strip()
        elif _LLM_CLIENT_TYPE == "openai":
            res = _OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )
            return res.choices[0].message.content.strip()

        return self._build_domain_explanation(dec, ledger_row, bank_row, invoice_row, settlement_row, matched_sources, missing_sources, risk_level, action, safe_auto_resolved)

    def _build_domain_explanation(
        self,
        dec: ReconciliationDecision,
        ledger_row: dict,
        bank_row: Optional[dict],
        invoice_row: Optional[dict],
        settlement_row: Optional[dict],
        matched_sources: list[str],
        missing_sources: list[str],
        risk_level: str,
        action: str,
        safe_auto_resolved: bool,
    ) -> str:
        l_amt = ledger_row.get("amount", "N/A")
        l_ref = ledger_row.get("reference", "N/A")
        l_cp = ledger_row.get("counterparty", "N/A")
        l_date = str(ledger_row.get("date", "")).split()[0] if ledger_row.get("date") else "N/A"

        matched_details = []
        if bank_row:
            matched_details.append(f"Bank Statement {dec.bank_match.record_id} (₹{bank_row.get('amount'):,.2f})")
        if invoice_row:
            matched_details.append(f"Invoice {dec.invoice_match.record_id} (₹{invoice_row.get('amount'):,.2f})")
        if settlement_row:
            matched_details.append(f"Settlement {dec.settlement_match.record_id} (₹{settlement_row.get('amount'):,.2f})")

        matched_str = ", ".join(matched_details) if matched_details else "None"
        missing_str = ", ".join(s.capitalize() for s in missing_sources)

        if safe_auto_resolved:
            if "invoice" in missing_sources:
                return (
                    f"Ledger entry {dec.ledger_id} for {l_cp} (₹{l_amt:,.2f}, Ref: {l_ref}, Date: {l_date}) "
                    f"is 100% verified across {matched_str} with zero amount discrepancy. "
                    f"The invoice record is pending vendor generation or accounting sync, but physical cash movement is fully confirmed by the bank feed. "
                    f"The Exception Agent has safely auto-resolved this low-risk non-cash timing lag without risk of financial misstatement. "
                    f"Action Assigned: {action}."
                )
            elif "settlement" in missing_sources:
                return (
                    f"Ledger entry {dec.ledger_id} for {l_cp} (₹{l_amt:,.2f}, Ref: {l_ref}, Date: {l_date}) "
                    f"is fully corroborated by {matched_str}. "
                    f"The payment gateway settlement batch is currently lagging, but bank statement verification confirms cash receipt. "
                    f"The Exception Agent has safely auto-resolved this entry as a low-risk settlement sync delay. "
                    f"Action Assigned: {action}."
                )

        if "bank" in missing_sources:
            return (
                f"Ledger entry {dec.ledger_id} for {l_cp} (₹{l_amt:,.2f}, Ref: {l_ref}, Date: {l_date}) "
                f"has corroboration across {matched_str}, but lacks a corresponding Bank Statement record. "
                f"Because bank statements represent primary cash movement controls, missing cash proof poses an unverified cash risk. "
                f"This transaction cannot be auto-resolved and is escalated to Finance Ops for manual bank feed verification. "
                f"Action Assigned: {action}."
            )

        if dec.status == "EXCEPTION":
            return (
                f"Ledger entry {dec.ledger_id} for {l_cp} (₹{l_amt:,.2f}, Ref: {l_ref}, Date: {l_date}) "
                f"triggered a reconciliation exception ({dec.reason}). Matched sources: {matched_str}. "
                f"The transaction requires manual audit due to potential reference ambiguity or conflicting amount fields. "
                f"Action Assigned: {action}."
            )

        return (
            f"Ledger entry {dec.ledger_id} for {l_cp} (₹{l_amt:,.2f}, Ref: {l_ref}, Date: {l_date}) "
            f"has partial source coverage (missing: {missing_str}) and requires manual investigation. "
            f"Action Assigned: {action}."
        )
