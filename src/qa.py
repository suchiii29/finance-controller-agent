"""Grounded Finance Controller Q&A over reconciliation decisions and audit events."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.agent import AgentDecision, AuditEvent, BatchResult
from src.llm_reviewer import GeminiExceptionReviewer


@dataclass
class GroundedAnswer:
    question: str
    category: str
    answer: str
    record: Optional[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    recommendation: Optional[str]
    human_review_required: Optional[bool]
    ai_status: str = "NOT_USED"
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinanceControllerQA:
    """Retrieval-first Q&A. Structured controller data remains factually authoritative."""

    def __init__(
        self,
        decisions: Sequence[AgentDecision] | BatchResult,
        audit_events: Optional[Sequence[AuditEvent]] = None,
        reviewer: Optional[GeminiExceptionReviewer] = None,
    ) -> None:
        if isinstance(decisions, BatchResult):
            batch_result = decisions
            self.decisions = list(batch_result.decisions)
            self.audit_events = list(batch_result.audit_events)
            if reviewer is None:
                reviewer = None
        else:
            self.decisions = list(decisions)
            self.audit_events = list(audit_events or [])
        self.reviewer = reviewer
        self._by_id = {decision.ledger_id: decision for decision in self.decisions}
        self._audit_by_id = {event.ledger_id: event for event in self.audit_events}

    @staticmethod
    def _category(question: str) -> str:
        text = question.lower()
        if "tax" in text:
            return "TAX"
        if "duplicate" in text or "collision" in text:
            return "EXCEPTION"
        if any(word in text for word in ("how many", "count", "number of")):
            return "SUMMARY"
        if re.search(r"led[- ]?\d{4}", text):
            return "RECORD"
        return "SUMMARY"

    @staticmethod
    def _compact(decision: AgentDecision, audit: Optional[AuditEvent]) -> Dict[str, Any]:
        tax_check = decision.evidence.get("verifications", {}).get("tax_check")
        return {
            "ledger_id": decision.ledger_id,
            "status": decision.status,
            "exception_type": decision.exception_type,
            "amount": decision.evidence.get("ledger_amount"),
            "date": decision.evidence.get("ledger_date"),
            "reference": decision.evidence.get("ledger_reference"),
            "matched_sources": decision.evidence.get("matched_sources", []),
            "missing_sources": decision.evidence.get("missing_sources", []),
            "verifications": {
                "tax_check": tax_check,
                "duplicate_check": decision.evidence.get("verifications", {}).get("dup_check"),
            },
            "recommended_action": decision.recommended_action,
            "human_review_required": decision.requires_human_review,
            "audit_event_id": audit.event_id if audit else None,
        }

    def _record_id(self, question: str) -> Optional[str]:
        match = re.search(r"led[- ]?(\d{4})", question.lower())
        return f"LED-{match.group(1)}" if match else None

    def answer_question(self, question: str) -> GroundedAnswer:
        start = time.perf_counter()
        category = self._category(question)
        ai_status = "NOT_USED"
        text = question.lower()
        evidence: List[Dict[str, Any]] = []
        record = None
        recommendation = None
        human_review = None

        if category == "RECORD":
            ledger_id = self._record_id(question)
            decision = self._by_id.get(ledger_id or "")
            if decision is None:
                answer = "I could not find that information in the available reconciliation records."
            else:
                record = self._compact(decision, self._audit_by_id.get(decision.ledger_id))
                evidence = [record]
                recommendation = decision.recommended_action
                human_review = decision.requires_human_review
                answer = f"{decision.ledger_id} is {decision.status}."
                if "why" in text or "unresolved" in text or "auto-resolved" in text:
                    if self.reviewer:
                        llm = self.reviewer.review_grounded_question(question, record)
                        if llm.get("status") == "SUCCESS":
                            answer = llm["answer"]
                            ai_status = "SUCCESS"
                        else:
                            answer = llm.get("answer", "AI explanation unavailable. The underlying controller decision remains unchanged.")
                            ai_status = llm.get("status", "UNAVAILABLE")
                    else:
                        ai_status = "NOT_USED"
                    return GroundedAnswer(question, category, answer, record, evidence, recommendation, human_review, ai_status, time.perf_counter() - start)
                ai_status = "NOT_USED"
            return GroundedAnswer(question, category, answer, record, evidence, recommendation, human_review, ai_status, time.perf_counter() - start)

        if category == "EXCEPTION":
            matches = [d for d in self.decisions if d.exception_type == "duplicate_reference"]
            evidence = [self._compact(d, self._audit_by_id.get(d.ledger_id)) for d in matches]
            answer = "Duplicate-reference exceptions: " + (", ".join(d.ledger_id for d in matches) if matches else "none found")
        elif category == "TAX":
            matches = [
                d for d in self.decisions
                if d.evidence.get("verifications", {}).get("tax_check", {}).get("status") == "TAX_MISMATCH"
            ]
            evidence = [self._compact(d, self._audit_by_id.get(d.ledger_id)) for d in matches]
            answer = "Tax mismatches: " + (", ".join(d.ledger_id for d in matches) if matches else "none found")
        else:
            if "human" in text or "review" in text:
                matches = [d for d in self.decisions if d.requires_human_review]
                answer = f"{len(matches)} records require human review."
            elif "missing bank" in text:
                matches = [d for d in self.decisions if "bank" in d.evidence.get("missing_sources", [])]
                answer = f"{len(matches)} records are missing bank evidence."
            elif "auto-resolv" in text:
                matches = [d for d in self.decisions if d.status == "SAFE_AUTO_RESOLVED"]
                answer = f"{len(matches)} records were safely auto-resolved."
            else:
                matches = []
                answer = "I could not find that information in the available reconciliation records."
            evidence = [self._compact(d, self._audit_by_id.get(d.ledger_id)) for d in matches]

        return GroundedAnswer(question, category, answer, None, evidence, None, None, "NOT_USED", time.perf_counter() - start)
