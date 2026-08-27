"""Bounded Gemini Interactions API controller agent over verified case data."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent import (
    AgentDecision,
    AuditEvent,
    tool_verify_amount,
    tool_verify_date,
    tool_verify_reference,
    tool_verify_tax_line,
)
from src.llm_reviewer import GeminiExceptionReviewer

MAX_AGENT_STEPS = 8
FORBIDDEN_TOOLS = {
    "update_ledger", "modify_transaction", "approve_payment", "send_payment",
    "delete_transaction", "modify_source_data", "access_ground_truth", "retrain_model",
}


@dataclass
class AgentStepAudit:
    interaction_id: Optional[str]
    tool_call_id: Optional[str]
    timestamp: str
    ledger_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result_status: str
    result_summary: Dict[str, Any]


@dataclass
class ControllerAgentResult:
    ledger_id: str
    final_decision: str
    explanation: str
    evidence_used: List[str]
    tools_called: List[str]
    recommendations: str
    confidence: float
    requires_human_review: bool
    status: str
    interaction_id: Optional[str]
    steps: List[AgentStepAudit] = field(default_factory=list)
    loop_detected: bool = False
    step_limit_reached: bool = False
    fallback_used: bool = False
    latency_seconds: float = 0.0
    failure_category: Optional[str] = None
    forbidden_tool_attempts: int = 0
    controller_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerAgentMetrics:
    agent_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tool_calls: int = 0
    max_tool_calls: int = 0
    loop_stops: int = 0
    step_limit_stops: int = 0
    gemini_calls: int = 0
    total_investigation_seconds: float = 0.0
    tool_successes: int = 0
    tool_failures: int = 0
    forbidden_tool_attempts: int = 0
    controller_overrides: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinanceControllerPolicy:
    """Authoritative policy applied after Gemini investigation."""

    @staticmethod
    def finalize(case: AgentDecision, proposed: str) -> Dict[str, Any]:
        if case.requires_human_review or case.status in {"PARTIAL", "EXCEPTION", "UNRESOLVED"}:
            return {"decision": "HUMAN_REVIEW_REQUIRED", "requires_human_review": True}
        return {"decision": case.status, "requires_human_review": False}


class BoundedGeminiControllerAgent:
    """Gemini-driven investigator with application-owned tools and hard bounds."""

    def __init__(
        self,
        decisions: Sequence[AgentDecision],
        audit_events: Sequence[AuditEvent],
        reviewer: GeminiExceptionReviewer,
        max_steps: int = MAX_AGENT_STEPS,
    ) -> None:
        self.decisions = {d.ledger_id: d for d in decisions}
        self.audit_events = {a.ledger_id: a for a in audit_events}
        self.reviewer = reviewer
        self.max_steps = max(1, max_steps)
        self.metrics = ControllerAgentMetrics()
        self.tool_calls: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_reconciliation_case": self.get_reconciliation_case,
            "find_candidate_records": self.find_candidate_records,
            "compare_candidate_evidence": self.compare_candidate_evidence,
            "verify_amount": self.verify_amount,
            "verify_date": self.verify_date,
            "verify_reference": self.verify_reference,
            "verify_tax": self.verify_tax,
            "inspect_duplicate_reference": self.inspect_duplicate_reference,
            "get_audit_history": self.get_audit_history,
            "get_exception_history": self.get_exception_history,
            "get_controller_policy": self.get_controller_policy,
            "summarize_case": self.summarize_case,
        }

    def _case(self, ledger_id: str) -> Optional[AgentDecision]:
        if not isinstance(ledger_id, str) or ledger_id not in self.decisions:
            return None
        return self.decisions[ledger_id]

    def get_reconciliation_case(self, ledger_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        if case is None:
            return {"status": "not_found", "ledger_id": ledger_id}
        return {"status": "success", "ledger_id": ledger_id, "case": {
            "status": case.status, "exception_type": case.exception_type,
            "source_ids": {"bank": case.bank_id, "invoice": case.invoice_id, "settlement": case.settlement_id},
            "controller_confidence": case.confidence,
            "requires_human_review": case.requires_human_review,
            "recommended_action": case.recommended_action,
        }}

    def find_candidate_records(self, ledger_id: str, source: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        if case is None or source not in {"bank", "invoice", "settlement"}:
            return {"status": "invalid_or_not_found", "ledger_id": ledger_id, "source": source, "candidates": []}
        selected = case.evidence.get(f"{source}_candidates", [])
        return {"status": "success", "ledger_id": ledger_id, "source": source, "candidates": selected[:5]}

    def compare_candidate_evidence(self, ledger_id: str, candidate_ids: List[str]) -> Dict[str, Any]:
        case = self._case(ledger_id)
        return {"status": "success" if case else "not_found", "ledger_id": ledger_id,
                "candidate_ids": candidate_ids[:5], "evidence": case.evidence if case else {}}

    def verify_amount(self, ledger_id: str, candidate_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        if case is None:
            return {"status": "not_found", "ledger_id": ledger_id}
        verification = case.evidence.get("amount_verification", case.evidence.get("verifications", {}).get("amount"))
        return {"status": "success", "ledger_id": ledger_id, "candidate_id": candidate_id,
            "verification_status": "available" if verification is not None else "not_recorded",
            "verification": verification}

    def verify_date(self, ledger_id: str, candidate_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        verification = case.evidence.get("date_verification", case.evidence.get("verifications", {}).get("date")) if case else None
        return {"status": "success" if case else "not_found", "ledger_id": ledger_id,
            "candidate_id": candidate_id, "verification_status": "available" if verification is not None else "not_recorded",
            "verification": verification}

    def verify_reference(self, ledger_id: str, candidate_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        verification = case.evidence.get("reference_verification", case.evidence.get("verifications", {}).get("reference")) if case else None
        return {"status": "success" if case else "not_found", "ledger_id": ledger_id,
            "candidate_id": candidate_id, "verification_status": "available" if verification is not None else "not_recorded",
            "verification": verification}

    def verify_tax(self, ledger_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        if case is None:
            return {"status": "not_found", "ledger_id": ledger_id}
        return {"status": "success", "ledger_id": ledger_id,
                "verification": case.evidence.get("verifications", {}).get("tax_check")}

    def inspect_duplicate_reference(self, ledger_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        if case is None:
            return {"status": "not_found", "ledger_id": ledger_id}
        duplicate = case.evidence.get("verifications", {}).get("dup_check", {})
        return {"status": "success", "ledger_id": ledger_id,
                "candidate_count": sum(len(v) for v in duplicate.get("collisions", {}).values()),
                "candidates": duplicate.get("collisions", {}), "evidence": duplicate}

    def get_audit_history(self, ledger_id: str) -> Dict[str, Any]:
        audit = self.audit_events.get(ledger_id)
        return {"status": "success" if audit else "not_found", "ledger_id": ledger_id,
                "audit": {"event_id": audit.event_id, "decision": audit.decision, "timestamp": audit.timestamp} if audit else None}

    def get_exception_history(self, ledger_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        return {"status": "success" if case else "not_found", "ledger_id": ledger_id,
                "exception_type": case.exception_type if case else None, "human_review": case.requires_human_review if case else None}

    def get_controller_policy(self, ledger_id: str = "") -> Dict[str, Any]:
        return {"status": "success", "policy": "Controller status and human-review requirements are authoritative; Gemini cannot change them."}

    def summarize_case(self, ledger_id: str) -> Dict[str, Any]:
        case = self._case(ledger_id)
        return self.get_reconciliation_case(ledger_id) if case else {"status": "not_found", "ledger_id": ledger_id}

    def tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for name in self.tool_calls:
            properties = {"ledger_id": {"type": "string"}}
            required = ["ledger_id"]
            if name == "find_candidate_records":
                properties["source"] = {"type": "string", "enum": ["bank", "invoice", "settlement"]}
                required.append("source")
            if name == "compare_candidate_evidence":
                properties["candidate_ids"] = {"type": "array", "items": {"type": "string"}}
                required.append("candidate_ids")
            if name in {"verify_amount", "verify_date", "verify_reference"}:
                properties["candidate_id"] = {"type": "string"}
                required.append("candidate_id")
            schemas.append({"type": "function", "name": name, "description": f"Read or verify controller evidence using {name}.",
                            "parameters": {"type": "object", "properties": properties, "required": required}})
        return schemas

    def _execute(self, name: str, arguments: Any) -> Dict[str, Any]:
        if name in FORBIDDEN_TOOLS or name not in self.tool_calls:
            return {"status": "tool_rejected", "reason": "Tool is not on the Finance Controller allowlist."}
        if not isinstance(arguments, dict):
            return {"status": "invalid_arguments", "reason": "Arguments must be a JSON object."}
        try:
            return self.tool_calls[name](**arguments)
        except (TypeError, ValueError, KeyError):
            return {"status": "invalid_arguments", "reason": "Tool arguments failed validation."}

    @staticmethod
    def _step_from(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        return value.model_dump() if hasattr(value, "model_dump") else None

    def investigate(self, ledger_id: str, goal: Optional[str] = None) -> ControllerAgentResult:
        started = time.perf_counter()
        self.metrics.agent_runs += 1
        case = self._case(ledger_id)
        if case is None:
            return ControllerAgentResult(ledger_id, "HUMAN_REVIEW_REQUIRED", "Case was not found in controller records.", [], [], "Manual verification", 0.0, True, "NOT_FOUND", None, latency_seconds=time.perf_counter() - started)
        if not self.reviewer.is_configured or self.reviewer.client is None:
            final = FinanceControllerPolicy.finalize(case, "")
            result = ControllerAgentResult(ledger_id, final["decision"], "Gemini unavailable; controller decision retained.", [], [], case.recommended_action, case.confidence, final["requires_human_review"], "GEMINI_UNAVAILABLE", None, fallback_used=True, latency_seconds=time.perf_counter() - started)
            self.metrics.failed_runs += 1
            self.metrics.total_investigation_seconds += result.latency_seconds
            return result

        interaction_id = None
        steps: List[AgentStepAudit] = []
        called: List[str] = []
        seen: Dict[str, int] = {}
        current_input: Any = goal or f"Investigate finance exception {ledger_id}. Use approved tools and cite their results."
        proposed = ""
        loop_detected = False
        limit_reached = False
        forbidden_attempts = 0
        try:
            for step_number in range(self.max_steps):
                kwargs: Dict[str, Any] = {"model": self.reviewer.model_name, "input": current_input, "tools": self.tool_schemas(),
                    "system_instruction": "You are a bounded finance investigator. Use only approved read/verification tools. Never alter records or override controller policy."}
                if interaction_id:
                    kwargs["previous_interaction_id"] = interaction_id
                response = self.reviewer.client.interactions.create(timeout=60.0, **kwargs)
                self.metrics.gemini_calls += 1
                interaction_id = getattr(response, "id", None) or (response.get("id") if isinstance(response, dict) else None)
                raw_steps = getattr(response, "steps", None) or (response.get("steps", []) if isinstance(response, dict) else [])
                calls = []
                for raw in raw_steps:
                    item = self._step_from(raw)
                    if item and item.get("type") == "function_call":
                        calls.append(item)
                if not calls:
                    proposed = getattr(response, "output_text", None) or (response.get("output_text", "") if isinstance(response, dict) else "") or "Investigation completed from verified evidence."
                    break
                results = []
                for call in calls:
                    name = call.get("name", "")
                    arguments = call.get("arguments", {})
                    key = json.dumps([name, arguments], sort_keys=True, default=str)
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > 1:
                        loop_detected = True
                        result = {"status": "loop_detected", "reason": "Repeated tool and arguments stopped."}
                    else:
                        result = self._execute(name, arguments)
                    forbidden_attempts += int(result.get("status") == "tool_rejected")
                    called.append(name)
                    steps.append(AgentStepAudit(interaction_id, call.get("id"), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ledger_id, name, arguments if isinstance(arguments, dict) else {}, result.get("status", "unknown"), {k: result[k] for k in result if k in {"status", "candidate_count", "ledger_id", "reason"}}))
                    results.append({"type": "function_result", "call_id": call.get("id", ""), "name": name, "result": result, "is_error": result.get("status") in {"tool_rejected", "invalid_arguments"}})
                    if loop_detected:
                        break
                if loop_detected:
                    break
                current_input = results
            else:
                limit_reached = True
        except Exception as error:
            final = FinanceControllerPolicy.finalize(case, "")
            failure_category = self.reviewer._classify_exception(error)
            result = ControllerAgentResult(ledger_id, final["decision"], "Gemini investigation unavailable; controller decision retained.", [], called, case.recommended_action, case.confidence, final["requires_human_review"], "INTERACTION_ERROR", interaction_id, steps, loop_detected, limit_reached, True, time.perf_counter() - started, failure_category, forbidden_attempts)
            self.metrics.failed_runs += 1
            self.metrics.total_tool_calls += len(steps)
            self.metrics.tool_successes += sum(step.result_status == "success" for step in steps)
            self.metrics.tool_failures += sum(step.result_status != "success" for step in steps)
            self.metrics.forbidden_tool_attempts += forbidden_attempts
            self.metrics.max_tool_calls = max(self.metrics.max_tool_calls, len(steps))
            self.metrics.loop_stops += int(loop_detected)
            self.metrics.step_limit_stops += int(limit_reached)
            self.metrics.total_investigation_seconds += result.latency_seconds
            return result

        final = FinanceControllerPolicy.finalize(case, proposed)
        status = "AGENT_STEP_LIMIT_REACHED" if limit_reached else "AGENT_LOOP_DETECTED" if loop_detected else "SUCCESS"
        unsafe_proposal = any(term in proposed.lower() for term in ("select candidate", "auto-resolve", "approve", "mark as matched"))
        controller_override = bool(unsafe_proposal and final["decision"] == "HUMAN_REVIEW_REQUIRED")
        result = ControllerAgentResult(ledger_id, final["decision"], proposed or "Investigation stopped; controller decision retained.", [s.tool_name for s in steps], called, case.recommended_action, case.confidence, final["requires_human_review"] or limit_reached or loop_detected, status, interaction_id, steps, loop_detected, limit_reached, False, time.perf_counter() - started, None, forbidden_attempts, controller_override)
        self.metrics.successful_runs += int(status == "SUCCESS")
        self.metrics.failed_runs += int(status != "SUCCESS")
        self.metrics.total_tool_calls += len(steps)
        self.metrics.tool_successes += sum(step.result_status == "success" for step in steps)
        self.metrics.tool_failures += sum(step.result_status != "success" for step in steps)
        self.metrics.forbidden_tool_attempts += forbidden_attempts
        self.metrics.controller_overrides += int(controller_override)
        self.metrics.max_tool_calls = max(self.metrics.max_tool_calls, len(steps))
        self.metrics.loop_stops += int(loop_detected)
        self.metrics.step_limit_stops += int(limit_reached)
        self.metrics.total_investigation_seconds += result.latency_seconds
        return result
