import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent import AgentDecision, AuditEvent
from src.controller_agent import BoundedGeminiControllerAgent, MAX_AGENT_STEPS
from src.llm_reviewer import GeminiExceptionReviewer


def decision(status="EXCEPTION", exception_type="duplicate_reference", human=True):
    return AgentDecision(
        ledger_id="LED-0034", bank_id="BNK-0034", invoice_id="INV-0034", settlement_id="STL-0034",
        status=status, confidence=0.9, matching_method="Tier-2", exception_type=exception_type,
        recommended_action="Escalate: Duplicate reference collision", requires_human_review=human,
        audit_event_id="AUD-0034", evidence={"ledger_amount": 100.0, "ledger_date": "2024-01-01",
        "ledger_reference": "REF-34", "verifications": {"dup_check": {"has_collision": True, "collisions": {"settlement": ["STL-0034", "STL-0035"]}},
        "tax_check": {"status": "TAX_MATCH"}}},
    )


class TestBoundedGeminiControllerAgent(unittest.TestCase):
    def setUp(self):
        self.audit = AuditEvent("AUD-0034", "2024-01-01T00:00:00Z", "LED-0034", {}, "EXCEPTION", 0.9, [], {}, "duplicate_reference", "Manual review", True)

    def reviewer(self):
        return GeminiExceptionReviewer(api_key="mock_key", model_name="gemini-3.7-flash")

    @patch("google.genai.Client")
    def test_multi_step_executes_tools_and_returns_results(self, client_cls):
        first = SimpleNamespace(id="INT-1", steps=[{"type": "function_call", "id": "CALL-1", "name": "get_reconciliation_case", "arguments": {"ledger_id": "LED-0034"}}])
        second = SimpleNamespace(id="INT-2", steps=[{"type": "function_call", "id": "CALL-2", "name": "inspect_duplicate_reference", "arguments": {"ledger_id": "LED-0034"}}])
        third = SimpleNamespace(id="INT-3", steps=[{"type": "function_call", "id": "CALL-3", "name": "compare_candidate_evidence", "arguments": {"ledger_id": "LED-0034", "candidate_ids": ["STL-0034", "STL-0035"]}}])
        fourth = SimpleNamespace(id="INT-4", steps=[{"type": "function_call", "id": "CALL-4", "name": "get_audit_history", "arguments": {"ledger_id": "LED-0034"}}])
        fifth = SimpleNamespace(id="INT-5", steps=[], output_text="Evidence remains ambiguous. Human review required.")
        client = MagicMock()
        client.interactions.create.side_effect = [first, second, third, fourth, fifth]
        client_cls.return_value = client
        agent = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer())
        result = agent.investigate("LED-0034")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.final_decision, "HUMAN_REVIEW_REQUIRED")
        self.assertEqual(result.tools_called, ["get_reconciliation_case", "inspect_duplicate_reference", "compare_candidate_evidence", "get_audit_history"])
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(len(result.steps), 4)
        self.assertFalse(result.controller_override)
        self.assertEqual(agent.metrics.controller_overrides, 0)
        self.assertTrue(agent.audit_events["LED-0034"].event_id.startswith("AUD-"))
        self.assertEqual(client.interactions.create.call_count, 5)
        self.assertEqual(client.interactions.create.call_args_list[1].kwargs["previous_interaction_id"], "INT-1")
        self.assertEqual(client.interactions.create.call_args_list[1].kwargs["input"][0]["type"], "function_result")
        self.assertEqual(client.interactions.create.call_args_list[3].kwargs["previous_interaction_id"], "INT-3")
        self.assertTrue(all(step.result_status == "success" for step in result.steps))

    @patch("google.genai.Client")
    def test_loop_detection_stops_repeated_call(self, client_cls):
        call = {"type": "function_call", "id": "CALL-1", "name": "get_reconciliation_case", "arguments": {"ledger_id": "LED-0034"}}
        client = MagicMock()
        client.interactions.create.side_effect = [SimpleNamespace(id="INT-1", steps=[call]), SimpleNamespace(id="INT-2", steps=[call])]
        client_cls.return_value = client
        result = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer()).investigate("LED-0034")
        self.assertTrue(result.loop_detected)
        self.assertEqual(result.status, "AGENT_LOOP_DETECTED")
        self.assertTrue(result.requires_human_review)

    @patch("google.genai.Client")
    def test_step_limit_stops(self, client_cls):
        call = {"type": "function_call", "id": "CALL-1", "name": "get_controller_policy", "arguments": {"ledger_id": "LED-0034"}}
        client = MagicMock()
        second_call = {**call, "id": "CALL-2", "arguments": {"ledger_id": ""}}
        client.interactions.create.side_effect = [SimpleNamespace(id="INT-1", steps=[call]), SimpleNamespace(id="INT-2", steps=[second_call])]
        client_cls.return_value = client
        result = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer(), max_steps=2).investigate("LED-0034")
        self.assertTrue(result.step_limit_reached)
        self.assertEqual(result.status, "AGENT_STEP_LIMIT_REACHED")
        self.assertEqual(client.interactions.create.call_count, 2)

    def test_forbidden_and_malformed_tools_rejected(self):
        agent = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer())
        self.assertEqual(agent._execute("send_payment", {"ledger_id": "LED-0034"})["status"], "tool_rejected")
        self.assertEqual(agent._execute("get_reconciliation_case", ["LED-0034"])["status"], "invalid_arguments")
        self.assertNotIn("send_payment", {tool["name"] for tool in agent.tool_schemas()})
        schemas = {tool["name"]: tool for tool in agent.tool_schemas()}
        self.assertIn("source", schemas["find_candidate_records"]["parameters"]["properties"])
        self.assertIn("candidate_id", schemas["verify_amount"]["parameters"]["properties"])

    @patch("google.genai.Client")
    def test_unsafe_gemini_proposal_is_overridden_by_controller(self, client_cls):
        final = SimpleNamespace(id="INT-1", steps=[], output_text="Select candidate A.")
        client = MagicMock()
        client.interactions.create.return_value = final
        client_cls.return_value = client
        agent = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer())
        result = agent.investigate("LED-0034")
        self.assertEqual(result.final_decision, "HUMAN_REVIEW_REQUIRED")
        self.assertTrue(result.controller_override)
        self.assertEqual(agent.metrics.controller_overrides, 1)

    @patch("google.genai.Client")
    def test_forbidden_model_call_is_executed_as_rejection(self, client_cls):
        forbidden = SimpleNamespace(id="INT-1", steps=[{"type": "function_call", "id": "CALL-1", "name": "update_ledger", "arguments": {"ledger_id": "LED-0034"}}])
        final = SimpleNamespace(id="INT-2", steps=[], output_text="I cannot modify records.")
        client = MagicMock()
        client.interactions.create.side_effect = [forbidden, final]
        client_cls.return_value = client
        agent = BoundedGeminiControllerAgent([decision()], [self.audit], self.reviewer())
        result = agent.investigate("LED-0034")
        self.assertEqual(result.steps[0].result_status, "tool_rejected")
        self.assertEqual(result.forbidden_tool_attempts, 1)
        self.assertEqual(agent.metrics.forbidden_tool_attempts, 1)
        self.assertEqual(client.interactions.create.call_count, 2)

    def test_unavailable_fallback_retains_controller(self):
        agent = BoundedGeminiControllerAgent([decision()], [self.audit], GeminiExceptionReviewer(api_key=""))
        result = agent.investigate("LED-0034")
        self.assertEqual(result.status, "GEMINI_UNAVAILABLE")
        self.assertEqual(result.final_decision, "HUMAN_REVIEW_REQUIRED")
        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.failure_category)
        self.assertEqual(agent.metrics.agent_runs, 1)
        self.assertEqual(agent.metrics.failed_runs, 1)


if __name__ == "__main__":
    unittest.main()
