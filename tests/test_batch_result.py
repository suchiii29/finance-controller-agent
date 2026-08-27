import json
import unittest
from pathlib import Path

from src.agent import FinanceControllerAgent
from src.qa import FinanceControllerQA
from src.report import generate_final_report


class TestBatchResultConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).parent.parent / "data"
        cls.result = FinanceControllerAgent(data_dir=cls.data_dir, api_key="").run_reconciliation_batch()

    def test_batch_result_is_authoritative_and_propagated(self):
        result = self.result
        self.assertTrue(result.run_id.startswith("RUN-"))
        self.assertEqual(len(result.decisions), result.summary.records_processed)
        self.assertEqual(len(result.audit_events), result.summary.records_processed)
        self.assertTrue(all(event.run_id == result.run_id for event in result.audit_events))
        self.assertEqual(result.summary.human_review if hasattr(result.summary, "human_review") else result.summary.escalated,
                         sum(d.requires_human_review for d in result.decisions))

    def test_report_json_and_qa_use_same_result(self):
        result = self.result
        json_path, report_path = generate_final_report(result)
        records = json.loads(json_path.read_text(encoding="utf-8"))
        expected = [d for d in result.decisions if d.requires_human_review or d.status in {"EXCEPTION", "UNRESOLVED"}]
        self.assertEqual(len(records), len(expected))
        self.assertTrue(all(record["run_id"] == result.run_id for record in records))
        report = report_path.read_text(encoding="utf-8")
        self.assertIn(result.run_id, report)
        self.assertIn(f"| Records processed | {result.summary.records_processed} |", report)
        qa = FinanceControllerQA(result)
        answer = qa.answer_question("How many cases require human review?")
        self.assertIn(str(sum(d.requires_human_review for d in result.decisions)), answer.answer)

    def test_dashboard_source_uses_batch_result(self):
        app_source = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
        self.assertIn("BatchResult", app_source)
        self.assertNotIn("ReconciliationMatcher()", app_source)
        self.assertNotIn("ExceptionAgent()", app_source)


if __name__ == "__main__":
    unittest.main()
