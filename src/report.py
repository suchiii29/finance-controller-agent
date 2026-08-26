"""
report.py
=========
Report generator for the AI Finance Controller project.

Produces:
1. Console summary output
2. data/exceptions.json (machine-readable structured exception export)
3. report.md (human-readable executive summary & detailed report)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.matcher import ReconciliationResult
from src.evaluate import EvaluationResult
from src.agent import ExceptionAnalysis

REPORT_DIR = Path(__file__).parent.parent / "reports"
DATA_DIR = Path(__file__).parent.parent / "data"


def generate_final_report(
    matcher_result: ReconciliationResult,
    evaluation_result: EvaluationResult,
    analyses: List[ExceptionAnalysis],
) -> Tuple[Path, Path]:
    """
    Generate final exports: exceptions.json and report.md.
    Returns paths to (exceptions_json_path, report_md_path).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "exceptions.json"
    md_path = Path(__file__).parent.parent / "report.md"

    # 1. Export exceptions.json
    export_data = [a.to_dict() for a in analyses]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # 2. Compute Category Breakdown
    total = matcher_result.total_processed
    deterministic_matched = matcher_result.status_counts.get("MATCHED", 0)
    agent_auto_resolved = sum(1 for a in analyses if a.safe_auto_resolved)
    escalated_exceptions = len(analyses) - agent_auto_resolved

    effective_resolved = deterministic_matched + agent_auto_resolved
    effective_match_rate = (effective_resolved / total * 100) if total > 0 else 0.0

    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for a in analyses:
        risk_counts[a.risk_level] = risk_counts.get(a.risk_level, 0) + 1

    md_content = f"""# Executive Reconciliation & Exception Report
**Razorpay AI Buildathon Track 04 — AI Finance Controller System**

---

## 1. Executive Summary

| Category / Metric | Count / Value | Percentage |
| :--- | :--- | :--- |
| **Total Anchor Records Processed** | {total} | 100.0% |
| **Fully Matched (Deterministic Rules)** | {deterministic_matched} | {deterministic_matched/total*100:.1f}% |
| **Safely Auto-Resolved (AI Agent)** | {agent_auto_resolved} | {agent_auto_resolved/total*100:.1f}% |
| **Still Escalated Exceptions (Finance Ops)** | {escalated_exceptions} | {escalated_exceptions/total*100:.1f}% |
| **Total Effective Automation Rate** | **{effective_resolved} / {total}** | **{effective_match_rate:.1f}%** |
| **Processing Throughput** | **{matcher_result.throughput_per_second:.0f} rec/sec** | — |

> **Financial Safety Statement:**
> The system operates under strict financial control policies. Only single non-cash timing lags with 100% Tier-1 corroboration and zero collision risk are safely auto-resolved ({agent_auto_resolved} cases). All {escalated_exceptions} remaining risk cases (such as missing bank statement feeds) are safely escalated to Finance Ops with detailed root-cause explanations.

---

## 2. Exception Risk Breakdown

- **HIGH Risk Cases**: {risk_counts['HIGH']} (Requires immediate manual audit / ops action)
- **MEDIUM Risk Cases**: {risk_counts['MEDIUM']} (Missing primary cash feeds - Bank statement gap)
- **LOW Risk Cases**: {risk_counts['LOW']} (Safely auto-resolved non-cash timing lags)

---

## 3. Exception & Resolution Register

| Ledger ID | Original Status | Agent Final Status | Risk Level | Actionable Recommendation | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for a in analyses:
        md_content += f"| `{a.ledger_id}` | {a.original_status} | **{a.final_status}** | **{a.risk_level}** | {a.recommended_action} | {a.explanation} |\n"

    md_content += f"""
---

## 4. System Performance against Ground Truth

- **Match Precision**: {evaluation_result.precision * 100:.1f}%
- **Match Recall**: {evaluation_result.recall * 100:.1f}%
- **F1 Score**: {evaluation_result.f1 * 100:.1f}%
- **False Negatives**: {evaluation_result.false_negatives}
- **False Positives**: {evaluation_result.false_positives}

---

## 5. System Architecture & Safety Controls

1. **Deterministic Layer**: Performs high-confidence multi-tier matching against observable source feeds.
2. **AI Exception Agent Layer**: Evaluates residual cases with strict guardrails:
   - *Never auto-resolves missing bank statement feeds.*
   - *Never auto-resolves when duplicate reference collisions exist.*
   - *Proposes safe auto-resolution only when 2 available feeds agree 100% on Tier-1 exact matching.*
3. **Audit Trail & Traceability**: Every decision is fully logged in `data/exceptions.json`.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path
