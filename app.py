"""Streamlit Finance Controller operations dashboard."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.agent import BatchResult, FinanceControllerAgent
from src.qa import FinanceControllerQA

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Finance Controller | Razorpay",
    page_icon="FC",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def run_reconciliation(custom_sources: Optional[dict[str, pd.DataFrame]] = None) -> BatchResult:
    """Execute the sole runtime reconciliation pipeline."""
    return FinanceControllerAgent().run_reconciliation_batch(custom_sources)


def _status_class(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def _decision_rows(result: BatchResult) -> list[dict]:
    return [
        {
            "Ledger ID": d.ledger_id,
            "Status": d.status,
            "Amount": d.evidence.get("ledger_amount"),
            "Date": d.evidence.get("ledger_date"),
            "Confidence": d.confidence,
            "Matching method": d.matching_method,
            "Bank": d.bank_id or "-",
            "Invoice": d.invoice_id or "-",
            "Settlement": d.settlement_id or "-",
            "Exception": d.exception_type,
            "Human review": "Required" if d.requires_human_review else "No",
        }
        for d in result.decisions
    ]


def _audit_rows(result: BatchResult) -> list[dict]:
    return [
        {
            "Timestamp": event.timestamp,
            "Event ID": event.event_id,
            "Run ID": event.run_id or result.run_id,
            "Ledger ID": event.ledger_id,
            "Decision": event.decision,
            "Exception": event.exception_type,
            "Recommendation": event.recommended_action,
            "Human review": "Required" if event.requires_human_review else "No",
            "Gemini": event.llm_status_code or "Not attempted",
        }
        for event in result.audit_events
    ]


def _metric(label: str, value: object, detail: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-detail'>{detail}</div></div>",
        unsafe_allow_html=True,
    )


def _render_overview(result: BatchResult) -> None:
    summary = result.summary
    st.markdown("## Finance Controller")
    st.markdown("<div class='page-kicker'>RECONCILIATION OPERATIONS</div>", unsafe_allow_html=True)
    st.markdown("# Latest reconciliation", unsafe_allow_html=True)
    st.markdown(
        f"<div class='run-strip'><span class='run-label'>RUN ID</span>"
        f"<code>{result.run_id}</code><span class='run-time'>{result.completed_at}</span></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]: _metric("Records processed", summary.records_processed, "source rows")
    with cols[1]: _metric("Matched", summary.matched, f"{summary.matched / summary.records_processed * 100:.1f}% of run" if summary.records_processed else "0%")
    with cols[2]: _metric("Human review", summary.escalated, "controller escalations")
    with cols[3]: _metric("Exceptions", summary.exceptions, "ambiguous or conflicting")

    st.markdown("### Reconciliation health")
    health_cols = st.columns(4)
    with health_cols[0]: _metric("Partial", summary.partial, "incomplete source coverage")
    with health_cols[1]: _metric("Unresolved", summary.unresolved, "no reliable correspondence")
    with health_cols[2]: _metric("Tax checks", summary.tax_checks, f"{summary.tax_matches} matched")
    with health_cols[3]: _metric("Agent throughput", f"{result.performance['agent_throughput']:.1f}/s", "records per second")

    st.markdown("### System status")
    statuses = st.columns(3)
    with statuses[0]:
        st.markdown(f"<div class='status-panel'><span class='status-dot ready'></span><b>Controller</b><span>Ready</span></div>", unsafe_allow_html=True)
    with statuses[1]:
        ml_state = "Ready" if result.ml.get("ml_available") else "Unavailable"
        ml_class = "ready" if result.ml.get("ml_available") else "warning"
        st.markdown(f"<div class='status-panel'><span class='status-dot {ml_class}'></span><b>ML matcher</b><span>{ml_state}</span></div>", unsafe_allow_html=True)
    with statuses[2]:
        gemini_state = "Configured" if result.gemini.get("configured") else "Unavailable"
        gemini_class = "ready" if result.gemini.get("configured") else "warning"
        st.markdown(f"<div class='status-panel'><span class='status-dot {gemini_class}'></span><b>Gemini</b><span>{gemini_state}</span></div>", unsafe_allow_html=True)

    st.markdown("### Recent cases requiring review")
    review_rows = [row for row in _decision_rows(result) if row["Human review"] == "Required"][:8]
    if review_rows:
        st.dataframe(pd.DataFrame(review_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No cases currently require human review.")


def _render_reconciliation(result: BatchResult) -> None:
    st.markdown("## Reconciliation")
    st.markdown("Review every controller decision from the latest runtime batch.")
    rows = _decision_rows(result)
    filters = st.columns([2, 1, 1])
    with filters[0]: query = st.text_input("Search ledger ID or exception", placeholder="LED-0034")
    with filters[1]: status = st.selectbox("Status", ["All"] + sorted({row["Status"] for row in rows}))
    with filters[2]: review = st.selectbox("Review", ["All", "Required", "No"])
    filtered = [row for row in rows if (not query or query.lower() in str(row).lower()) and (status == "All" or row["Status"] == status) and (review == "All" or row["Human review"] == review)]
    st.caption(f"Showing {len(filtered)} of {len(rows)} controller decisions")
    st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)


def _render_exceptions(result: BatchResult) -> None:
    st.markdown("## Exceptions")
    st.markdown("Controller decisions retained for finance-operations review. No write-back actions are available.")
    cases = [d for d in result.decisions if d.requires_human_review]
    if not cases:
        st.success("No human-review cases in this run.")
        return
    for case in cases:
        llm_status = (case.llm_review or {}).get("status_code", "Not attempted")
        with st.expander(f"{case.ledger_id}  |  {case.status}  |  {case.exception_type}", expanded=False):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"**Controller recommendation**  \n{case.recommended_action}")
                st.markdown(f"**Evidence**  \n{case.evidence}")
                explanation = (case.llm_review or {}).get("explanation") or "AI explanation unavailable; the controller decision remains unchanged."
                st.markdown(f"**Gemini status: {llm_status}**  \n{explanation}")
            with right:
                st.metric("Amount", f"₹{case.evidence.get('ledger_amount', 0):,.2f}")
                st.metric("Controller confidence", f"{case.confidence:.2f}")
                st.write(f"Date: {case.evidence.get('ledger_date', '-')}")
                st.write(f"Audit event: `{case.audit_event_id}`")


def _render_agent(result: BatchResult) -> None:
    st.markdown("## Agent Activity")
    st.markdown("Gemini is an explanation layer. The controller remains authoritative.")
    cols = st.columns(4)
    with cols[0]: _metric("Configured", "Yes" if result.gemini.get("configured") else "No", "Gemini availability")
    with cols[1]: _metric("Eligible cases", result.gemini.get("eligible_cases", 0), "exception reasoning")
    with cols[2]: _metric("Successful reviews", result.gemini.get("successful_reviews", 0), "validated responses")
    with cols[3]: _metric("Fallback cases", result.gemini.get("fallback_cases", 0), "controller retained")
    tool_steps = []
    for event in result.audit_events:
        if event.tools_used and any("review_exception" in tool for tool in event.tools_used):
            tool_steps.append({"Ledger ID": event.ledger_id, "Audit event": event.event_id, "Gemini status": event.llm_status_code or "Not attempted", "Controller": event.decision})
    if tool_steps:
        st.dataframe(pd.DataFrame(tool_steps), use_container_width=True, hide_index=True)
    else:
        st.info("No autonomous Gemini tool activity was recorded in this batch. Gemini exception status is shown per case in Exceptions.")


def _render_audit(result: BatchResult) -> None:
    st.markdown("## Audit Trail")
    st.markdown(f"Run `{result.run_id}` · {len(result.audit_events)} immutable audit events")
    st.dataframe(pd.DataFrame(_audit_rows(result)), use_container_width=True, hide_index=True)


def _render_reports(result: BatchResult) -> None:
    st.markdown("## Reports")
    st.markdown(f"Generated from runtime run `{result.run_id}`. Offline evaluation is separate.")
    report_file = ROOT / "report.md"
    json_file = DATA_DIR / "exceptions.json"
    for path, label, mime in [(report_file, "Download runtime report", "text/markdown"), (json_file, "Download exception export", "application/json")]:
        if path.exists():
            st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)
        else:
            st.info(f"{path.name} is not available yet.")


def _render_qa(result: BatchResult) -> None:
    st.markdown("## Ask Finance Controller")
    st.markdown("Ask about the current runtime result. Factual answers come from structured controller data.")
    question = st.text_input("Question", placeholder="Why was LED-0034 escalated?", key="controller_question")
    if question:
        answer = FinanceControllerQA(result).answer_question(question)
        st.markdown(f"<div class='answer-panel'><div class='answer-status'>{answer.category} · {answer.ai_status}</div><div class='answer-text'>{answer.answer}</div></div>", unsafe_allow_html=True)
        if answer.evidence:
            st.dataframe(pd.DataFrame(answer.evidence), use_container_width=True, hide_index=True)


def main() -> None:
    if "batch_result" not in st.session_state:
        st.session_state.batch_result = None

    st.markdown("<header class='app-header'><div class='brand-lockup'><div><div class='brand-name'>Finance Controller</div><div class='brand-sub'>Razorpay operations</div></div></div><div class='header-context'>Internal finance operations</div></header>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Controller workspace</div>", unsafe_allow_html=True)
        section = st.radio("Navigate", ["Overview", "Reconciliation", "Exceptions", "Agent Activity", "Audit Trail", "Reports", "Ask Finance Controller"], label_visibility="collapsed")
        st.divider()
        st.markdown("<div class='sidebar-label'>Data source</div>", unsafe_allow_html=True)
        input_mode = st.radio("Data source", ["Demo data", "Upload source files"], label_visibility="collapsed")
        custom_data = None
        if input_mode == "Upload source files":
            uploads = {name: st.file_uploader(f"{label} CSV", type=["csv"], key=f"upload_{name}") for name, label in [("ledger", "Ledger"), ("bank", "Bank statements"), ("invoice", "Invoices"), ("settlement", "Settlements")]}
            missing = [label for (name, label), upload in zip([("ledger", "Ledger"), ("bank", "Bank statements"), ("invoice", "Invoices"), ("settlement", "Settlements")], uploads.values()) if upload is None]
            if not missing:
                custom_data = {name: pd.read_csv(upload) for name, upload in uploads.items()}
            else:
                st.caption("Required: " + ", ".join(missing))
        if st.button("Run reconciliation", type="primary", use_container_width=True):
            if input_mode == "Upload source files" and custom_data is None:
                st.error("Upload all four source CSVs before running reconciliation.")
            else:
                with st.spinner("Running controller pipeline..."):
                    st.session_state.batch_result = run_reconciliation(custom_data)
                st.rerun()
        if st.session_state.batch_result:
            st.divider()
            st.caption(f"Latest run\n{st.session_state.batch_result.run_id}")

    result = st.session_state.batch_result
    if result is None:
        st.markdown("<main class='empty-state'><div class='page-kicker'>FINANCE CONTROLLER</div><h1>Run a reconciliation batch</h1><p>Choose demo data or upload the four source files, then run the controller to populate this workspace.</p></main>", unsafe_allow_html=True)
        return

    if section == "Overview": _render_overview(result)
    elif section == "Reconciliation": _render_reconciliation(result)
    elif section == "Exceptions": _render_exceptions(result)
    elif section == "Agent Activity": _render_agent(result)
    elif section == "Audit Trail": _render_audit(result)
    elif section == "Reports": _render_reports(result)
    else: _render_qa(result)


if __name__ == "__main__":
    main()
