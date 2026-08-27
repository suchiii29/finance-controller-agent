"""
app.py
======
AI Finance Controller Agent — Streamlit Web Application
Razorpay AI Buildathon Track 04
"""

import time
import textwrap
import pandas as pd
import streamlit as st
from pathlib import Path

from src.matcher import ReconciliationMatcher
from src.evaluate import evaluate
from src.agent import ExceptionAgent
from src.report import generate_final_report

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Finance Controller Agent | AI Reconciliation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External CSS
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.html(css_path)

# ---------------------------------------------------------------------------
# Pipeline Execution Helper
# ---------------------------------------------------------------------------
def run_reconciliation(custom_sources=None):
    """Run full reconciliation pipeline."""
    matcher = ReconciliationMatcher()
    if custom_sources:
        matcher.load_sources_from_dict(custom_sources)
    else:
        matcher.load_sources()

    matcher_res = matcher.reconcile()

    source_records = {
        "ledger": {r["record_id"]: r for r in matcher._ledger_records},
        "bank": {r["record_id"]: r for r in matcher._bank_records},
        "invoice": {r["record_id"]: r for r in matcher._invoice_records},
        "settlement": {r["record_id"]: r for r in matcher._settlement_records},
    }

    agent = ExceptionAgent()
    analyses = agent.analyze_residuals(matcher_res.decisions, source_records)
    eval_res = evaluate(matcher_res)

    return matcher_res, source_records, analyses, eval_res


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main():
    # 1. TOP INFORMATION STRIP
    st.html(textwrap.dedent("""
    <div class="fc-strip">
        <div class="fc-strip-left">
            <span class="fc-pill">Finance Controller</span>
            <span class="fc-strip-msg">AI-powered finance operations</span>
            <span class="fc-strip-sub">&nbsp;•&nbsp; Reconcile records, surface exceptions, and close workflows faster.</span>
        </div>
        <div>
            <a href="#workspace" class="fc-strip-btn">Explore Controller</a>
        </div>
    </div>
    """))

    # 2. MAIN NAVIGATION & 3. MEGA MENU
    st.html(textwrap.dedent("""
    <div class="fc-nav">
        <div class="fc-nav-brand">
            <div class="fc-logo-mark">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="5" width="20" height="14" rx="2"/>
                    <line x1="2" y1="10" x2="22" y2="10"/>
                </svg>
            </div>
            <span class="fc-nav-name">Finance Controller Agent</span>
        </div>

        <div class="fc-nav-links">
            <div class="fc-dropdown">
                <span class="fc-nav-link">
                    Controller
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left:4px; vertical-align:middle;">
                        <path d="M6 9l6 6 6-6"/>
                    </svg>
                </span>
                <div class="fc-megamenu">
                    <div class="fc-menu-heading">FINANCE OPERATIONS</div>
                    <div class="fc-menu-cols">
                        <div>
                            <a href="#workspace" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Reconciliation</div>
                                    <div class="fc-menu-desc">Match financial records across sources</div>
                                </div>
                            </a>
                            <a href="#exceptions" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Exception Center</div>
                                    <div class="fc-menu-desc">Review unresolved finance issues</div>
                                </div>
                            </a>
                            <a href="#workspace" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Settlement Review</div>
                                    <div class="fc-menu-desc">Verify settlement consistency</div>
                                </div>
                            </a>
                        </div>

                        <div>
                            <a href="#workspace" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Smart Matching</div>
                                    <div class="fc-menu-desc">Find high-confidence record relationships</div>
                                </div>
                            </a>
                            <a href="#exceptions" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Variance Detection</div>
                                    <div class="fc-menu-desc">Surface amount, date & tax discrepancies</div>
                                </div>
                            </a>
                            <a href="#exceptions" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Tax Verification</div>
                                    <div class="fc-menu-desc">Check tax-line consistency</div>
                                </div>
                            </a>
                        </div>

                        <div>
                            <a href="#audit-log" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Audit Trail</div>
                                    <div class="fc-menu-desc">Track every reconciliation decision</div>
                                </div>
                            </a>
                            <a href="#analytics" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Finance Insights</div>
                                    <div class="fc-menu-desc">Understand reconciliation performance</div>
                                </div>
                            </a>
                            <a href="#audit-log" class="fc-menu-item">
                                <div class="fc-menu-icon">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                </div>
                                <div>
                                    <div class="fc-menu-label">Controller Assistant</div>
                                    <div class="fc-menu-desc">Ask grounded questions on operations</div>
                                </div>
                            </a>
                        </div>
                    </div>
                    <div class="fc-menu-footer">
                        Enterprise multi-source reconciliation engine • Track 04 Buildathon
                    </div>
                </div>
            </div>
            <a href="#workspace" class="fc-nav-link">Reconciliation</a>
            <a href="#exceptions" class="fc-nav-link">Exceptions</a>
            <a href="#analytics" class="fc-nav-link">Analytics</a>
            <a href="#audit-log" class="fc-nav-link">Audit Trail</a>
        </div>

        <div class="fc-nav-right">
            <a href="#workspace" class="fc-btn-primary">Open Controller</a>
        </div>
    </div>
    """))

    # 4. HERO SECTION & 5. HERO VISUAL
    st.html(textwrap.dedent("""
    <div class="fc-hero-wrap">
        <div class="fc-hero">
            <div>
                <div class="fc-eyebrow">AI FINANCE CONTROLLER</div>
                <h1 class="fc-h1">Close your finance ops loop <span class="accent">with AI.</span></h1>
                <p class="fc-hero-sub">Reconcile financial records across sources, resolve discrepancies safely, and know exactly what still needs human review.</p>
                <div class="fc-cta-row">
                    <a href="#workspace" class="fc-cta-p">Open Controller</a>
                    <a href="#exceptions" class="fc-cta-s">View Exceptions</a>
                </div>
                <div class="fc-trust">
                    <strong>100+ records</strong> &nbsp;•&nbsp; measured accuracy &nbsp;•&nbsp; auditable decisions
                </div>
            </div>

            <div class="fc-vis">
                <div class="fc-vis-grid">
                    <div class="fc-vis-card">
                        <div class="fc-vis-src">BANK</div>
                        <div class="fc-vis-amt">₹4,500.00</div>
                        <span class="badge badge-ok">✓ Matched</span>
                    </div>
                    <div class="fc-vis-card">
                        <div class="fc-vis-src">LEDGER</div>
                        <div class="fc-vis-amt">₹4,500.00</div>
                        <span class="badge badge-ok">✓ Matched</span>
                    </div>
                    <div class="fc-vis-card">
                        <div class="fc-vis-src">INVOICE</div>
                        <div class="fc-vis-amt">₹4,500.00</div>
                        <span class="badge badge-ok">✓ Matched</span>
                    </div>
                    <div class="fc-vis-card">
                        <div class="fc-vis-src">SETTLEMENT</div>
                        <div class="fc-vis-amt">₹4,450.00</div>
                        <span class="badge badge-err">⚠️ Exception</span>
                    </div>
                </div>
                <div class="fc-vis-footer">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    <span><strong>₹50.00 variance detected</strong> — Gateway fee deduction flagged for audit review</span>
                </div>
            </div>
        </div>
    </div>
    """))

    # 6. DARK FEATURE SECTION BELOW HERO
    st.html(textwrap.dedent("""
    <div class="fc-dark">
        <div class="fc-dark-inner">
            <h2>Built for finance teams that need answers, not guesswork.</h2>
            <p class="sub">Automate the repeatable work. Surface the exceptions. Keep every decision traceable.</p>

            <div class="fc-feat-grid">
                <div class="fc-feat">
                    <div class="fc-feat-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    </div>
                    <h4>Reconcile faster</h4>
                    <p>Execute multi-way deterministic matching across bank statements, ledger, invoices, and gateway settlements in seconds.</p>
                </div>

                <div class="fc-feat">
                    <div class="fc-feat-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    </div>
                    <h4>Surface exceptions</h4>
                    <p>Detect amount mismatches, missing statement feeds, duplicate reference collisions, and timing lags automatically.</p>
                </div>

                <div class="fc-feat">
                    <div class="fc-feat-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    </div>
                    <h4>Stay audit-ready</h4>
                    <p>Every match and resolution generates a deterministic, plain-English audit log with clear financial reasoning.</p>
                </div>

                <div class="fc-feat">
                    <div class="fc-feat-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 1 0 7.75"/></svg>
                    </div>
                    <h4>Know what needs review</h4>
                    <p>Zero unverified cash resolutions. High-risk cash discrepancies are strictly escalated to human controllers.</p>
                </div>
            </div>
        </div>
    </div>
    """))

    # SIDEBAR CONFIGURATION
    st.sidebar.html(textwrap.dedent("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
            <div style="width:32px; height:32px; background:#2F5BFF; border-radius:8px; display:flex; align-items:center; justify-content:center;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>
            </div>
            <span style="font-weight:700; color:#0B1B33; font-size:16px;">Agent Controls</span>
        </div>
    """))

    input_mode = st.sidebar.radio(
        "Choose Data Source:",
        ["Use demo data (recommended for first run)", "Upload custom files"],
        index=0
    )

    custom_data = None
    if input_mode == "Upload custom files":
        st.sidebar.markdown("---")
        st.sidebar.html("<span style='font-size:13px; font-weight:700; color:#0B1B33;'>Upload Source CSVs</span>")
        led_file = st.sidebar.file_uploader("Ledger CSV", type=["csv"])
        bank_file = st.sidebar.file_uploader("Bank Statements CSV", type=["csv"])
        inv_file = st.sidebar.file_uploader("Invoices CSV", type=["csv"])
        stl_file = st.sidebar.file_uploader("Settlements CSV", type=["csv"])

        if led_file and bank_file and inv_file and stl_file:
            custom_data = {
                "ledger": pd.read_csv(led_file),
                "bank": pd.read_csv(bank_file),
                "invoice": pd.read_csv(inv_file),
                "settlement": pd.read_csv(stl_file),
            }
            st.sidebar.success("All 4 custom files uploaded successfully!")
        else:
            st.sidebar.warning("Please upload all 4 files to run custom reconciliation.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Track 04: Razorpay AI Buildathon")

    # 7. PRODUCT DASHBOARD SECTION
    st.html('<div id="workspace" class="fc-dash-wrap">')

    st.html(textwrap.dedent("""
    <div class="fc-dash-header">
        <h2>Reconciliation Controller Workspace</h2>
        <p>Run end-to-end reconciliation across Ledger, Bank, Invoice, and Gateway feeds.</p>
    </div>
    """))

    # Big Action Button
    run_clicked = st.button("⚡ Run AI Reconciliation Agent", type="primary", use_container_width=True)

    # State initialization & Auto-run demo mode
    if "ran" not in st.session_state:
        st.session_state.ran = False

    if run_clicked:
        with st.status("AI Agent Reconciling Financial Records...", expanded=True) as status:
            st.write("📥 Loading and normalising records from source systems...")
            time.sleep(0.3)
            st.write("🔍 Running 4-way deterministic reconciliation matcher...")
            time.sleep(0.3)
            st.write("🤖 AI Exception Agent evaluating residual cases & safety guardrails...")
            time.sleep(0.3)
            st.write("📊 Generating audit trail & resolution plan...")

            matcher_res, source_records, analyses, eval_res = run_reconciliation(custom_data)

            st.session_state.matcher_res = matcher_res
            st.session_state.source_records = source_records
            st.session_state.analyses = analyses
            st.session_state.eval_res = eval_res
            st.session_state.ran = True

            status.update(label="Reconciliation Complete!", state="complete", expanded=False)

    if not st.session_state.ran and input_mode == "Use demo data (recommended for first run)":
        matcher_res, source_records, analyses, eval_res = run_reconciliation(None)
        st.session_state.matcher_res = matcher_res
        st.session_state.source_records = source_records
        st.session_state.analyses = analyses
        st.session_state.eval_res = eval_res
        st.session_state.ran = True

    if st.session_state.ran:
        matcher_res = st.session_state.matcher_res
        analyses = st.session_state.analyses
        eval_res = st.session_state.eval_res

        total = matcher_res.total_processed
        det_matched = matcher_res.status_counts.get("MATCHED", 0)
        agent_auto = sum(1 for a in analyses if a.safe_auto_resolved)
        escalated = len(analyses) - agent_auto
        effective_resolved = det_matched + agent_auto
        auto_pct = (effective_resolved / total * 100) if total > 0 else 0.0

        # KPI Dashboard Cards
        st.html(textwrap.dedent(f"""
        <div id="analytics" class="fc-kpi-grid">
            <div class="fc-kpi">
                <div class="fc-kpi-lbl">RECORDS PROCESSED</div>
                <div class="fc-kpi-val">{total}</div>
                <div class="fc-kpi-delta delta-neu">100% Data Ingested</div>
            </div>
            <div class="fc-kpi">
                <div class="fc-kpi-lbl">SAFELY RESOLVED</div>
                <div class="fc-kpi-val">{effective_resolved}</div>
                <div class="fc-kpi-delta delta-pos">↑ {auto_pct:.1f}% Automated</div>
            </div>
            <div class="fc-kpi">
                <div class="fc-kpi-lbl">NEEDS HUMAN REVIEW</div>
                <div class="fc-kpi-val">{escalated}</div>
                <div class="fc-kpi-delta delta-neg">Escalated Exceptions</div>
            </div>
            <div class="fc-kpi">
                <div class="fc-kpi-lbl">MATCHING THROUGHPUT</div>
                <div class="fc-kpi-val">{matcher_res.throughput_per_second:.0f}</div>
                <div class="fc-kpi-delta delta-neu">records / second</div>
            </div>
        </div>

        <div class="fc-safety-box">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2F5BFF" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            <div class="fc-safety-txt">
                <strong>Financial Control & Cash Safety Guarantee:</strong>
                The AI Agent automatically resolves low-risk non-cash timing lags (e.g. missing invoice or settlement) ONLY when physical cash movement is 100% verified by the bank statement feed. Missing bank statement feeds and duplicate reference collisions are strictly escalated to human reviewers.
            </div>
        </div>
        """))

        st.html('<hr class="fc-divider">')

        # Tabbed Workspace
        tab_escalated, tab_resolved, tab_agent_log = st.tabs([
            f"⚠️ Exceptions Needing Review ({escalated})",
            f"✅ Safely Resolved Cases ({effective_resolved})",
            "🧠 Agent Decision Policy & Reasoning Log"
        ])

        # TAB 1: ESCALATED EXCEPTIONS
        with tab_escalated:
            st.html('<div id="exceptions"></div>')
            st.markdown("### Cases Escalated to Finance Operations")
            st.caption("These entries pose potential cash control risks (such as missing bank statement feeds) and require human verification.")

            escalated_cases = [a for a in analyses if not a.safe_auto_resolved]

            for case in escalated_cases:
                with st.expander(f"🔴 Ledger ID: {case.ledger_id} | Risk: {case.risk_level} | Action: {case.recommended_action}", expanded=True):
                    # Use 65% / 35% ratio for generous explanation space
                    c1, c2 = st.columns([65, 35])
                    with c1:
                        st.markdown("**Agent Explanation (Plain English):**")
                        st.write(case.detailed_explanation)
                        st.markdown("<div style='margin-top: 12px;'><strong>Recommended Action:</strong></div>", unsafe_allow_html=True)
                        st.warning(case.recommended_action)
                    with c2:
                        st.markdown("**Key Fields:**")
                        ev = case.evidence_summary["ledger"]
                        missing_feeds_str = ", ".join(case.missing_sources) if case.missing_sources else "None"
                        st.html(textwrap.dedent(f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; font-size:13px; line-height:1.7;">
                            <div><strong style="color:#0B1B33;">Ledger ID:</strong> <code style="word-break:break-word;">{case.ledger_id}</code></div>
                            <div><strong style="color:#0B1B33;">Amount:</strong> <code style="word-break:break-word;">₹{ev['amount']:,.2f}</code></div>
                            <div><strong style="color:#0B1B33;">Date:</strong> <code style="word-break:break-word;">{ev['date']}</code></div>
                            <div><strong style="color:#0B1B33;">Reference:</strong> <code style="word-break:break-word;">{ev['reference']}</code></div>
                            <div><strong style="color:#0B1B33;">Counterparty:</strong> <span style="color:#334155; word-break:break-word;">{ev['counterparty']}</span></div>
                            <div><strong style="color:#0B1B33;">Missing Feed:</strong> <span style="color:#DC2626; font-weight:600; word-break:break-word;">{missing_feeds_str}</span></div>
                        </div>
                        """))

        # TAB 2: RESOLVED CASES
        with tab_resolved:
            st.markdown("### Reconciled & Auto-Resolved Transactions")
            st.caption(f"Includes {det_matched} deterministic 4-way matches and {agent_auto} AI Agent safe auto-resolutions.")

            filter_type = st.radio(
                "Filter Resolved Cases:",
                ["All Resolved", f"AI Agent Auto-Resolved ({agent_auto})", f"Deterministic Matched ({det_matched})"],
                horizontal=True
            )

            auto_cases = [a for a in analyses if a.safe_auto_resolved]

            if filter_type == f"AI Agent Auto-Resolved ({agent_auto})":
                for case in auto_cases:
                    with st.expander(f"🟢 Ledger ID: {case.ledger_id} | Auto-Resolved | Action: {case.recommended_action}", expanded=False):
                        st.write(case.detailed_explanation)
                        st.success(case.recommended_action)
            elif filter_type == f"Deterministic Matched ({det_matched})":
                st.success(f"{det_matched} records fully matched across all 4 systems (Ledger, Bank, Invoice, Settlement) with 100% confidence.")
            else:
                st.write(f"Total {effective_resolved} transactions resolved successfully ({det_matched} deterministic + {agent_auto} AI auto-resolved).")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Ledger ID": a.ledger_id,
                            "Resolution Mode": "AI Agent Safe Resolution" if a.safe_auto_resolved else "Deterministic 4-Way Match",
                            "Action": a.recommended_action,
                            "Missing Feed": ", ".join(a.missing_sources) if a.missing_sources else "None"
                        }
                        for a in auto_cases
                    ]),
                    use_container_width=True
                )

        # TAB 3: AGENT DECISION LOG
        with tab_agent_log:
            st.html('<div id="audit-log"></div>')
            st.markdown("### 🧠 AI Agent Decision Policy & Reasoning Log")
            st.markdown("""
            The AI Exception Agent operates under a strict, auditable financial safety framework:

            1. **Rule #1: Cash Control Supremacy**:
               - Bank statements represent physical cash flow.
               - If a Bank Statement record is missing, the Agent **never** auto-resolves, regardless of Invoice or Settlement agreement.

            2. **Rule #2: Non-Cash Timing Lag Auto-Resolution**:
               - If Bank Statement + Settlement match 100% on Tier-1, but the Invoice is absent, the Agent safely auto-resolves as an un-invoiced timing lag.
               - If Bank Statement + Invoice match 100% on Tier-1, but Settlement batch is lagging, the Agent safely auto-resolves as a settlement sync delay.

            3. **Rule #3: Collision & Variance Escalation**:
               - Any duplicate reference collision or amount mismatch triggers immediate human escalation.
            """)

            st.markdown("#### Real Decision Examples:")
            for case in analyses[:4]:
                badge = "🟢 SAFE AUTO-RESOLVE" if case.safe_auto_resolved else "🔴 HUMAN ESCALATION"
                st.markdown(f"**{case.ledger_id}** → `{badge}`")
                st.caption(case.detailed_explanation)
                st.markdown("---")

    st.html('</div>')

    # 15. FOOTER
    st.html(textwrap.dedent("""
    <div class="fc-footer">
        Finance Controller Agent &nbsp;•&nbsp; Razorpay AI Buildathon Track 04 &nbsp;•&nbsp; Reconcile records, surface exceptions & close workflows faster.
    </div>
    """))


if __name__ == "__main__":
    main()
