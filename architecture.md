# Architecture – Finance Controller Agent

## Overview

The system reconciles financial records from four independent source systems against a canonical ledger of transactions. Exceptions are flagged and escalated to a human-in-the-loop agent for resolution.

---

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                     Data Sources                        │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐ │
│  │Bank Statements│  │  Ledger  │  │ Invoices │  │ STL │ │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └──┬──┘ │
└─────────┼───────────────┼─────────────┼────────────┼────┘
          │               │             │            │
          └───────────────┴──────┬──────┴────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Ingestor / Normalizer  │
                    │  (schema standardisation)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Matcher Engine       │
                    │  (rule + fuzzy + ML)     │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
  ┌────────────▼──────┐  ┌───────▼──────┐  ┌──────▼──────────┐
  │  Matched Records  │  │  Exceptions  │  │ Unresolved Items │
  └────────────┬──────┘  └───────┬──────┘  └──────┬──────────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    AI Reconciliation     │
                    │    Agent (Gemini)         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Report Generator       │
                    │   + Dashboard            │
                    └─────────────────────────┘
```

---

## Data Flow

1. **Generate / Ingest** – Raw CSVs are loaded from the four source systems. Each carries source-specific field names and date formats.
2. **Normalise** – A thin normalisation layer maps every source to a common schema: `{date, amount, counterparty, reference, tax_line}`.
3. **Match** – A three-tier matcher attempts to link records:
   - **Exact**: reference + amount + date match within tolerance
   - **Fuzzy**: counterparty name similarity, ±3-day date window, ±2% amount tolerance
   - **ML / Embedding**: for residual ambiguous cases
4. **Classify** – Each canonical transaction is assigned one of:
   - `MATCHED` – all four sources agree within tolerance
   - `PARTIAL` – some sources missing or ambiguous
   - `EXCEPTION` – known discrepancy type detected
   - `UNRESOLVED` – requires human review
5. **Agent** – A Gemini-powered agent reviews `EXCEPTION` and `UNRESOLVED` items, generating explanations and resolution recommendations.
6. **Report / Dashboard** – Outputs a structured reconciliation report and interactive dashboard.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fixed seed (42) for data gen | Full reproducibility across environments |
| Deterministic IDs (BNK-XXXX, LED-XXXX, …) | Simplifies ground-truth linking without leaking info during inference |
| `canonical_id` dropped from source CSVs | Prevents the matcher from "cheating" – it must infer links from content |
| ~20% exception rate | Mirrors realistic enterprise reconciliation failure rates |
| Ground truth in separate file | Evaluation-only; not available at inference time |

---

## Technology Stack (Planned)

| Layer | Technology |
|-------|-----------|
| Data generation | Python + pandas + Faker |
| Matching engine | Python + pandas + rapidfuzz |
| AI agent | Google Gemini API (gemini-1.5-pro) |
| Report generation | Jinja2 + WeasyPrint |
| Dashboard | Streamlit or Gradio |
| Deployment | Cloud Run / local |
