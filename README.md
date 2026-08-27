# Finance Controller Agent

> 

An AI-assisted multi-source financial reconciliation system that identifies mismatches, anomalies, and exceptions across bank statements, ledger entries, invoices, and payment settlements.

---

## Project Structure

```
finance-controller-agent/
├── data/                        # Generated CSV datasets (git-ignored)
│   ├── bank_statements.csv
│   ├── ledger.csv
│   ├── invoices.csv
│   ├── settlements.csv
│   └── ground_truth.csv         # Evaluation only – not used during inference
├── src/
│   └── generate_data.py         # Synthetic data generator (Milestone 1)
├── main.py
├── requirements.txt
├── README.md
└── architecture.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (Milestone 1)

```bash
python -m src.generate_data
```

Or via the top-level entry point:

```bash
python main.py
```

---

## Dataset Design

| Source               | Rows (nominal) | Key fields |
|----------------------|---------------|------------|
| `bank_statements.csv`  | ~100          | `bank_txn_id`, `value_date`, `amount`, `counterparty`, `reference`, `bank_channel` |
| `ledger.csv`           | ~100          | `ledger_entry_id`, `posting_date`, `txn_date`, `amount`, `counterparty`, `tax_line`, `cost_center` |
| `invoices.csv`         | ~100          | `invoice_id`, `invoice_date`, `due_date`, `amount`, `counterparty`, `tax_line`, `invoice_status` |
| `settlements.csv`      | ~100          | `settlement_id`, `settlement_date`, `amount`, `counterparty`, `utr_number`, `payment_mode`, `status` |
| `ground_truth.csv`     | 100           | Evaluation mapping – canonical ↔ source IDs, `match_status`, `exception_type` |

### Exception Types Planted (~20%)

| Type | Description |
|------|-------------|
| `amount_mismatch` | Amount differs by ±0.50–50.00 in one source |
| `date_drift` | Date shifted 1–3 days in one source |
| `missing_reference` | `reference` field stripped from one source |
| `duplicate_reference` | Previous row's reference reused |
| `missing_tax_line` | `tax_line` absent in ledger or invoice |
| `fuzzy_counterparty` | Company name variation (Ltd vs Limited, etc.) |
| `missing_source_record` | Entire record absent from one source |

---

## Reproducibility

The generator uses a fixed random seed of **42**. Running with the same seed always produces the same dataset.

---

## Milestones

- [x] **Milestone 1** – Synthetic data generation & validation
- [ ] **Milestone 2** – Multi-source matcher
- [ ] **Milestone 3** – AI reconciliation agent
- [ ] **Milestone 4** – Report generation
- [ ] **Milestone 5** – Dashboard

## Controller Boundaries

- **Deterministic verification** checks amount, date, reference, and invoice-versus-ledger tax values. Tax arithmetic is performed by Python; missing or mismatched tax is escalated and never inferred.
- **ML matching** scores residual candidates only; it does not approve accounting outcomes.
- **Agent orchestration** retains the authoritative reconciliation status, recommendation, human-review requirement, and audit event.
- **Gemini reasoning** explains verified exception evidence with strict structured validation and safe fallback. It cannot change controller decisions.
- **Ask the Finance Controller** retrieves only relevant `AgentDecision` and `AuditEvent` records. Python retrieval is the factual source of truth; Gemini is optional explanation over that evidence and never receives the entire dataset.
