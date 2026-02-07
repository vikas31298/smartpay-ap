# SmartPay AP - Invoice Reconciliation Platform

## Overview

SmartPay AP automates invoice reconciliation for Acme Manufacturing's accounts payable operations. The platform processes supplier invoices, matches them against POs, classifies mismatches, and generates dispute emails.

## Features

- Invoice-PO matching using hybrid rule-based and ML approaches
- Mismatch classification (price variance, quantity variance, tax issues, missing PO)
- Natural language explanations for discrepancies
- Automated dispute email generation
- LangGraph-based workflow with human-in-the-loop approval

## Quick Start

```bash
pip install -r requirements.txt

# Run the workflow
python -m src.agentic_workflow --invoice-id INV00001

# Run tests
pytest tests/ -v
```

## Project Structure

```
smartpay-ap/
├── src/
│   ├── data_loader.py        # For Data loading utilities
│   ├── matching_model.py     # Invoice-PO matching models
│   ├── evaluation.py         # Metrics and visualizations
│   └── agentic_workflow.py   # LangGraph workflows
├── notebooks/
│   └── matching_model_notebook.ipynb
├── docs/
│   ├── architecture_deck.pptx
│   └── responsible_ai_brief.pptx
├── data/
│   ├── invoices.csv
│   ├── po_grn.csv
│   └── labelled_mismatches.csv
└── tests/
    └── test_matching.py
```

## Usage

### Matching Model

```python
from src.data_loader import load_all_data
from src.matching_model import InvoicePOMatcher

invoices, po_grn, labels = load_all_data('data')

model = InvoicePOMatcher()
model.fit(invoices, po_grn, labels)

result = model.predict('INV00001')
print(result['mismatch_type'])
print(result['explanation'])
```

### Agentic Workflow

```bash
python -m src.agentic_workflow --demo
python -m src.agentic_workflow --invoice-id INV00001
python -m src.agentic_workflow --batch INV00001 INV00002 INV00003
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 93.75% |
| F1 Score (macro) | 0.95 |
| F1 Score (weighted) | 0.94 |

## Configuration

```python
model = InvoicePOMatcher(
    exact_tolerance_pct=0.1,   
    fuzzy_tolerance_pct=5.0,   
    use_ml_classification=True
)
```

## Architecture

See `docs/architecture_deck.pptx` for the full architecture (12 slides) covering:
- Data layer (multi-source ingestion, data lake)
- ML/AI layer (matching model, classification)
- Gen-AI & Agent layer (LangGraph orchestration)
- Multi-cloud deployment (Azure + AWS)
- Security & compliance (GDPR)

## Testing

```bash
pytest tests/ -v
```
