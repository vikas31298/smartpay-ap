import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    load_invoices,
    load_po_grn,
    load_labelled_mismatches,
    aggregate_invoice_totals,
    merge_invoice_po,
)
from matching_model import InvoicePOMatcher
from evaluation import (
    evaluate_matching,
    evaluate_po_matching,
    evaluate_variance_detection,
)


@pytest.fixture
def sample_invoices():
    return pd.DataFrame({
        "invoice_id": ["INV001", "INV001", "INV002", "INV003", "INV003"],
        "invoice_date": pd.to_datetime(["2024-01-15"] * 5),
        "vendor_id": ["V001", "V001", "V001", "V002", "V002"],
        "vendor_name": ["Vendor A", "Vendor A", "Vendor A", "Vendor B", "Vendor B"],
        "currency": ["USD", "USD", "USD", "USD", "USD"],
        "line_item_number": [1, 2, 1, 1, 2],
        "item_code": ["ITM001", "ITM002", "ITM003", "ITM004", "ITM005"],
        "description": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"],
        "quantity": [10, 5, 20, 15, 8],
        "unit_price": [100.0, 50.0, 75.0, 200.0, 150.0],
        "line_total": [1000.0, 250.0, 1500.0, 3000.0, 1200.0],
    })


@pytest.fixture
def sample_po_grn():
    return pd.DataFrame({
        "po_number": ["PO001", "PO002", "PO003"],
        "po_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
        "vendor_id": ["V001", "V001", "V002"],
        "vendor_name": ["Vendor A", "Vendor A", "Vendor B"],
        "po_total": [1250.0, 1500.0, 4200.0],
        "currency": ["USD", "USD", "USD"],
        "grn_number": ["GRN001", "GRN002", "GRN003"],
        "grn_date": pd.to_datetime(["2024-01-12", "2024-01-10", "2024-01-14"]),
    })


@pytest.fixture
def sample_labels():
    return pd.DataFrame({
        "invoice_id": ["INV001", "INV002", "INV003"],
        "po_number": ["PO001", "PO002", "PO003"],
        "mismatch_type": ["EXACT_MATCH", "EXACT_MATCH", "EXACT_MATCH"],
        "invoice_value": [1250.0, 1500.0, 4200.0],
        "po_value": [1250.0, 1500.0, 4200.0],
        "difference": [0.0, 0.0, 0.0],
    })


@pytest.fixture
def data_dir():
    return Path(__file__).parent.parent / "data"


class TestDataLoader:
    def test_load_invoices_from_file(self, data_dir):
        filepath = data_dir / "invoices.csv"
        if filepath.exists():
            df = load_invoices(filepath)
            assert len(df) > 0
            assert "invoice_id" in df.columns
            assert "line_total" in df.columns
            assert df["invoice_date"].dtype == "datetime64[ns]"

    def test_load_po_grn_from_file(self, data_dir):
        filepath = data_dir / "po_grn.csv"
        if filepath.exists():
            df = load_po_grn(filepath)
            assert len(df) > 0
            assert "po_number" in df.columns
            assert "po_total" in df.columns

    def test_load_labels_from_file(self, data_dir):
        filepath = data_dir / "labelled_mismatches.csv"
        if filepath.exists():
            df = load_labelled_mismatches(filepath)
            assert len(df) > 0
            assert "mismatch_type" in df.columns

    def test_aggregate_invoice_totals(self, sample_invoices):
        agg = aggregate_invoice_totals(sample_invoices)

        assert len(agg) == 3
        assert "invoice_total" in agg.columns
        assert "line_count" in agg.columns

        inv001 = agg[agg["invoice_id"] == "INV001"].iloc[0]
        assert inv001["invoice_total"] == 1250.0
        assert inv001["line_count"] == 2

    def test_merge_invoice_po(self, sample_invoices, sample_po_grn):
        inv_agg = aggregate_invoice_totals(sample_invoices)
        merged = merge_invoice_po(inv_agg, sample_po_grn, tolerance_pct=5.0)

        assert len(merged) == 3
        assert "matched_po_number" in merged.columns
        assert "match_type" in merged.columns

    def test_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            load_invoices("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            load_po_grn("nonexistent.csv")


class TestInvoicePOMatcher:
    def test_model_initialization(self):
        model = InvoicePOMatcher()
        assert model.exact_tolerance_pct == 0.1
        assert model.fuzzy_tolerance_pct == 5.0
        assert model.use_ml_classification is True

    def test_model_fit(self, sample_invoices, sample_po_grn, sample_labels):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        assert model._is_fitted is True
        assert model.invoices_agg is not None
        assert len(model.invoice_lookup) > 0

    def test_model_predict_exact_match(self, sample_invoices, sample_po_grn, sample_labels):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        result = model.predict("INV001")

        assert "matched_po" in result
        assert "match_confidence" in result
        assert "mismatch_type" in result
        assert "variance_amount" in result
        assert "explanation" in result

    def test_model_predict_unknown_invoice(self, sample_invoices, sample_po_grn, sample_labels):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        result = model.predict("INV999")

        assert result["matched_po"] is None
        assert result["mismatch_type"] == "MISSING_PO"

    def test_model_predict_not_fitted(self):
        model = InvoicePOMatcher()

        with pytest.raises(RuntimeError):
            model.predict("INV001")

    def test_model_predict_batch(self, sample_invoices, sample_po_grn, sample_labels):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        results = model.predict_batch(["INV001", "INV002"])

        assert len(results) == 2
        assert all("mismatch_type" in r for r in results)

    def test_feature_importance(self, sample_invoices, sample_po_grn, sample_labels):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        importance = model.get_feature_importance()
        assert isinstance(importance, dict)

    def test_model_save_load(self, sample_invoices, sample_po_grn, sample_labels, tmp_path):
        model = InvoicePOMatcher()
        model.fit(sample_invoices, sample_po_grn, sample_labels)

        filepath = tmp_path / "model.pkl"
        model.save(filepath)
        assert filepath.exists()

        loaded_model = InvoicePOMatcher.load(filepath)
        assert loaded_model._is_fitted is True

        orig_result = model.predict("INV001")
        loaded_result = loaded_model.predict("INV001")
        assert orig_result["mismatch_type"] == loaded_result["mismatch_type"]


class TestEvaluation:
    def test_evaluate_matching_basic(self):
        y_true = ["EXACT_MATCH", "PRICE_VARIANCE", "EXACT_MATCH"]
        y_pred = ["EXACT_MATCH", "PRICE_VARIANCE", "PRICE_VARIANCE"]

        metrics = evaluate_matching(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["accuracy"] == 2/3

    def test_evaluate_matching_all_correct(self):
        y_true = ["EXACT_MATCH", "PRICE_VARIANCE", "QUANTITY_VARIANCE"]
        y_pred = ["EXACT_MATCH", "PRICE_VARIANCE", "QUANTITY_VARIANCE"]

        metrics = evaluate_matching(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_evaluate_po_matching(self):
        predicted = ["PO001", "PO002", None, "PO004"]
        actual = ["PO001", "PO003", None, None]

        metrics = evaluate_po_matching(predicted, actual)

        assert "matching_accuracy" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics
        assert "true_positives" in metrics

        assert metrics["true_positives"] == 1
        assert metrics["true_negatives"] == 1

    def test_evaluate_variance_detection(self):
        predicted = [100.0, 250.0, 0.0]
        actual = [100.0, 240.0, 10.0]

        metrics = evaluate_variance_detection(predicted, actual)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "accuracy_within_tolerance" in metrics
        assert metrics["mae"] > 0

    def test_evaluate_empty_lists(self):
        metrics = evaluate_po_matching([], [])

        assert metrics["matching_accuracy"] == 0.0
        assert metrics.get("total_samples", 0) == 0


class TestAgenticWorkflow:
    def test_create_audit_entry(self):
        from agentic_workflow import create_audit_entry

        entry = create_audit_entry("test_action", "SUCCESS", {"key": "value"})

        assert "timestamp" in entry
        assert entry["action"] == "test_action"
        assert entry["status"] == "SUCCESS"
        assert entry["details"]["key"] == "value"

    def test_mock_llm_explanation(self):
        from agentic_workflow import MockLLM

        llm = MockLLM()

        match_result = {
            "mismatch_type": "PRICE_VARIANCE",
            "variance_amount": 100.0,
            "matched_po": "PO001",
            "match_confidence": 0.85
        }

        explanation = llm.generate_explanation(match_result)

        assert len(explanation) > 0
        assert "variance" in explanation.lower() or "price" in explanation.lower()

    def test_mock_llm_email_generation(self):
        from agentic_workflow import MockLLM

        llm = MockLLM()

        invoice_data = {
            "invoice_id": "INV001",
            "vendor_name": "Test Vendor",
            "invoice_total": 1000.0
        }
        po_data = {
            "po_number": "PO001",
            "po_total": 950.0
        }
        match_result = {
            "mismatch_type": "PRICE_VARIANCE",
            "variance_amount": 50.0
        }

        email = llm.generate_dispute_email(
            invoice_data, po_data, match_result, "Test explanation"
        )

        assert "INV001" in email
        assert "Subject:" in email
        assert "Test Vendor" in email

    def test_data_extraction_node_invalid_invoice(self):
        from agentic_workflow import data_extraction_node, ReconciliationState

        state: ReconciliationState = {
            "invoice_id": "INVALID",
            "invoice_data": None,
            "po_data": None,
            "match_result": None,
            "mismatch_explanation": None,
            "dispute_email_draft": None,
            "human_approval": None,
            "workflow_status": "INITIATED",
            "audit_trail": [],
            "error_message": None,
            "retry_count": 0
        }

        result = data_extraction_node(state)

        assert result["workflow_status"] == "FAILED"
        assert result["error_message"] is not None
        assert len(result["audit_trail"]) > 0


class TestIntegration:
    def test_full_pipeline(self, data_dir):
        invoices_path = data_dir / "invoices.csv"
        po_grn_path = data_dir / "po_grn.csv"
        labels_path = data_dir / "labelled_mismatches.csv"

        if not all(p.exists() for p in [invoices_path, po_grn_path, labels_path]):
            pytest.skip("Data files not found")

        invoices = load_invoices(invoices_path)
        po_grn = load_po_grn(po_grn_path)
        labels = load_labelled_mismatches(labels_path)

        model = InvoicePOMatcher()
        model.fit(invoices, po_grn, labels)

        sample_id = invoices["invoice_id"].iloc[0]
        result = model.predict(sample_id)

        assert result is not None
        assert "mismatch_type" in result
        assert result["mismatch_type"] in [
            "EXACT_MATCH", "PRICE_VARIANCE", "QUANTITY_VARIANCE",
            "TAX_MISCODE", "MISSING_PO"
        ]

    def test_evaluation_on_real_data(self, data_dir):
        labels_path = data_dir / "labelled_mismatches.csv"

        if not labels_path.exists():
            pytest.skip("Labels file not found")

        labels = load_labelled_mismatches(labels_path)

        y_true = labels["mismatch_type"].tolist()
        y_pred = labels["mismatch_type"].tolist()

        metrics = evaluate_matching(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0


class TestEdgeCases:
    def test_empty_dataframe_aggregation(self):
        empty_df = pd.DataFrame(columns=[
            "invoice_id", "invoice_date", "vendor_id", "vendor_name",
            "currency", "line_item_number", "item_code", "description",
            "quantity", "unit_price", "line_total"
        ])

        result = aggregate_invoice_totals(empty_df)
        assert len(result) == 0

    def test_single_line_invoice(self):
        df = pd.DataFrame({
            "invoice_id": ["INV001"],
            "invoice_date": pd.to_datetime(["2024-01-15"]),
            "vendor_id": ["V001"],
            "vendor_name": ["Vendor A"],
            "currency": ["USD"],
            "line_item_number": [1],
            "item_code": ["ITM001"],
            "description": ["Item 1"],
            "quantity": [10],
            "unit_price": [100.0],
            "line_total": [1000.0],
        })

        result = aggregate_invoice_totals(df)

        assert len(result) == 1
        assert result.iloc[0]["invoice_total"] == 1000.0
        assert result.iloc[0]["line_count"] == 1

    def test_mismatched_lengths_evaluation(self):
        with pytest.raises(ValueError):
            evaluate_po_matching(["PO001", "PO002"], ["PO001"])

    def test_zero_po_total(self):
        invoices = pd.DataFrame({
            "invoice_id": ["INV001"],
            "invoice_date": pd.to_datetime(["2024-01-15"]),
            "vendor_id": ["V001"],
            "vendor_name": ["Vendor A"],
            "currency": ["USD"],
            "line_item_number": [1],
            "item_code": ["ITM001"],
            "description": ["Item 1"],
            "quantity": [10],
            "unit_price": [100.0],
            "line_total": [1000.0],
        })

        po_grn = pd.DataFrame({
            "po_number": ["PO001"],
            "po_date": pd.to_datetime(["2024-01-01"]),
            "vendor_id": ["V001"],
            "vendor_name": ["Vendor A"],
            "po_total": [0.0],
            "currency": ["USD"],
            "grn_number": ["GRN001"],
            "grn_date": pd.to_datetime(["2024-01-14"]),
        })

        inv_agg = aggregate_invoice_totals(invoices)
        result = merge_invoice_po(inv_agg, po_grn)

        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
