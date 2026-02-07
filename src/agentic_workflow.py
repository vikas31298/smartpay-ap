import logging
from datetime import datetime
from typing import Literal, Optional, Dict, Any, List, Annotated
from dataclasses import dataclass, field
from pathlib import Path
import json
import argparse
import numpy as np

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Using mock implementation.")

try:
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from typing import TypedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReconciliationState(TypedDict):
    invoice_id: str
    invoice_data: Optional[Dict[str, Any]]
    po_data: Optional[Dict[str, Any]]
    match_result: Optional[Dict[str, Any]]
    mismatch_explanation: Optional[str]
    dispute_email_draft: Optional[str]
    human_approval: Optional[Literal["pending", "approved", "rejected"]]
    workflow_status: str
    audit_trail: List[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int


class MockLLM:
    def generate_explanation(self, match_result: Dict[str, Any]) -> str:
        mismatch_type = match_result.get("mismatch_type", "UNKNOWN")
        variance = match_result.get("variance_amount", 0)
        matched_po = match_result.get("matched_po", "N/A")
        confidence = match_result.get("match_confidence", 0)

        explanations = {
            "EXACT_MATCH": (
                f"The invoice has been successfully matched to PO {matched_po} "
                f"with {confidence:.1%} confidence. All values align within tolerance."
            ),
            "PRICE_VARIANCE": (
                f"A price variance of ${abs(variance):,.2f} was detected when comparing "
                f"the invoice to PO {matched_po}. This may be due to:\n"
                "- Updated pricing not reflected in the original PO\n"
                "- Volume discount adjustments\n"
                "- Currency conversion differences\n"
                "Recommend: Review unit prices on invoice vs. PO line items."
            ),
            "QUANTITY_VARIANCE": (
                f"A quantity variance resulting in ${abs(variance):,.2f} difference was found "
                f"against PO {matched_po}. Possible causes:\n"
                "- Partial shipment not matching full PO quantity\n"
                "- Additional items delivered beyond PO scope\n"
                "- GRN/delivery note discrepancies\n"
                "Recommend: Cross-reference with goods receipt note and delivery documentation."
            ),
            "TAX_MISCODE": (
                f"Tax calculation discrepancy of ${abs(variance):,.2f} identified "
                f"compared to PO {matched_po}. This could indicate:\n"
                "- Incorrect tax rate applied\n"
                "- Tax jurisdiction mismatch\n"
                "- Missing or incorrect tax exemption\n"
                "Recommend: Verify tax codes and rates with finance team."
            ),
            "MISSING_PO": (
                "No matching Purchase Order was found for this invoice. "
                "This could be due to:\n"
                "- Invoice submitted without prior PO\n"
                "- PO number incorrectly referenced\n"
                "- Service/maintenance invoice requiring different process\n"
                "Recommend: Request PO reference from vendor or route to non-PO approval workflow."
            ),
        }

        return explanations.get(mismatch_type, f"Unknown mismatch type: {mismatch_type}")

    def generate_dispute_email(
        self,
        invoice_data: Dict[str, Any],
        po_data: Optional[Dict[str, Any]],
        match_result: Dict[str, Any],
        explanation: str
    ) -> str:
        invoice_id = invoice_data.get("invoice_id", "Unknown")
        vendor_name = invoice_data.get("vendor_name", "Vendor")
        invoice_total = invoice_data.get("invoice_total", 0)
        mismatch_type = match_result.get("mismatch_type", "UNKNOWN")
        variance = match_result.get("variance_amount", 0)

        po_number = po_data.get("po_number", "N/A") if po_data else "N/A"
        po_total = po_data.get("po_total", 0) if po_data else 0

        email = f"""Subject: Invoice Discrepancy - {invoice_id} - Action Required

Dear {vendor_name} Accounts Team,

We have received invoice {invoice_id} and during our reconciliation process,
we identified a discrepancy that requires your attention.

INVOICE DETAILS:
- Invoice Number: {invoice_id}
- Invoice Total: ${invoice_total:,.2f}
- Reference PO: {po_number}
- PO Total: ${po_total:,.2f}

DISCREPANCY IDENTIFIED:
- Type: {mismatch_type.replace('_', ' ').title()}
- Variance Amount: ${abs(variance):,.2f}

EXPLANATION:
{explanation}

REQUESTED ACTION:
Please review the above discrepancy and provide one of the following:
1. Corrected invoice reflecting the PO terms
2. Supporting documentation justifying the variance
3. Credit note for the difference

We kindly request a response within 5 business days to avoid payment delays.

For any questions, please contact our AP team at ap-queries@acme-manufacturing.com.

Best regards,
Accounts Payable Team
Acme Manufacturing

---
This email was auto-generated by SmartPay AP.
Reference: {invoice_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return email


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


def create_audit_entry(
    action: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "status": status,
        "details": convert_to_native(details) if details else {}
    }


def data_extraction_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Extracting data for invoice: {state['invoice_id']}")

    invoice_id = state["invoice_id"]
    audit_trail = state.get("audit_trail", [])

    if not invoice_id or not invoice_id.startswith("INV"):
        return {
            **state,
            "workflow_status": "FAILED",
            "error_message": f"Invalid invoice ID format: {invoice_id}",
            "audit_trail": audit_trail + [
                create_audit_entry("data_extraction", "FAILED", {"error": "Invalid invoice ID"})
            ]
        }

    try:
        from .matching_model import InvoicePOMatcher
        from .data_loader import load_all_data

        data_dir = Path(__file__).parent.parent / "data"
        invoices, po_grn, _ = load_all_data(data_dir)

        from .data_loader import aggregate_invoice_totals
        inv_agg = aggregate_invoice_totals(invoices)
        inv_row = inv_agg[inv_agg["invoice_id"] == invoice_id]

        if len(inv_row) == 0:
            return {
                **state,
                "workflow_status": "FAILED",
                "error_message": f"Invoice {invoice_id} not found",
                "audit_trail": audit_trail + [
                    create_audit_entry("data_extraction", "FAILED", {"error": "Invoice not found"})
                ]
            }

        invoice_data = inv_row.iloc[0].to_dict()
        if "invoice_date" in invoice_data:
            invoice_data["invoice_date"] = str(invoice_data["invoice_date"])
        invoice_data = convert_to_native(invoice_data)

        return {
            **state,
            "invoice_data": invoice_data,
            "workflow_status": "DATA_EXTRACTED",
            "audit_trail": audit_trail + [
                create_audit_entry("data_extraction", "SUCCESS", {
                    "invoice_id": invoice_id,
                    "invoice_total": invoice_data.get("invoice_total"),
                    "vendor_id": invoice_data.get("vendor_id")
                })
            ]
        }

    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        return {
            **state,
            "workflow_status": "FAILED",
            "error_message": str(e),
            "audit_trail": audit_trail + [
                create_audit_entry("data_extraction", "FAILED", {"error": str(e)})
            ]
        }


def matching_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Matching invoice: {state['invoice_id']}")

    invoice_id = state["invoice_id"]
    invoice_data = state.get("invoice_data", {})
    audit_trail = state.get("audit_trail", [])

    if not invoice_data:
        return {
            **state,
            "workflow_status": "FAILED",
            "error_message": "No invoice data available for matching",
            "audit_trail": audit_trail + [
                create_audit_entry("matching", "FAILED", {"error": "No invoice data"})
            ]
        }

    try:
        from .matching_model import InvoicePOMatcher
        from .data_loader import load_all_data

        data_dir = Path(__file__).parent.parent / "data"
        invoices, po_grn, labels = load_all_data(data_dir)

        matcher = InvoicePOMatcher()
        matcher.fit(invoices, po_grn, labels)

        match_result = matcher.predict(invoice_id)

        po_data = None
        if match_result.get("matched_po"):
            po_row = po_grn[po_grn["po_number"] == match_result["matched_po"]]
            if len(po_row) > 0:
                po_data = po_row.iloc[0].to_dict()
                for key in ["po_date", "grn_date"]:
                    if key in po_data:
                        po_data[key] = str(po_data[key])

        match_result = convert_to_native(match_result)
        po_data = convert_to_native(po_data) if po_data else None

        return {
            **state,
            "match_result": match_result,
            "po_data": po_data,
            "workflow_status": "MATCHED",
            "audit_trail": audit_trail + [
                create_audit_entry("matching", "SUCCESS", {
                    "mismatch_type": match_result.get("mismatch_type"),
                    "matched_po": match_result.get("matched_po"),
                    "confidence": match_result.get("match_confidence"),
                    "variance": match_result.get("variance_amount")
                })
            ]
        }

    except Exception as e:
        logger.error(f"Matching failed: {e}")
        return {
            **state,
            "workflow_status": "FAILED",
            "error_message": str(e),
            "audit_trail": audit_trail + [
                create_audit_entry("matching", "FAILED", {"error": str(e)})
            ]
        }


def explanation_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Generating explanation for: {state['invoice_id']}")

    match_result = state.get("match_result", {})
    audit_trail = state.get("audit_trail", [])

    llm = MockLLM()
    explanation = llm.generate_explanation(match_result)

    return {
        **state,
        "mismatch_explanation": explanation,
        "workflow_status": "EXPLAINED",
        "audit_trail": audit_trail + [
            create_audit_entry("explanation_generation", "SUCCESS", {
                "explanation_length": len(explanation)
            })
        ]
    }


def email_generation_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Generating dispute email for: {state['invoice_id']}")

    invoice_data = state.get("invoice_data", {})
    po_data = state.get("po_data")
    match_result = state.get("match_result", {})
    explanation = state.get("mismatch_explanation", "")
    audit_trail = state.get("audit_trail", [])

    llm = MockLLM()
    email_draft = llm.generate_dispute_email(
        invoice_data, po_data, match_result, explanation
    )

    return {
        **state,
        "dispute_email_draft": email_draft,
        "workflow_status": "EMAIL_DRAFTED",
        "human_approval": "pending",
        "audit_trail": audit_trail + [
            create_audit_entry("email_generation", "SUCCESS", {
                "email_length": len(email_draft)
            })
        ]
    }


def human_approval_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Awaiting human approval for: {state['invoice_id']}")

    match_result = state.get("match_result", {})
    audit_trail = state.get("audit_trail", [])
    current_approval = state.get("human_approval", "pending")

    confidence = match_result.get("match_confidence", 0)

    if current_approval == "pending":
        if confidence >= 0.7:
            approval_decision = "approved"
            decision_reason = f"Auto-approved (confidence {confidence:.1%} >= 70%)"
        elif confidence >= 0.5:
            approval_decision = "approved"
            decision_reason = f"Simulated human approval (confidence {confidence:.1%})"
        else:
            approval_decision = "rejected"
            decision_reason = f"Rejected due to low confidence ({confidence:.1%})"

        return {
            **state,
            "human_approval": approval_decision,
            "workflow_status": f"APPROVAL_{approval_decision.upper()}",
            "audit_trail": audit_trail + [
                create_audit_entry("human_approval", approval_decision.upper(), {
                    "confidence": confidence,
                    "reason": decision_reason
                })
            ]
        }

    return state


def payment_trigger_node(state: ReconciliationState) -> ReconciliationState:
    logger.info(f"Triggering payment for: {state['invoice_id']}")

    invoice_id = state["invoice_id"]
    invoice_data = state.get("invoice_data", {})
    match_result = state.get("match_result", {})
    audit_trail = state.get("audit_trail", [])

    payment_reference = f"PAY-{invoice_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return {
        **state,
        "workflow_status": "PAYMENT_TRIGGERED",
        "audit_trail": audit_trail + [
            create_audit_entry("payment_trigger", "SUCCESS", {
                "payment_reference": payment_reference,
                "amount": invoice_data.get("invoice_total"),
                "matched_po": match_result.get("matched_po")
            })
        ]
    }


def route_after_matching(state: ReconciliationState) -> str:
    match_result = state.get("match_result", {})
    mismatch_type = match_result.get("mismatch_type", "")

    if mismatch_type == "EXACT_MATCH":
        logger.info("Routing to payment (exact match)")
        return "match"
    else:
        logger.info(f"Routing to explanation ({mismatch_type})")
        return "mismatch"


def route_after_approval(state: ReconciliationState) -> str:
    approval = state.get("human_approval", "pending")

    if approval == "approved":
        return "approved"
    elif approval == "rejected":
        return "rejected"
    else:
        return "pending"


def create_reconciliation_graph():
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, returning None")
        return None

    workflow = StateGraph(ReconciliationState)

    workflow.add_node("extract_data", data_extraction_node)
    workflow.add_node("match_invoice", matching_node)
    workflow.add_node("explain_mismatch", explanation_node)
    workflow.add_node("generate_email", email_generation_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("trigger_payment", payment_trigger_node)

    workflow.set_entry_point("extract_data")

    workflow.add_edge("extract_data", "match_invoice")

    workflow.add_conditional_edges(
        "match_invoice",
        route_after_matching,
        {
            "mismatch": "explain_mismatch",
            "match": "trigger_payment"
        }
    )

    workflow.add_edge("explain_mismatch", "generate_email")
    workflow.add_edge("generate_email", "human_approval")

    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "approved": "trigger_payment",
            "rejected": END,
            "pending": "human_approval"
        }
    )

    workflow.add_edge("trigger_payment", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_reconciliation_demo(
    invoice_id: str,
    verbose: bool = True
) -> ReconciliationState:
    if verbose:
        print("=" * 60)
        print(f"SMARTPAY AP - RECONCILIATION WORKFLOW")
        print(f"Invoice: {invoice_id}")
        print("=" * 60)

    initial_state: ReconciliationState = {
        "invoice_id": invoice_id,
        "invoice_data": None,
        "po_data": None,
        "match_result": None,
        "mismatch_explanation": None,
        "dispute_email_draft": None,
        "human_approval": None,
        "workflow_status": "INITIATED",
        "audit_trail": [create_audit_entry("workflow_start", "INITIATED", {"invoice_id": invoice_id})],
        "error_message": None,
        "retry_count": 0
    }

    graph = create_reconciliation_graph()

    if graph is None:
        if verbose:
            print("\n[Fallback Mode - Running without LangGraph]\n")

        state = initial_state
        state = data_extraction_node(state)

        if state["workflow_status"] != "FAILED":
            state = matching_node(state)

        if state["workflow_status"] != "FAILED":
            match_result = state.get("match_result", {})
            if match_result.get("mismatch_type") != "EXACT_MATCH":
                state = explanation_node(state)
                state = email_generation_node(state)
                state = human_approval_node(state)

            if state.get("human_approval") == "approved" or match_result.get("mismatch_type") == "EXACT_MATCH":
                state = payment_trigger_node(state)

        final_state = state
    else:
        config = {"configurable": {"thread_id": invoice_id}}
        final_state = None

        for event in graph.stream(initial_state, config):
            if verbose:
                for node_name, node_state in event.items():
                    status = node_state.get("workflow_status", "UNKNOWN")
                    print(f"[{node_name}] Status: {status}")
            final_state = list(event.values())[0]

    if verbose:
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)

        print(f"\nFinal Status: {final_state.get('workflow_status')}")

        if final_state.get("error_message"):
            print(f"Error: {final_state['error_message']}")

        match_result = final_state.get("match_result", {})
        if match_result:
            print(f"\nMatch Result:")
            print(f"  - Type: {match_result.get('mismatch_type')}")
            print(f"  - Matched PO: {match_result.get('matched_po')}")
            print(f"  - Confidence: {match_result.get('match_confidence', 0):.1%}")
            print(f"  - Variance: ${match_result.get('variance_amount', 0):,.2f}")

        if final_state.get("mismatch_explanation"):
            print(f"\nExplanation:\n{final_state['mismatch_explanation']}")

        if final_state.get("dispute_email_draft"):
            print(f"\nDispute Email Draft:\n{'-' * 40}")
            print(final_state["dispute_email_draft"][:500] + "...")

        print(f"\nAudit Trail ({len(final_state.get('audit_trail', []))} entries):")
        for entry in final_state.get("audit_trail", []):
            print(f"  [{entry['timestamp']}] {entry['action']}: {entry['status']}")

    return final_state


def run_batch_reconciliation(
    invoice_ids: List[str],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    results = []

    for i, invoice_id in enumerate(invoice_ids):
        print(f"\nProcessing {i + 1}/{len(invoice_ids)}: {invoice_id}")

        try:
            state = run_reconciliation_demo(invoice_id, verbose=verbose)
            results.append({
                "invoice_id": invoice_id,
                "status": state.get("workflow_status"),
                "mismatch_type": state.get("match_result", {}).get("mismatch_type"),
                "approval": state.get("human_approval"),
                "error": state.get("error_message")
            })
        except Exception as e:
            results.append({
                "invoice_id": invoice_id,
                "status": "ERROR",
                "error": str(e)
            })

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    status_counts = {}
    for r in results:
        status = r.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SmartPay AP - Invoice Reconciliation Workflow"
    )
    parser.add_argument(
        "--invoice-id",
        type=str,
        help="Invoice ID to process (e.g., INV00001)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample invoices"
    )
    parser.add_argument(
        "--batch",
        type=str,
        nargs="+",
        help="Process multiple invoice IDs"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.demo:
        print("Running SmartPay AP Demo...")
        sample_ids = ["INV00001", "INV00010", "INV00050"]

        for inv_id in sample_ids:
            print("\n" + "=" * 60)
            run_reconciliation_demo(inv_id, verbose=True)

    elif args.batch:
        run_batch_reconciliation(args.batch, verbose=args.verbose)

    elif args.invoice_id:
        run_reconciliation_demo(args.invoice_id, verbose=True)

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python -m src.agentic_workflow --invoice-id INV00001")
        print("  python -m src.agentic_workflow --demo")
        print("  python -m src.agentic_workflow --batch INV00001 INV00002 INV00003")


if __name__ == "__main__":
    main()
