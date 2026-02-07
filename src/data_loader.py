import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_invoices(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Invoice file not found: {filepath}")

    logger.info(f"Loading invoices from {filepath}")
    df = pd.read_csv(filepath)

    required_cols = [
        "invoice_id", "invoice_date", "vendor_id", "vendor_name",
        "currency", "line_item_number", "item_code", "description",
        "quantity", "unit_price", "line_total"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["line_total"] = pd.to_numeric(df["line_total"], errors="coerce")

    logger.info(f"Loaded {len(df)} invoice line items across {df['invoice_id'].nunique()} invoices")
    return df


def load_po_grn(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PO/GRN file not found: {filepath}")

    logger.info(f"Loading PO/GRN data from {filepath}")
    df = pd.read_csv(filepath)

    required_cols = [
        "po_number", "po_date", "vendor_id", "vendor_name",
        "po_total", "currency", "grn_number", "grn_date"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["po_date"] = pd.to_datetime(df["po_date"])
    df["grn_date"] = pd.to_datetime(df["grn_date"])
    df["po_total"] = pd.to_numeric(df["po_total"], errors="coerce")

    logger.info(f"Loaded {len(df)} PO/GRN records")
    return df


def load_labelled_mismatches(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Labelled mismatches file not found: {filepath}")

    logger.info(f"Loading labelled mismatches from {filepath}")
    df = pd.read_csv(filepath)

    required_cols = [
        "invoice_id", "po_number", "mismatch_type",
        "invoice_value", "po_value", "difference"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["invoice_value"] = pd.to_numeric(df["invoice_value"], errors="coerce")
    df["po_value"] = pd.to_numeric(df["po_value"], errors="coerce")
    df["difference"] = pd.to_numeric(df["difference"], errors="coerce")

    valid_types = {"PRICE_VARIANCE", "QUANTITY_VARIANCE", "TAX_MISCODE", "MISSING_PO", "EXACT_MATCH"}
    invalid_types = set(df["mismatch_type"].dropna().unique()) - valid_types
    if invalid_types:
        logger.warning(f"Found unexpected mismatch types: {invalid_types}")

    logger.info(f"Loaded {len(df)} labelled mismatch records")
    logger.info(f"Mismatch type distribution:\n{df['mismatch_type'].value_counts()}")
    return df


def aggregate_invoice_totals(invoices_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating invoice line items to invoice totals")

    agg_df = invoices_df.groupby(
        ["invoice_id", "invoice_date", "vendor_id", "vendor_name", "currency"]
    ).agg(
        invoice_total=("line_total", "sum"),
        line_count=("line_item_number", "count"),
        avg_unit_price=("unit_price", "mean"),
        total_quantity=("quantity", "sum"),
    ).reset_index()

    agg_df["invoice_total"] = agg_df["invoice_total"].round(2)
    agg_df["avg_unit_price"] = agg_df["avg_unit_price"].round(2)

    logger.info(f"Aggregated {len(invoices_df)} line items into {len(agg_df)} invoices")
    return agg_df


def merge_invoice_po(
    invoices_agg: pd.DataFrame,
    po_grn: pd.DataFrame,
    tolerance_pct: float = 5.0,
    exact_tolerance_pct: float = 0.1
) -> pd.DataFrame:
    logger.info(f"Matching invoices to POs (tolerance: {tolerance_pct}%)")

    inv = invoices_agg.copy()
    po = po_grn.copy()

    inv["matched_po_number"] = None
    inv["match_type"] = "UNMATCHED"
    inv["po_total"] = np.nan
    inv["po_date"] = pd.NaT
    inv["grn_number"] = None
    inv["grn_date"] = pd.NaT
    inv["total_difference"] = np.nan
    inv["percentage_difference"] = np.nan

    used_pos = set()

    for idx, inv_row in inv.iterrows():
        po_candidates = po[
            (po["vendor_id"] == inv_row["vendor_id"]) &
            (po["currency"] == inv_row["currency"]) &
            (~po["po_number"].isin(used_pos))
        ]

        if len(po_candidates) == 0:
            continue

        po_candidates = po_candidates.copy()
        po_candidates["diff"] = abs(po_candidates["po_total"] - inv_row["invoice_total"])
        po_candidates["diff_pct"] = (po_candidates["diff"] / po_candidates["po_total"] * 100)

        exact_matches = po_candidates[po_candidates["diff_pct"] <= exact_tolerance_pct]

        if len(exact_matches) > 0:
            best_match = exact_matches.loc[exact_matches["diff"].idxmin()]
            inv.at[idx, "matched_po_number"] = best_match["po_number"]
            inv.at[idx, "match_type"] = "EXACT_MATCH"
            inv.at[idx, "po_total"] = best_match["po_total"]
            inv.at[idx, "po_date"] = best_match["po_date"]
            inv.at[idx, "grn_number"] = best_match["grn_number"]
            inv.at[idx, "grn_date"] = best_match["grn_date"]
            inv.at[idx, "total_difference"] = inv_row["invoice_total"] - best_match["po_total"]
            inv.at[idx, "percentage_difference"] = best_match["diff_pct"]
            used_pos.add(best_match["po_number"])

    unmatched_mask = inv["match_type"] == "UNMATCHED"

    for idx, inv_row in inv[unmatched_mask].iterrows():
        po_candidates = po[
            (po["vendor_id"] == inv_row["vendor_id"]) &
            (po["currency"] == inv_row["currency"]) &
            (~po["po_number"].isin(used_pos))
        ]

        if len(po_candidates) == 0:
            inv.at[idx, "match_type"] = "MISSING_PO"
            continue

        po_candidates = po_candidates.copy()
        po_candidates["diff"] = abs(po_candidates["po_total"] - inv_row["invoice_total"])
        po_candidates["diff_pct"] = (po_candidates["diff"] / po_candidates["po_total"] * 100)

        fuzzy_matches = po_candidates[po_candidates["diff_pct"] <= tolerance_pct]

        if len(fuzzy_matches) > 0:
            best_match = fuzzy_matches.loc[fuzzy_matches["diff"].idxmin()]
            inv.at[idx, "matched_po_number"] = best_match["po_number"]
            inv.at[idx, "match_type"] = "FUZZY_MATCH"
            inv.at[idx, "po_total"] = best_match["po_total"]
            inv.at[idx, "po_date"] = best_match["po_date"]
            inv.at[idx, "grn_number"] = best_match["grn_number"]
            inv.at[idx, "grn_date"] = best_match["grn_date"]
            inv.at[idx, "total_difference"] = inv_row["invoice_total"] - best_match["po_total"]
            inv.at[idx, "percentage_difference"] = best_match["diff_pct"]
            used_pos.add(best_match["po_number"])
        else:
            closest = po_candidates.loc[po_candidates["diff"].idxmin()]
            inv.at[idx, "matched_po_number"] = closest["po_number"]
            inv.at[idx, "match_type"] = "VARIANCE_DETECTED"
            inv.at[idx, "po_total"] = closest["po_total"]
            inv.at[idx, "po_date"] = closest["po_date"]
            inv.at[idx, "grn_number"] = closest["grn_number"]
            inv.at[idx, "grn_date"] = closest["grn_date"]
            inv.at[idx, "total_difference"] = inv_row["invoice_total"] - closest["po_total"]
            inv.at[idx, "percentage_difference"] = closest["diff_pct"]
            used_pos.add(closest["po_number"])

    inv["days_between"] = (inv["invoice_date"] - inv["po_date"]).dt.days

    match_summary = inv["match_type"].value_counts()
    logger.info(f"Matching complete:\n{match_summary}")

    return inv


def load_all_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    invoices = load_invoices(data_dir / "invoices.csv")
    po_grn = load_po_grn(data_dir / "po_grn.csv")
    labels = load_labelled_mismatches(data_dir / "labelled_mismatches.csv")
    return invoices, po_grn, labels


def create_training_dataset(
    invoices_df: pd.DataFrame,
    po_grn_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Creating training dataset")

    inv_agg = aggregate_invoice_totals(invoices_df)
    merged = merge_invoice_po(inv_agg, po_grn_df)

    training_data = merged.merge(
        labels_df[["invoice_id", "mismatch_type"]],
        on="invoice_id",
        how="left"
    )

    training_data.loc[
        (training_data["mismatch_type"].isna()) &
        (training_data["match_type"] == "EXACT_MATCH"),
        "mismatch_type"
    ] = "EXACT_MATCH"

    training_data.loc[
        (training_data["mismatch_type"].isna()) &
        (training_data["match_type"] == "MISSING_PO"),
        "mismatch_type"
    ] = "MISSING_PO"

    logger.info(f"Created training dataset with {len(training_data)} samples")
    return training_data


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    print("Loading data files...")
    invoices, po_grn, labels = load_all_data(data_dir)

    print(f"\nInvoice Summary:")
    print(f"Total line items: {len(invoices)}")
    print(f"Unique invoices: {invoices['invoice_id'].nunique()}")
    print(f"Unique vendors: {invoices['vendor_id'].nunique()}")

    print(f"\nPO/GRN Summary:")
    print(f"Total POs: {len(po_grn)}")

    print(f"\nLabel Summary:")
    print(labels["mismatch_type"].value_counts())

    inv_agg = aggregate_invoice_totals(invoices)
    merged = merge_invoice_po(inv_agg, po_grn)

    print(f"\nMatch Results:")
    print(merged["match_type"].value_counts())
