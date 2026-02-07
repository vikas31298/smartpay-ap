import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    invoice_id: str
    matched_po: Optional[str]
    match_confidence: float
    mismatch_type: str
    variance_amount: float
    explanation: str
    features: Dict[str, Any] = field(default_factory=dict)


class InvoicePOMatcher:
    EXACT_MATCH = "EXACT_MATCH"
    PRICE_VARIANCE = "PRICE_VARIANCE"
    QUANTITY_VARIANCE = "QUANTITY_VARIANCE"
    TAX_MISCODE = "TAX_MISCODE"
    MISSING_PO = "MISSING_PO"

    MISMATCH_TYPES = [EXACT_MATCH, PRICE_VARIANCE, QUANTITY_VARIANCE, TAX_MISCODE, MISSING_PO]

    def __init__(
        self,
        exact_tolerance_pct: float = 0.1,
        fuzzy_tolerance_pct: float = 5.0,
        use_ml_classification: bool = True,
        random_state: int = 42
    ):
        self.exact_tolerance_pct = exact_tolerance_pct
        self.fuzzy_tolerance_pct = fuzzy_tolerance_pct
        self.use_ml_classification = use_ml_classification
        self.random_state = random_state

        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []

        self.invoices_agg = None
        self.po_grn = None
        self.po_lookup = {}
        self.invoice_lookup = {}

        self._is_fitted = False

    def _aggregate_invoices(self, invoices_df: pd.DataFrame) -> pd.DataFrame:
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

        return agg_df

    def _create_features(self, invoice_row: pd.Series, po_row: Optional[pd.Series]) -> Dict[str, Any]:
        features = {}

        features["invoice_total"] = invoice_row.get("invoice_total", 0)
        features["line_count"] = invoice_row.get("line_count", 0)
        features["avg_unit_price"] = invoice_row.get("avg_unit_price", 0)
        features["total_quantity"] = invoice_row.get("total_quantity", 0)

        if po_row is not None:
            features["po_total"] = po_row.get("po_total", 0)

            total_diff = features["invoice_total"] - features["po_total"]
            features["total_difference"] = total_diff
            features["abs_difference"] = abs(total_diff)

            if features["po_total"] > 0:
                features["percentage_difference"] = abs(total_diff) / features["po_total"] * 100
            else:
                features["percentage_difference"] = 100.0

            if pd.notna(invoice_row.get("invoice_date")) and pd.notna(po_row.get("po_date")):
                inv_date = pd.to_datetime(invoice_row["invoice_date"])
                po_date = pd.to_datetime(po_row["po_date"])
                features["days_between"] = (inv_date - po_date).days
            else:
                features["days_between"] = 0

            features["vendor_match"] = 1 if invoice_row.get("vendor_id") == po_row.get("vendor_id") else 0
            features["currency_match"] = 1 if invoice_row.get("currency") == po_row.get("currency") else 0
            features["has_po"] = 1
        else:
            features["po_total"] = 0
            features["total_difference"] = features["invoice_total"]
            features["abs_difference"] = features["invoice_total"]
            features["percentage_difference"] = 100.0
            features["days_between"] = 0
            features["vendor_match"] = 0
            features["currency_match"] = 0
            features["has_po"] = 0

        return features

    def _find_best_po_match(self, invoice_row: pd.Series) -> Tuple[Optional[pd.Series], str, float]:
        vendor_id = invoice_row["vendor_id"]
        currency = invoice_row["currency"]
        invoice_total = invoice_row["invoice_total"]

        candidates = self.po_grn[
            (self.po_grn["vendor_id"] == vendor_id) &
            (self.po_grn["currency"] == currency)
        ]

        if len(candidates) == 0:
            return None, self.MISSING_PO, 0.0

        candidates = candidates.copy()
        candidates["diff"] = abs(candidates["po_total"] - invoice_total)
        candidates["diff_pct"] = candidates["diff"] / candidates["po_total"] * 100

        best_idx = candidates["diff"].idxmin()
        best_po = candidates.loc[best_idx]
        best_diff_pct = best_po["diff_pct"]

        if best_diff_pct <= self.exact_tolerance_pct:
            match_type = self.EXACT_MATCH
            confidence = 1.0 - (best_diff_pct / self.exact_tolerance_pct) * 0.1
        elif best_diff_pct <= self.fuzzy_tolerance_pct:
            match_type = "FUZZY_MATCH"
            confidence = 0.9 - (best_diff_pct / self.fuzzy_tolerance_pct) * 0.3
        else:
            match_type = "VARIANCE_DETECTED"
            confidence = max(0.1, 0.6 - (best_diff_pct / 100) * 0.5)

        return best_po, match_type, confidence

    def _prepare_training_data(
        self,
        invoices_df: pd.DataFrame,
        po_grn_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []

        inv_agg = self._aggregate_invoices(invoices_df)
        inv_lookup = inv_agg.set_index("invoice_id").to_dict("index")

        po_by_vendor = po_grn_df.groupby(["vendor_id", "currency"])

        for _, label_row in labels_df.iterrows():
            invoice_id = label_row["invoice_id"]
            mismatch_type = label_row["mismatch_type"]

            if invoice_id not in inv_lookup:
                continue

            inv_data = inv_lookup[invoice_id]
            inv_series = pd.Series(inv_data)
            inv_series["invoice_id"] = invoice_id

            vendor_id = inv_data.get("vendor_id")
            currency = inv_data.get("currency")

            po_row = None
            try:
                vendor_pos = po_by_vendor.get_group((vendor_id, currency))
                if len(vendor_pos) > 0:
                    vendor_pos = vendor_pos.copy()
                    vendor_pos["diff"] = abs(vendor_pos["po_total"] - inv_data["invoice_total"])
                    best_idx = vendor_pos["diff"].idxmin()
                    po_row = vendor_pos.loc[best_idx]
            except KeyError:
                pass

            features = self._create_features(inv_series, po_row)

            X_list.append([
                features["percentage_difference"],
                features["abs_difference"],
                features["days_between"],
                features["vendor_match"],
                features["currency_match"],
                features["has_po"],
                features["line_count"],
                features["invoice_total"],
            ])
            y_list.append(mismatch_type)

        return np.array(X_list), np.array(y_list)

    def fit(
        self,
        invoices_df: pd.DataFrame,
        po_grn_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> "InvoicePOMatcher":
        logger.info("Fitting InvoicePOMatcher...")

        self.invoices_agg = self._aggregate_invoices(invoices_df)
        self.po_grn = po_grn_df.copy()

        self.invoice_lookup = self.invoices_agg.set_index("invoice_id").to_dict("index")
        self.po_lookup = self.po_grn.set_index("po_number").to_dict("index")

        X, y = self._prepare_training_data(invoices_df, po_grn_df, labels_df)

        if len(X) == 0:
            logger.warning("No training data available")
            self._is_fitted = True
            return self

        logger.info(f"Training on {len(X)} samples")

        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight="balanced"
        )
        self.classifier.fit(X_scaled, y_encoded)

        self.feature_columns = [
            "percentage_difference", "abs_difference", "days_between",
            "vendor_match", "currency_match", "has_po", "line_count", "invoice_total"
        ]

        self._is_fitted = True
        logger.info("Fitting complete")

        return self

    def _classify_mismatch_ml(self, features: Dict[str, Any]) -> Tuple[str, float]:
        if self.classifier is None:
            return self._classify_mismatch_rules(features)

        X = np.array([[
            features["percentage_difference"],
            features["abs_difference"],
            features["days_between"],
            features["vendor_match"],
            features["currency_match"],
            features["has_po"],
            features["line_count"],
            features["invoice_total"],
        ]])

        X_scaled = self.scaler.transform(X)
        pred = self.classifier.predict(X_scaled)[0]
        proba = self.classifier.predict_proba(X_scaled)[0]

        mismatch_type = self.label_encoder.inverse_transform([pred])[0]
        confidence = proba.max()

        return mismatch_type, confidence

    def _classify_mismatch_rules(self, features: Dict[str, Any]) -> Tuple[str, float]:
        if features["has_po"] == 0:
            return self.MISSING_PO, 0.95

        pct_diff = features["percentage_difference"]
        abs_diff = features["abs_difference"]

        if pct_diff <= self.exact_tolerance_pct:
            return self.EXACT_MATCH, 0.99

        if pct_diff < 3:
            return self.TAX_MISCODE, 0.7
        elif pct_diff < 10:
            if abs_diff > 1000:
                return self.QUANTITY_VARIANCE, 0.65
            else:
                return self.PRICE_VARIANCE, 0.65
        else:
            return self.QUANTITY_VARIANCE, 0.6

    def _generate_explanation(
        self,
        invoice_row: pd.Series,
        po_row: Optional[pd.Series],
        mismatch_type: str,
        features: Dict[str, Any]
    ) -> str:
        invoice_id = invoice_row.get("invoice_id", "Unknown")
        invoice_total = features.get("invoice_total", 0)

        if mismatch_type == self.EXACT_MATCH:
            po_number = po_row["po_number"] if po_row is not None else "Unknown"
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) exactly matches "
                f"PO {po_number} within {self.exact_tolerance_pct}% tolerance."
            )

        if mismatch_type == self.MISSING_PO:
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) has no matching PO found "
                f"for vendor {invoice_row.get('vendor_id', 'Unknown')}."
            )

        po_number = po_row["po_number"] if po_row is not None else "Unknown"
        po_total = features.get("po_total", 0)
        diff = features.get("total_difference", 0)
        pct_diff = features.get("percentage_difference", 0)

        direction = "over" if diff > 0 else "under"

        if mismatch_type == self.PRICE_VARIANCE:
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) is ${abs(diff):,.2f} ({pct_diff:.1f}%) "
                f"{direction} PO {po_number} (${po_total:,.2f}). "
                f"Likely caused by unit price differences."
            )
        elif mismatch_type == self.QUANTITY_VARIANCE:
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) is ${abs(diff):,.2f} ({pct_diff:.1f}%) "
                f"{direction} PO {po_number} (${po_total:,.2f}). "
                f"Likely caused by quantity differences."
            )
        elif mismatch_type == self.TAX_MISCODE:
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) differs from PO {po_number} "
                f"(${po_total:,.2f}) by ${abs(diff):,.2f} ({pct_diff:.1f}%). "
                f"Pattern suggests tax calculation discrepancy."
            )
        else:
            return (
                f"Invoice {invoice_id} (${invoice_total:,.2f}) has variance of "
                f"${abs(diff):,.2f} ({pct_diff:.1f}%) from closest PO {po_number}."
            )

    def predict(self, invoice_id: str) -> Dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if invoice_id not in self.invoice_lookup:
            return {
                "matched_po": None,
                "match_confidence": 0.0,
                "mismatch_type": self.MISSING_PO,
                "variance_amount": 0.0,
                "explanation": f"Invoice {invoice_id} not found in data."
            }

        inv_data = self.invoice_lookup[invoice_id]
        inv_series = pd.Series(inv_data)
        inv_series["invoice_id"] = invoice_id

        po_row, match_type, match_confidence = self._find_best_po_match(inv_series)

        features = self._create_features(inv_series, po_row)

        if match_type == self.EXACT_MATCH:
            mismatch_type = self.EXACT_MATCH
            classification_confidence = 0.99
        elif match_type == self.MISSING_PO:
            mismatch_type = self.MISSING_PO
            classification_confidence = 0.95
        else:
            if self.use_ml_classification and self.classifier is not None:
                mismatch_type, classification_confidence = self._classify_mismatch_ml(features)
            else:
                mismatch_type, classification_confidence = self._classify_mismatch_rules(features)

        final_confidence = match_confidence * 0.6 + classification_confidence * 0.4

        explanation = self._generate_explanation(inv_series, po_row, mismatch_type, features)

        return {
            "matched_po": po_row["po_number"] if po_row is not None else None,
            "match_confidence": round(final_confidence, 3),
            "mismatch_type": mismatch_type,
            "variance_amount": round(features.get("total_difference", 0), 2),
            "explanation": explanation,
            "features": features
        }

    def predict_batch(self, invoice_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(inv_id) for inv_id in invoice_ids]

    def get_feature_importance(self) -> Dict[str, float]:
        if self.classifier is None:
            return {}

        importance = dict(zip(
            self.feature_columns,
            self.classifier.feature_importances_
        ))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "invoices_agg": self.invoices_agg,
                "po_grn": self.po_grn,
                "invoice_lookup": self.invoice_lookup,
                "po_lookup": self.po_lookup,
                "exact_tolerance_pct": self.exact_tolerance_pct,
                "fuzzy_tolerance_pct": self.fuzzy_tolerance_pct,
                "is_fitted": self._is_fitted,
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "InvoicePOMatcher":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(
            exact_tolerance_pct=data["exact_tolerance_pct"],
            fuzzy_tolerance_pct=data["fuzzy_tolerance_pct"]
        )
        model.classifier = data["classifier"]
        model.label_encoder = data["label_encoder"]
        model.scaler = data["scaler"]
        model.feature_columns = data["feature_columns"]
        model.invoices_agg = data["invoices_agg"]
        model.po_grn = data["po_grn"]
        model.invoice_lookup = data["invoice_lookup"]
        model.po_lookup = data["po_lookup"]
        model._is_fitted = data["is_fitted"]

        logger.info(f"Model loaded from {filepath}")
        return model


def train_and_evaluate(data_dir: str | Path) -> Tuple[InvoicePOMatcher, Dict[str, Any]]:
    from .data_loader import load_all_data

    data_dir = Path(data_dir)

    invoices, po_grn, labels = load_all_data(data_dir)

    train_labels, test_labels = train_test_split(
        labels, test_size=0.2, random_state=42, stratify=labels["mismatch_type"]
    )

    model = InvoicePOMatcher()
    model.fit(invoices, po_grn, train_labels)

    y_true = []
    y_pred = []

    for _, row in test_labels.iterrows():
        result = model.predict(row["invoice_id"])
        y_true.append(row["mismatch_type"])
        y_pred.append(result["mismatch_type"])

    from sklearn.metrics import classification_report, accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return model, {"accuracy": accuracy, "classification_report": report}


if __name__ == "__main__":
    from data_loader import load_all_data

    data_dir = Path(__file__).parent.parent / "data"

    print("Loading data...")
    invoices, po_grn, labels = load_all_data(data_dir)

    print("\nTraining model...")
    model = InvoicePOMatcher()
    model.fit(invoices, po_grn, labels)

    print("\nFeature Importance:")
    for feat, imp in model.get_feature_importance().items():
        print(f"  {feat}: {imp:.4f}")

    print("\nSample Predictions:")
    sample_ids = labels["invoice_id"].head(5).tolist()

    for inv_id in sample_ids:
        result = model.predict(inv_id)
        print(f"\n{inv_id}:")
        print(f"  Matched PO: {result['matched_po']}")
        print(f"  Confidence: {result['match_confidence']:.2%}")
        print(f"  Type: {result['mismatch_type']}")
        print(f"  Variance: ${result['variance_amount']:,.2f}")
