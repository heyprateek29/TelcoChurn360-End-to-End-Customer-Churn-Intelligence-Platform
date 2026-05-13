"""
api/model_loader.py
-------------------
Loads the trained pipeline once at startup and keeps it in memory.

Why load at startup and not per-request?
  Loading a .pkl file takes ~200ms.
  If you load it per-request, every API call pays that cost.
  FastAPI's lifespan pattern loads it once and reuses it — 
  this is the production-correct approach.
"""

import os
import joblib
import pandas as pd
from typing import Optional


MODEL_PATH   = "models/churn_model.pkl"
META_PATH    = "models/feature_meta.pkl"
MODEL_VERSION = "1.0.0"


class ModelStore:
    """
    Singleton-style store that holds the loaded pipeline and metadata.
    FastAPI's app.state will hold one instance of this across all requests.
    """

    def __init__(self):
        self.pipeline     = None
        self.feature_meta = None
        self.model_type   = "not loaded"
        self.feature_count = 0

    def load(self) -> None:
        """Load model pipeline and feature metadata from disk."""

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Run Phase 4 first: python src/train.py"
            )

        self.pipeline     = joblib.load(MODEL_PATH)
        self.feature_meta = joblib.load(META_PATH)
        self.model_type   = type(
            self.pipeline.named_steps["classifier"]
        ).__name__
        self.feature_count = len(self.feature_meta["all_features"])

        print(f"  Model loaded : {self.model_type}")
        print(f"  Features     : {self.feature_count}")
        print(f"  Version      : {MODEL_VERSION}")

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, features_df: pd.DataFrame) -> tuple[int, float]:
        """
        Run prediction on a single-row DataFrame.
        Returns (predicted_class, churn_probability).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call .load() first.")

        # predict_proba returns [[prob_class0, prob_class1]]
        proba = self.pipeline.predict_proba(features_df)[0]
        churn_prob   = float(proba[1])
        churn_class  = int(self.pipeline.predict(features_df)[0])

        return churn_class, churn_prob


# ─────────────────────────────────────────────
# BUSINESS LOGIC HELPERS
# ─────────────────────────────────────────────

def get_risk_tier(churn_probability: float) -> str:
    """Map churn probability to a business risk tier."""
    if churn_probability >= 0.70:
        return "Critical"
    elif churn_probability >= 0.50:
        return "High"
    elif churn_probability >= 0.30:
        return "Medium"
    else:
        return "Low"


def get_recommendation(churn_probability: float,
                        contract: str,
                        tenure: int,
                        internet_service: str,
                        monthly_charges: float,
                        support_score: int,
                        num_services: int) -> str:
    """
    Return a plain-English retention recommendation
    based on the customer's profile and predicted risk.
    """

    if churn_probability < 0.30:
        return "Low risk — standard engagement. Consider upsell opportunities."

    # High risk logic — prioritise by contract type and tenure
    if contract == "Month-to-month":
        if tenure <= 6:
            return (
                "URGENT: New high-risk customer. "
                "Offer annual contract with 20% first-year discount + onboarding call."
            )
        elif tenure <= 12:
            return (
                "Offer annual contract upgrade with 15% discount. "
                "Emphasise savings vs month-to-month over 12 months."
            )
        elif internet_service == "Fiber optic" and monthly_charges > 75:
            return (
                "High-spend fiber customer at risk. "
                "Proactive tech support call + loyalty rate review recommended."
            )
        else:
            return (
                "Send contract upgrade campaign. "
                "Highlight stability benefits and loyalty rewards."
            )

    if support_score == 0:
        return (
            "No support services — upsell tech support bundle. "
            "Engaged customers with support churn at half the rate."
        )

    if num_services <= 2:
        return (
            "Low service adoption — offer bundled services trial. "
            "More services = higher switching cost = lower churn."
        )

    return (
        f"Monitor closely (churn probability: {churn_probability:.0%}). "
        "Personalised outreach recommended."
    )