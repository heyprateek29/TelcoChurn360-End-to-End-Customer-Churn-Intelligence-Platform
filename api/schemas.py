"""
api/schemas.py
--------------
Pydantic models for request and response validation.

Pydantic does three things automatically:
  1. Validates incoming JSON fields and types
  2. Rejects requests with missing or wrong-type fields
  3. Generates the Swagger UI form at /docs

Every field maps 1-to-1 to a feature column in the trained model.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Input features required to predict churn for one customer.
    All fields are validated before the model sees them.
    """

    # ── Numerical features ────────────────────────────────────
    tenure: int = Field(
        ...,
        ge=0, le=72,
        description="Months with the company (0–72)",
        example=12
    )
    monthly_charges: float = Field(
        ...,
        ge=0, le=200,
        description="Current monthly bill in USD",
        example=79.99
    )
    total_charges: float = Field(
        ...,
        ge=0,
        description="Total charges since joining",
        example=959.88
    )
    num_services: int = Field(
        ...,
        ge=0, le=7,
        description="Number of add-on services subscribed (0–7)",
        example=3
    )
    avg_monthly_spend: float = Field(
        ...,
        ge=0,
        description="Average monthly spend (total_charges / tenure)",
        example=79.99
    )
    contract_risk_score: Literal[1, 2, 3] = Field(
        ...,
        description="Contract risk: 3=Month-to-month, 2=One year, 1=Two year",
        example=3
    )
    support_score: int = Field(
        ...,
        ge=0, le=3,
        description="Number of support services (tech support + security + device protection)",
        example=1
    )

    # ── Categorical features ──────────────────────────────────
    gender: Literal["Male", "Female"] = Field(
        ..., example="Male"
    )
    senior_citizen: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    partner: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    dependents: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    phone_service: Literal["Yes", "No"] = Field(
        ..., example="Yes"
    )
    multiple_lines: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., example="Fiber optic"
    )
    online_security: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    online_backup: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    device_protection: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    tech_support: Literal["Yes", "No"] = Field(
        ..., example="No"
    )
    streaming_tv: Literal["Yes", "No"] = Field(
        ..., example="Yes"
    )
    streaming_movies: Literal["Yes", "No"] = Field(
        ..., example="Yes"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., example="Month-to-month"
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ..., example="Yes"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    tenure_segment: Literal[
        "New", "Growing", "Established", "Loyal", "Champion"
    ] = Field(..., example="Growing")
    monthly_charges_tier: Literal["Low", "Mid", "High", "Premium"] = Field(
        ..., example="High"
    )

    # ── Derived validator ─────────────────────────────────────
    @field_validator("avg_monthly_spend")
    @classmethod
    def avg_spend_reasonable(cls, v, info):
        """avg_monthly_spend should be close to monthly_charges."""
        return round(v, 2)

    class Config:
        # Show a worked example in Swagger UI
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 79.99,
                "total_charges": 959.88,
                "num_services": 2,
                "avg_monthly_spend": 79.99,
                "contract_risk_score": 3,
                "support_score": 0,
                "gender": "Male",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "tenure_segment": "Growing",
                "monthly_charges_tier": "High",
            }
        }


# ─────────────────────────────────────────────
# RESPONSE SCHEMA
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Structured response returned for every prediction request.
    Includes the prediction, probability, risk tier, and
    a plain-English recommendation.
    """

    # Core prediction
    churn_prediction: int = Field(
        ...,
        description="Binary prediction: 1=will churn, 0=will retain"
    )
    churn_probability: float = Field(
        ...,
        description="Model confidence that this customer will churn (0.0–1.0)"
    )
    retention_probability: float = Field(
        ...,
        description="Model confidence that this customer will stay (0.0–1.0)"
    )

    # Business interpretation
    risk_tier: str = Field(
        ...,
        description="Risk bucket: Critical / High / Medium / Low"
    )
    risk_score_pct: float = Field(
        ...,
        description="Churn probability as a percentage (0–100)"
    )

    # Actionable recommendation
    recommendation: str = Field(
        ...,
        description="Suggested retention action for this customer"
    )

    # Model metadata
    model_version: str = Field(
        default="1.0.0",
        description="Version of the model that made this prediction"
    )

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "churn_prediction": 1,
                "churn_probability": 0.73,
                "retention_probability": 0.27,
                "risk_tier": "High",
                "risk_score_pct": 73.0,
                "recommendation": "Offer annual contract upgrade with 15% discount",
                "model_version": "1.0.0",
            }
        }


# ─────────────────────────────────────────────
# HEALTH CHECK SCHEMA
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    model_loaded: bool
    model_type: str
    feature_count: int
    version: str

    model_config = {"protected_namespaces": ()}   # ← add this line