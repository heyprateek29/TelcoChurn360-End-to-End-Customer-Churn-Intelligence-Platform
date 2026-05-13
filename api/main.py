"""
api/main.py
-----------
Phase 7: FastAPI churn prediction service.

Endpoints:
  GET  /              → Welcome message
  GET  /health        → Model status and metadata
  POST /predict       → Predict churn for one customer
  POST /predict/batch → Predict churn for multiple customers

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs    ← Swagger UI (interactive testing)
    http://localhost:8000/redoc  ← ReDoc (clean documentation)
"""

from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from api.schemas import CustomerFeatures, PredictionResponse, HealthResponse
from api.model_loader import (
    ModelStore, MODEL_VERSION,
    get_risk_tier, get_recommendation
)


# ─────────────────────────────────────────────
# APP LIFESPAN — Load model at startup
# ─────────────────────────────────────────────

model_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: runs setup before the API starts accepting requests,
    and cleanup when it shuts down.
    This is FastAPI's recommended pattern (replaces @app.on_event).
    """
    print("\n" + "─" * 45)
    print("  TelcoChurn360 API — Starting up")
    print("─" * 45)
    model_store.load()
    print("  API ready. Visit http://localhost:8000/docs")
    print("─" * 45 + "\n")

    yield  # API runs here

    print("\nShutting down — model store cleared.")


# ─────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────

app = FastAPI(
    title="TelcoChurn360 — Churn Prediction API",
    description="""
## Customer Churn Prediction Service

Predicts the probability that a telecom customer will churn,
and returns a risk tier plus a plain-English retention recommendation.

### How to use
1. Send a **POST /predict** request with the customer's features
2. Receive churn probability, risk tier, and retention action
3. Use **GET /health** to confirm the model is loaded

### Model
- Algorithm: Random Forest (300 trees)
- Features: 25 raw → 67 encoded
- ROC-AUC: ~0.84 on held-out test set
- Training data: IBM Telco Customer Churn dataset (7,043 customers)
    """,
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ── CORS — allow Streamlit dashboard to call the API ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MIDDLEWARE — Request timing
# ─────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header to every response."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    return response


# ─────────────────────────────────────────────
# ENDPOINT 1: Root
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Welcome message and link to docs."""
    return {
        "service":     "TelcoChurn360 Prediction API",
        "version":     MODEL_VERSION,
        "status":      "running",
        "docs":        "http://localhost:8000/docs",
        "health":      "http://localhost:8000/health",
        "predict":     "POST http://localhost:8000/predict",
    }


# ─────────────────────────────────────────────
# ENDPOINT 2: Health check
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """
    Health check — confirms the model is loaded and ready.
    Use this in CI/CD pipelines and monitoring dashboards.
    """
    return HealthResponse(
        status        = "healthy" if model_store.is_loaded() else "degraded",
        model_loaded  = model_store.is_loaded(),
        model_type    = model_store.model_type,
        feature_count = model_store.feature_count,
        version       = MODEL_VERSION,
    )


# ─────────────────────────────────────────────
# HELPER: Build features DataFrame
# ─────────────────────────────────────────────

def build_features_df(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert the Pydantic model to a single-row DataFrame
    with column names matching exactly what the model was trained on.
    """
    feature_meta = model_store.feature_meta
    all_features = feature_meta["all_features"]

    # Convert Pydantic model to dict, then to DataFrame
    data = customer.model_dump()
    df   = pd.DataFrame([data])

    # Select only the columns the model expects, in the right order
    return df[all_features]


# ─────────────────────────────────────────────
# ENDPOINT 3: Single prediction
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    **Input:** Customer features (demographics, services, contract, billing)

    **Output:**
    - `churn_prediction`: 1 = will churn, 0 = will retain
    - `churn_probability`: model confidence (0.0–1.0)
    - `risk_tier`: Critical / High / Medium / Low
    - `recommendation`: plain-English retention action
    """

    if not model_store.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the API server."
        )

    try:
        # Build feature DataFrame
        features_df = build_features_df(customer)

        # Run prediction
        churn_class, churn_prob = model_store.predict(features_df)

        # Derive business outputs
        risk_tier      = get_risk_tier(churn_prob)
        recommendation = get_recommendation(
            churn_probability = churn_prob,
            contract          = customer.contract,
            tenure            = customer.tenure,
            internet_service  = customer.internet_service,
            monthly_charges   = customer.monthly_charges,
            support_score     = customer.support_score,
            num_services      = customer.num_services,
        )

        return PredictionResponse(
            churn_prediction     = churn_class,
            churn_probability    = round(churn_prob, 4),
            retention_probability= round(1 - churn_prob, 4),
            risk_tier            = risk_tier,
            risk_score_pct       = round(churn_prob * 100, 1),
            recommendation       = recommendation,
            model_version        = MODEL_VERSION,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# ENDPOINT 4: Batch prediction
# ─────────────────────────────────────────────

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    """
    Predict churn for a list of customers in one request.

    **Input:** List of customer feature objects (max 500)

    **Output:** List of prediction results in the same order as input.

    Use this for bulk scoring — e.g. scoring all customers nightly
    and loading results into a CRM.
    """

    if not model_store.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(customers) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 500 customers per request."
        )

    if len(customers) == 0:
        raise HTTPException(
            status_code=400,
            detail="Request body must contain at least one customer."
        )

    try:
        results = []

        for i, customer in enumerate(customers):
            features_df     = build_features_df(customer)
            churn_class, churn_prob = model_store.predict(features_df)
            risk_tier       = get_risk_tier(churn_prob)
            recommendation  = get_recommendation(
                churn_probability = churn_prob,
                contract          = customer.contract,
                tenure            = customer.tenure,
                internet_service  = customer.internet_service,
                monthly_charges   = customer.monthly_charges,
                support_score     = customer.support_score,
                num_services      = customer.num_services,
            )

            results.append({
                "index":               i,
                "churn_prediction":    churn_class,
                "churn_probability":   round(churn_prob, 4),
                "retention_probability": round(1 - churn_prob, 4),
                "risk_tier":           risk_tier,
                "risk_score_pct":      round(churn_prob * 100, 1),
                "recommendation":      recommendation,
            })

        return {
            "batch_size":  len(customers),
            "predictions": results,
            "summary": {
                "predicted_churn":    sum(r["churn_prediction"] for r in results),
                "predicted_retain":   sum(1 - r["churn_prediction"] for r in results),
                "avg_churn_prob":     round(
                    sum(r["churn_probability"] for r in results) / len(results), 4
                ),
                "critical_count":     sum(1 for r in results if r["risk_tier"] == "Critical"),
                "high_count":         sum(1 for r in results if r["risk_tier"] == "High"),
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )