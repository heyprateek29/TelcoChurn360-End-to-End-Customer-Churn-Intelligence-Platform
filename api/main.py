"""
PHASE 7: FASTAPI APPLICATION
Path: api/main.py
Run: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL & METADATA
# ─────────────────────────────────────────────

try:
    with open("models/churn_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open("models/model_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# ─────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="TelcoChurn360 API",
    description="End-to-End Customer Churn Prediction & Retention Actions",
    version="1.0.0"
)

# Add CORS middleware (allow frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# PYDANTIC MODELS (Request/Response Schemas)
# ─────────────────────────────────────────────

class CustomerData(BaseModel):
    """Single customer prediction request"""
    customer_id: Optional[str] = None
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    tenure: int = Field(..., ge=0, le=72)
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0, le=50000)
    
    @validator('contract')
    def validate_contract(cls, v):
        valid = ["Month-to-month", "One year", "Two year"]
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v


class FeatureImportance(BaseModel):
    """Feature importance item"""
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    """Single prediction response"""
    customer_id: Optional[str]
    prediction_timestamp: str
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_score: float
    retention_action: str
    expected_impact: str
    top_churn_drivers: List[FeatureImportance]
    model_confidence: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData]
    return_driver_analysis: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_records: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    total_revenue_at_risk: float
    processing_time_seconds: float
    predictions: Optional[List[PredictionResponse]] = None


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def encode_customer(customer_dict: dict) -> pd.DataFrame:
    """Encode customer data for model prediction"""
    
    # Create DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Drop customer_id if present
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    # Encode categorical features
    for col, encoder in metadata['encoders'].items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except Exception as e:
                logger.error(f"Encoding error for {col}: {str(e)}")
                raise
    
    # Reorder to match training features
    df = df[metadata['feature_names']]
    
    return df


def get_retention_action(
    churn_prob: float, 
    tenure: int, 
    monthly_charges: float,
    num_services: int,
    contract: str
) -> str:
    """Generate business-ready retention recommendation"""
    
    if churn_prob >= 0.6:
        if tenure <= 6:
            return "🚨 URGENT: Intensive 30-day onboarding restart + daily check-ins"
        elif monthly_charges >= 75:
            return "🚨 URGENT: Executive retention call for high-value customer at risk"
        else:
            return "🚨 URGENT: Retention offer (discount/service upgrade)"
    
    elif churn_prob >= 0.4:
        if num_services <= 2:
            return "🟡 Proactive: Recommend tech support + security bundle (cross-sell)"
        elif contract == "Month-to-month":
            return "🟡 Proactive: Incentivize upgrade to 1-year contract"
        else:
            return "🟡 Proactive: Monitor engagement, plan retention campaign"
    
    else:
        if num_services < 3:
            return "🟢 Opportunity: Upsell additional services for expansion revenue"
        else:
            return "🟢 Loyalty: Include in VIP retention program"


def calculate_risk_score(churn_prob: float) -> float:
    """Convert probability to risk score (0-100)"""
    return churn_prob * 100


def get_top_drivers(indices: np.ndarray, values: np.ndarray, top_n: int = 3) -> List[FeatureImportance]:
    """Get top churn drivers from model"""
    
    sorted_idx = np.argsort(np.abs(values[0]))[::-1][:top_n]
    drivers = []
    
    for idx in sorted_idx:
        feature_name = metadata['feature_names'][idx]
        importance = float(np.abs(values[0, idx]))
        drivers.append(FeatureImportance(feature=feature_name, importance=importance))
    
    return drivers


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_auc": metadata['test_auc'],
        "model_accuracy": metadata['test_acc'],
        "model_f1": metadata['test_f1']
    }


@app.get("/model/info")
def model_info():
    """Get model metadata"""
    return {
        "model_name": "Random Forest Churn Classifier",
        "model_version": "1.0.0",
        "test_auc": metadata['test_auc'],
        "test_accuracy": metadata['test_acc'],
        "test_precision": metadata['test_precision'],
        "test_recall": metadata['test_recall'],
        "test_f1": metadata['test_f1'],
        "feature_count": len(metadata['feature_names']),
        "features": metadata['feature_names'],
        "created_at": "2024-01-15"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerData,
    customer_id: Optional[str] = Query(None)
):
    """Single customer churn prediction"""
    
    try:
        # Use provided customer_id or generate one
        cust_id = customer_id or customer.customer_id or "UNKNOWN"
        
        # Encode customer data
        X = encode_customer(customer.dict())
        
        # Predict
        churn_prob = float(model.predict_proba(X)[0, 1])
        churn_class = int(model.predict(X)[0])
        
        # Determine risk level
        if churn_prob >= 0.6:
            risk_level = "HIGH"
        elif churn_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get retention action
        retention_action = get_retention_action(
            churn_prob, 
            customer.tenure,
            customer.monthly_charges,
            0,  # Will be calculated from customer data
            customer.contract
        )
        
        # Calculate expected impact
        annual_revenue = customer.monthly_charges * 12
        expected_impact = f"{'HIGH' if churn_prob >= 0.6 else 'MEDIUM' if churn_prob >= 0.4 else 'LOW'} RISK: ${annual_revenue:,.2f}/year at risk"
        
        # Get top drivers (using feature importance as proxy)
        top_drivers = []
        for feat, imp in sorted(
            metadata['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]:
            top_drivers.append(FeatureImportance(feature=feat, importance=imp))
        
        # Model confidence
        model_confidence = max(model.predict_proba(X)[0])
        
        return PredictionResponse(
            customer_id=cust_id,
            prediction_timestamp=datetime.now().isoformat(),
            churn_probability=churn_prob,
            predicted_class=churn_class,
            risk_level=risk_level,
            risk_score=calculate_risk_score(churn_prob),
            retention_action=retention_action,
            expected_impact=expected_impact,
            top_churn_drivers=top_drivers,
            model_confidence=model_confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Batch customer predictions"""
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        total_revenue_at_risk = 0
        
        for customer in request.customers:
            # Get single prediction
            pred = predict(customer)
            predictions.append(pred)
            
            # Aggregate stats
            if pred.risk_level == "HIGH":
                high_risk += 1
            elif pred.risk_level == "MEDIUM":
                medium_risk += 1
            else:
                low_risk += 1
            
            total_revenue_at_risk += float(pred.expected_impact.split('$')[1].split('/')[0].replace(',', '')) / 12
        
        elapsed = time.time() - start_time
        
        return BatchPredictionResponse(
            total_records=len(request.customers),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            total_revenue_at_risk=total_revenue_at_risk,
            processing_time_seconds=elapsed,
            predictions=predictions if request.return_driver_analysis else None
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ─────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "Welcome to TelcoChurn360 API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "endpoints": {
            "predict": "POST /predict?customer_id=X",
            "batch": "POST /predict/batch"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
