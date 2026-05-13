"""
test_api.py
-----------
Local test script for the FastAPI endpoints.

Run this AFTER starting the API server:
    # Terminal 1:
    uvicorn api.main:app --reload --port 8000

    # Terminal 2:
    python test_api.py

Tests:
  1. Health check
  2. High-risk customer prediction
  3. Low-risk customer prediction
  4. Batch prediction (3 customers)
  5. Validation error test (bad input)
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def pretty(data: dict) -> None:
    print(json.dumps(data, indent=2))


# ─────────────────────────────────────────────
# TEST 1: Health check
# ─────────────────────────────────────────────

def test_health():
    separator("TEST 1: Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status code : {r.status_code}")
    pretty(r.json())
    assert r.status_code == 200, "Health check failed"
    assert r.json()["model_loaded"] is True, "Model not loaded"
    print("  ✓ PASSED")


# ─────────────────────────────────────────────
# TEST 2: High-risk customer
# ─────────────────────────────────────────────

def test_high_risk_customer():
    separator("TEST 2: High-Risk Customer Prediction")

    # Profile: New customer, month-to-month, fiber, no support services
    # Expectation: High churn probability
    customer = {
        "tenure": 3,
        "monthly_charges": 95.50,
        "total_charges": 286.50,
        "num_services": 1,
        "avg_monthly_spend": 95.50,
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
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "tenure_segment": "New",
        "monthly_charges_tier": "Premium",
    }

    r = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status code : {r.status_code}")
    result = r.json()
    pretty(result)

    assert r.status_code == 200
    assert result["churn_probability"] > 0.4, \
        f"Expected high churn prob, got {result['churn_probability']}"
    print("  ✓ PASSED — high churn probability as expected")


# ─────────────────────────────────────────────
# TEST 3: Low-risk customer
# ─────────────────────────────────────────────

def test_low_risk_customer():
    separator("TEST 3: Low-Risk Customer Prediction")

    # Profile: Long tenure, two-year contract, all support services
    # Expectation: Low churn probability
    customer = {
        "tenure": 60,
        "monthly_charges": 55.00,
        "total_charges": 3300.00,
        "num_services": 6,
        "avg_monthly_spend": 55.00,
        "contract_risk_score": 1,
        "support_score": 3,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
        "tenure_segment": "Champion",
        "monthly_charges_tier": "Mid",
    }

    r = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status code : {r.status_code}")
    result = r.json()
    pretty(result)

    assert r.status_code == 200
    assert result["churn_probability"] < 0.4, \
        f"Expected low churn prob, got {result['churn_probability']}"
    print("  ✓ PASSED — low churn probability as expected")


# ─────────────────────────────────────────────
# TEST 4: Batch prediction
# ─────────────────────────────────────────────

def test_batch():
    separator("TEST 4: Batch Prediction (3 customers)")

    customers = [
        # Customer 1: High risk
        {
            "tenure": 2, "monthly_charges": 90.0, "total_charges": 180.0,
            "num_services": 1, "avg_monthly_spend": 90.0,
            "contract_risk_score": 3, "support_score": 0,
            "gender": "Male", "senior_citizen": "No", "partner": "No",
            "dependents": "No", "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No",
            "online_backup": "No", "device_protection": "No",
            "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "tenure_segment": "New", "monthly_charges_tier": "Premium",
        },
        # Customer 2: Medium risk
        {
            "tenure": 18, "monthly_charges": 70.0, "total_charges": 1260.0,
            "num_services": 3, "avg_monthly_spend": 70.0,
            "contract_risk_score": 3, "support_score": 1,
            "gender": "Female", "senior_citizen": "No", "partner": "Yes",
            "dependents": "No", "phone_service": "Yes", "multiple_lines": "Yes",
            "internet_service": "DSL", "online_security": "Yes",
            "online_backup": "No", "device_protection": "No",
            "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
            "contract": "Month-to-month", "paperless_billing": "No",
            "payment_method": "Mailed check",
            "tenure_segment": "Established", "monthly_charges_tier": "High",
        },
        # Customer 3: Low risk
        {
            "tenure": 55, "monthly_charges": 45.0, "total_charges": 2475.0,
            "num_services": 5, "avg_monthly_spend": 45.0,
            "contract_risk_score": 1, "support_score": 3,
            "gender": "Male", "senior_citizen": "No", "partner": "Yes",
            "dependents": "Yes", "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "DSL", "online_security": "Yes",
            "online_backup": "Yes", "device_protection": "Yes",
            "tech_support": "Yes", "streaming_tv": "Yes", "streaming_movies": "No",
            "contract": "Two year", "paperless_billing": "No",
            "payment_method": "Credit card (automatic)",
            "tenure_segment": "Champion", "monthly_charges_tier": "Low",
        },
    ]

    r = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    print(f"Status code : {r.status_code}")
    result = r.json()
    print(f"Batch size  : {result['batch_size']}")
    print(f"Summary     : {result['summary']}")
    print("\nPer-customer results:")
    for p in result["predictions"]:
        print(f"  [{p['index']}] prob={p['churn_probability']:.2f}  "
              f"tier={p['risk_tier']:<10}  "
              f"action={p['recommendation'][:50]}...")

    assert r.status_code == 200
    assert result["batch_size"] == 3
    print("  ✓ PASSED")


# ─────────────────────────────────────────────
# TEST 5: Validation error
# ─────────────────────────────────────────────

def test_validation_error():
    separator("TEST 5: Validation Error (bad input)")

    bad_payload = {
        "tenure": 999,           # Invalid: max is 72
        "contract": "Weekly",    # Invalid: not in allowed values
    }

    r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    print(f"Status code : {r.status_code}")
    print(f"Response    : {r.json()['detail'][0]['msg']}")

    assert r.status_code == 422, \
        f"Expected 422 Unprocessable Entity, got {r.status_code}"
    print("  ✓ PASSED — FastAPI correctly rejected bad input")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  TelcoChurn360 — API Test Suite")
    print("=" * 50)
    print("  Make sure the API is running:")
    print("  uvicorn api.main:app --reload --port 8000")

    try:
        test_health()
        test_high_risk_customer()
        test_low_risk_customer()
        test_batch()
        test_validation_error()

        print("\n" + "=" * 50)
        print("  ALL TESTS PASSED ✓")
        print("=" * 50 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n  ERROR: Could not connect to API.")
        print("  Is the server running?")
        print("  Start it with: uvicorn api.main:app --reload --port 8000\n")