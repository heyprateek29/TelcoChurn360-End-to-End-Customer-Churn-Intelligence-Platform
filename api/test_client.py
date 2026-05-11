"""
PHASE 7: API TEST CLIENT
Path: api/test_client.py
Run: python api/test_client.py
"""

import time
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# SAMPLE CUSTOMERS
# ─────────────────────────────────────────────

HIGH_RISK_CUSTOMER = {
    "customer_id": "CUST-HIGH-001",
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "tenure": 3,
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
    "monthly_charges": 89.5,
    "total_charges": 268.5
}

MEDIUM_RISK_CUSTOMER = {
    "customer_id": "CUST-MED-001",
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "Yes",
    "tenure": 24,
    "phone_service": "Yes",
    "multiple_lines": "Yes",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "One year",
    "paperless_billing": "No",
    "payment_method": "Mailed check",
    "monthly_charges": 45.3,
    "total_charges": 1087.2
}

LOW_RISK_CUSTOMER = {
    "customer_id": "CUST-LOW-001",
    "gender": "Male",
    "senior_citizen": "Yes",
    "partner": "Yes",
    "dependents": "Yes",
    "tenure": 60,
    "phone_service": "Yes",
    "multiple_lines": "Yes",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Two year",
    "paperless_billing": "Yes",
    "payment_method": "Bank transfer",
    "monthly_charges": 98.5,
    "total_charges": 5910.0
}

# ─────────────────────────────────────────────
# TEST FUNCTIONS
# ─────────────────────────────────────────────

def print_header(title):
    """Print formatted test header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_success(message):
    """Print success message"""
    print(f"  ✅ {message}")


def print_error(message):
    """Print error message"""
    print(f"  ❌ {message}")


def test_health_check():
    """Test 1: Health check endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model AUC: {data['model_auc']:.4f}")
        print(f"  Model Accuracy: {data['model_accuracy']:.4f}")
        
        print_success("Health check passed")
        return True
    
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        print("  Make sure API is running: uvicorn api.main:app --reload")
        return False


def test_model_info():
    """Test 2: Model info endpoint"""
    print_header("TEST 2: Model Information")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"  Model: {data['model_name']}")
        print(f"  Version: {data['model_version']}")
        print(f"  Test AUC: {data['test_auc']:.4f}")
        print(f"  Features: {data['feature_count']}")
        
        print_success("Model info retrieved")
        return True
    
    except Exception as e:
        print_error(f"Failed to get model info: {str(e)}")
        return False


def test_single_prediction(customer, label):
    """Test single customer prediction"""
    print(f"\n  Testing {label}...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict?customer_id={customer['customer_id']}",
            json=customer
        )
        response.raise_for_status()
        
        pred = response.json()
        
        print(f"    Probability: {pred['churn_probability']:.2%}")
        print(f"    Risk Level: {pred['risk_level']}")
        print(f"    Risk Score: {pred['risk_score']:.1f}")
        print(f"    Impact: {pred['expected_impact']}")
        print(f"    Action: {pred['retention_action']}")
        
        return True
    
    except Exception as e:
        print_error(f"Prediction failed: {str(e)}")
        return False


def test_predictions():
    """Test 3-5: Individual predictions"""
    print_header("TEST 3-5: Individual Predictions")
    
    success = True
    success &= test_single_prediction(HIGH_RISK_CUSTOMER, "HIGH Risk")
    success &= test_single_prediction(MEDIUM_RISK_CUSTOMER, "MEDIUM Risk")
    success &= test_single_prediction(LOW_RISK_CUSTOMER, "LOW Risk")
    
    if success:
        print_success("All individual predictions passed")
    
    return success


def test_batch_prediction():
    """Test 6: Batch prediction endpoint"""
    print_header("TEST 6: Batch Prediction")
    
    try:
        payload = {
            "customers": [
                HIGH_RISK_CUSTOMER,
                MEDIUM_RISK_CUSTOMER,
                LOW_RISK_CUSTOMER
            ],
            "return_driver_analysis": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload
        )
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        data = response.json()
        print(f"  Total Records: {data['total_records']}")
        print(f"  High Risk: {data['high_risk_count']}")
        print(f"  Medium Risk: {data['medium_risk_count']}")
        print(f"  Low Risk: {data['low_risk_count']}")
        print(f"  Total Revenue at Risk: ${data['total_revenue_at_risk']:,.2f}")
        print(f"  Processing Time: {data['processing_time_seconds']:.3f}s")
        
        print_success(f"Batch prediction passed ({elapsed:.2f}s)")
        return True
    
    except Exception as e:
        print_error(f"Batch prediction failed: {str(e)}")
        return False


def test_error_handling():
    """Test 7: Error handling"""
    print_header("TEST 7: Error Handling")
    
    try:
        # Send invalid data
        bad_customer = HIGH_RISK_CUSTOMER.copy()
        bad_customer['tenure'] = 999  # Invalid (max 72)
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=bad_customer
        )
        
        if response.status_code == 422:  # Validation error
            print_success("Error handling works correctly (validation errors caught)")
            return True
        else:
            print_error("Expected validation error but got different response")
            return False
    
    except Exception as e:
        print_error(f"Error handling test failed: {str(e)}")
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("  TELCOCHURN360 API - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\n  This test validates that the FastAPI server is running")
    print("  and all endpoints function correctly.\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Predictions", test_predictions),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        results.append((name, test_func()))
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! API is production-ready.")
    else:
        print("\n  ⚠️  Some tests failed. Please check the output above.")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        exit(1)
