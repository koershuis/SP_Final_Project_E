import sys
import os
from fastapi.testclient import TestClient

# PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 2. IMPORT API
from api.main import app

# 3. WE CREATE THE TEST CLIENT
client = TestClient(app)

# --- TEST 1: Verify healthy patient of Low Risk ---
def test_predict_low_risk():
    # Data from a young, healthy person (expected class 0)
    healthy_patient = {
        "sex": "F",
        "age": 25,
        "education": 4.0,
        "currentSmoker": "No",
        "cigsPerDay": 0.0,
        "BPMeds": 0.0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
        "diabetes": 0,
        "totChol": 130.0,
        "sysBP": 100.0,
        "diaBP": 60.0,
        "BMI": 19.5,
        "heartRate": 55.0,
        "glucose": 70.0
    }

    # We simulate sending a POST to /predict
    response = client.post("/predict", json=healthy_patient)

    # ASSERTS
    # 1. That the API responds with “200 OK” (no error)
    assert response.status_code == 200
    
    # 2. Let the prediction be 0 (Healthy)
    json_data = response.json()
    assert json_data["prediction"] == 0
    assert json_data["diagnosis"] == "Low Risk"

# --- TEST 2: Verify sick patient of High Risk ---
def test_predict_high_risk():
    # Data on an elderly person with risk factors (expected class 1)
    sick_patient = {
        "sex": "M",
        "age": 68,
        "education": 1.0,
        "currentSmoker": "Yes",
        "cigsPerDay": 40.0,
        "BPMeds": 1.0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 1,
        "totChol": 290.0,
        "sysBP": 185.0,
        "diaBP": 105.0,
        "BMI": 35.0,
        "heartRate": 95.0,
        "glucose": 200.0
    }

    response = client.post("/predict", json=sick_patient)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert "High Risk" in response.json()["diagnosis"]

# --- TEST 3: Verify that it validates the required data ---
def test_invalid_input():
    # Send incomplete JSON (missing many required fields)
    bad_data = {"sex": "M"} 

    response = client.post("/predict", json=bad_data)

    # We expect a 422 error (Unprocessable Entity) from FastAPI
    assert response.status_code == 422