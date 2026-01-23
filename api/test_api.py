import sys
import os
import pytest
from fastapi.testclient import TestClient

# --- CONFIGURACIÓN DE RUTAS ---
# Subimos un nivel para encontrar la raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from api.main import app

# --- FIXTURE MÁGICA (La Solución al 503) ---
# Esta función actúa como "llave de contacto"
@pytest.fixture(scope="module")
def client():
    # El bloque 'with' fuerza a que se ejecute el lifespan (carga el modelo)
    with TestClient(app) as c:
        yield c
    # Al salir del bloque, se apaga correctamente

# --- TEST 1: Caso Sano ---
def test_predict_low_risk(client): # <--- Pasamos 'client' como argumento
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
    
    response = client.post("/predict", json=healthy_patient)
    assert response.status_code == 200
    assert response.json()["prediction"] == 0

# --- TEST 2: Caso Riesgo Alto ---
def test_predict_high_risk(client):
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

# --- TEST 3: Error 422 ---
def test_invalid_input(client):
    bad_data = {"sex": "Robot"} 
    response = client.post("/predict", json=bad_data)
    assert response.status_code == 422