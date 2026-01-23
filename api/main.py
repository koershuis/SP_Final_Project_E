import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from contextlib import asynccontextmanager

from src.cleanning import clean_data


# --- PATH CONFIGURATION ---
# We add the root folder to the system to find 'src'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


# --- MODEL MANAGEMENT ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load model when turning on
    model_path = os.path.join(BASE_DIR, "models", "best_model.joblib")
    
    if os.path.exists(model_path):
        ml_models["pipeline"] = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    else:
        print(f"ERROR: Model not found in {model_path}")    
    yield
    
    # 2. clean memory when turning off
    ml_models.clear()
    print("API apagada.")

# --- INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting 10-year coronary heart disease risk based on patient data",
    version="2.0",
    lifespan=lifespan
)

# --- DATA DEFINITION (INPUT) ---
class PatientInput(BaseModel):
    sex: str              # "M" o "F"
    age: int
    education: float
    currentSmoker: str    # "Yes" o "No"
    cigsPerDay: float
    BPMeds: float
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

    # This serves to display an example in the automatic documentation.
    class Config:
        json_schema_extra = {
            "example": {
                "sex": "M",
                "education": 2.0,
                "age": 50,
                "currentSmoker": "Yes",
                "cigsPerDay": 20.0,
                "BPMeds": 0.0,
                "prevalentStroke": 0,
                "prevalentHyp": 0,
                "diabetes": 0,
                "totChol": 250.0,
                "sysBP": 130.0,
                "diaBP": 80.0,
                "BMI": 28.0,
                "heartRate": 75.0,
                "glucose": 85.0
            }
        }

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_heart_disease(patient: PatientInput):
    pipeline = ml_models.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="The model is not loaded.")

    # 1. Convert what arrives (JSON) to Pandas DataFrame
    input_data = patient.model_dump()
    df = pd.DataFrame([input_data])

    # 2. Cleaning / Preprocessing
    try:
        df_clean = clean_data(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in data cleaning: {str(e)}")

    # 3. Verification after cleaning
    if df_clean.empty:
        raise HTTPException(status_code=400, detail="The data provided was discarded due to cleaning (possible null values in BMI or HeartRate).")

    # 4. Ordering columns
    expected_cols = ['sex', 'age', 'currentSmoker', 
                     'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'cigsPerDay', 
                     'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
    df_final = df_clean[expected_cols]

    # 5. Predict
    try:
        prediction = pipeline.predict(df_final)[0]      # Returns [0] or [1]
        probability = pipeline.predict_proba(df_final)[0] # Returns probabilities [Class0, Class1]
        risk_percent = round(probability[1] * 100, 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    # 6. Return reply
    return {
        "prediction": int(prediction),
        "risk_probability": f"{risk_percent}%",
        "diagnosis": "High Risk of Coronary Heart Disease" if prediction == 1 else "Low Risk"
    }


# --- STARTUP (For Docker and testing) ---
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 allows access from outside the container
    uvicorn.run(app, host="0.0.0.0", port=8000)