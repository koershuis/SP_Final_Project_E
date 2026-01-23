from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

from src.cleanning import clean_data

# 1. Initialize application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting 10-year coronary heart disease risk based on patient data",
    version="1.0"
)

# 2. Load model and scaler
# It is necessary to load also 'scaler' because
# new data must be normalize in the same way as training.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = None
scaler = None

# We use this event to load the models when the API is turned on.
@app.on_event("startup")
def load_models():
    global model, scaler
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model correctly loaded.")
    else:
        print("⚠️ WARNING: model.pkl or scaler.pkl not found in the models/ folder.")
        print("The API will work but will fail when attempting to predict.")

# 3. Define the "order" (What information should the user send?)
# These are the exact columns (except for Target).
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
        schema_extra = {
            "example": {
                "sex": "M",
                "age": 48,
                "education": 1.0,
                "currentSmoker": "Yes",
                "cigsPerDay": 20.0,
                "BPMeds": 0.0,
                "prevalentStroke": 0,
                "prevalentHyp": 0,
                "diabetes": 0,
                "totChol": 245.0,
                "sysBP": 127.5,
                "diaBP": 80.0,
                "BMI": 25.34,
                "heartRate": 75.0,
                "glucose": 70.0
            }
        }

# 4. Test route (to see if the API is live)
@app.get("/")
def home():
    return {"status": "online", "message": "Welcome to the Heart Disease Prediction API."}

# 5. The Primary Endpoint: Predicting
@app.post("/predict")
def predict_heart_disease(patient: PatientInput):

    if not model or not scaler:
        raise HTTPException(status_code=500, detail="The model is not loaded. Run validation.py first.")

    # 1. Convert what arrives (JSON) to Pandas DataFrame
    input_data = patient.dict()
    df = pd.DataFrame([input_data])

    # 2. Cleaning / Preprocessing
    try:
        df_clean = clean_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data cleaning: {str(e)}")

    # 3. Ensure column order (Critical)
    column_order = ['sex', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
                    'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                    'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    try:
        df = df[column_order]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Error en las columnas: {e}")

    # 4. Scale the data (Put it on the same scale as the training)
    data_scaled = scaler.transform(df)

    # 5. Predict
    prediction = model.predict(data_scaled)      # Returns [0] or [1]
    probability = model.predict_proba(data_scaled) # Returns probabilities [[0.2, 0.8]]

    # 6. Return reply
    risk_prob = probability[0][1] #  Probability of class 1 (high risk)
    
    return {
        "prediction_class": int(prediction[0]),
        "risk_probability": float(risk_prob),
        "diagnosis": "High Risk of Coronary Heart Disease" if prediction[0] == 1 else "Low Risk"
    }