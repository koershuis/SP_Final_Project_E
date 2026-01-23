import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw csv.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced cleaning: 
    --> mapping to 1/0 the binary variable 'currentSmoker'
    --> mapping to 1/0 the binary variable 'sex'
    --> GLUCOSE: Missing glucose values were imputed using group-specific medians based on the patient's diabetes status (diabetic vs. non-diabetic) to ensure clinical consistency
    --> cigsPerDay: Missing data for daily cigarette consumption was imputed by assigning a value of zero to non-smokers and applying the group-specific median to current smokers.
    -->EDUCATION: Considering the significant number of missing values and the limited medical relevance this information provides to the model, the decision was made to drop this column entirely.
    --> BMI and Heart Rate: Records with missing values were removed due to their low frequency relative to the 4,000+ total records.
    --> BPMed: Missing BPMeds entries were imputed using the mode of the patient's hypertension group (prevalentHyp)
    --> totChol: was filled with the overall median
    """
    df_clean = df.copy()

    # 1. Remove unnecessary columns (as per your notebook)
    if 'education' in df_clean.columns:
        df_clean.drop(columns=['education'], inplace=True)

    # 2. Advanced Glucose imputation (based on diabetes status)
    if 'glucose' in df_clean.columns and 'diabetes' in df_clean.columns:
        df_clean['glucose'] = df_clean['glucose'].fillna(
            df_clean.groupby('diabetes')['glucose'].transform('median')
        )

    # 3. cigsPerDay imputation (based on current smoking status)
    if 'cigsPerDay' in df_clean.columns and 'currentSmoker' in df_clean.columns:
        # Ensure currentSmoker is consistent before grouping if necessary
        df_clean['cigsPerDay'] = df_clean['cigsPerDay'].fillna(
            df_clean.groupby('currentSmoker')['cigsPerDay'].transform('median')
        )

    # 4. BPMeds imputation (using mode based on prevalent hypertension)
    if 'BPMeds' in df_clean.columns and 'prevalentHyp' in df_clean.columns:
        df_clean['BPMeds'] = df_clean['BPMeds'].fillna(
            df_clean.groupby('prevalentHyp')['BPMeds'].transform(lambda x: x.mode()[0])
        )

    # 5. Simple totChol imputation (global median)
    if 'totChol' in df_clean.columns:
        df_clean['totChol'] = df_clean['totChol'].fillna(df_clean['totChol'].median())

    # 6. Removal of rows with critical nulls (BMI and heartRate)
    df_clean.dropna(subset=['BMI', 'heartRate'], inplace=True)

    # 7. Encoding: Categorical Variables (Text -> Integers)
    if 'sex' in df_clean.columns:
        df_clean['sex'] = df_clean['sex'].map({'M': 1, 'F': 0})

    if 'currentSmoker' in df_clean.columns:
        # If values are Yes/No, map them. If they are already 1/0, do nothing.
        if df_clean['currentSmoker'].dtype == object:
            df_clean['currentSmoker'] = df_clean['currentSmoker'].map({'Yes': 1, 'No': 0})

    # 8. Final cleaning and type conversion
    df_clean.dropna(inplace=True)
    
    # Convert columns that should be binary to integers
    for col in ['sex', 'currentSmoker', 'BPMeds', 'diabetes', 'prevalentHyp']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)

    return df_clean