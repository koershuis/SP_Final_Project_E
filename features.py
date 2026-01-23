import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def identify_continuous_variables(df):
    """Identify continuous numeric variables (exclude binary 0/1 variables)."""
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    continuous_cols = []
    for col in all_numeric_cols:
        unique_vals = df[col].dropna().unique()
        # Only continuous if not just 0 and 1
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            continuous_cols.append(col)
    
    return continuous_cols


def normalize_data(df, continuous_cols):
    """Normalize continuous variables using MinMaxScaler."""
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    return df_normalized


def remove_correlated_variables(df, variable_groups, threshold=0.7):
    """
    Remove highly correlated variables within predefined groups.
    Keep the variable with lower average correlation to others.
    """
    all_vars_to_remove = set()
    
    for group_name, variables in variable_groups.items():
        # Only analyze variables that exist in dataframe
        available_vars = [v for v in variables if v in df.columns]
        if len(available_vars) < 2:
            continue
        
        # Calculate correlation matrix
        corr_matrix = df[available_vars].corr()
        
        # Calculate average correlation for each variable
        avg_corr = corr_matrix.abs().mean()
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if upper_triangle[i, j] and abs(corr_matrix.loc[col1, col2]) > threshold:
                    # Remove variable with higher average correlation
                    if avg_corr[col1] > avg_corr[col2]:
                        all_vars_to_remove.add(col1)
                    else:
                        all_vars_to_remove.add(col2)
    
    return sorted(list(all_vars_to_remove))


def create_cleaned_dataframe(df_normalized, continuous_cols, vars_to_remove):
    """Create cleaned dataframe by removing redundant variables."""
    # Get columns to keep
    continuous_to_keep = [col for col in continuous_cols if col not in vars_to_remove]
    non_continuous_cols = [col for col in df_normalized.columns if col not in continuous_cols]
    all_cols_to_keep = non_continuous_cols + continuous_to_keep
    
    cleaned_df = df_normalized[all_cols_to_keep].copy()
    return cleaned_df


def main(input_file='coronary_disease_clean.csv', 
         output_normalized='normalized_dataframe.csv',
         output_cleaned='cleaned_dataframe.csv'):
    """Execute the complete pipeline."""
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Dataset shape: {df.shape}\n")
    
    # 1. Identify continuous variables
    continuous_cols = identify_continuous_variables(df)
    print(f"1. Identified {len(continuous_cols)} continuous variables")
    
    # 2. Normalize continuous variables
    df_normalized = normalize_data(df, continuous_cols)
    print(f"2. Normalized continuous variables using MinMaxScaler")
    
    # 3. Define variable groups for correlation analysis
    variable_groups = {
        'Blood Pressure': ['sysBP', 'diaBP'],
        'Metabolic': ['totChol', 'glucose', 'BMI'],
        'Lifestyle': ['cigsPerDay', 'BMI', 'age']
    }
    
    # 4. Remove correlated variables
    vars_to_remove = remove_correlated_variables(df_normalized, variable_groups, threshold=0.7)
    print(f"3. Feature selection - removed {len(vars_to_remove)} redundant variable(s): {vars_to_remove if vars_to_remove else 'None'}")
    
    # 5. Create cleaned dataframe
    cleaned_df = create_cleaned_dataframe(df_normalized, continuous_cols, vars_to_remove)
    print(f"4. Created cleaned dataframe: {df.shape} → {cleaned_df.shape}")
    
    # 6. Save results
    df_normalized.to_csv(output_normalized, index=False)
    cleaned_df.to_csv(output_cleaned, index=False)
    print(f"\n✓ Saved: {output_normalized}")
    print(f"✓ Saved: {output_cleaned}")
    
    return df_normalized, cleaned_df


if __name__ == "__main__":
    df_normalized, cleaned_df = main()