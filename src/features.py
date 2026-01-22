'''
Data Normalization and Correlation Analysis
Group Project - Step 3

Objectives:

    Normalize all variables using z-score or min-max scaling (with justification)
    Analyze correlations among fundamental frequency, Jitter, and Shimmer variables
    Keep representative variables and remove redundant ones

Inputs: Cleaned DataFrame
Outputs: Normalized DataFrame, cleaned_df
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('coronary_disease_clean.csv')

# Display basic information
print("Dataset Shape:", df.shape)

display(df.head())

## 2. Data Preparation
# Get all column names
all_columns = df.columns.tolist()

# Identify numeric columns##
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Identify non-numeric columns (categorical like 'sex', 'education' if string)
non_numeric_categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Separate numeric columns into binary categorical vs continuous
binary_categorical_cols = []
continuous_numeric_cols = []

for col in all_numeric_cols:
    unique_vals = df[col].dropna().unique()
    # Check if variable only contains 0 and 1 (binary categorical)
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        binary_categorical_cols.append(col)
    else:
        continuous_numeric_cols.append(col)

# Store continuous numeric for normalization
numeric_cols = continuous_numeric_cols

# Store ALL categorical variables (both non-numeric and binary numeric)
all_categorical_cols = non_numeric_categorical_cols + binary_categorical_cols

### 3. Check data distribution

print("Statistical Summary:")
display(df[numeric_cols].describe())

# Check for outliers using IQR method
print("\nOutlier Detection (IQR Method):")
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_summary[col] = len(outliers)

outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier Count'])
outlier_df['Outlier %'] = (outlier_df['Outlier Count'] / len(df) * 100).round(2)
display(outlier_df.sort_values('Outlier Count', ascending=False).head(10))


### 4. Normalization
# Create a copy of the dataframe to preserve original data
df_normalized = df.copy()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization ONLY to continuous numeric columns
# All categorical columns (both binary and non-numeric) remain unchanged
df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])


### 5. Correlation Analysis for all groups

# We define these variable groups based on physiological relationships
variable_groups = {
    'Blood Pressure': {
        'variables': ['sysBP', 'diaBP']
    },
    'Metabolic': {
        'variables': ['totChol', 'glucose', 'BMI']
    },
    'Lifestyle': {
        'variables': ['cigsPerDay', 'BMI', 'age']
    }
}


# Display and verify groups
for group_name, group_info in variable_groups.items():
    print(f"\n{group_name}:")
    if group_name == 'All Numeric':
        print(f"  Variables: [All {len(group_info['variables'])} numeric variables]")
    else:
        print(f"  Variables: {group_info['variables']}")
        
available_groups = {}
for group_name, group_info in variable_groups.items():
    if group_name == 'All Numeric':
        available_vars = all_numeric_for_correlation
    else:
        available_vars = [v for v in group_info['variables'] if v in df_normalized.columns]
    
    if len(available_vars) >= 2:
        available_groups[group_name] = {
            'variables': available_vars,
        }
        print(f"\n {group_name}: {len(available_vars)} variables found")
        if group_name != 'All Numeric':
            print(f"  {available_vars}")
    else:
        print(f"\n {group_name}: Insufficient variables (found {len(available_vars)})")
              
# Store correlation matrices for each group
correlation_results = {}

for group_name, group_info in available_groups.items():
    print(f"CORRELATION ANALYSIS: {group_name.upper()}")
    print(f"Variables analyzed: {len(group_info['variables'])}")
    
    # Calculate correlation matrix
    corr_matrix = df_normalized[group_info['variables']].corr()
    correlation_results[group_name] = corr_matrix
    
    # Display correlation matrix
    print(f"\nCorrelation Matrix:")
    display(corr_matrix.round(3))
    
    # Visualize correlation matrix (only for manageable sizes)
    if len(group_info['variables']) <= 15:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1) 
        plt.tight_layout()
        plt.show()
    else:
        print(f"  (Skipping visualization - too many variables: {len(group_info['variables'])})")
        
        
### 6. Identify High Correlations

# Function to find highly correlated variable pairs
def find_high_correlations(corr_matrix, threshold=0.7):
    # Get upper triangle of correlation matrix to avoid duplicates
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find pairs with correlation > threshold
    high_corr_pairs = []
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            corr_value = upper_triangle.loc[index, column]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'Variable 1': index,
                    'Variable 2': column,
                    'Correlation': corr_value
                })
    
    # Return empty DataFrame with correct columns if no pairs found
    if len(high_corr_pairs) == 0:
        return pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Correlation'])
    
    return pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                      key=abs, 
                                                      ascending=False)

# Find high correlations in each group
high_corr_summary = {}

for group_name, corr_matrix in correlation_results.items():
    print(f"HIGH CORRELATIONS: {group_name.upper()}")
    
    # Try threshold 0.6 first
    high_corr_df = find_high_correlations(corr_matrix, threshold=0.7)
    
    if len(high_corr_df) > 0:
        print(f"\nHighly Correlated Pairs (|r| > 0.7): {len(high_corr_df)}")
        display(high_corr_df)
        high_corr_summary[group_name] = {'threshold': 0.7, 'pairs': len(high_corr_df)}
    else:
        print("\nNo pairs found with |r| > 0.7")
        print("Checking with threshold |r| > 0.6...")
        high_corr_df = find_high_correlations(corr_matrix, threshold=0.6)
        if len(high_corr_df) > 0:
            display(high_corr_df)
            high_corr_summary[group_name] = {'threshold': 0.6, 'pairs': len(high_corr_df)}
        else:
            print("No pairs found with |r| > 0.6")
            print("✓ All variables in this group are sufficiently independent")
            high_corr_summary[group_name] = {'threshold': 0.6, 'pairs': 0}
            
            
### 7. Variable selection 

def select_representative_variables(corr_matrix, threshold=0.6):

    avg_corr = corr_matrix.abs().mean().sort_values(ascending=True)
    variables_to_remove = set()
    
    # Get upper triangle to avoid duplicates
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # For each highly correlated pair, remove the one with higher avg correlation
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            corr_value = upper_triangle.loc[index, column]
            if abs(corr_value) > threshold:
                if avg_corr[index] > avg_corr[column]:
                    variables_to_remove.add(index)
                else:
                    variables_to_remove.add(column)
    
    variables_to_keep = set(corr_matrix.columns) - variables_to_remove
    return sorted(list(variables_to_keep)), sorted(list(variables_to_remove))

print("VARIABLE SELECTION")


# Track variables to remove across all groups
all_vars_to_remove = set()

# Apply selection to each group
for group_name, corr_matrix in correlation_results.items():
    threshold = high_corr_summary[group_name]['threshold']
    
    if high_corr_summary[group_name]['pairs'] > 0:
        vars_to_keep, vars_to_remove = select_representative_variables(corr_matrix, threshold)
        all_vars_to_remove.update(vars_to_remove)
        
        print(f"\n{group_name}:")
        print(f"  ✗ Remove: {vars_to_remove}")
        print(f"  ✓ Keep: {vars_to_keep}")

# Get all continuous variables analyzed
all_analyzed_vars = set()
for group_info in available_groups.values():
    all_analyzed_vars.update(group_info['variables'])

# Final lists
vars_to_remove = sorted(list(all_vars_to_remove))
vars_to_keep = sorted(list(all_analyzed_vars - all_vars_to_remove))

# Add variables not in any group
vars_not_in_groups = set(numeric_cols) - all_analyzed_vars
if vars_not_in_groups:
    vars_to_keep.extend(sorted(list(vars_not_in_groups)))

print("FINAL RESULTS")
print(f"Total continuous variables: {len(numeric_cols)}")
print(f"Removed: {len(vars_to_remove)} - {vars_to_remove if vars_to_remove else 'None'}")
print(f"Kept: {len(vars_to_keep)}")
print(f"{'='*70}")



### 8. Create Cleaned Dataframe and Export files

# Get all columns to keep (non-numeric + selected numeric variables)
non_numeric_cols = [col for col in df_normalized.columns if col not in numeric_cols]
all_cols_to_keep = non_numeric_cols + vars_to_keep

# Create cleaned dataframe
cleaned_df = df_normalized[all_cols_to_keep].copy()

print(f"Original shape: {df_normalized.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")
print(f"Removed {len(vars_to_remove)} redundant variable(s)")

# Display first few rows
display(cleaned_df.head())


# Save normalized dataframe
df_normalized.to_csv('normalized_dataframe.csv', index=False)

# Save cleaned dataframe
cleaned_df.to_csv('cleaned_dataframe.csv', index=False)

