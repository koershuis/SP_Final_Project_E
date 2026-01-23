import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### VARIABLES BINARIAS: ###

# Función que agrupa el DataFrame en grupo_cols y calcula la media de las variables numéricas
def group_and_average(df, group_cols):
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = numeric_cols.difference(group_cols)
    
    return (
        df
        .groupby(group_cols)[numeric_cols]
        .mean()
        .reset_index() 
    )

### VARIABLES CONTÍNUAS: ###

def aggregate_continuous_by_bins(df, group_col, continuous_col, bins=None, labels=None):
    """
    Crea grupos (bins) de una variable continua y calcula promedio de otra variable.
    """
    if bins is None:
        bins = [0, 29, 39, 49, 59, 69, 79, 120]
    if labels is None:
        labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    df[f'{continuous_col}_group'] = pd.cut(df[continuous_col], bins=bins, labels=labels)
    
    grouped = df.groupby(f'{continuous_col}_group')[group_col].mean().reset_index()
    return grouped
  
### SCATTERPLOT POR CATEGORÍA ###
  
def scatter_by_category(df, x, y, category):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=category,
        alpha=0.7
    )
    plt.title(f'{y} vs {x} agrupado por {category}')
    plt.tight_layout()
    plt.show()

