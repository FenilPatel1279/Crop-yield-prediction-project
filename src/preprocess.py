import pandas as pd
import numpy as np

def clean_yield_data(df: pd.DataFrame):
    """
    Clean raw FAOSTAT data, pivot, filter, and compute yield per area if necessary.
    """
    # Example: filter element = 'Yield'
    df_yield = df[df['Element'] == 'Yield']
    df_yield = df_yield[['Area', 'Year', 'Value']].rename(columns={'Value':'yield'})
    df_yield['Year'] = df_yield['Year'].astype(int)
    return df_yield

def add_user_features(df: pd.DataFrame, rainfall: float, temperature: float, fertilizer: float, population: int):
    """
    Given user input features, append them (or merge) into the dataframe for prediction.
    """
    # one‐row input
    user_df = pd.DataFrame({
        'rainfall': [rainfall],
        'temperature': [temperature],
        'fertilizer': [fertilizer],
        'population': [population]
    })
    return user_df
