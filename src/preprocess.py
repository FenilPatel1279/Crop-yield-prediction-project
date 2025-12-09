import pandas as pd
import numpy as np

def clean_crop_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter FAOSTAT crop yield data.
    Example: keep element = 'Yield', pivot years, select relevant columns.
    """
    # Example column names may vary; adjust accordingly
    df2 = df[df['Element'] == 'Yield'].copy()
    df2 = df2[['Area', 'Year', 'Value']].rename(columns={'Value': 'Yield'})
    df2['Year'] = df2['Year'].astype(int)
    # Could pivot if multiple years or items
    return df2

def clean_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean population dataset.
    """
    # Example: assuming columns like 'Country', 'Year', 'Population'
    df2 = df.rename(columns={'Country': 'Area'})
    # Keep relevant year(s)
    return df2

def merge_features(crop_df: pd.DataFrame, pop_df: pd.DataFrame,
                   rainfall: float, temperature: float, fertilizer: float, year: int, area: str) -> pd.DataFrame:
    """
    Merge the crop_df and pop_df for a given area+year, then add user-features rainfall, temperature, fertilizer.
    """
    df_c = crop_df[(crop_df['Area'] == area) & (crop_df['Year'] == year)].copy()
    df_p = pop_df[(pop_df['Area'] == area) & (pop_df['Year'] == year)].copy()
    if df_c.empty or df_p.empty:
        raise ValueError("No data for area/year combination")
    # take first row
    yield_val = df_c['Yield'].values[0]
    pop_val = df_p['Population'].values[0]
    df_feat = pd.DataFrame({
        'Area': [area],
        'Year': [year],
        'Yield': [yield_val],
        'Population': [pop_val],
        'Rainfall': [rainfall],
        'Temperature': [temperature],
        'Fertilizer': [fertilizer]
    })
    return df_feat

def prepare_training_data(crop_df: pd.DataFrame, pop_df: pd.DataFrame,
                          rainfall_data: pd.DataFrame, temperature_data: pd.DataFrame,
                          fertilizer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create full training dataset by merging crop yield, population, rainfall, temp, fertilizer features.
    For simplicity assume rainfall_data, temperature_data, fertilizer_data have same indexing by area/year.
    """
    # This is a placeholder for your actual feature engineering
    df = crop_df.merge(pop_df, on=['Area','Year'], how='left')
    df = df.merge(rainfall_data, on=['Area','Year'], how='left')
    df = df.merge(temperature_data, on=['Area','Year'], how='left')
    df = df.merge(fertilizer_data, on=['Area','Year'], how='left')
    # Rename columns
    df = df.rename(columns={
        'Value_x': 'Yield',
        'Population': 'Population',
        'Rainfall': 'Rainfall',
        'Temperature': 'Temperature',
        'Fertilizer': 'Fertilizer'
    })
    # Drop NA
    df = df.dropna()
    return df
