import pandas as pd
import os

def load_faostat_crop_data(url, save_path='data/raw/faostat_crop_yield.csv'):
    """Download and load FAOSTAT crop yield data directly from URL."""
    df = pd.read_csv(url, encoding='latin1')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print("✅ FAOSTAT crop yield data saved at:", save_path)
    return df

def load_kaggle_population_data(url, save_path='data/raw/world_population.csv'):
    """Load Kaggle population dataset."""
    df = pd.read_csv(url, encoding='latin1')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print("✅ Population data saved at:", save_path)
    return df

def clean_crop_data(df):
    """Filter and rename key columns."""
    df = df[df['Element'] == 'Yield']
    df = df.rename(columns={'Area': 'Country', 'Value': 'Yield (hg/ha)'})
    df = df[['Country', 'Year', 'Item', 'Yield (hg/ha)']]
    df.to_csv('data/processed/crop_yield_processed.csv', index=False)
    print("✅ Cleaned crop data saved.")
    return df

def clean_population_data(df):
    """Simplify population dataset."""
    df = df.rename(columns={'Country': 'Country', 'Year': 'Year', 'Population': 'Population'})
    df = df[['Country', 'Year', 'Population']]
    df.to_csv('data/processed/population_processed.csv', index=False)
    print("✅ Cleaned population data saved.")
    return df
