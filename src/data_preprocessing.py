import pandas as pd
import numpy as np
import os

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

FAOSTAT_PATH = os.path.join(PROCESSED_DIR, "canada_crop_yield.csv")
POP_PATH = os.path.join(RAW_DIR, "population_processed.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "canada_final_dataset.csv")


# LOAD FAOSTAT CANADA CROP DATA
def load_crop_data():
    print(" Loading FAOSTAT Canada crop dataset...")
    df = pd.read_csv(FAOSTAT_PATH)

    # Clean column names
    df.columns = df.columns.str.lower().str.strip()

    # Keep only wheat + maize
    df = df[df["crop"].isin(["wheat", "maize"])]

    print("Crops included:", df["crop"].unique())
    print("Rows after filtering:", df.shape)

    return df


# LOAD CANADA POPULATION FROM KAGGLE

def load_population():
    print(" Loading Kaggle population dataset...")

    df = pd.read_csv(POP_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Canada-only
    df = df[df["country/territory"].str.lower() == "canada"]

    # Identify year columns (e.g., '2020_population')
    pop_cols = [c for c in df.columns if c.endswith("population")]

    # Convert from wide → long format
    melt_df = df.melt(
        value_vars=pop_cols,
        var_name="year_raw",
        value_name="population"
    )

    # Extract the year number from the column name
    melt_df["year"] = melt_df["year_raw"].str.extract(r"(\d{4})").astype(int)

    melt_df = melt_df[["year", "population"]].drop_duplicates()

    print(" Loaded population rows:", melt_df.shape)

    return melt_df


# ADD REALISTIC ENVIRONMENTAL FEATURES

def generate_realistic_environment(df):
    print("🌦 Adding realistic environmental simulation...")

    np.random.seed(42)

    rainfall = []
    temperature = []
    fertilizer = []

    for crop in df["crop"]:
        if crop == "wheat":
            rainfall.append(np.random.normal(550, 120))
            temperature.append(np.random.normal(18, 4))
            fertilizer.append(np.random.normal(150, 25))

        elif crop == "maize":
            rainfall.append(np.random.normal(750, 180))
            temperature.append(np.random.normal(22, 4))
            fertilizer.append(np.random.normal(190, 30))

    # Clip to realistic physical ranges
    df["rainfall"] = np.clip(rainfall, 200, 1200)
    df["temperature"] = np.clip(temperature, 5, 35)
    df["fertilizer"] = np.clip(fertilizer, 80, 250)

    return df


# RUN FULL PREPROCESSING PIPELINE

def run_preprocessing():
    print("Running preprocessing pipeline...\n")

    crop_df = load_crop_data()
    pop_df = load_population()

    # Merge crop yield + population by year
    merged = pd.merge(crop_df, pop_df, on="year", how="left")

    # FIX MISSING POPULATION VALUES (NaN)
    merged["population"] = merged["population"].interpolate(method="linear")  # smooth
    merged["population"] = merged["population"].fillna(method="bfill")         # backfill
    merged["population"] = merged["population"].fillna(method="ffill")         # forward fill
    merged["population"] = merged["population"].fillna(38_000_000)             # fallback
    merged["population"] = merged["population"].astype(int)

    # ADD ENVIRONMENT FEATURES
    merged = generate_realistic_environment(merged)

    # SAVE FINAL DATASET
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(" Final dataset shape:", merged.shape)
    print(f" Saved final dataset to: {OUTPUT_PATH}")


# MAIN

if __name__ == "__main__":
    run_preprocessing()

