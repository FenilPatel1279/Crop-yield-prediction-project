import pandas as pd
import numpy as np
import os

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

FAOSTAT_PATH = os.path.join(PROCESSED_DIR, "canada_crop_yield.csv")

# NEW population file
POP_PATH = os.path.join(PROCESSED_DIR, "canada_population_yearly.csv")

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "canada_final_dataset.csv")


# ---------------------------------------------------
# LOAD CANADA CROP DATA
# ---------------------------------------------------
def load_crop_data():
    print(" Loading FAOSTAT Canada crop dataset...")
    df = pd.read_csv(FAOSTAT_PATH)

    df.columns = df.columns.str.lower().str.strip()

    # Keep only wheat + maize
    df = df[df["crop"].isin(["wheat", "maize"])]

    print("Crops included:", df["crop"].unique())
    print("Rows after filtering:", df.shape)

    return df


# ---------------------------------------------------
# LOAD THE CLEANED YEARLY CANADA POPULATION DATA
# ---------------------------------------------------
def load_population():
    print(" Loading cleaned yearly Canada population dataset...")

    df = pd.read_csv(POP_PATH)
    df.columns = df.columns.str.lower().str.strip()

    # Ensure correct format → year & population
    if not {"year", "population"}.issubset(df.columns):
        raise KeyError("❌ canada_population_yearly.csv must have columns: year, population")

    print(" Loaded population rows:", df.shape)

    return df


# ---------------------------------------------------
# ADD REALISTIC ENVIRONMENTAL FEATURES
# ---------------------------------------------------
def generate_realistic_environment(df):
    print(" Adding realistic environmental simulation...")

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

    df["rainfall"] = np.clip(rainfall, 200, 1200)
    df["temperature"] = np.clip(temperature, 5, 35)
    df["fertilizer"] = np.clip(fertilizer, 80, 250)

    return df


# ---------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# ---------------------------------------------------
def run_preprocessing():
    print("Running preprocessing pipeline...\n")

    crop_df = load_crop_data()
    pop_df = load_population()

    # Merge crop yield + population by year
    merged = pd.merge(crop_df, pop_df, on="year", how="left")

    # Fix missing population values
    merged["population"] = merged["population"].interpolate()
    merged["population"] = merged["population"].bfill().ffill()
    merged["population"] = merged["population"].astype(int)

    # Add simulated environment values
    merged = generate_realistic_environment(merged)

    # Save output
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(" Final dataset shape:", merged.shape)
    print(f" Saved final dataset to: {OUTPUT_PATH}")


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
