# ==============================================================
# Crop Yield Model Training (Optimized Version)
# ==============================================================

import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# --------------------------------------------------------------
# Step 1: Resolve Dynamic Paths
# --------------------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
model_dir = os.path.join(base_dir, "models")

crop_path = os.path.join(data_dir, "crop_yield_processed.csv")
pop_path = os.path.join(data_dir, "population_processed.csv")

# --------------------------------------------------------------
# Step 2: Validate Data Files
# --------------------------------------------------------------
if not os.path.exists(crop_path):
    raise FileNotFoundError(f"❌ Missing file: {crop_path}")
if not os.path.exists(pop_path):
    raise FileNotFoundError(f"❌ Missing file: {pop_path}")

print("✅ Data files found. Loading...")

# --------------------------------------------------------------
# Step 3: Load and Clean Data
# --------------------------------------------------------------
crop_df = pd.read_csv(crop_path, encoding="latin1")
pop_df = pd.read_csv(pop_path, encoding="latin1")

# Normalize column names
crop_df.columns = crop_df.columns.str.strip().str.lower()
pop_df.columns = pop_df.columns.str.strip().str.lower()

# Prepare Population Data
year_columns = [c for c in pop_df.columns if any(y in c for y in ["1970", "1980", "1990", "2000", "2010", "2020", "2022"])]

if year_columns:
    pop_melted = pop_df.melt(
        id_vars=["country/territory"],
        value_vars=year_columns,
        var_name="year_raw",
        value_name="population"
    )
    pop_melted["year"] = pop_melted["year_raw"].str.extract(r"(\d{4})").astype(int)
    pop_melted.rename(columns={"country/territory": "area"}, inplace=True)
    pop_melted = pop_melted[["area", "year", "population"]]
else:
    pop_melted = pop_df.rename(columns={"country": "area"})

# Filter Crop Data
crop_df = crop_df[crop_df["element"].str.lower().eq("yield")]
crop_df = crop_df[["area", "year", "item", "value"]].rename(columns={"value": "yield"})

# Merge Crop and Population Data
df = pd.merge(crop_df, pop_melted, on=["area", "year"], how="inner")
print(f"✅ Merged dataset shape: {df.shape}")

# --------------------------------------------------------------
# Step 4: Add Environmental Features (Simulated)
# --------------------------------------------------------------
np.random.seed(42)
df["rainfall"] = np.random.uniform(200, 1000, len(df))       # mm
df["temperature"] = np.random.uniform(10, 35, len(df))       # °C
df["fertilizer"] = np.random.uniform(100, 250, len(df))      # kg/ha

# Adjust Yield based on environment
df["yield"] = (
    df["yield"]
    + 0.3 * (df["rainfall"] - 600)
    - 0.5 * abs(df["temperature"] - 22)
    + 0.4 * (df["fertilizer"] - 180)
)

df = df.dropna()

# --------------------------------------------------------------
# Step 5: Train-Test Split
# --------------------------------------------------------------
X = df[["population", "rainfall", "temperature", "fertilizer"]]
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------------
# Step 6: Model Training (Optimized)
# --------------------------------------------------------------
print("\n🚀 Training RandomForestRegressor model...")
start_time = time.time()

model = RandomForestRegressor(
    n_estimators=50,    # reduced number of trees
    max_depth=12,       # limit depth for speed
    n_jobs=-1,          # use all CPU cores
    random_state=42
)

model.fit(X_train, y_train)

train_time = time.time() - start_time
r2 = model.score(X_test, y_test)

print(f"✅ Model trained successfully (R² = {r2:.3f})")
print(f"⏱️ Training completed in {train_time:.2f} seconds")

# --------------------------------------------------------------
# Step 7: Save Model
# --------------------------------------------------------------
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "crop_yield_model.pkl")

joblib.dump(model, model_path)
print(f"💾 Model saved at: {model_path}")

# --------------------------------------------------------------
# Step 8: Final Summary
# --------------------------------------------------------------
print("\n🎯 Training Summary:")
print(f"📊 Samples used: {len(df):,}")
print(f"🔍 Features: {list(X.columns)}")
print(f"📈 R² Score: {r2:.3f}")
print(f"💾 Model Path: {model_path}")
