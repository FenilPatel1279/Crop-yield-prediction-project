import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib, os, re

# --- Load datasets ---
crop = pd.read_csv("data/processed/crop_yield_processed.csv", encoding="latin1")
pop = pd.read_csv("data/processed/population_processed.csv", encoding="latin1")

# --- Standardize column names ---
crop.columns = crop.columns.str.strip().str.lower()
pop.columns = pop.columns.str.strip().str.lower()

# --- Fix population dataset (melt by year columns) ---
year_columns = [c for c in pop.columns if re.search(r"\d{4}", c)]  # pick columns like "1970 population"
pop_melted = pop.melt(
    id_vars=["country/territory"],
    value_vars=year_columns,
    var_name="year_raw",
    value_name="population"
)
# Extract numeric year
pop_melted["year"] = pop_melted["year_raw"].str.extract(r"(\d{4})").astype(int)
pop_melted.rename(columns={"country/territory": "area"}, inplace=True)
pop_melted = pop_melted[["area", "year", "population"]]
pop_melted["population"] = pd.to_numeric(pop_melted["population"], errors="coerce")
print(f"✅ Melted population dataset shape: {pop_melted.shape}")

# --- Filter crop data (Yield only) ---
crop = crop[crop["element"].str.lower().eq("yield")]
crop = crop[["area", "year", "item", "value"]].rename(columns={"value": "yield"})

# --- Merge datasets ---
df = pd.merge(crop, pop_melted, on=["area", "year"], how="inner")
print(f"✅ Merged dataset shape: {df.shape}")

# --- Add placeholder features ---
for col in ["rainfall", "temperature", "fertilizer"]:
    df[col] = 100  # dummy constant

# --- Prepare features and target ---
X = df[["population", "rainfall", "temperature", "fertilizer"]]
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Model trained. R²: {model.score(X_test, y_test):.3f}")

# --- Save model ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crop_yield_model.pkl")
print("💾 Model saved → models/crop_yield_model.pkl")
