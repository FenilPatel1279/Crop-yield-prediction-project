import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_PATH = "data/processed/canada_final_dataset.csv"
MODEL_PATH = "models/crop_yield_model.pkl"


def train_model():
    print("📥 Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    # ---- Required columns ----
    required = ["year", "crop", "yield_hg_ha", "population", "rainfall", "temperature", "fertilizer"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"❌ Missing column in dataset: {col}")

    # ---- Features + Target ----
    X = df[["crop", "population", "rainfall", "temperature", "fertilizer"]]
    y = df["yield_hg_ha"]

    # ---- Encode crop ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("crop_enc", OneHotEncoder(handle_unknown="ignore"), ["crop"])
        ],
        remainder="passthrough"
    )

    # ---- RandomForest model ----
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # ---- Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training RandomForest model...")
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"✔ Training complete (R² = {score:.3f})")

    # ---- Save model ----
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"💾 Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
