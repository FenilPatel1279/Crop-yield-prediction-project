import pandas as pd
import numpy as np
import joblib
import os
<<<<<<< HEAD

from sklearn.model_selection import train_test_split
=======
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
>>>>>>> d26e9788998ca4b6ef6641cf450c19eac2d3947f
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

<<<<<<< HEAD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Try loading XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    print(" XGBoost not installed. Skipping XGBoost model.")
    HAS_XGB = False


DATA_PATH = "data/processed/canada_final_dataset.csv"
MODEL_DIR = "models"


def evaluate_model(name, model, X_test, y_test):
    """Print evaluation metrics for any model."""
    preds = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mse = mean_squared_error(y_test, preds)

    print(f"\n📌 RESULTS FOR {name}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MSE      : {mse:.2f}")
    print("-" * 50)

    return {
        "model": name,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mse": mse
    }


def train_all_models():
    print("📥 Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    required = ["crop", "yield_hg_ha", "population", "rainfall", "temperature", "fertilizer"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f" Missing column: {col}")

    # Features + Target
    X = df[["crop", "population", "rainfall", "temperature", "fertilizer"]]
    y = df["yield_hg_ha"]

    # OneHot encode crop
    preprocessor = ColumnTransformer(
        transformers=[("crop_enc", OneHotEncoder(handle_unknown="ignore"), ["crop"])],
        remainder="passthrough"
    )

    # Split
=======

DATA_PATH = "data/processed/canada_final_dataset.csv"
MODEL_PATH = "models/crop_yield_model.pkl"


def train_model():
    print(" Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    # Required columns 
    required = ["year", "crop", "yield_hg_ha", "population", "rainfall", "temperature", "fertilizer"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f" Missing column in dataset: {col}")

    # Features + Target 
    X = df[["crop", "population", "rainfall", "temperature", "fertilizer"]]
    y = df["yield_hg_ha"]

    # Encode crop 
    preprocessor = ColumnTransformer(
        transformers=[
            ("crop_enc", OneHotEncoder(handle_unknown="ignore"), ["crop"])
        ],
        remainder="passthrough"
    )

    # RandomForest model
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
>>>>>>> d26e9788998ca4b6ef6641cf450c19eac2d3947f
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

<<<<<<< HEAD
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------
    # 1️⃣ LINEAR REGRESSION
    # --------------------------
    lr_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    print("\n▶ Training Linear Regression...")
    lr_pipeline.fit(X_train, y_train)

    evaluate_model("Linear Regression", lr_pipeline, X_test, y_test)
    joblib.dump(lr_pipeline, f"{MODEL_DIR}/model_lr.pkl")


    # --------------------------
    # 2️⃣ RANDOM FOREST REGRESSOR
    # --------------------------
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])

    print("\n▶ Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)

    evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
    joblib.dump(rf_pipeline, f"{MODEL_DIR}/model_rf.pkl")


    # --------------------------
    # 3️⃣ XGBOOST REGRESSOR
    # --------------------------
    if HAS_XGB:
        xgb_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror"
            ))
        ])

        print("\n▶ Training XGBoost...")
        xgb_pipeline.fit(X_train, y_train)

        evaluate_model("XGBoost", xgb_pipeline, X_test, y_test)
        joblib.dump(xgb_pipeline, f"{MODEL_DIR}/model_xgb.pkl")

    else:
        print("⚠️ Skipping XGBoost — not installed.")


    print("\n🎉 Training completed for all models!")
    print(f"📁 Models saved in folder: {MODEL_DIR}")


if __name__ == "__main__":
    train_all_models()
=======
    print("Training RandomForest model...")
    pipeline.fit(X_train, y_train)

    # Model accuracy metrics
    score = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n Training complete (R² = {score:.3f})")
    print("\n Model Evaluation Metrics ")
    print(f"R² Score: {score:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print("\n")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f" Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
>>>>>>> d26e9788998ca4b6ef6641cf450c19eac2d3947f
