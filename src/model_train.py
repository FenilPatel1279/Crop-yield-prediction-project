import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Try loading XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    print("XGBoost not installed. Skipping XGBoost model.")
    HAS_XGB = False


DATA_PATH = "data/processed/canada_final_dataset.csv"
MODEL_DIR = "models"


def separator():
    print("\n" + "-" * 70 + "\n")


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mse = mean_squared_error(y_test, preds)

    print(f"{name} Results")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  MSE      : {mse:.2f}")
    separator()

    return {
        "model": name,
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mse": round(mse, 2)
    }


def train_all_models():
    print("Loading processed dataset...\n")

    df = pd.read_csv(DATA_PATH)

    required = ["crop", "yield_hg_ha", "population", "rainfall", "temperature", "fertilizer"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    X = df[["crop", "population", "rainfall", "temperature", "fertilizer"]]
    y = df["yield_hg_ha"]

    preprocessor = ColumnTransformer(
        transformers=[("crop_enc", OneHotEncoder(handle_unknown="ignore"), ["crop"])],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    results = []

    # Linear Regression
    print("Training Linear Regression...")
    lr_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    results.append(evaluate_model("Linear Regression", lr_pipeline, X_test, y_test))
    joblib.dump(lr_pipeline, f"{MODEL_DIR}/model_lr.pkl")

    # Random Forest
    print("Training Random Forest...")
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf_pipeline, X_test, y_test))
    joblib.dump(rf_pipeline, f"{MODEL_DIR}/model_rf.pkl")

    # XGBoost
    if HAS_XGB:
        print("Training XGBoost...")
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
        xgb_pipeline.fit(X_train, y_train)
        results.append(evaluate_model("XGBoost", xgb_pipeline, X_test, y_test))
        joblib.dump(xgb_pipeline, f"{MODEL_DIR}/model_xgb.pkl")
    else:
        print("Skipping XGBoost (not installed).\n")

    # Summary Table
    print("Training Summary")
    print("-" * 70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    print("\nModels saved in folder:", MODEL_DIR)
    print("Training Completed.\n")


if __name__ == "__main__":
    train_all_models()
