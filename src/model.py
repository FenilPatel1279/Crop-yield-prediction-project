import pandas as pd
import joblib


class CropYieldModel:
    def __init__(self, model_path="models/crop_yield_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, crop, population, rainfall, temperature, fertilizer):
        df = pd.DataFrame({
            "crop": [crop],
            "population": [population],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "fertilizer": [fertilizer]
        })

        pred = self.model.predict(df)[0]
        return pred
