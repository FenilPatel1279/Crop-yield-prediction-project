import joblib
import pandas as pd
from preprocess import merge_features  # or add_user_features

def load_model(model_path: str = '../models/crop_yield_model.pkl'):
    return joblib.load(model_path)

def predict_yield(input_df: pd.DataFrame):
    model = load_model()
    X = input_df[['Population','Rainfall','Temperature','Fertilizer']]
    pred = model.predict(X)[0]
    return pred

def evaluate_surplus(predicted_yield: float, population: int, demand_per_person: float = 1.0):
    """
    Assess whether predicted yield meets demand.
    Returns dictionary with status, amount, export/import advice.
    """
    total_demand = population * demand_per_person
    if predicted_yield > total_demand:
        surplus = predicted_yield - total_demand
        return {'status': 'surplus', 'amount': surplus, 'advice': 'Can export'}
    else:
        deficit = total_demand - predicted_yield
        return {'status': 'deficit', 'amount': deficit, 'advice': 'Need to import'}

def build_input_df(area: str, year: int, pop_df: pd.DataFrame, rainfall: float, temperature: float, fertilizer: float):
    """
    Build a DataFrame with one row, merging population for the area/year with user features.
    """
    pop_val = pop_df[(pop_df['Area']==area) & (pop_df['Year']==year)]['Population'].values
    if len(pop_val)==0:
        raise ValueError("Population data not found for area/year")
    population = pop_val[0]
    input_df = pd.DataFrame({
        'Area': [area],
        'Year': [year],
        'Population': [population],
        'Rainfall': [rainfall],
        'Temperature': [temperature],
        'Fertilizer': [fertilizer]
    })
    return input_df
