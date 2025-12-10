.
**
🌾 Crop Yield Prediction
📌 Overview**

This project predicts crop yield (hg/ha) for Canada using machine learning.
The prediction is based on:

Crop type

Population

Rainfall

Temperature

Fertilizer usage

The project includes data preprocessing, model training, saving trained models, and a dashboard for user prediction.

**📁 Project Structure
**
Your project folder looks like this:

CROP_YIELD_PREDICTION/
│
├── data/
│   ├── processed/
│   │   ├── canada_crop_yield.csv
│   │   ├── canada_final_dataset.csv
│   │   └── canada_population_yearly.csv
│   └── raw/
│       └── population_processed.csv
│
├── models/
│   ├── crop_yield_model.pkl
│   ├── model_lr.pkl
│   ├── model_rf.pkl
│   └── model_xgb.pkl
│
├── notebooks/
│   ├── cropyeld.ipynb
│   └── population.ipynb
│
├── src/
│   ├── dashboard_app.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_train.py
│   ├── model.py
│   ├── predictor.py
│   └── preprocess.py
│
├── templates/
│   └── (HTML files for Flask app)
│
├── streamlitapp.py
├── requirements.txt
└── README.md

**🔧 What Each File Does**
data/

Contains raw and cleaned datasets used for training.

models/

Stores saved ML model files (.pkl) after training.

notebooks/

Jupyter notebooks for EDA and preprocessing.

src/

Main source code files:

model_train.py → trains all ML models

model.py → loads a saved model and makes predictions

dashboard_app.py → Flask dashboard

data_loader.py → loads dataset

data_preprocessing.py → cleans and prepares data

predictor.py → additional prediction utilities

templates/

HTML templates for Flask UI.

**▶️ How to Train Models**

Run this command:

python src/model_train.py


This will:

Train Linear Regression

Train Random Forest

Train XGBoost 

Print model performance

Save all models inside the models/ folder

**📊 Model Metrics**

Each model prints:

R² Score

MAE

RMSE

MSE

This helps compare performance.

**🤖 How to Make a Prediction**

Example using model.py:

from model import CropYieldModel

model = CropYieldModel("models/model_rf.pkl")

prediction = model.predict(
    crop="maize",
    population=30000000,
    rainfall=800,
    temperature=18,
    fertilizer=150
)

print(prediction)

🖥 Running the Dashboard
**python src/dashboard_app.py
**
Streamlit App
**streamlit run streamlitapp.py**

📦 Install Requirements

Install dependencies:

pip install -r requirements.txt

**To see the Dashboard**
streamlit run streamitapp.py 

If you want XGBoost:

pip install xgboost

**⭐ Summary
**
This project predicts crop yield using ML models.

Includes full data preprocessing, training, and evaluation.

Models are stored in the models/ directory.

Dashboard allows user-friendly prediction.
