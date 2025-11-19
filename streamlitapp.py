import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
from src.model import CropYieldModel

st.set_page_config(page_title="🇨🇦 Canada Crop Yield Predictor", layout="wide")

DATA_PATH = "data/processed/canada_final_dataset.csv"
MODEL_PATH = "models/crop_yield_model.pkl"

df = pd.read_csv(DATA_PATH)
model = CropYieldModel(model_path=MODEL_PATH)

st.title("🇨🇦 Canada Crop Yield Prediction Dashboard")
st.write("Predict yield for **Wheat** and **Maize (Corn)** using climate, fertilizer type, and population inputs.")

# =================================================================
# REALISTIC TEMPERATURE PENALTY
# =================================================================
def apply_temperature_penalty(temp, crop):
    if crop == "wheat":
        ideal_min, ideal_max = 15, 25
    else:
        ideal_min, ideal_max = 18, 30

    if temp < -5 or temp > 45:
        return 0

    if ideal_min <= temp <= ideal_max:
        return 1.0

    if ideal_min - 5 <= temp < ideal_min:
        return 0.7
    if ideal_max < temp <= ideal_max + 5:
        return 0.7

    if ideal_min - 10 <= temp < ideal_min - 5:
        return 0.4
    if ideal_max + 5 < temp <= ideal_max + 10:
        return 0.4

    return 0.1


# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Input Settings")

crop_type = st.sidebar.selectbox("Crop Type", ["wheat", "maize"])
population = st.sidebar.number_input("Population", min_value=1_000_000, max_value=50_000_000, value=38_000_000)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 1500, 600)
temperature = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
fertilizer = st.sidebar.slider("Fertilizer (kg/ha)", 50, 250, 150)

# Fertilizer Type
fertilizer_type = st.sidebar.selectbox(
    "Fertilizer Type",
    ["Organic", "Chemical", "Mixed"]
)

# Fertilizer multipliers
fertilizer_factor_dict = {
    "Organic": 0.9,
    "Chemical": 1.2,
    "Mixed": 1.0
}
fert_factor = fertilizer_factor_dict[fertilizer_type]  # FIXED NAME

# Farmland area
area_ha = st.sidebar.number_input(
    "Farmland Area (ha)",
    min_value=1000,
    max_value=20_000_000,
    value=10_000_000,
    step=1000
)

predict_btn = st.sidebar.button("Predict Yield")

# ---------------------------
# Prediction
# ---------------------------
if predict_btn:
    pred = model.predict(
        crop=crop_type,
        population=population,
        rainfall=rainfall,
        temperature=temperature,
        fertilizer=fertilizer
    )

    # AGRONOMY RULES
    if rainfall < 50:
        pred = 0
    else:
        if rainfall < 200:
            pred *= 0.4

        temp_penalty = apply_temperature_penalty(temperature, crop_type)
        pred *= temp_penalty

        if fertilizer < 80:
            pred *= 0.5

    # Apply fertilizer factor
    pred *= fert_factor

    pred = max(pred, 0)

    st.subheader("📈 Predicted Yield")
    st.metric(f"{crop_type.upper()} Yield (hg/ha)", f"{pred:,.2f}")

    # -----------------------------------------------------------
    # REALISTIC IMPORT/EXPORT CALCULATION
    # -----------------------------------------------------------
    if crop_type == "wheat":
        need_per_person = 0.067  # tons per person per year
    else:
        need_per_person = 0.035

    # Convert yield to tons/ha
    yield_ton_per_ha = pred / 10_000

    # Total production
    total_production = yield_ton_per_ha * area_ha

    # National consumption
    need_tons = population * need_per_person

    difference = total_production - need_tons

    st.write("### 🧮 National Balance Summary")
    if difference >= 0:
        st.success(f"✔ Canada can export approx. {difference:,.0f} tons.")
    else:
        st.error(f"❌ Canada must import approx. {abs(difference):,.0f} tons.")

    # -----------------------------------------------------------
    # YEARLY TREND GRAPH
    # -----------------------------------------------------------
    st.subheader(f"📊 Yield Trend for {crop_type}")

    crop_df = df[df["crop"] == crop_type]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=crop_df, x="year", y="yield_hg_ha", ax=ax)
    ax.set_title(f"{crop_type.capitalize()} Yield Over Time")
    ax.set_ylabel("Yield (hg/ha)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)


# =================================================================
#  FERTILIZER RESPONSE CURVE
# =================================================================
st.subheader("🌾 Yield Response to Fertilizer (Smooth Curve)")

fert_range = np.arange(50, 251, 5)
wheat_y = []
maize_y = []

def compute_final_yield(crop, f):
    y = model.predict(
        crop=crop,
        population=population,
        rainfall=rainfall,
        temperature=temperature,
        fertilizer=f
    )

    if rainfall < 50:
        return 0
    if rainfall < 200:
        y *= 0.4

    y *= apply_temperature_penalty(temperature, crop)

    if f < 80:
        y *= 0.5

    y *= fert_factor  # FIXED fertilizer factor usage

    return max(y, 0)

for f in fert_range:
    wheat_y.append(compute_final_yield("wheat", f))
    maize_y.append(compute_final_yield("maize", f))

wheat_smooth = savgol_filter(wheat_y, window_length=11, polyorder=3)
maize_smooth = savgol_filter(maize_y, window_length=11, polyorder=3)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(fert_range, wheat_smooth, label="Wheat (Smooth)", linewidth=2)
ax2.plot(fert_range, maize_smooth, label="Maize (Smooth)", linewidth=2)

ax2.set_xlabel("Fertilizer Amount (kg/ha)")
ax2.set_ylabel("Predicted Yield (hg/ha)")
ax2.set_title("Smooth Fertilizer Response Curve (Wheat vs Maize)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

st.pyplot(fig2)
