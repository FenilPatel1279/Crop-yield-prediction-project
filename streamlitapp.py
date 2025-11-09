import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page setup ---
st.set_page_config(page_title="🌾 Crop Yield Prediction Dashboard", layout="wide")

# --- Load model ---
if not os.path.exists("models/crop_yield_model.pkl"):
    st.error("❌ Model file not found. Please run src/model_train.py first.")
    st.stop()

model = joblib.load("models/crop_yield_model.pkl")

# --- Load population data ---
pop_df = pd.read_csv("data/processed/population_processed.csv", encoding="latin1")
pop_df.columns = pop_df.columns.str.strip().str.lower()

# --- Title and description ---
st.title("🌾 Crop Yield Prediction Dashboard — Canada")
st.markdown("""
Use this dashboard to simulate how **rainfall**, **temperature**, **fertilizer type**, and **population size**  
affect Canada's crop yield and whether the production can meet national food demand.
""")

# --- Sidebar Controls ---
st.sidebar.header("Input Controls")

rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1000.0, 500.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 45.0, 20.0)

# Fertilizer configuration
st.sidebar.subheader("💧 Fertilizer Configuration")

fertilizer_type = st.sidebar.selectbox(
    "Select Fertilizer Type",
    ("Organic (Expensive, High Quality - Lower Yield)", 
     "Chemical (Cheaper, High Yield - Lower Quality)", 
     "Mixed (Balanced - Moderate Yield & Quality)")
)

if "Organic" in fertilizer_type:
    default_fert = 150.0
    quality_factor = 0.9
elif "Chemical" in fertilizer_type:
    default_fert = 200.0
    quality_factor = 1.2
else:
    default_fert = 175.0
    quality_factor = 1.0

fertilizer = st.sidebar.number_input(
    "Fertilizer Amount (kg/ha)", 
    min_value=100.0, 
    max_value=250.0, 
    value=default_fert, 
    step=5.0,
    help="Adjust fertilizer quantity to observe yield impact."
)

st.sidebar.caption("💡 Organic = costly but eco-friendly | Chemical = cheap but high yield | Mixed = balanced approach")

# --- Population Input ---
st.sidebar.subheader("👥 Population Settings")
population = st.sidebar.number_input(
    "Enter Population",
    min_value=1_000_000,
    max_value=100_000_000,
    value=38_000_000,
    step=1_000_000,
    help="Adjust the population to simulate total demand changes."
)

# --- Predict Button ---
if st.sidebar.button("🔮 Predict Crop Yield"):

    # --- Constants ---
    country = "Canada"
    avg_demand_per_person = 500.0  # hg/year per person
    farmland_area = 40_000_000  # hectares (approximation)

    # --- Prepare model input ---
    X_input = pd.DataFrame({
        "population": [population],
        "rainfall": [rainfall],
        "temperature": [temperature],
        "fertilizer": [fertilizer]
    })

    # --- Model Prediction ---
    pred_yield = model.predict(X_input)[0] * quality_factor

    # --- 🌦️ Apply Realistic Agricultural Rules ---
    # No rainfall → no yield
    if rainfall < 50:
        pred_yield = 0
    # Very low or high temperature → big drop
    elif temperature < 5 or temperature > 40:
        pred_yield *= 0.3
    # Too little fertilizer → lower yield
    elif fertilizer < 120:
        pred_yield *= 0.7
    # Too much fertilizer → overuse penalty
    elif fertilizer > 220:
        pred_yield *= 0.8

    # --- Compute Totals ---
    total_production = pred_yield * farmland_area  # hg
    total_demand = population * avg_demand_per_person

    # Convert to metric tons (1 ton = 10,000 hg)
    total_production_tons = total_production / 10_000
    total_demand_tons = total_demand / 10_000
    difference_tons = abs(total_production_tons - total_demand_tons)
    status = "Surplus" if total_production_tons > total_demand_tons else "Deficit"

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌾 Predicted Yield (hg/ha)", f"{pred_yield:,.2f}")
    col2.metric("🏡 Total Production (metric tons)", f"{total_production_tons:,.0f}")
    col3.metric("📊 Status", status, delta=f"{difference_tons:,.0f} tons")
    col4.metric("👥 Population", f"{population:,.0f}")

    # --- Status message ---
    if status == "Surplus":
        st.success(f"✅ Surplus of {difference_tons:,.0f} tons — Canada can export crops.")
    else:
        st.error(f"⚠️ Deficit of {difference_tons:,.0f} tons — increase rainfall, improve soil, or import crops.")

    # --- Explanation ---
    with st.expander("ℹ️ How the Prediction Works"):
        st.write("""
        The model uses **Random Forest Regression**, trained on FAOSTAT crop yield and Kaggle population data.  
        We also applied **realistic constraints**:
        - 🌧️ Rainfall < 50 mm → crops fail (yield = 0).  
        - 🌡️ Temperature < 5°C or > 40°C → yield drops 70%.  
        - 💩 Fertilizer too low or too high → reduced efficiency.  
        This ensures predictions follow real-world agricultural logic.
        """)

    # --- Charts ---
    colA, colB = st.columns(2)

    # Chart 1 — Production vs Demand
    with colA:
        st.subheader("📊 Production vs Demand Ratio")
        labels = ["Total Production", "Total Demand"]
        values = [total_production_tons, total_demand_tons]
        fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))
        ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                 colors=["#6ab04c", "#f0932b"], textprops={"fontsize": 8})
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)

    # Chart 2 — Fertilizer Yield Comparison
    with colB:
        st.subheader("🌿 Fertilizer Yield Impact")
        fert_types = ["Organic", "Chemical", "Mixed"]
        fert_factors = [0.9, 1.2, 1.0]
        fert_values = [pred_yield / quality_factor * f for f in fert_factors]
        fig2, ax2 = plt.subplots(figsize=(3, 2.3))
        sns.barplot(x=fert_types, y=fert_values,
                    palette=["#badc58", "#eb4d4b", "#22a6b3"], ax=ax2)
        ax2.set_ylabel("Yield (hg/ha)", fontsize=8)
        ax2.set_xlabel("Fertilizer Type", fontsize=8)
        ax2.tick_params(axis="both", labelsize=8)
        for i, val in enumerate(fert_values):
            ax2.text(i, val + (val * 0.02), f"{val:,.0f}", ha="center", fontsize=8)
        st.pyplot(fig2, use_container_width=True)

    # Chart 3 — Yield vs Fertilizer Line
    st.subheader("📈 Yield vs Fertilizer Quantity")
    fert_range = range(100, 251, 10)
    pred_range = []
    for f in fert_range:
        pred = model.predict(pd.DataFrame({
            "population": [population],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "fertilizer": [f]
        }))[0] * quality_factor
        # Apply same realism logic
        if rainfall < 50:
            pred = 0
        elif temperature < 5 or temperature > 40:
            pred *= 0.3
        pred_range.append(pred)

    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    ax3.plot(fert_range, pred_range, color="#0984e3", linewidth=2)
    ax3.set_xlabel("Fertilizer (kg/ha)", fontsize=8)
    ax3.set_ylabel("Predicted Yield (hg/ha)", fontsize=8)
    ax3.tick_params(axis="both", labelsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig3, use_container_width=True)

    # --- Download results ---
    result_df = pd.DataFrame({
        "Population": [population],
        "Rainfall (mm)": [rainfall],
        "Temperature (°C)": [temperature],
        "Fertilizer Type": [fertilizer_type],
        "Fertilizer (kg/ha)": [fertilizer],
        "Predicted Yield (hg/ha)": [pred_yield],
        "Total Production (tons)": [total_production_tons],
        "Total Demand (tons)": [total_demand_tons],
        "Status": [status]
    })
    st.download_button("📥 Download Results", result_df.to_csv(index=False), file_name="crop_yield_results.csv")

else:
    st.info("👈 Adjust rainfall, temperature, fertilizer, and population — then click **Predict Crop Yield**.")
