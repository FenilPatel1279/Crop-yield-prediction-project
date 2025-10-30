import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page setup ---
st.set_page_config(page_title="Crop Yield Prediction Dashboard", layout="wide")

# --- Load model ---
if not os.path.exists("models/crop_yield_model.pkl"):
    st.error("❌ Model file not found. Please run src/model_train.py first.")
    st.stop()

model = joblib.load("models/crop_yield_model.pkl")

# --- Load population data ---
pop_df = pd.read_csv("data/processed/population_processed.csv", encoding="latin1")
pop_df.columns = pop_df.columns.str.strip().str.lower()

# --- Title and description ---
st.title("🌾 Crop Yield Prediction Dashboard (Canada)")
st.markdown("Use this dashboard to analyze how rainfall, temperature, and fertilizer affect **Canada’s crop yield**.")

# --- Sidebar Controls ---
st.sidebar.header("Input Controls")

rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1000.0, 500.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)

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

# --- Predict Button ---
if st.sidebar.button("🔮 Predict Crop Yield"):
    # --- Prepare constants ---
    country = "Canada"
    avg_demand_per_person = 500.0
    population = 38_000_000

    # --- Prepare input ---
    X_input = pd.DataFrame({
        "population": [population],
        "rainfall": [rainfall],
        "temperature": [temperature],
        "fertilizer": [fertilizer]
    })

    # --- Prediction ---
    pred_yield = model.predict(X_input)[0] * quality_factor
    total_demand = population * avg_demand_per_person
    status = "Surplus" if pred_yield > total_demand else "Deficit"

    # --- KPI Cards ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 Predicted Yield (hg/ha)", f"{pred_yield:,.2f}")
    col2.metric("👥 Population", f"{population:,.0f}")
    col3.metric("📊 Status", status, delta_color="off" if status=="Surplus" else "inverse")

    # --- Expander Explanation ---
    with st.expander("ℹ️ How the Prediction Works"):
        st.write("""
        This model uses **Random Forest Regression**, trained on FAOSTAT and Kaggle population data.
        - Inputs: rainfall, temperature, fertilizer type & quantity, and population.
        - Output: estimated yield (hg/ha).
        Fertilizer type affects both **yield quantity** and **quality** via internal multipliers.
        """)

    # --- Charts ---
    colA, colB = st.columns(2)

    # Chart 1 — Pie chart (Production vs Demand)
    with colA:
        st.subheader("📊 Production vs Demand Ratio")
        labels = ["Predicted Yield", "Total Demand"]
        values = [pred_yield, total_demand]
        fig1, ax1 = plt.subplots(figsize=(2.3, 2.3))
        ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                 colors=["#6ab04c", "#f0932b"], textprops={"fontsize": 8})
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)

    # Chart 2 — Fertilizer comparison
    with colB:
        st.subheader("🌿 Fertilizer Yield Impact")
        fert_types = ["Organic", "Chemical", "Mixed"]
        fert_factors = [0.9, 1.2, 1.0]
        fert_values = [pred_yield / quality_factor * f for f in fert_factors]
        fig2, ax2 = plt.subplots(figsize=(2.8, 2.3))
        sns.barplot(x=fert_types, y=fert_values,
                    palette=["#badc58", "#eb4d4b", "#22a6b3"], ax=ax2)
        ax2.set_ylabel("Yield (hg/ha)", fontsize=8)
        ax2.set_xlabel("Fertilizer Type", fontsize=8)
        ax2.tick_params(axis="both", labelsize=8)
        ax2.set_title("Yield Comparison", fontsize=9)
        for i, val in enumerate(fert_values):
            ax2.text(i, val + (val * 0.02), f"{val:,.0f}", ha="center", fontsize=8)
        st.pyplot(fig2, use_container_width=True)

    # Chart 3 — Yield vs Fertilizer Quantity line chart
    st.subheader("📈 Yield vs Fertilizer Quantity")
    fert_range = range(100, 251, 10)
    pred_range = [
        model.predict(pd.DataFrame({
            "population": [population],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "fertilizer": [f]
        }))[0] * quality_factor for f in fert_range
    ]
    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    ax3.plot(fert_range, pred_range, color="#0984e3", linewidth=2)
    ax3.set_xlabel("Fertilizer (kg/ha)", fontsize=8)
    ax3.set_ylabel("Predicted Yield (hg/ha)", fontsize=8)
    ax3.tick_params(axis="both", labelsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig3, use_container_width=True)

    # --- Downloadable Results ---
    result_df = pd.DataFrame({
        "Fertilizer Type": [fertilizer_type],
        "Fertilizer (kg/ha)": [fertilizer],
        "Predicted Yield": [pred_yield],
        "Population": [population],
        "Status": [status]
    })
    st.download_button("📥 Download Results", result_df.to_csv(index=False), file_name="crop_yield_results.csv")

else:
    st.info("👈 Adjust inputs and click **Predict Crop Yield** to see results.")
