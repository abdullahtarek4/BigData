import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Urban Traffic Analytics Dashboard",
    layout="wide"
)

st.title("🚦 Big Data Project: Weather Impact on Urban Traffic")
st.markdown("### Interactive Dashboard for Analysis & Prediction")
st.markdown("---")

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        df_merged = pd.read_csv(os.path.join(script_dir, "final_merged_dataset.csv"))
        df_sim = pd.read_csv(os.path.join(script_dir, "simulation_results.csv"))
        df_factors = pd.read_csv(
            os.path.join(script_dir, "factor_analysis_loadings.csv"),
            index_col=0
        )
        return df_merged, df_sim, df_factors

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None


df_merged, df_sim, df_factors = load_data()

if df_merged is None:
    st.stop()

# ===============================
# Sidebar
# ===============================
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Go to:",
    [
        "📊 Data Overview",
        "🎲 Monte Carlo Simulation",
        "📐 Factor Analysis"
    ]
)

# ======================================================
# 1️⃣ DATA OVERVIEW
# ======================================================
if options == "📊 Data Overview":

    st.header("1. Dataset Overview (Weather + Traffic)")
    st.dataframe(df_merged.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_merged))
    col2.metric("Avg Traffic Speed", f"{df_merged['avg_speed_kmh'].mean():.1f} km/h")
    col3.metric("Avg Rainfall", f"{df_merged['rain_mm'].mean():.1f} mm")

    st.markdown("---")
    st.subheader("Exploratory Visualizations")

    # Rain vs Speed
    fig1 = px.scatter(
        df_merged,
        x="rain_mm",
        y="avg_speed_kmh",
        color="congestion_level",
        title="Impact of Rainfall on Traffic Speed"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Accidents by Weather
    fig2 = px.box(
        df_merged,
        x="weather_condition",
        y="accident_count",
        title="Accident Distribution by Weather Condition"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Bubble Chart
    fig3 = px.scatter(
        df_merged,
        x="rain_mm",
        y="accident_count",
        size="vehicle_count",
        color="weather_condition",
        title="Rainfall vs Accidents (Bubble = Vehicle Count)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Congestion Distribution
    fig4 = px.pie(
        df_merged,
        names="congestion_level",
        title="Traffic Congestion Level Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# 2️⃣ MONTE CARLO SIMULATION
# ======================================================
elif options == "🎲 Monte Carlo Simulation":

    st.header("2. Monte Carlo Simulation Analysis")
    st.markdown(
        "Simulation based on **10,000 iterations** predicting traffic congestion risk under **heavy rain**."
    )

    high_risk_prob = (df_sim["congestion_risk"] > 0.7).mean() * 100
    avg_risk = df_sim["congestion_risk"].mean()

    c1, c2 = st.columns(2)
    c1.metric("High Congestion Probability (>70%)", f"{high_risk_prob:.2f}%")
    c2.metric("Average Risk Score", f"{avg_risk:.2f}")

    st.markdown("---")

    # Histogram
    fig_hist = px.histogram(
        df_sim,
        x="congestion_risk",
        nbins=50,
        title="Distribution of Traffic Congestion Risk"
    )
    fig_hist.add_vline(
        x=0.7,
        line_dash="dash",
        annotation_text="High Risk Threshold"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box Plot
    fig_box = px.box(
        df_sim,
        y="congestion_risk",
        title="Spread of Congestion Risk"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Risk Categories
    df_sim["risk_category"] = pd.cut(
        df_sim["congestion_risk"],
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    fig_bar = px.bar(
        df_sim["risk_category"].value_counts().reset_index(),
        x="risk_category",
        y="count",
        title="Congestion Risk Categories"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # CDF Curve
    sorted_risk = df_sim["congestion_risk"].sort_values().reset_index(drop=True)
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=sorted_risk,
        y=sorted_risk.index / len(sorted_risk),
        mode="lines"
    ))
    fig_cdf.update_layout(
        title="Cumulative Probability of Congestion Risk",
        xaxis_title="Risk Score",
        yaxis_title="Probability"
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

# ======================================================
# 3️⃣ FACTOR ANALYSIS
# ======================================================
elif options == "📐 Factor Analysis":

    st.header("3. Factor Analysis Interpretation")
    st.markdown("Understanding hidden latent factors influencing traffic behavior.")

    # Heatmap
    fig_heat = px.imshow(
        df_factors,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Factor Loadings Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Factor Importance
    factor_strength = df_factors.abs().mean().reset_index()
    factor_strength.columns = ["Factor", "Average Loading"]

    fig_factor = px.bar(
        factor_strength,
        x="Factor",
        y="Average Loading",
        title="Overall Importance of Each Factor"
    )
    st.plotly_chart(fig_factor, use_container_width=True)

    # Variable Contribution
    selected_factor = st.selectbox(
        "Select Factor",
        df_factors.columns
    )

    fig_var = px.bar(
        df_factors[selected_factor].sort_values().reset_index(),
        x=selected_factor,
        y="index",
        orientation="h",
        title=f"Variable Contributions to {selected_factor}"
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Radar Chart
    fig_radar = go.Figure()
    for factor in df_factors.columns:
        fig_radar.add_trace(go.Scatterpolar(
            r=df_factors[factor],
            theta=df_factors.index,
            fill="toself",
            name=factor
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Radar Chart of Factor Loadings"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.info("""
    **Interpretation:**
    - Factor 1 → Traffic Flow 
    - Factor 2 → Weather Severity
    - Factor 3 → Risk & Safety  
    """)

# ===============================
# Footer
# ===============================
st.sidebar.markdown("---")
st.sidebar.info("Big Data Final Project – 2024")

