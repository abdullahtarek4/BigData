import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# === إعدادات الصفحة ===
st.set_page_config(page_title="Urban Traffic Analytics Dashboard", layout="wide")

st.title("🚦 Big Data Project: Weather Impact on Urban Traffic")
st.markdown("### Interactive Dashboard for Analysis & Prediction")
st.markdown("---")

# === دالة تحميل البيانات ===
@st.cache_data
def load_data():
    # محاولة تحميل الملفات التي أنتجناها سابقاً
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        df_merged = pd.read_csv(os.path.join(script_dir, 'final_merged_dataset.csv'))
        df_sim = pd.read_csv(os.path.join(script_dir, 'simulation_results.csv'))
        df_factors = pd.read_csv(os.path.join(script_dir, 'factor_analysis_loadings.csv'), index_col=0)
        return df_merged, df_sim, df_factors
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None

df_merged, df_sim, df_factors = load_data()

if df_merged is None:
    st.error("❌ Error: Could not find the CSV files. Please run the previous analysis codes first!")
    st.stop()

# === القائمة الجانبية (Sidebar) ===
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", ["📊 Data Overview", "🎲 Monte Carlo Simulation", "bmi Factor Analysis"])

# ==========================================
# 1. قسم استعراض البيانات (Data Overview)
# ==========================================
if options == "📊 Data Overview":
    st.header("1. Merged Dataset (Weather + Traffic)")
    
    # عرض أول 5 صفوف
    st.dataframe(df_merged.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_merged))
    col2.metric("Avg Traffic Speed", f"{df_merged['avg_speed_kmh'].mean():.1f} km/h")
    col3.metric("Avg Rainfall", f"{df_merged['rain_mm'].mean():.1f} mm")

    st.subheader("Visualizations")
    
    # رسم بياني 1: العلاقة بين المطر وسرعة السيارات
    fig1 = px.scatter(df_merged, x="rain_mm", y="avg_speed_kmh", 
                      color="congestion_level", 
                      title="Impact of Rain on Traffic Speed",
                      labels={"rain_mm": "Rainfall (mm)", "avg_speed_kmh": "Avg Speed (km/h)"})
    st.plotly_chart(fig1, use_container_width=True)

    # رسم بياني 2: الحوادث حسب حالة الطقس
    fig2 = px.box(df_merged, x="weather_condition", y="accident_count", 
                  title="Accident Distribution by Weather Condition")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 2. قسم المحاكاة (Monte Carlo Simulation)
# ==========================================
elif options == "🎲 Monte Carlo Simulation":
    st.header("2. Monte Carlo Simulation Results")
    st.markdown("Simulation of **10,000 runs** to predict traffic congestion risk under **Heavy Rain** scenario.")

    # حساب الاحتمالات
    high_risk_prob = (df_sim['congestion_risk'] > 0.7).mean() * 100
    avg_risk = df_sim['congestion_risk'].mean()

    # عرض بطاقات الأرقام (KPIs)
    c1, c2 = st.columns(2)
    c1.metric("High Congestion Probability (>70%)", f"{high_risk_prob:.2f}%", delta_color="inverse")
    c2.metric("Average Risk Score", f"{avg_risk:.2f}")

    # رسم هستوجرام لتوزيع المخاطر
    fig_hist = px.histogram(df_sim, x="congestion_risk", nbins=50, 
                            title="Distribution of Traffic Congestion Risk",
                            color_discrete_sequence=['#FF5733'])
    fig_hist.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="High Risk Threshold")
    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 3. قسم تحليل العوامل (Factor Analysis)
# ==========================================
elif options == "bmi Factor Analysis":
    st.header("3. Factor Analysis Interpretation")
    st.markdown("Identifying hidden latent factors driving the traffic behavior.")
    
    st.subheader("Factor Loadings Heatmap")
    st.write("Values closer to **1.0** or **-1.0** indicate strong influence.")

    # رسم Heatmap
    fig_heat = px.imshow(df_factors, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r",
                         title="Correlation between Variables and Hidden Factors")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info("""
    **Interpretation Guide:**
    * **Factor 1:** Likely represents 'Weather Severity' (High loadings on Rain/Wind).
    * **Factor 2:** Likely represents 'Traffic Flow' (High loadings on Speed/Vehicle Count).
    * **Factor 3:** Likely represents 'Risk Factor' (High loadings on Accidents/Visibility).
    """)

# تذييل الصفحة
st.sidebar.markdown("---")
st.sidebar.info("Big Data Final Project - 2024")
