import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="LDP-XAI Dashboard", layout="wide", initial_sidebar_state="expanded")

# This CSS forces the text to be dark inside white cards regardless of the theme
st.markdown("""
<style>
    /* Card styling */
    div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], .stTabs {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #f0f2f6;
    }

    /* Force all text inside metrics and tabs to dark slate */
    div[data-testid="stMetric"] *, .stTabs * {
        color: #1e293b !important;
    }

    /* Ensuring the headers and subheaders inside cards are visible */
    h1, h2, h3, p {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'patient_log' not in st.session_state:
    st.session_state.patient_log = pd.DataFrame(columns=['Time', 'Age', 'Gender', 'Risk_Score', 'Alert_Level'])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model = pickle.load(open('liver_model.pkl', 'rb'))
    explainer = pickle.load(open('explainer.pkl', 'rb'))
    return model, explainer

model, explainer = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🩺 LDP-XAI Admin")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation Menu", ["🧑‍⚕️ New Patient Scan", "📊 Live System Dashboard"])
st.sidebar.markdown("---")

# ==========================================
# PAGE 1: NEW PATIENT SCAN
# ==========================================
if menu == "🧑‍⚕️ New Patient Scan":
    st.title("Clinical Risk Assessment")
    st.markdown("Enter patient biomarkers to generate an AI-assisted diagnostic prediction.")

    with st.expander("📝 Enter Patient Clinical Markers", expanded=True):
        col1, col2 = st.columns(2)

        # ALL DEFAULT VALUES SET TO 0
        with col1:
            age = st.slider("Age", 0, 100, 0)

            # GENDER DROPDOWN UPDATED
            gender = st.selectbox("Gender", ["Select", "Male", "Female", "Transgender"])

            tb = st.number_input("Total Bilirubin", 0.0, 40.0, 0.0, step=0.1)
            db = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.0, step=0.1)
            alp = st.slider("Alkaline Phosphotase", 0, 2000, 0)

        with col2:
            alt = st.slider("SGPT (ALT)", 0, 2000, 0)
            ast = st.slider("SGOT (AST)", 0, 2000, 0)
            tp = st.number_input("Total Proteins", 0.0, 10.0, 0.0, step=0.1)
            alb = st.number_input("Albumin", 0.0, 10.0, 0.0, step=0.1)
            ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0, 0.0, step=0.1)

    # Encode Gender for the XGBoost Model (Model trained on Male=1, Female=0)
    gender_encoded = 1 if gender == "Male" else 0

    input_data = pd.DataFrame([[age, gender_encoded, tb, db, alp, alt, ast, tp, alb, ag_ratio]],
                              columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                                       'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                                       'Albumin', 'Albumin_and_Globulin_Ratio'])

    if st.button("🚀 Analyze Risk Metrics", type="primary", use_container_width=True):

        # VALIDATION CHECK: Prevent prediction if Gender is "Select"
        if gender == "Select":
            st.error("⚠️ Please select a valid Gender (Male, Female, or Transgender) before analyzing.")
        else:
            with st.spinner("Processing clinical data..."):
                prob = model.predict_proba(input_data)[0][1] * 100

                # Determine Alert Level
                if prob < 30: alert = "Low Risk 🟢"
                elif 30 <= prob < 70: alert = "Medium Risk 🟡"
                else: alert = "High Risk 🔴"

                # Log to Session State
                new_record = pd.DataFrame([{
                    'Time': datetime.now().strftime("%H:%M:%S"),
                    'Age': age, 'Gender': gender, 'Risk_Score': round(prob, 2), 'Alert_Level': alert
                }])
                st.session_state.patient_log = pd.concat([new_record, st.session_state.patient_log], ignore_index=True)

                st.markdown("---")

                tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Overview", "🧠 SHAP Explainability", "📋 Next Steps"])

                with tab1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = prob, title = {'text': "Liver Disease Probability (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1e293b"},
                            'steps': [
                                {'range': [0, 30], 'color': "#10b981"},
                                {'range': [30, 70], 'color': "#facc15"},
                                {'range': [70, 100], 'color': "#ef4444"}]
                        }))
                    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with tab2:
                    st.markdown("### Feature Contribution Analysis")
                    shap_values = explainer(input_data)
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig_shap, bbox_inches='tight')
                    plt.close(fig_shap) # Clean up memory

                with tab3:
                    st.markdown("### Clinical Recommendations")
                    if prob >= 70:
                        st.error("Urgent Hepatologist Consultation Required.")
                    elif 30 <= prob < 70:
                        st.warning("Follow-up required. Recommend lifestyle changes and secondary blood panel.")
                    else:
                        st.success("Patient is in healthy parameters.")

                    csv = input_data.copy()
                    csv['Risk_Score'] = prob
                    st.download_button(label="📥 Download Patient Record (CSV)", data=csv.to_csv(index=False), file_name="patient_result.csv", mime="text/csv")

# ==========================================
# PAGE 2: LIVE SYSTEM DASHBOARD
# ==========================================
elif menu == "📊 Live System Dashboard":
    st.title("Admin Overview")
    logs = st.session_state.patient_log

    if logs.empty:
        st.info("No scans performed in this session yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Scans This Session", len(logs))
        m2.metric("High Risk Cases", len(logs[logs['Risk_Score'] >= 70]))
        m3.metric("Avg Risk Score", f"{logs['Risk_Score'].mean():.1f}%")

        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pie = px.pie(logs, names='Alert_Level', title="Risk Distribution",
                             color='Alert_Level',
                             color_discrete_map={"Low Risk 🟢":"#10b981", "Medium Risk 🟡":"#facc15", "High Risk 🔴":"#ef4444"})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### Recent Activity Log")
            st.dataframe(logs, use_container_width=True)
