# 🩺 LDP-XAI Dashboard: Clinical Decision Support System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ldp-xai-dashboard-ex3uqsotyqtwvjfcj3rt8p.streamlit.app/)

##  Overview
The early diagnosis of liver disease is critical for patient survival, yet standard machine learning diagnostics often function as opaque "black boxes," lacking the transparency required for physicians to trust their predictions. 

The **LDP-XAI Dashboard** is an integrated and fully transparent Clinical Decision Support System (CDSS) built to solve this. It shifts the clinical paradigm from *problem-solver* to *problem-predictor* by combining highly accurate ensemble learning with Explainable AI (XAI).

 **[Experience the Live Dashboard Here](https://ldp-xai-dashboard-ex3uqsotyqtwvjfcj3rt8p.streamlit.app/)**

##  Key Features
* **Predictive Engine:** Powered by an Extreme Gradient Boosting (XGBoost) algorithm that achieved a **94.0% predictive accuracy** on unseen data.
* **Algorithmic Transparency:** Seamlessly integrates **SHAP (SHapley Additive exPlanations)** to generate interactive Waterfall plots, decoding the mathematical reasoning behind every prediction for the clinician.
* **Bias Mitigation:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) during training to completely eliminate the natural class imbalance found in medical datasets, ensuring fair and unbiased learning.
* **Clinical UI:** Deployed via Streamlit with a focus on cognitive load, featuring dynamic risk gauging (Green/Yellow/Red alerts) and automated clinical recommendations.

##  Technology Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Explainable AI:** SHAP (TreeExplainer)
* **Data Processing:** Pandas, Numpy, Imbalanced-learn (SMOTE)
* **Deployment & UI:** Streamlit, Plotly

##  How to Run Locally
If you wish to run this application on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Eswar003-sai/LDP-XAI-Dashboard.git](https://github.com/Eswar003-sai/LDP-XAI-Dashboard.git)

   # 1. Clone the repository
git clone https://github.com/Eswar003-sai/LDP-XAI-Dashboard.git

# 2. Navigate into the folder
cd LDP-XAI-Dashboard

# 3. Install dependencies (Add it here)
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
