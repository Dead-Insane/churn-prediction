# ==============================================
# Churn Prediction - Streamlit App
# Author: Divya Prakash
# Description: Predicts customer churn risk and
#              calculates business impact
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    with open('../src/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../src/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../src/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter customer deatails to predict churn risk and estimate business impact.")
st.divider()

st.sidebar.header("🧾 Customer Details")

#Demographics 
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

#Services
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No Phone Service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optics", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No Internet Service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No Internet Service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No Internet Service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No Internet Service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No Internet Service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No Internet Service"])

#Account Info
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer (Automatic)", "Credit Card (Automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = monthly_charges * tenure

predict_btn = st.sidebar.button("🔍 Predict Churn", use_container_width=True)

def preprocess_input():
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges   
        }
    input_df = pd.DataFrame([input_dict])
    
    #Binary Encoding
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map(binary_map)
        
    #One-Hot encode Multiple class Columns
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=multi_cols, drop_first=True)
    
    #Scale Numerical Columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    #Align Columns with Training Data
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    return input_df


#Output Metrics

if predict_btn:
    try:
        input_data = preprocess_input()
        churn_prob = model.predict_proba(input_data)[:,1][0]
        churn_pred = model.predict(input_data)[0]
#Rsik Level Logic
        if churn_prob < 0.4:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
    
        elif churn_prob < 0.65:
            risk_level = "🟡 Medium Risk"
            risk_color = "orange"
    
        else:
            risk_level = "🔴 High Risk"
            risk_color = "red"
        
        st.subheader("🎯 Prediction Result")  
        col1, col2, col3 = st.columns(3)
          
        with col1:
            st.metric(label="Churn Prediction", value="Will Churn ❌" if churn_pred == 1 else "Will Stay ✅")
            
        with col2:
            st.metric(label="Churn Probability", value=f"{churn_prob:.1%}")
            
        with col3:
            st.metric(label="Risk Level", value=risk_level)
            
        st.divider()

        st.subheader("💰 Business Impact Estimate")

        col4, col5, col6 = st.columns(3)

        monthly_loss = monthly_charges
        annual_loss = monthly_loss * 12
        savable = annual_loss * 0.30

        with col4:
            st.metric("MOnthly Revenue at Risk", f"${monthly_loss:.2f}")
            
        with col5:
            st.metric("Annual Revenue at Risk", f"${annual_loss:.2f}")
            
        with col6:
            st.metric("Estimated Retainable Revenue", f"${savable:.2f}", help="Assuming 30% retention success rate ")
            
        st.divider()

        st.subheader("📈 Churn Risk Gauge")

        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(["Churn Risk"], [churn_prob], color=risk_color, height=0.4)
        ax.barh(["Churn Risk"], [1-churn_prob], left=[churn_prob], color='lightgray', height=0.4)
        ax.set_xlim(0,1)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
        ax.set_xlabel("Probability")
        ax.set_title(f"Churn Probability: {churn_prob:.1%}")
        ax.legend()
        st.pyplot(fig)

        st.divider()

        st.subheader("⚠️ Key Risk Factors for This Customer")

        risk_factors = []

        if contract == "Month-to-Month":
            risk_factors.append("📌 Month-to-month contract - highest churn risk contract type")
            
        if tenure < 12:
            risk_factors.append("📌 Low tenure (< 12 months) - early-stage customers churn more")
            
        if monthly_charges > 70:
            risk_factors.append("📌 High monthly charges - may feel the service is overpriced")
            
        if internet_service == "Fiber Optics":
            risk_factors.append("📌 Fiber optic subscriber - historically higher churn in this segment")

        if payment_method == "Electronic Check":
            risk_factors.append("📌 Pays by electronic check - correlated with higher churn")
            
        if risk_factors:
            for factors in risk_factors:
                st.warning(factors)
                
        else:
            st.success("✅ No major risk factors detected for this customer.")
            
        st.divider()
        st.caption("Built by Divya Prakash | Telco Customer Churn Prediction Project | Logistic Regression Model (ROC-AUC: 0.835)")
    
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info("Check that model.pkl, scaler.pkl and feature_columns.pkl are all in the src/ folder")
        
else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to see results.")
