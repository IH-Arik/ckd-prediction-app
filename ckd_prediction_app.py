import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CKD Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏥 Chronic Kidney Disease (CKD) Prediction System")
st.markdown("""
This app predicts the likelihood of CKD occurrence within 35 months based on clinical parameters.
The model uses Random Forest classifier trained on medical data.
""")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('pone_disease_prediction_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("📋 Patient Information")

def create_input_fields():
    """Create input fields for all features"""
    
    # Demographics
    st.sidebar.subheader("👤 Demographics")
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=0)
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=50, step=1)
    
    # Medical History
    st.sidebar.subheader("📋 Medical History")
    history_diabetes = st.sidebar.selectbox("History of Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_chd = st.sidebar.selectbox("History of Coronary Heart Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_vascular = st.sidebar.selectbox("History of Vascular Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_smoking = st.sidebar.selectbox("History of Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_htn = st.sidebar.selectbox("History of Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_dld = st.sidebar.selectbox("History of Dyslipidemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_obesity = st.sidebar.selectbox("History of Obesity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    # Medications
    st.sidebar.subheader("💊 Medications")
    dld_meds = st.sidebar.selectbox("Dyslipidemia Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    dm_meds = st.sidebar.selectbox("Diabetes Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    htn_meds = st.sidebar.selectbox("Hypertension Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    acei_arb = st.sidebar.selectbox("ACEI/ARB Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    # Clinical Measurements
    st.sidebar.subheader("🔬 Clinical Measurements")
    cholesterol = st.sidebar.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
    triglycerides = st.sidebar.slider("Triglycerides (mmol/L)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    hba1c = st.sidebar.slider("HbA1c (%)", min_value=4.0, max_value=15.0, value=6.0, step=0.1)
    creatinine = st.sidebar.slider("Creatinine (μmol/L)", min_value=50, max_value=200, value=80, step=1)
    egfr = st.sidebar.slider("eGFR (mL/min/1.73m²)", min_value=30, max_value=200, value=90, step=1)
    sbp = st.sidebar.slider("Systolic BP (mmHg)", min_value=80, max_value=200, value=120, step=1)
    dbp = st.sidebar.slider("Diastolic BP (mmHg)", min_value=40, max_value=120, value=80, step=1)
    bmi = st.sidebar.slider("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    
    # Study ID (required by model)
    study_id = st.sidebar.number_input("Study ID", min_value=1, max_value=10000, value=1, step=1)
    
    return {
        'StudyID': study_id,
        'Gender': gender,
        'AgeBaseline': age,
        'Age.3.categories': age // 10,  # Age category
        'HistoryDiabetes': history_diabetes,
        'HistoryCHD': history_chd,
        'HistoryVascular': history_vascular,
        'HistorySmoking': history_smoking,
        'HistoryHTN ': history_htn,  # Note the space
        'HistoryDLD': history_dld,
        'HistoryObesity': history_obesity,
        'DLDmeds': dld_meds,
        'DMmeds': dm_meds,
        'HTNmeds': htn_meds,
        'ACEIARB': acei_arb,
        'CholesterolBaseline': cholesterol,
        'TriglyceridesBaseline': triglycerides,
        'HgbA1C': hba1c,
        'CreatnineBaseline': creatinine,
        'eGFRBaseline': egfr,
        'sBPBaseline': sbp,
        'dBPBaseline': dbp,
        'BMIBaseline': bmi,
        'TimeToEventMonths': 35  # Default prediction timeframe
    }

def create_feature_importance_display():
    """Display feature importance information"""
    st.subheader("📊 Feature Importance")
    
    # Clinical importance based on medical knowledge
    importance_data = {
        'Feature': ['eGFR', 'Creatinine', 'Age', 'Diabetes History', 'HbA1c', 
                   'Blood Pressure', 'BMI', 'Cholesterol', 'Smoking History'],
        'Importance': [0.9, 0.85, 0.7, 0.8, 0.75, 0.6, 0.5, 0.4, 0.6],
        'Clinical Relevance': ['Very High', 'Very High', 'High', 'Very High', 'High', 
                           'Medium', 'Medium', 'Medium', 'High']
    }
    
    df_importance = pd.DataFrame(importance_data)
    st.dataframe(df_importance, use_container_width=True)
    
    st.info("""
    🏥 **Clinical Note**: eGFR (estimated Glomerular Filtration Rate) and Creatinine are 
    the most important predictors of kidney function. Diabetes and hypertension are major risk factors.
    """)

def main():
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
        return
    
    # Create input fields
    input_data = create_input_fields()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Input Summary")
        
        # Display current inputs in a nice format
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        # Prediction button
        if st.button("🔮 Predict CKD Risk", type="primary", use_container_width=True):
            make_prediction(input_data)
    
    with col2:
        create_feature_importance_display()
    
    # Disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer**: This app is for educational purposes only and should not be used 
    for actual medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
    """)

def make_prediction(input_data):
    """Make prediction using the loaded model"""
    
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("🔴 **High Risk**: CKD predicted within 35 months")
            else:
                st.success("🟢 **Low Risk**: No CKD predicted within 35 months")
        
        with col2:
            # Probability display
            prob_ckd = probability[1] * 100
            prob_no_ckd = probability[0] * 100
            
            st.metric("CKD Probability", f"{prob_ckd:.1f}%")
            st.metric("No CKD Probability", f"{prob_no_ckd:.1f}%")
        
        # Risk assessment
        st.subheader("📈 Risk Assessment")
        
        if prob_ckd < 20:
            st.success("🟢 **Very Low Risk**: Patient has excellent prognosis")
        elif prob_ckd < 40:
            st.info("🟡 **Low Risk**: Regular monitoring recommended")
        elif prob_ckd < 60:
            st.warning("🟠 **Moderate Risk**: Consult nephrologist recommended")
        elif prob_ckd < 80:
            st.error("🔴 **High Risk**: Immediate medical attention needed")
        else:
            st.error("🔴 **Very High Risk**: Urgent medical intervention required")
        
        # Recommendations
        st.subheader("💡 Clinical Recommendations")
        
        if prob_ckd > 50:
            st.write("""
            🏥 **Recommended Actions:**
            - Consult with a nephrologist immediately
            - Regular kidney function tests (every 3 months)
            - Strict blood pressure control (<130/80 mmHg)
            - Blood sugar management if diabetic
            - Avoid nephrotoxic medications
            - Consider dietary modifications (low protein, low sodium)
            """)
        else:
            st.write("""
            ✅ **Preventive Measures:**
            - Annual kidney function screening
            - Maintain healthy blood pressure
            - Control blood sugar levels
            - Regular exercise and healthy diet
            - Avoid smoking and excessive alcohol
            - Stay hydrated
            """)
        
        # Display raw probabilities
        st.subheader("📊 Detailed Probabilities")
        
        prob_df = pd.DataFrame({
            'Outcome': ['No CKD', 'CKD'],
            'Probability': [prob_no_ckd, prob_ckd]
        })
        
        st.bar_chart(prob_df.set_index('Outcome'), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please check your input values and try again.")

if __name__ == "__main__":
    main()
