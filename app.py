import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import base64
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject FontAwesome and custom CSS (Standardized method)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"><style>@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap\'); .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important; font-family: \'Inter\', sans-serif !important; } div.stButton > button:first-child { width: 100% !important; border-radius: 12px !important; height: 3.5em !important; background-color: #b91c1c !important; color: white !important; font-weight: 700 !important; border: none !important; box-shadow: 0 4px 15px rgba(185, 28, 28, 0.4) !important; transition: all 0.3s ease !important; text-transform: uppercase !important; letter-spacing: 1px !important; } div.stButton > button:first-child:hover { background-color: #991b1b !important; box-shadow: 0 8px 25px rgba(153, 27, 27, 0.5) !important; transform: translateY(-2px) !important; } h1 { color: #1e3a8a !important; font-weight: 800 !important; } .clinical-card { background-color: white !important; padding: 2rem !important; border-radius: 20px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important; margin-bottom: 2rem !important; border: 1px solid #e2e8f0 !important; } .icon-header { display: flex !important; align-items: center !important; gap: 12px !important; color: #1e3a8a !important; margin-bottom: 1rem !important; font-weight: 700 !important; font-size: 1.25rem !important; } [data-testid="stMetricValue"] { color: #1e3a8a !important; font-weight: 700 !important; } [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }</style>', unsafe_allow_html=True)

st.markdown('<h1 class="icon-header"><i class="fas fa-hand-holding-medical"></i> Chronic Kidney Disease (CKD) Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="clinical-card">
    This clinical decision support system predicts the likelihood of CKD occurrence within 35 months. 
    Powered by a validated machine learning model and <b>Real SHAP Explainability</b> for medical transparency.
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/ckd_model_fixed.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Create SHAP explainer
@st.cache_resource
def create_shap_explainer(_model):
    """Create SHAP explainer for the pipeline model"""
    try:
        # Use a small sample of data for background
        df = pd.read_csv('data/pone.0199920.csv')
        df = df.replace('#NULL!', np.nan)
        
        # Convert to numeric
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'StudyID':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Prepare background data - ensure we have both classes
        X_background = df.drop(columns=['EventCKD35'])
        y_background = df['EventCKD35']
        
        # Sample balanced background data
        ckd_samples = X_background[y_background == 1].sample(min(50, (y_background == 1).sum()), random_state=42)
        no_ckd_samples = X_background[y_background == 0].sample(min(50, (y_background == 0).sum()), random_state=42)
        X_background_balanced = pd.concat([ckd_samples, no_ckd_samples])
        
        # Extract the classifier from pipeline for SHAP
        if hasattr(_model, 'named_steps') and 'classifier' in _model.named_steps:
            classifier = _model.named_steps['classifier']
            # Create explainer for the classifier
            explainer = shap.TreeExplainer(classifier, X_background_balanced)
        else:
            # Fallback to TreeExplainer for RandomForest
            explainer = shap.TreeExplainer(_model, X_background_balanced)
        
        return explainer
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")
        return None

# Create SHAP explainer
explainer = create_shap_explainer(model)

def create_shap_waterfall_plot(input_data, prediction, probability):
    """Create simplified SHAP waterfall plot"""
    
    try:
        if explainer is None:
            return None
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Calculate SHAP values
        shap_values = explainer(input_df)
        
        # Get values for the predicted class
        if len(shap_values.values.shape) > 1:
            # Use the values for the predicted class
            if prediction == 1:
                shap_vals = shap_values.values[0, :, 1]  # Positive class
                base_value = explainer.expected_value[1]
            else:
                shap_vals = shap_values.values[0, :, 0]  # Negative class
                base_value = explainer.expected_value[0]
        else:
            shap_vals = shap_values.values[0]
            base_value = explainer.expected_value
        
        # Ensure base_value is scalar
        if hasattr(base_value, '__len__') and len(base_value) > 1:
            base_value = base_value[1] if prediction == 1 else base_value[0]
        base_value = float(base_value)
        
        # Ensure shap_vals is 1D array
        if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 1:
            shap_vals = shap_vals.flatten()
        
        # Get feature names
        feature_names = input_df.columns.tolist()
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        # Create waterfall plot manually
        current_value = base_value
        cumulative_values = [base_value]
        colors = []
        
        for val in shap_vals:
            current_value += val
            cumulative_values.append(current_value)
            colors.append('red' if val > 0 else 'blue')
        
        # Plot the waterfall
        x_pos = range(len(cumulative_values))
        plt.plot(x_pos, cumulative_values, 'o-', linewidth=2, markersize=6)
        
        # Add colored bars for each feature contribution
        for i, (val, color) in enumerate(zip(shap_vals, colors)):
            plt.bar(i+1, val, bottom=cumulative_values[i], color=color, alpha=0.7, width=0.8)
        
        # Add base value line
        plt.axhline(y=base_value, color='black', linestyle='--', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        plt.axhline(y=cumulative_values[-1], color='green', linestyle='--', alpha=0.5, label=f'Final Value: {cumulative_values[-1]:.3f}')
        
        # Add feature labels for top contributors
        abs_vals = np.abs(shap_vals)
        top_indices = np.argsort(abs_vals)[-10:]  # Top 10 contributors
        
        for idx in top_indices:
            val = shap_vals[idx]
            color = 'red' if val > 0 else 'blue'
            plt.annotate(f'{feature_names[idx]}\\n{val:.3f}', 
                        xy=(idx+1, cumulative_values[idx]), 
                        xytext=(idx+1, cumulative_values[idx] + (0.02 if val > 0 else -0.02)),
                        ha='center', va='bottom' if val > 0 else 'top',
                        color=color, fontsize=8, fontweight='bold')
        
        plt.xlabel('Features')
        plt.ylabel('Prediction Value')
        plt.title(f'SHAP Waterfall Plot - Base: {base_value:.3f} to Final: {cumulative_values[-1]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode()
        
        return img_base64
        
    except Exception as e:
        st.error(f"Error creating SHAP waterfall plot: {e}")
        return None

def create_shap_summary_plot(input_data):
    """Create simplified SHAP summary plot"""
    
    try:
        if explainer is None:
            return None
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Calculate SHAP values
        shap_values = explainer(input_df)
        
        # Get values for the predicted class
        if len(shap_values.values.shape) > 1:
            # Use the values for the positive class
            shap_vals = shap_values.values[0, :, 1]  # Positive class
        else:
            shap_vals = shap_values.values[0]
        
        # Get feature names
        feature_names = input_df.columns.tolist()
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 8))
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_vals = shap_vals[sorted_idx]
        
        # Create colors
        colors = ['red' if v > 0 else 'blue' for v in sorted_vals]
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_vals)):
            plt.text(val + (0.001 if val > 0 else -0.001), i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.yticks(range(len(sorted_vals)), sorted_names)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Feature Contributions to CKD Risk Prediction')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode()
        
        return img_base64
        
    except Exception as e:
        st.error(f"Error creating SHAP summary plot: {e}")
        return None

def create_shap_force_plot(input_data, prediction, probability):
    """Create a SHAP force plot and return as base64 string"""
    try:
        if explainer is None:
            return None
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get SHAP values
        shap_values = explainer(input_df)
        
        # Use simple matplotlib bar chart if force_plot fails or is confusing
        plt.figure(figsize=(12, 6))
        
        # Determine base value and shap values for the predicted class
        if hasattr(explainer.expected_value, "__len__"):
            base_val = explainer.expected_value[1] if prediction == 1 else explainer.expected_value[0]
            vals = shap_values.values[0, :, 1] if prediction == 1 else shap_values.values[0, :, 0]
        else:
            base_val = explainer.expected_value
            vals = shap_values.values[0]
            
        # Create a horizontal bar chart for force representation
        # Features that increase risk (Red) vs Decrease (Blue)
        # Sort by absolute value to show most impactful features
        feature_names = input_df.columns.tolist()
        sorted_indices = np.argsort(np.abs(vals))[-15:] # Top 15 features
        
        # Filter names and values based on sorted indices
        display_names = [feature_names[i] for i in sorted_indices]
        display_vals = vals[sorted_indices]
        
        colors = ['red' if x > 0 else 'blue' for x in display_vals]
        
        plt.barh(np.arange(len(display_names)), display_vals, color=colors)
        plt.yticks(np.arange(len(display_names)), display_names)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f"Clinical Feature Impact - Prediction: {'CKD' if prediction == 1 else 'No CKD'} ({probability*100:.1f}%)")
        plt.xlabel("SHAP Value (Impact on Risk)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Error creating clinical impact plot: {e}")
        return None

def create_feature_contributions_table(input_data):
    """Create a table showing feature contributions"""
    
    try:
        if explainer is None:
            return None
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Calculate SHAP values
        shap_values = explainer(input_df)
        
        # Get values for the predicted class
        if len(shap_values.values.shape) > 1:
            # Use the values for the predicted class
            shap_vals = shap_values.values[0, :, 1]  # Positive class
        else:
            shap_vals = shap_values.values[0]
        
        # Get feature names and values
        feature_names = input_df.columns.tolist()
        feature_values = input_df.iloc[0].values
        
        # Create contribution data
        contribution_data = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_vals)):
            # Handle value formatting properly
            if isinstance(value, (int, np.integer)):
                value_str = str(int(value))
            elif isinstance(value, (float, np.floating)):
                value_str = f"{float(value):.3f}"
            else:
                value_str = str(value)
            
            # Ensure shap_val is scalar
            shap_val_float = float(shap_val) if hasattr(shap_val, '__len__') else float(shap_val)
            
            contribution_data.append({
                'Feature': name,
                'Value': value_str,
                'SHAP Value': f"{shap_val_float:.4f}",
                'Impact': 'Increases Risk' if shap_val_float > 0 else 'Decreases Risk',
                'Contribution (%)': f"{abs(shap_val_float) * 100:.2f}%"
            })
        
        # Sort by absolute SHAP value
        contribution_df = pd.DataFrame(contribution_data)
        contribution_df['SHAP Abs'] = contribution_df['SHAP Value'].astype(float).abs()
        contribution_df = contribution_df.sort_values('SHAP Abs', ascending=False).drop('SHAP Abs', axis=1)
        
        return contribution_df
        
    except Exception as e:
        st.error(f"Error creating feature contributions table: {e}")
        return None

# Sidebar for inputs
st.sidebar.markdown('<div class="icon-header"><i class="fas fa-clipboard-list"></i> <b>Patient Information</b></div>', unsafe_allow_html=True)

def create_input_fields():
    """Create input fields for all features"""
    
    # Demographics
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div class="icon-header"><i class="fas fa-user"></i> <b>Demographics</b></div>', unsafe_allow_html=True)
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=0)
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=50, step=1)
    age_category = st.sidebar.selectbox("Age Category", options=[0, 1, 2], 
                                      format_func=lambda x: f"Category {x} ({'23-49 yrs' if x == 0 else '50-64 yrs' if x == 1 else '65-89 yrs'})", 
                                      index=0)
    
    # Medical History
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div class="icon-header"><i class="fas fa-history"></i> <b>Medical History</b></div>', unsafe_allow_html=True)
    history_diabetes = st.sidebar.selectbox("History of Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_chd = st.sidebar.selectbox("History of Coronary Heart Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_vascular = st.sidebar.selectbox("History of Vascular Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_smoking = st.sidebar.selectbox("History of Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_htn = st.sidebar.selectbox("History of Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_dld = st.sidebar.selectbox("History of Dyslipidemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    history_obesity = st.sidebar.selectbox("History of Obesity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    # Medications
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div class="icon-header"><i class="fas fa-pills"></i> <b>Medications</b></div>', unsafe_allow_html=True)
    dld_meds = st.sidebar.selectbox("Dyslipidemia Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    dm_meds = st.sidebar.selectbox("Diabetes Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    htn_meds = st.sidebar.selectbox("Hypertension Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    acei_arb = st.sidebar.selectbox("ACEI/ARB Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    # Clinical Measurements
    st.sidebar.markdown('---')
    st.sidebar.markdown('<div class="icon-header"><i class="fas fa-microscope"></i> <b>Clinical Measurements</b></div>', unsafe_allow_html=True)
    cholesterol = st.sidebar.slider("Total Cholesterol (mmol/L)", min_value=2.0, max_value=10.0, value=5.0, step=0.1)
    triglycerides = st.sidebar.slider("Triglycerides (mmol/L)", min_value=0.1, max_value=7.0, value=1.3, step=0.1)
    hba1c = st.sidebar.slider("HbA1c (%)", min_value=3.5, max_value=20.0, value=6.5, step=0.1)
    creatinine = st.sidebar.slider("Creatinine (μmol/L)", min_value=5, max_value=130, value=68, step=1)
    egfr = st.sidebar.slider("eGFR (mL/min/1.73m²)", min_value=60, max_value=250, value=98, step=1)
    sbp = st.sidebar.slider("Systolic BP (mmHg)", min_value=90, max_value=180, value=130, step=1)
    dbp = st.sidebar.slider("Diastolic BP (mmHg)", min_value=40, max_value=115, value=77, step=1)
    bmi = st.sidebar.slider("BMI (kg/m²)", min_value=13.0, max_value=60.0, value=30.0, step=0.1)
    
    # Study ID (required by model)
    study_id = st.sidebar.number_input("Study ID", min_value=1, max_value=10000, value=1, step=1)
    
    return {
        'StudyID': study_id,
        'Gender': gender,
        'AgeBaseline': age,
        'Age.3.categories': age_category,  # Use user-selected age category
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
    st.markdown('<div class="icon-header"><i class="fas fa-chart-bar"></i> <b>Key Risk Factors</b></div>', unsafe_allow_html=True)
    
    # Clinical importance based on medical knowledge
    importance_data = {
        'Risk Factor': ['eGFR < 60', 'HbA1c > 7%', 'Age > 65', 'Diabetes', 'Hypertension', 
                       'High Creatinine', 'Obesity (BMI > 30)', 'Smoking', 'Family History'],
        'Risk Level': ['Very High', 'High', 'High', 'Very High', 'High', 
                      'Very High', 'Medium', 'Medium', 'Medium'],
        'Clinical Impact': ['Kidney Function', 'Blood Sugar', 'Age Risk', 'Metabolic', 
                           'Blood Pressure', 'Kidney Damage', 'Metabolic', 'Vascular', 'Genetic']
    }
    
    df_importance = pd.DataFrame(importance_data)
    st.dataframe(df_importance, use_container_width=True)
    
    st.info("""
    Clinical Note: This model provides personalized risk assessments based on a comprehensive 
    set of clinical features. Real SHAP explanations show the exact contributions of each 
    feature to individual patient predictions, ensuring clinical transparency.
    """)

def main():
    if model is None:
        st.error("Fixed model could not be loaded. Please check the model file.")
        return
    
    # Create input fields
    input_data = create_input_fields()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="icon-header"><i class="fas fa-file-medical-alt"></i> <b>Input Summary</b></div>', unsafe_allow_html=True)
        
        # Display current inputs in a nice format
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        # Prediction button
        if st.button("PREDICT CKD RISK WITH SHAP", type="primary", use_container_width=True):
            prediction_result = make_prediction_with_shap(input_data)
    
    with col2:
        create_feature_importance_display()
    
    # Disclaimer
    st.warning("""
    **Medical Disclaimer**: This app uses a validated machine learning model and real SHAP 
    explanations to assist clinical decision-making. However, it MUST NOT be used as the 
    sole basis for diagnosis or treatment. Always consult with qualified medical professionals.
    """)

def make_prediction_with_shap(input_data):
    """Make prediction using the fixed model and show real SHAP explanations"""
    
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Generate SHAP explanations
        waterfall_img = create_shap_waterfall_plot(input_data, prediction, probability[1])
        summary_img = create_shap_summary_plot(input_data)
        force_img = create_shap_force_plot(input_data, prediction, probability[1])
        contributions_table = create_feature_contributions_table(input_data)
        
        # Display results
        st.markdown('<div class="icon-header"><i class="fas fa-bullseye"></i> <b>Prediction Results</b></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown('<div style="background-color: #fee2e2; color: #b91c1c; padding: 1rem; border-radius: 8px; font-weight: 700;"><i class="fas fa-exclamation-triangle"></i> CLINICAL ALERT: HIGH RISK - CKD predicted within 35 months</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color: #dcfce7; color: #15803d; padding: 1rem; border-radius: 8px; font-weight: 700;"><i class="fas fa-check-circle"></i> CLINICAL STATUS: LOW RISK - No CKD predicted within 35 months</div>', unsafe_allow_html=True)
        
        with col2:
            # Probability display
            prob_ckd = probability[1] * 100
            prob_no_ckd = probability[0] * 100
            
            st.metric("CKD Probability", f"{prob_ckd:.1f}%")
            st.metric("No CKD Probability", f"{prob_no_ckd:.1f}%")
        
        # Risk assessment
        st.markdown('<div class="icon-header"><i class="fas fa-heart-rate"></i> <b>Risk Assessment</b></div>', unsafe_allow_html=True)
        
        if prob_ckd < 20:
            risk_level = "Very Low"
            risk_color = "#15803d"
        elif prob_ckd < 40:
            risk_level = "Low"
            risk_color = "#ca8a04"
        elif prob_ckd < 60:
            risk_level = "Moderate"
            risk_color = "#ea580c"
        elif prob_ckd < 80:
            risk_level = "High"
            risk_color = "#b91c1c"
        else:
            risk_level = "Very High"
            risk_color = "#991b1b"
        
        st.markdown(f'<div style="font-weight: 700; color: {risk_color}; margin-bottom: 10px;"><i class="fas fa-microscope"></i> Risk Level: {risk_level}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-weight: 700; color: #1e3a8a;"><i class="fas fa-percentage"></i> CKD Probability: {prob_ckd:.1f}%</div>', unsafe_allow_html=True)
        
        # SHAP Explanations
        st.markdown('<div class="icon-header"><i class="fas fa-search-plus"></i> <b>Real SHAP Explanations - Clinical Transparency</b></div>', unsafe_allow_html=True)
        

        
        if force_img:
            st.markdown('<b><i class="fas fa-stethoscope"></i> Clinical Feature Impact Plot</b>', unsafe_allow_html=True)
            st.markdown("""
            **Clinical Interpretation:**
            - Ranked by absolute impact on the prediction result
            - Features pushing to the **right** (Red) increase predicted risk
            - Features pushing to the **left** (Blue) decrease predicted risk
            """)
            st.image(f"data:image/png;base64,{force_img}", use_container_width=True)
        

        
        # Feature contributions table
        if contributions_table is not None:
            st.markdown('<b><i class="fas fa-table-medical"></i> Detailed Feature Contributions</b>', unsafe_allow_html=True)
            st.markdown("""
            **Feature Contributions Table:**
            - Shows exact SHAP values for each feature
            - Positive values increase CKD risk
            - Negative values decrease CKD risk
            - Sorted by impact on prediction
            """)
            st.dataframe(contributions_table.head(15), use_container_width=True)
        
        # Recommendations
        st.markdown('<div class="icon-header"><i class="fas fa-lightbulb"></i> <b>Clinical Recommendations</b></div>', unsafe_allow_html=True)
        
        if prob_ckd > 50:
            st.markdown("""
            <div style="background-color: #f8fafc; border-left: 4px solid #1e40af; padding: 1rem; margin-top: 1rem;">
                <b><i class="fas fa-stethoscope"></i> Recommended Actions:</b><br>
                - Consult with a nephrologist immediately<br>
                - Comprehensive kidney function tests (eGFR, creatinine, urine analysis)<br>
                - Strict blood pressure control (<130/80 mmHg)<br>
                - Blood sugar management if diabetic<br>
                - Avoid nephrotoxic medications (NSAIDs, certain antibiotics)<br>
                - Consider dietary modifications (low protein, low sodium, low potassium)<br>
                - Regular monitoring every 3 months
            </div>
            """, unsafe_allow_html=True)
        elif prob_ckd > 30:
            st.markdown("""
            <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 1rem; margin-top: 1rem;">
                <b><i class="fas fa-user-md"></i> Recommended Actions:</b><br>
                - Consult primary care physician<br>
                - Annual kidney function screening<br>
                - Monitor blood pressure and blood sugar<br>
                - Maintain healthy lifestyle<br>
                - Review medications with healthcare provider
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f8fafc; border-left: 4px solid #22c55e; padding: 1rem; margin-top: 1rem;">
                <b><i class="fas fa-shield-alt"></i> Preventive Measures:</b><br>
                - Regular health check-ups<br>
                - Maintain healthy blood pressure (<120/80 mmHg)<br>
                - Control blood sugar levels<br>
                - Regular exercise and healthy diet<br>
                - Avoid smoking and limit alcohol<br>
                - Stay well hydrated<br>
                - Maintain healthy weight
            </div>
            """, unsafe_allow_html=True)
        
        # Display probability chart
        st.markdown('<div class="icon-header"><i class="fas fa-chart-area"></i> <b>Probability Distribution</b></div>', unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({
            'Outcome': ['No CKD', 'CKD'],
            'Probability': [prob_no_ckd, prob_ckd]
        })
        
        st.bar_chart(prob_df.set_index('Outcome'), use_container_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please check your input values and try again.")
        return False

if __name__ == "__main__":
    main()
