import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('pone_disease_prediction_model.pkl')

# StudyID 100 data from image
data_100 = {
    'StudyID': 100,
    'Gender': 0,
    'AgeBaseline': 44,
    'Age.3.categories': 2,
    'HistoryDiabetes': 0,
    'HistoryCHD': 0,
    'HistoryVascular': 0,
    'HistorySmoking': 0,
    'HistoryHTN ': 1,
    'HistoryDLD': 0,
    'HistoryObesity': 0,
    'DLDmeds': 1,
    'DMmeds': 1,
    'HTNmeds': 1,
    'ACEIARB': 1,
    'CholesterolBaseline': 0.97,
    'TriglyceridesBaseline': 4.2,
    'HgbA1C': 5.5,
    'CreatnineBaseline': 99.7,
    'eGFRBaseline': 99.7,
    'sBPBaseline': 105.4,
    'dBPBaseline': 142,
    'BMIBaseline': 39,
    'TimeToEventMonths': 100
}

print('=== STUDY ID 100 CKD PREDICTION TEST ===')
print('Patient Data:')
for key, value in data_100.items():
    print(f'  {key}: {value}')

# Convert to DataFrame
input_df = pd.DataFrame([data_100])

# Make prediction
try:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    prob_ckd = probability[1] * 100
    prob_no_ckd = probability[0] * 100
    
    print(f'\n=== PREDICTION RESULTS ===')
    if prediction == 1:
        print('CKD Prediction: YES')
    else:
        print('CKD Prediction: NO')
    print(f'CKD Probability: {prob_ckd:.1f}%')
    print(f'No CKD Probability: {prob_no_ckd:.1f}%')
    
    if prediction == 1:
        print('HIGH RISK: CKD predicted')
    else:
        print('LOW RISK: No CKD predicted')
    
    # Risk assessment
    if prob_ckd < 20:
        risk_level = 'Very Low'
    elif prob_ckd < 40:
        risk_level = 'Low'
    elif prob_ckd < 60:
        risk_level = 'Moderate'
    elif prob_ckd < 80:
        risk_level = 'High'
    else:
        risk_level = 'Very High'
    
    print(f'Risk Level: {risk_level}')
    
    # Key factors analysis
    print(f'\n=== KEY FACTORS ANALYSIS ===')
    print(f'Age: {data_100["AgeBaseline"]} years')
    print(f'Hypertension: Yes')
    print(f'Diabetes Meds: Yes')
    print(f'Hypertension Meds: Yes')
    print(f'ACEI/ARB: Yes')
    print(f'HbA1c: {data_100["HgbA1C"]}% (Normal)')
    print(f'BMI: {data_100["BMIBaseline"]} (High)')
    print(f'eGFR: {data_100["eGFRBaseline"]} (Normal)')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
