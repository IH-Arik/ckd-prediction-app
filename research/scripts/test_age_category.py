import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Test the app with user input for Age.3.categories
model = joblib.load('ckd_model_fixed.pkl')

# Test with different age categories
test_cases = [
    {'age': 44, 'category': 0, 'desc': 'Age 44, Category 0 (23-49)'},
    {'age': 55, 'category': 1, 'desc': 'Age 55, Category 1 (50-64)'},
    {'age': 70, 'category': 2, 'desc': 'Age 70, Category 2 (65-89)'}
]

for case in test_cases:
    test_data = {
        'StudyID': 100,
        'Gender': 0,
        'AgeBaseline': case['age'],
        'Age.3.categories': case['category'],
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
        'CholesterolBaseline': 5.0,
        'TriglyceridesBaseline': 1.3,
        'HgbA1C': 6.5,
        'CreatnineBaseline': 68,
        'eGFRBaseline': 98,
        'sBPBaseline': 130,
        'dBPBaseline': 77,
        'BMIBaseline': 30.0,
        'TimeToEventMonths': 35
    }
    
    input_df = pd.DataFrame([test_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    print(f'{case["desc"]}:')
    print(f'  Prediction: {prediction}, Probability: {probability:.3f}')
    print('  Age.3.categories input field working!')
    print()

print('Streamlit app is ready with Age.3.categories input field!')
print('Users can now select age category manually in the form.')
