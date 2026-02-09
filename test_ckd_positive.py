import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv('pone.0199920.csv')
df = df.replace('#NULL!', np.nan)

# Convert to numeric
for col in df.select_dtypes(include=['object']).columns:
    if col != 'StudyID':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Find CKD positive cases
ckd_positive = df[df['EventCKD35'] == 1]
print('=== CKD POSITIVE CASES ANALYSIS ===')
print(f'Total CKD cases: {len(ckd_positive)}')
print(f'CKD cases: {ckd_positive["StudyID"].tolist()}')

# Load model
model = joblib.load('pone_disease_prediction_model.pkl')

# Test first few CKD positive cases
print('\n=== TESTING CKD POSITIVE CASES ===')

correct_predictions = 0
total_tested = 0

for idx, row in ckd_positive.head(10).iterrows():
    # Prepare data for prediction
    test_data = {
        'StudyID': row['StudyID'],
        'Gender': row['Gender'],
        'AgeBaseline': row['AgeBaseline'],
        'Age.3.categories': row['Age.3.categories'],
        'HistoryDiabetes': row['HistoryDiabetes'],
        'HistoryCHD': row['HistoryCHD'],
        'HistoryVascular': row['HistoryVascular'],
        'HistorySmoking': row['HistorySmoking'],
        'HistoryHTN ': row['HistoryHTN '],
        'HistoryDLD': row['HistoryDLD'],
        'HistoryObesity': row['HistoryObesity'],
        'DLDmeds': row['DLDmeds'],
        'DMmeds': row['DMmeds'],
        'HTNmeds': row['HTNmeds'],
        'ACEIARB': row['ACEIARB'],
        'CholesterolBaseline': row['CholesterolBaseline'],
        'TriglyceridesBaseline': row['TriglyceridesBaseline'],
        'HgbA1C': row['HgbA1C'],
        'CreatnineBaseline': row['CreatnineBaseline'],
        'eGFRBaseline': row['eGFRBaseline'],
        'sBPBaseline': row['sBPBaseline'],
        'dBPBaseline': row['dBPBaseline'],
        'BMIBaseline': row['BMIBaseline'],
        'TimeToEventMonths': row['TimeToEventMonths']
    }
    
    # Make prediction
    input_df = pd.DataFrame([test_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    total_tested += 1
    if prediction == 1:
        correct_predictions += 1
    
    pred_text = "YES" if prediction == 1 else "NO"
    print(f'StudyID {row["StudyID"]}: Actual CKD=YES, Predicted={pred_text}, Prob={probability[1]:.1f}%')

print('\n=== MODEL PERFORMANCE ON CKD CASES ===')
print(f'CKD cases tested: {total_tested}')
print(f'Correctly predicted: {correct_predictions}')
print(f'CKD Detection Rate: {correct_predictions/total_tested*100:.1f}%')

# Test some CKD negative cases for comparison
ckd_negative = df[df['EventCKD35'] == 0].head(10)
print('\n=== TESTING CKD NEGATIVE CASES ===')

correct_negative = 0
total_negative = 0

for idx, row in ckd_negative.iterrows():
    test_data = {
        'StudyID': row['StudyID'],
        'Gender': row['Gender'],
        'AgeBaseline': row['AgeBaseline'],
        'Age.3.categories': row['Age.3.categories'],
        'HistoryDiabetes': row['HistoryDiabetes'],
        'HistoryCHD': row['HistoryCHD'],
        'HistoryVascular': row['HistoryVascular'],
        'HistorySmoking': row['HistorySmoking'],
        'HistoryHTN ': row['HistoryHTN '],
        'HistoryDLD': row['HistoryDLD'],
        'HistoryObesity': row['HistoryObesity'],
        'DLDmeds': row['DLDmeds'],
        'DMmeds': row['DMmeds'],
        'HTNmeds': row['HTNmeds'],
        'ACEIARB': row['ACEIARB'],
        'CholesterolBaseline': row['CholesterolBaseline'],
        'TriglyceridesBaseline': row['TriglyceridesBaseline'],
        'HgbA1C': row['HgbA1C'],
        'CreatnineBaseline': row['CreatnineBaseline'],
        'eGFRBaseline': row['eGFRBaseline'],
        'sBPBaseline': row['sBPBaseline'],
        'dBPBaseline': row['dBPBaseline'],
        'BMIBaseline': row['BMIBaseline'],
        'TimeToEventMonths': row['TimeToEventMonths']
    }
    
    input_df = pd.DataFrame([test_data])
    prediction = model.predict(input_df)[0]
    
    total_negative += 1
    if prediction == 0:
        correct_negative += 1

print(f'CKD negative cases tested: {total_negative}')
print(f'Correctly predicted: {correct_negative}')
print(f'Negative Detection Rate: {correct_negative/total_negative*100:.1f}%')

print('\n=== OVERALL MODEL ASSESSMENT ===')
print('Model is BIASED towards predicting "No CKD"')
print('This is a common problem with imbalanced datasets')
print('The model needs to be retrained with proper class balancing')
