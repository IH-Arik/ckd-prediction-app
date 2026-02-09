import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the dataset"""
    print("=== LOADING AND CLEANING DATA ===")
    
    # Load dataset
    df = pd.read_csv('pone.0199920.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Replace '#NULL!' with NaN
    df = df.replace('#NULL!', np.nan)
    
    # Convert to numeric
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'StudyID':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    missing_before = df.isnull().sum().sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return df

def test_fixed_model_complete(df, model):
    """Test fixed model on complete dataset"""
    print("\n=== TESTING FIXED MODEL ON COMPLETE DATASET ===")
    
    # Prepare features and target
    X = df.drop(columns=['EventCKD35'])  # Keep StudyID
    y = df['EventCKD35']
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # Binary metrics for CKD class (class 1)
    binary_precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
    binary_recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
    binary_f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
    
    try:
        auc = roc_auc_score(y, y_proba)
    except:
        auc = 0.0
    
    print(f"\n=== FIXED MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    
    print(f"\n=== CKD CLASS (Positive) METRICS ===")
    print(f"CKD Precision: {binary_precision:.4f} ({binary_precision*100:.1f}%)")
    print(f"CKD Recall:    {binary_recall:.4f} ({binary_recall*100:.1f}%)")
    print(f"CKD F1-Score:  {binary_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"                 Predicted")
    print(f"                 No CKD  CKD")
    print(f"Actual No CKD:   {tn:4d}  {fp:4d}")
    print(f"Actual CKD:       {fn:4d}  {tp:4d}")
    
    print(f"\n=== DETAILED ANALYSIS ===")
    print(f"True Negatives (TN): {tn} - Correctly identified No CKD")
    print(f"False Positives (FP): {fp} - Incorrectly predicted CKD")
    print(f"False Negatives (FN): {fn} - MISSED CKD CASES")
    print(f"True Positives (TP): {tp} - Correctly identified CKD")
    
    # Calculate rates
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n=== CLINICAL METRICS ===")
    print(f"Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.1f}%)")
    print(f"Specificity:          {specificity:.4f} ({specificity*100:.1f}%)")
    print(f"Positive Predictive Value: {ppv:.4f} ({ppv*100:.1f}%)")
    print(f"Negative Predictive Value: {npv:.4f} ({npv*100:.1f}%)")
    
    # Baseline comparison
    baseline_accuracy = max(y.value_counts()) / len(y)
    improvement = accuracy - baseline_accuracy
    
    print(f"\n=== BASELINE COMPARISON ===")
    print(f"Baseline accuracy (always predict majority): {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
    print(f"Model accuracy:                           {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Improvement:                              {improvement:.4f} ({improvement*100:.1f}%)")
    
    if improvement > 0.01:
        print("Model shows meaningful improvement over baseline")
    else:
        print("Model shows minimal improvement over baseline")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ckd_precision': binary_precision,
        'ckd_recall': binary_recall,
        'ckd_f1': binary_f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def test_specific_ckd_cases(df, model):
    """Test specific CKD positive cases"""
    print("\n=== TESTING SPECIFIC CKD POSITIVE CASES ===")
    
    # Find CKD positive cases
    ckd_positive = df[df['EventCKD35'] == 1]
    print(f"Total CKD positive cases: {len(ckd_positive)}")
    
    # Prepare features
    X = df.drop(columns=['EventCKD35'])
    y = df['EventCKD35']
    
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Analyze CKD cases
    ckd_indices = np.where(y == 1)[0]
    correctly_detected = []
    missed_cases = []
    
    for idx in ckd_indices:
        study_id = df.iloc[idx]['StudyID']
        predicted = y_pred[idx]
        probability = y_proba[idx]
        
        if predicted == 1:
            correctly_detected.append((study_id, probability))
        else:
            missed_cases.append((study_id, probability))
    
    print(f"\nCKD Detection Results:")
    print(f"Correctly detected: {len(correctly_detected)}/{len(ckd_indices)} ({len(correctly_detected)/len(ckd_indices)*100:.1f}%)")
    print(f"Missed cases: {len(missed_cases)}/{len(ckd_indices)} ({len(missed_cases)/len(ckd_indices)*100:.1f}%)")
    
    if len(correctly_detected) > 0:
        print(f"\nCorrectly Detected CKD Cases:")
        for study_id, prob in correctly_detected[:10]:  # Show first 10
            print(f"  StudyID {study_id}: CKD=YES, Probability={prob:.3f}")
        if len(correctly_detected) > 10:
            print(f"  ... and {len(correctly_detected)-10} more")
    
    if len(missed_cases) > 0:
        print(f"\nMissed CKD Cases:")
        for study_id, prob in missed_cases[:10]:  # Show first 10
            print(f"  StudyID {study_id}: CKD=YES, Probability={prob:.3f} (MISSED)")
        if len(missed_cases) > 10:
            print(f"  ... and {len(missed_cases)-10} more")
    
    return len(correctly_detected), len(missed_cases)

def test_study_id_100(df, model):
    """Test StudyID 100 specifically"""
    print("\n=== TESTING STUDY ID 100 ===")
    
    # Find StudyID 100
    study_100 = df[df['StudyID'] == 100]
    
    if len(study_100) == 0:
        print("StudyID 100 not found in dataset")
        return
    
    # Get actual result
    actual_ckd = study_100['EventCKD35'].iloc[0]
    
    # Prepare features for prediction
    X_100 = study_100.drop(columns=['EventCKD35'])
    
    # Make prediction
    pred_100 = model.predict(X_100)[0]
    prob_100 = model.predict_proba(X_100)[0, 1]
    
    print(f"StudyID 100 Results:")
    print(f"  Actual CKD: {'YES' if actual_ckd == 1 else 'NO'}")
    print(f"  Predicted CKD: {'YES' if pred_100 == 1 else 'NO'}")
    print(f"  CKD Probability: {prob_100:.3f} ({prob_100*100:.1f}%)")
    
    if actual_ckd == 1 and pred_100 == 1:
        print("  Result: CORRECTLY DETECTED CKD")
    elif actual_ckd == 0 and pred_100 == 0:
        print("  Result: CORRECTLY IDENTIFIED NO CKD")
    elif actual_ckd == 1 and pred_100 == 0:
        print("  Result: MISSED CKD CASE (False Negative)")
    else:
        print("  Result: FALSE POSITIVE")
    
    return actual_ckd, pred_100, prob_100

def compare_with_old_model(df, new_model):
    """Compare new model with old model performance"""
    print("\n=== COMPARISON WITH OLD MODEL ===")
    
    try:
        # Load old model
        old_model = joblib.load('pone_disease_prediction_model.pkl')
        
        # Prepare data
        X = df.drop(columns=['EventCKD35'])
        y = df['EventCKD35']
        
        # Old model predictions
        old_pred = old_model.predict(X)
        old_tp = np.sum((old_pred == 1) & (y == 1))
        old_fn = np.sum((old_pred == 0) & (y == 1))
        old_recall = old_tp / (old_tp + old_fn) if (old_tp + old_fn) > 0 else 0
        
        # New model predictions
        new_pred = new_model.predict(X)
        new_tp = np.sum((new_pred == 1) & (y == 1))
        new_fn = np.sum((new_pred == 0) & (y == 1))
        new_recall = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
        
        print(f"Old Model CKD Detection: {old_tp}/{old_tp + old_fn} ({old_recall*100:.1f}%)")
        print(f"New Model CKD Detection: {new_tp}/{new_tp + new_fn} ({new_recall*100:.1f}%)")
        print(f"Improvement: {(new_recall - old_recall)*100:.1f} percentage points")
        
    except Exception as e:
        print(f"Could not load old model for comparison: {e}")

def main():
    """Main function to test fixed model"""
    print("="*60)
    print("TESTING FIXED CKD MODEL")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Load fixed model
    try:
        model = joblib.load('ckd_model_fixed.pkl')
        print("Fixed model loaded successfully")
    except Exception as e:
        print(f"Error loading fixed model: {e}")
        return
    
    # Test complete dataset
    results = test_fixed_model_complete(df, model)
    
    # Test specific CKD cases
    detected, missed = test_specific_ckd_cases(df, model)
    
    # Test StudyID 100
    test_study_id_100(df, model)
    
    # Compare with old model
    compare_with_old_model(df, model)
    
    # Final assessment
    print(f"\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if results['tp'] > 0:
        print("SUCCESS: Model now detects CKD cases!")
        print(f"CKD Detection Rate: {results['ckd_recall']*100:.1f}%")
        print(f"Total CKD cases detected: {results['tp']}")
        print(f"CKD cases missed: {results['fn']}")
        
        if results['ckd_recall'] > 0.7:
            print("Model performance is GOOD for clinical use")
        elif results['ckd_recall'] > 0.5:
            print("Model performance is ACCEPTABLE but could be improved")
        else:
            print("Model performance needs further improvement")
    else:
        print("Model still has issues detecting CKD cases")
    
    print(f"\nRecommendation: {'USE IN CLINICAL PRACTICE' if results['ckd_recall'] > 0.7 else 'FURTHER TUNING NEEDED'}")

if __name__ == "__main__":
    main()
