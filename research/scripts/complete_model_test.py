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

def test_model_on_dataset(df, model):
    """Test model on entire dataset"""
    print("\n=== TESTING MODEL ON ENTIRE DATASET ===")
    
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
    
    print(f"\n=== OVERALL PERFORMANCE METRICS ===")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    
    print(f"\n=== CKD CLASS (Positive) METRICS ===")
    print(f"CKD Precision: {binary_precision:.4f}")
    print(f"CKD Recall:    {binary_recall:.4f}")
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
    print(f"False Negatives (FN): {fn} - MISSED CKD CASES!")
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
    
    if improvement < 0:
        print("WARNING: Model performs WORSE than baseline!")
    elif improvement < 0.01:
        print("WARNING: Model shows minimal improvement over baseline")
    else:
        print("Model shows meaningful improvement")
    
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

def analyze_prediction_distribution(y_true, y_pred, y_proba):
    """Analyze the distribution of predictions"""
    print(f"\n=== PREDICTION DISTRIBUTION ANALYSIS ===")
    
    # Overall prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"Prediction distribution:")
    for val, count in zip(unique, counts):
        label = "No CKD" if val == 0 else "CKD"
        print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # Probability distribution for actual CKD cases
    ckd_indices = np.where(y_true == 1)[0]
    if len(ckd_indices) > 0:
        ckd_probs = y_proba[ckd_indices]
        print(f"\nCKD cases probability distribution:")
        print(f"  Mean probability: {np.mean(ckd_probs):.4f}")
        print(f"  Max probability: {np.max(ckd_probs):.4f}")
        print(f"  Min probability: {np.min(ckd_probs):.4f}")
        print(f"  Cases with prob > 0.5: {np.sum(ckd_probs > 0.5)}")
        print(f"  Cases with prob > 0.3: {np.sum(ckd_probs > 0.3)}")
        print(f"  Cases with prob > 0.1: {np.sum(ckd_probs > 0.1)}")
    
    # Probability distribution for actual No CKD cases
    no_ckd_indices = np.where(y_true == 0)[0]
    if len(no_ckd_indices) > 0:
        no_ckd_probs = y_proba[no_ckd_indices]
        print(f"\nNo CKD cases probability distribution:")
        print(f"  Mean probability: {np.mean(no_ckd_probs):.4f}")
        print(f"  Max probability: {np.max(no_ckd_probs):.4f}")
        print(f"  Min probability: {np.min(no_ckd_probs):.4f}")

def main():
    """Main function to run complete model test"""
    print("=" * 60)
    print("COMPLETE MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Load model
    try:
        model = joblib.load('pone_disease_prediction_model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test model
    results = test_model_on_dataset(df, model)
    
    # Additional analysis
    X = df.drop(columns=['EventCKD35'])  # Keep StudyID
    y = df['EventCKD35']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    analyze_prediction_distribution(y, y_pred, y_proba)
    
    # Final assessment
    print(f"\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    if results['tp'] == 0:
        print("CRITICAL FAILURE: Model detects 0 CKD cases!")
        print("Model is NOT clinically usable")
        print("100% False Negative Rate")
    elif results['ckd_recall'] < 0.5:
        print("POOR PERFORMANCE: Low CKD detection rate")
        print("Model needs significant improvement")
    elif results['ckd_recall'] < 0.7:
        print("MODERATE PERFORMANCE: Acceptable but could be better")
    else:
        print("GOOD PERFORMANCE: Model detects CKD effectively")
    
    print(f"\nRecommendation: {'RETRAIN MODEL' if results['tp'] == 0 else 'MODEL NEEDS IMPROVEMENT' if results['ckd_recall'] < 0.7 else 'MODEL IS USABLE'}")

if __name__ == "__main__":
    main()
