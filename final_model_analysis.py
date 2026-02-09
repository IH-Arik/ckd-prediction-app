import pandas as pd
import numpy as np

def clean_dataset():
    """Clean the dataset and handle missing values"""
    print("="*60)
    print("CLEANING AND ANALYZING DATASET")
    print("="*60)
    
    df = pd.read_csv('pone.0199920.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Replace '#NULL!' with NaN
    df = df.replace('#NULL!', np.nan)
    
    # Check missing values after replacement
    print(f"Missing values after cleaning:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Convert object columns to numeric where possible
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'StudyID':  # Keep StudyID as is
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values - for simplicity, use median for numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values after filling: {df.isnull().sum().sum()}")
    
    return df

def analyze_dataset(df):
    """Analyze the cleaned dataset"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    target_col = 'EventCKD35'
    y = df[target_col]
    X = df.drop(columns=[target_col, 'StudyID'])
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Features: {X.shape[1]}")
    print(f"Number of Samples: {df.shape[0]}")
    
    # Target variable analysis
    print(f"\nTarget Variable: {target_col}")
    print(f"Class Distribution:")
    print(f"  Class 0 (No CKD): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Class 1 (CKD):     {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Imbalance Ratio: 1:{sum(y == 0)/sum(y == 1):.1f}")
    
    # Baseline accuracy
    majority_class = y.value_counts().idxmax()
    baseline_accuracy = sum(y == majority_class) / len(y)
    
    print(f"\nBaseline Performance:")
    print(f"  Majority Class: {majority_class}")
    print(f"  Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
    
    # Feature correlations with target
    print(f"\nFeature Correlations with Target:")
    correlations = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            corr = np.abs(X[col].corr(y))
            correlations[col] = corr
    
    # Sort by correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 Most Correlated Features:")
    for i, (feature, corr) in enumerate(sorted_correlations[:10]):
        print(f"  {i+1:2d}. {feature}: {corr:.3f}")
    
    return X, y, baseline_accuracy

def model_performance_expectations(baseline_accuracy, y):
    """Provide realistic model performance expectations"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EXPECTATIONS")
    print("="*60)
    
    print(f"Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
    
    print(f"\nPerformance Benchmarks:")
    print(f"  Random Guess:     ~50.0%")
    print(f"  Baseline Model:   {baseline_accuracy*100:.1f}% (always predict majority)")
    print(f"  Simple Model:     {baseline_accuracy*100 + 2:.1f}% - {baseline_accuracy*100 + 5:.1f}%")
    print(f"  Good Model:       {baseline_accuracy*100 + 5:.1f}% - {baseline_accuracy*100 + 10:.1f}%")
    print(f"  Excellent Model:  > {baseline_accuracy*100 + 10:.1f}%")
    
    print(f"\nKey Challenges:")
    print(f"  - High class imbalance (1:{sum(y == 0)/sum(y == 1):.1f} ratio)")
    print(f"  - Need for proper evaluation metrics (F1, AUC, not just accuracy)")
    print(f"  - Should use techniques like SMOTE, class weights")

def model_loading_instructions():
    """Provide clear instructions for loading the model"""
    print("\n" + "="*60)
    print("MODEL LOADING INSTRUCTIONS")
    print("="*60)
    
    print("To load and test your pone_disease_prediction_model.pkl:")
    print("\n1. Install required packages:")
    print("   pip install scikit-learn pandas numpy imbalanced-learn")
    
    print("\n2. Use this code to load and test:")
    print("""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Clean and prepare data
df = pd.read_csv('pone.0199920.csv')
df = df.replace('#NULL!', np.nan)

# Convert to numeric
for col in df.select_dtypes(include=['object']).columns:
    if col != 'StudyID':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Prepare features and target
X = df.drop(columns=['EventCKD35', 'StudyID'])
y = df['EventCKD35']

# Load model
try:
    with open('pone_disease_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    
    # Make predictions
    y_pred = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
        print("Model provides probability estimates")
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    print(f"\\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\\nClassification Report:")
    print(classification_report(y, y_pred))
    
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure imbalanced-learn is installed: pip install imbalanced-learn")
""")

def main():
    # Clean and analyze dataset
    df_clean = clean_dataset()
    X, y, baseline_acc = analyze_dataset(df_clean)
    
    # Provide performance expectations
    model_performance_expectations(baseline_acc, y)
    
    # Give loading instructions
    model_loading_instructions()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print("1. Dataset: 491 samples, 24 features")
    print("2. Target: CKD occurrence within 35 months")
    print("3. Imbalance: 1:7.8 ratio (88.6% vs 11.4%)")
    print("4. Baseline accuracy: 88.6% (always predict 'no CKD')")
    print("5. Key predictors: eGFR, Creatinine, Age, Diabetes")
    print("6. Model requires imbalanced-learn to load")
    print("7. Good model should achieve >93% accuracy")
    print("8. Focus on F1-score and AUC, not just accuracy")

if __name__ == "__main__":
    main()
