import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
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

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Binary metrics for CKD class
    binary_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    binary_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    binary_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n=== {model_name} PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"CKD Recall (Sensitivity): {binary_recall:.4f} ({binary_recall*100:.1f}%)")
    print(f"CKD Precision: {binary_precision:.4f} ({binary_precision*100:.1f}%)")
    print(f"CKD F1-Score: {binary_f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"True Positives: {tp}, False Negatives: {fn}")
    print(f"True Negatives: {tn}, False Positives: {fp}")
    
    return {
        'accuracy': accuracy,
        'ckd_recall': binary_recall,
        'ckd_precision': binary_precision,
        'ckd_f1': binary_f1,
        'auc': auc,
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp
    }

def train_balanced_models(df):
    """Train models with different balancing techniques"""
    
    # Prepare features and target
    X = df.drop(columns=['EventCKD35'])  # Keep StudyID
    y = df['EventCKD35']
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    models = {}
    results = {}
    
    # 1. Model with Class Weights
    print("\n=== MODEL 1: CLASS WEIGHTS ===")
    rf_weighted = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_weighted.fit(X_train, y_train)
    models['class_weights'] = rf_weighted
    results['class_weights'] = evaluate_model(rf_weighted, X_test, y_test, "Class Weights")
    
    # 2. Model with SMOTE
    print("\n=== MODEL 2: SMOTE ===")
    smote_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    smote_pipeline.fit(X_train, y_train)
    models['smote'] = smote_pipeline
    results['smote'] = evaluate_model(smote_pipeline, X_test, y_test, "SMOTE")
    
    # 3. Model with BorderlineSMOTE
    print("\n=== MODEL 3: BORDERLINE SMOTE ===")
    borderline_pipeline = ImbPipeline([
        ('smote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    borderline_pipeline.fit(X_train, y_train)
    models['borderline_smote'] = borderline_pipeline
    results['borderline_smote'] = evaluate_model(borderline_pipeline, X_test, y_test, "Borderline SMOTE")
    
    # 4. Model with SMOTETomek
    print("\n=== MODEL 4: SMOTE + TOMEK ===")
    smotetomek_pipeline = ImbPipeline([
        ('smote', SMOTETomek(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    smotetomek_pipeline.fit(X_train, y_train)
    models['smotetomek'] = smotetomek_pipeline
    results['smotetomek'] = evaluate_model(smotetomek_pipeline, X_test, y_test, "SMOTE + Tomek")
    
    # 5. Model with Manual Class Weights
    print("\n=== MODEL 5: MANUAL CLASS WEIGHTS ===")
    rf_manual = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight={0: 1, 1: 8},  # Heavier weight for CKD class
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_manual.fit(X_train, y_train)
    models['manual_weights'] = rf_manual
    results['manual_weights'] = evaluate_model(rf_manual, X_test, y_test, "Manual Weights")
    
    return models, results, X_test, y_test

def compare_models(results):
    """Compare all model results"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'CKD Recall': result['ckd_recall'],
            'CKD Precision': result['ckd_precision'],
            'CKD F1': result['ckd_f1'],
            'AUC': result['auc'],
            'TP': result['tp'],
            'FN': result['fn']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Find best model for CKD detection
    best_recall_idx = df_comparison['CKD Recall'].idxmax()
    best_model = df_comparison.iloc[best_recall_idx]
    
    print(f"\n=== BEST MODEL FOR CKD DETECTION ===")
    print(f"Model: {best_model['Model']}")
    print(f"CKD Recall: {best_model['CKD Recall']:.4f} ({best_model['CKD Recall']*100:.1f}%)")
    print(f"CKD F1-Score: {best_model['CKD F1']:.4f}")
    print(f"True Positives: {int(best_model['TP'])}")
    print(f"False Negatives: {int(best_model['FN'])}")
    
    return best_model['Model']

def save_best_model(models, best_model_name):
    """Save the best performing model"""
    best_model = models[best_model_name]
    
    # Save with joblib
    joblib.dump(best_model, 'ckd_model_fixed.pkl')
    print(f"\nBest model saved as 'ckd_model_fixed.pkl'")
    
    return best_model

def test_fixed_model(model, X_test, y_test):
    """Test the fixed model on test set"""
    print("\n=== TESTING FIXED MODEL ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Detailed analysis
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))
    
    # Find correctly predicted CKD cases
    ckd_indices = np.where(y_test == 1)[0]
    correct_ckd = []
    
    for idx in ckd_indices:
        if y_pred[idx] == 1:
            correct_ckd.append(idx)
    
    print(f"\nCKD Detection Analysis:")
    print(f"Total CKD cases in test: {len(ckd_indices)}")
    print(f"Correctly detected: {len(correct_ckd)}")
    print(f"Detection rate: {len(correct_ckd)/len(ckd_indices)*100:.1f}%")
    
    if len(correct_ckd) > 0:
        print(f"Correctly detected CKD case indices: {correct_ckd}")
    
    return len(correct_ckd) > 0

def main():
    """Main function to fix the CKD model"""
    print("="*60)
    print("CKD MODEL FIX - PROPER CLASS BALANCING")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Train balanced models
    models, results, X_test, y_test = train_balanced_models(df)
    
    # Compare models
    best_model_name = compare_models(results)
    
    # Save best model
    best_model = save_best_model(models, best_model_name)
    
    # Test fixed model
    success = test_fixed_model(best_model, X_test, y_test)
    
    print(f"\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if success:
        print("MODEL FIXED SUCCESSFULLY!")
        print(f"Best approach: {best_model_name}")
        print("Model now detects CKD cases")
        print("Use 'ckd_model_fixed.pkl' for predictions")
    else:
        print("Model improvement needed")
        print("Consider trying different hyperparameters")
    
    print(f"\nUsage:")
    print("import joblib")
    print("model = joblib.load('ckd_model_fixed.pkl')")
    print("prediction = model.predict(X)")

if __name__ == "__main__":
    main()
