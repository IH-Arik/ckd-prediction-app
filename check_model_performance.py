import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn metrics
try:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix)
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. Basic evaluation only.")
    SKLEARN_AVAILABLE = False

def simple_accuracy(y_true, y_pred):
    """Simple accuracy calculation without sklearn"""
    return np.mean(y_true == y_pred)

def main():
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Load the model
    try:
        with open('pone_disease_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        if hasattr(model, 'named_steps'):
            print("Model pipeline steps:")
            for name, step in model.named_steps.items():
                print(f"  - {name}: {type(step).__name__}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load the dataset
    try:
        df = pd.read_csv('pone.0199920.csv')
        print("\n✓ Dataset loaded successfully")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Find target column
        target_col = None
        for col in df.columns:
            if 'class' in col.lower() or 'target' in col.lower() or 'disease' in col.lower() or 'ckd' in col.lower():
                target_col = col
                break
        
        if target_col:
            print(f"Target column identified: {target_col}")
        else:
            # Assume last column is target
            target_col = df.columns[-1]
            print(f"Assuming last column as target: {target_col}")
        
        # Show basic info
        print(f"\nTarget distribution:")
        print(df[target_col].value_counts())
        print(f"\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Prepare data for prediction
    try:
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Make predictions
        y_pred = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            print("✓ Model has predict_proba method")
        else:
            y_proba = None
            print("⚠ Model does not have predict_proba method")
        
        print(f"\nPredictions shape: {y_pred.shape}")
        print(f"Unique predictions: {np.unique(y_pred)}")
        
        # Calculate metrics
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            print(f"\n{'='*40}")
            print("PERFORMANCE METRICS")
            print(f"{'='*40}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            if y_proba is not None and len(np.unique(y)) == 2:
                try:
                    auc = roc_auc_score(y, y_proba)
                    print(f"ROC AUC:   {auc:.4f}")
                except:
                    print("ROC AUC:   Could not calculate")
            
            # Detailed classification report
            print(f"\n{'='*40}")
            print("CLASSIFICATION REPORT")
            print(f"{'='*40}")
            print(classification_report(y, y_pred))
            
            # Confusion matrix
            print(f"{'='*40}")
            print("CONFUSION MATRIX")
            print(f"{'='*40}")
            cm = confusion_matrix(y, y_pred)
            print(cm)
            
        else:
            # Simple metrics without sklearn
            accuracy = simple_accuracy(y, y_pred)
            print(f"\nSimple Accuracy: {accuracy:.4f}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            print(f"\n{'='*40}")
            print("FEATURE IMPORTANCE")
            print(f"{'='*40}")
            importances = model.feature_importances_
            feature_names = X.columns
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            print("Top 10 most important features:")
            for i in range(min(10, len(indices))):
                print(f"{i+1:2d}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
