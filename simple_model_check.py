import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Load the dataset first to understand structure
    try:
        df = pd.read_csv('pone.0199920.csv')
        print("Dataset loaded successfully")
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
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Try to load model with different approach
    try:
        # First try to see if we can import required modules
        import sys
        import importlib
        
        # Try to import imblearn
        try:
            import imblearn
            print("imblearn is available")
        except ImportError:
            print("imblearn is not available - cannot load model")
            print("Model was trained with imblearn dependencies")
            print("Please install imblearn: pip install imbalanced-learn")
            return
        
        # Now try to load the model
        import pickle
        with open('pone_disease_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        if hasattr(model, 'named_steps'):
            print("Model pipeline steps:")
            for name, step in model.named_steps.items():
                print(f"  - {name}: {type(step).__name__}")
        
        # Try to make predictions
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Make predictions
        y_pred = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            print("Model has predict_proba method")
        else:
            y_proba = None
            print("Model does not have predict_proba method")
        
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Unique predictions: {np.unique(y_pred)}")
        
        # Calculate basic metrics
        accuracy = np.mean(y == y_pred)
        print(f"\nBasic Accuracy: {accuracy:.4f}")
        
        # Try sklearn metrics
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                     roc_auc_score, classification_report, confusion_matrix)
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            print(f"\n{'='*40}")
            print("DETAILED PERFORMANCE METRICS")
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
            
            print(f"\n{'='*40}")
            print("CLASSIFICATION REPORT")
            print(f"{'='*40}")
            print(classification_report(y, y_pred))
            
            print(f"{'='*40}")
            print("CONFUSION MATRIX")
            print(f"{'='*40}")
            cm = confusion_matrix(y, y_pred)
            print(cm)
            
        except ImportError:
            print("sklearn not available for detailed metrics")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
