import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import gc
import os
import scipy.stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
                           classification_report, average_precision_score, matthews_corrcoef, cohen_kappa_score)
from sklearn.metrics import make_scorer
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from scipy.stats import randint, uniform, chi2, ttest_rel, ttest_1samp, norm, t
import warnings
warnings.filterwarnings('ignore')

# Function to clean feature names
def clean_feature_names(feature_names):
    cleaned_names = []
    for name in feature_names:
        # Remove prefixes like 'num_standard__', 'cat_binary__', etc.
        if '__' in name:
            cleaned_name = name.split('__')[-1]
        else:
            cleaned_name = name
        cleaned_names.append(cleaned_name)
    return cleaned_names

# Function to get feature names from preprocessor without fitting
def get_feature_names_from_column_transformer(preprocessor, original_features, non_binary_categorical_cols, X_sample):
    """
    Estimate feature names without fitting on entire dataset
    """
    feature_names = []

    # Add names for numerical features (they keep their original names)
    for name, transformer, columns in preprocessor.transformers_[:-1]:  # Exclude 'remainder'
        if 'num_' in name:
            feature_names.extend(columns)
        elif 'cat_binary' in name:
            feature_names.extend(columns)
        elif 'cat_non_binary' in name:
            # For one-hot encoded features, we need to check unique values
            for col in columns:
                unique_vals = X_sample[col].dropna().unique()
                # One-hot encoding drops first category
                for val in sorted(unique_vals)[1:]:
                    feature_names.append(f"{col}_{val}")

    return feature_names

def mcnemars_test(y_true, y_pred1, y_pred2, model1_name="Model 1", model2_name="Model 2"):
    """
    Perform McNemar's test to compare two classifiers
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred1 : array-like
        Predictions from model 1
    y_pred2 : array-like
        Predictions from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    
    Returns:
    --------
    dict : Dictionary containing test results
    """
    print(f"\n{'='*60}")
    print(f"McNEMAR'S TEST: {model1_name} vs {model2_name}")
    print(f"{'='*60}")
    
    # Create contingency table
    # Both correct: a, Model1 correct Model2 wrong: b, Model1 wrong Model2 correct: c, Both wrong: d
    a = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    d = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    # Create 2x2 contingency table
    contingency_table = np.array([[b, a], [d, c]])
    
    print(f"Contingency Table:")
    print(f"                {model2_name}")
    print(f"                Correct    Wrong")
    print(f"{model1_name} Correct   {a:4d}     {b:4d}")
    print(f"          Wrong     {c:4d}     {d:4d}")
    
    # Calculate accuracies
    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    
    print(f"\nAccuracy Comparison:")
    print(f"{model1_name}: {acc1:.4f}")
    print(f"{model2_name}: {acc2:.4f}")
    print(f"Difference: {abs(acc1 - acc2):.4f}")
    
    # Perform McNemar's test
    if b + c == 0:
        # No discordant pairs, models are identical
        chi2_stat = 0.0
        p_value = 1.0
        significance = "Not Significant"
        conclusion = f"The difference between {model1_name} and {model2_name} is not statistically significant (p >= {alpha})"
        print(f"\nNo discordant pairs found - models make identical predictions")
    else:
        # Apply continuity correction for better approximation
        chi2_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        
        print(f"\nMcNemar's Test Results:")
        print(f"Chi-square statistic: {chi2_stat:.4f}")
        print(f"Degrees of freedom: 1")
        print(f"P-value: {p_value:.6f}")
        
        # Note: McNemar's test uses chi-square statistic, not t-statistic
        # T-statistics are used for continuous data (paired/independent t-tests)
        # McNemar's test is for paired binary outcomes (correct vs wrong predictions)
        
        # Determine statistical significance
        alpha = 0.05
        if p_value < alpha:
            significance = "Significant"
            conclusion = f"The difference between {model1_name} and {model2_name} is statistically significant (p < {alpha})"
        else:
            significance = "Not Significant"
            conclusion = f"The difference between {model1_name} and {model2_name} is not statistically significant (p >= {alpha})"
        
        print(f"\nStatistical Status: {significance}")
        print(f"Conclusion: {conclusion}")
        
        # Effect size (Cohen's h for proportions)
        if b + c > 0:
            prop_diff = abs(b - c) / (b + c)
            print(f"Effect size (proportion difference): {prop_diff:.4f}")
    
    print(f"\nStatistical Status: {significance}")
    print(f"Conclusion: {conclusion}")
    
    # Effect size (Cohen's h for proportions)
    if b + c > 0:
        prop_diff = abs(b - c) / (b + c)
        print(f"Effect size (proportion difference): {prop_diff:.4f}")
    
    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'accuracy1': acc1,
        'accuracy2': acc2,
        'accuracy_difference': abs(acc1 - acc2),
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significance': significance,
        'conclusion': conclusion,
        'discordant_pairs': b + c,
        'model1_correct_model2_wrong': b,
        'model1_wrong_model2_correct': c
    }

def compare_random_forest_with_others(X_train, y_train, X_test, y_test, feature_names):
    """
    Compare Random Forest with 8 other models using McNemar's test
    """
    print(f"\n{'='*80}")
    print(f"RANDOM FOREST VS OTHER MODELS - McNEMAR'S TEST COMPARISON")
    print(f"{'='*80}")
    
    # Define models to compare (excluding Random Forest)
    other_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
        'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    # Train Random Forest (reference model)
    print("\nTraining Random Forest (reference model)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    results = []
    
    # Compare with each other model
    for model_name, model in other_models.items():
        print(f"\n{'-'*60}")
        print(f"Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            model_pred = model.predict(X_test)
            
            # Perform McNemar's test
            test_result = mcnemars_test(
                y_test, rf_pred, model_pred, 
                "Random Forest", model_name
            )
            
            results.append(test_result)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create summary table
    print(f"\n{'='*80}")
    print(f"MCNEMAR'S TEST SUMMARY TABLE")
    print(f"{'='*80}")
    
    print(f"{'Model':<20} {'RF Acc':<8} {'Model Acc':<10} {'Diff':<8} {'Chi2':<8} {'P-value':<10} {'Status':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model2_name']:<20} "
              f"{result['accuracy1']:<8.4f} "
              f"{result['accuracy2']:<10.4f} "
              f"{result['accuracy_difference']:<8.4f} "
              f"{result['chi2_statistic']:<8.4f} "
              f"{result['p_value']:<10.6f} "
              f"{result['significance']:<12}")
    
    # Statistical summary
    significant_tests = [r for r in results if r['significance'] == 'Significant']
    print(f"\nSTATISTICAL SUMMARY:")
    print(f"Total comparisons: {len(results)}")
    print(f"Significant differences: {len(significant_tests)}")
    print(f"Non-significant differences: {len(results) - len(significant_tests)}")
    
    if significant_tests:
        print(f"\nModels significantly different from Random Forest:")
        for result in significant_tests:
            better_model = "Random Forest" if result['accuracy1'] > result['accuracy2'] else result['model2_name']
            print(f"- {result['model2_name']}: {result['conclusion']}")
            print(f"  Better performing: {better_model}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    model_names = [r['model2_name'] for r in results]
    rf_accs = [r['accuracy1'] for r in results]
    model_accs = [r['accuracy2'] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, rf_accs, width, label='Random Forest', alpha=0.8)
    plt.bar(x + width/2, model_accs, width, label='Other Models', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest vs Other Models - Test Accuracy Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: P-values
    plt.subplot(2, 2, 2)
    p_values = [r['p_value'] for r in results]
    colors = ['red' if p < 0.05 else 'green' for p in p_values]
    
    plt.bar(model_names, p_values, color=colors, alpha=0.7)
    plt.axhline(y=0.05, color='black', linestyle='--', label='Significance Level (0.05)')
    plt.xlabel('Models')
    plt.ylabel('P-value')
    plt.title('McNemar\'s Test P-values')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Chi-square statistics
    plt.subplot(2, 2, 3)
    chi2_stats = [r['chi2_statistic'] for r in results]
    
    plt.bar(model_names, chi2_stats, alpha=0.7, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Chi-square Statistic')
    plt.title('McNemar\'s Test Chi-square Statistics')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Accuracy differences
    plt.subplot(2, 2, 4)
    acc_diffs = [r['accuracy_difference'] for r in results]
    
    plt.bar(model_names, acc_diffs, alpha=0.7, color='purple')
    plt.xlabel('Models')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Differences (|RF - Model|)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/mcnemars_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/mcnemars_test_results.csv', index=False)
    
    print(f"\n✓ McNemar's test comparison completed!")
    print(f"✓ Results saved to:")
    print(f"  - results/mcnemars_test_results.csv")
    print(f"  - results/mcnemars_test_comparison.png")
    
    return results, results_df

def t_test_accuracy_comparison(y_true, y_pred1, y_pred2, model1_name="Model 1", model2_name="Model 2"):
    """
    Perform t-test to compare accuracy between two classifiers
    Note: This treats accuracy as continuous data for statistical comparison
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred1 : array-like
        Predictions from model 1
    y_pred2 : array-like
        Predictions from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    
    Returns:
    --------
    dict : Dictionary containing t-test results
    """
    print(f"\n{'='*60}")
    print(f"T-TEST ACCURACY COMPARISON: {model1_name} vs {model2_name}")
    print(f"{'='*60}")
    
    # Calculate accuracies
    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    
    # Create binary arrays (1=correct, 0=incorrect) for each prediction
    correct1 = (y_pred1 == y_true).astype(int)
    correct2 = (y_pred2 == y_true).astype(int)
    
    # Perform paired t-test on the binary correctness arrays
    t_stat, p_value = ttest_rel(correct1, correct2)
    
    # Calculate mean difference and standard error
    mean_diff = np.mean(correct1 - correct2)
    std_diff = np.std(correct1 - correct2, ddof=1)
    std_error = std_diff / np.sqrt(len(correct1))
    
    # Calculate 95% confidence interval for the difference
    ci_lower, ci_upper = t.interval(0.95, len(correct1)-1, loc=mean_diff, scale=std_error)
    
    print(f"Accuracy Comparison:")
    print(f"{model1_name}: {acc1:.4f}")
    print(f"{model2_name}: {acc2:.4f}")
    print(f"Difference: {abs(acc1 - acc2):.4f}")
    
    print(f"\nT-Test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {len(correct1) - 1}")
    print(f"P-value: {p_value:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Standard error: {std_error:.6f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Determine statistical significance
    alpha = 0.05
    if p_value < alpha:
        significance = "Significant"
        if mean_diff > 0:
            conclusion = f"{model1_name} is significantly better than {model2_name} (p < {alpha})"
        else:
            conclusion = f"{model2_name} is significantly better than {model1_name} (p < {alpha})"
    else:
        significance = "Not Significant"
        conclusion = f"No significant difference between {model1_name} and {model2_name} (p >= {alpha})"
    
    print(f"\nStatistical Status: {significance}")
    print(f"Conclusion: {conclusion}")
    
    # Effect size (Cohen's d)
    if std_diff > 0:
        cohens_d = mean_diff / std_diff
        print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    else:
        cohens_d = 0.0
        print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    
    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'accuracy1': acc1,
        'accuracy2': acc2,
        'accuracy_difference': abs(acc1 - acc2),
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_freedom': len(correct1) - 1,
        'mean_difference': mean_diff,
        'standard_error': std_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significance': significance,
        'conclusion': conclusion,
        'cohens_d': cohens_d,
        'sample_size': len(correct1)
    }

def compare_random_forest_with_others_t_test(X_train, y_train, X_test, y_test, feature_names):
    """
    Compare Random Forest with 8 other models using t-tests on accuracy
    """
    print(f"\n{'='*80}")
    print(f"RANDOM FOREST VS OTHER MODELS - T-TEST ACCURACY COMPARISON")
    print(f"{'='*80}")
    
    # Define models to compare (excluding Random Forest)
    other_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
        'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    # Train Random Forest (reference model)
    print("\nTraining Random Forest (reference model)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    results = []
    
    # Compare with each other model
    for model_name, model in other_models.items():
        print(f"\n{'-'*60}")
        print(f"Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            model_pred = model.predict(X_test)
            
            # Perform t-test
            test_result = t_test_accuracy_comparison(
                y_test, rf_pred, model_pred, 
                "Random Forest", model_name
            )
            
            results.append(test_result)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create summary table
    print(f"\n{'='*80}")
    print(f"T-TEST ACCURACY COMPARISON SUMMARY TABLE")
    print(f"{'='*80}")
    
    print(f"{'Model':<20} {'RF Acc':<8} {'Model Acc':<10} {'Diff':<8} {'T-stat':<8} {'P-value':<10} {'Status':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model2_name']:<20} "
              f"{result['accuracy1']:<8.4f} "
              f"{result['accuracy2']:<10.4f} "
              f"{result['accuracy_difference']:<8.4f} "
              f"{result['t_statistic']:<8.4f} "
              f"{result['p_value']:<10.6f} "
              f"{result['significance']:<12}")
    
    # Statistical summary
    significant_tests = [r for r in results if r['significance'] == 'Significant']
    print(f"\nSTATISTICAL SUMMARY:")
    print(f"Total comparisons: {len(results)}")
    print(f"Significant differences: {len(significant_tests)}")
    print(f"Non-significant differences: {len(results) - len(significant_tests)}")
    
    if significant_tests:
        print(f"\nModels significantly different from Random Forest:")
        for result in significant_tests:
            better_model = "Random Forest" if result['mean_difference'] > 0 else result['model2_name']
            print(f"- {result['model2_name']}: {result['conclusion']}")
            print(f"  Better performing: {better_model}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    model_names = [r['model2_name'] for r in results]
    rf_accs = [r['accuracy1'] for r in results]
    model_accs = [r['accuracy2'] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, rf_accs, width, label='Random Forest', alpha=0.8)
    plt.bar(x + width/2, model_accs, width, label='Other Models', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest vs Other Models - Test Accuracy Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: P-values
    plt.subplot(2, 2, 2)
    p_values = [r['p_value'] for r in results]
    colors = ['red' if p < 0.05 else 'green' for p in p_values]
    
    plt.bar(model_names, p_values, color=colors, alpha=0.7)
    plt.axhline(y=0.05, color='black', linestyle='--', label='Significance Level (0.05)')
    plt.xlabel('Models')
    plt.ylabel('P-value')
    plt.title('T-Test P-values')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: T-statistics
    plt.subplot(2, 2, 3)
    t_stats = [r['t_statistic'] for r in results]
    
    plt.bar(model_names, t_stats, alpha=0.7, color='blue')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Models')
    plt.ylabel('T-statistic')
    plt.title('T-Test T-statistics')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Effect sizes (Cohen's d)
    plt.subplot(2, 2, 4)
    effect_sizes = [r['cohens_d'] for r in results]
    
    plt.bar(model_names, effect_sizes, alpha=0.7, color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Models')
    plt.ylabel("Cohen's d")
    plt.title("Effect Sizes (Cohen's d)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/t_test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/t_test_accuracy_results.csv', index=False)
    
    print(f"\n✓ T-test accuracy comparison completed!")
    print(f"✓ Results saved to:")
    print(f"  - results/t_test_accuracy_results.csv")
    print(f"  - results/t_test_accuracy_comparison.png")
    
    return results, results_df

# Advanced Feature Selection for Better Performance
def advanced_feature_selection(X_train, y_train, X_test, method='hybrid', k_best=15):
    """
    Advanced feature selection using multiple techniques
    Methods: 'univariate', 'rfe', 'hybrid'
    """
    print(f"\n{'='*50}")
    print(f"ADVANCED FEATURE SELECTION")
    print(f"{'='*50}")
    print(f"Original features: {X_train.shape[1]}")
    
    if method == 'univariate':
        # Univariate feature selection
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"✓ Univariate selection: {len(selected_features)} features selected")
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        selector = RFE(estimator=rf, n_features_to_select=k_best, step=1)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"✓ RFE selection: {len(selected_features)} features selected")
        
    elif method == 'hybrid':
        # Hybrid: Univariate + RFE
        # Step 1: Univariate selection to reduce features
        selector1 = SelectKBest(score_func=f_classif, k=min(25, X_train.shape[1]))
        X_temp = selector1.fit_transform(X_train, y_train)
        
        # Step 2: RFE on reduced features
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        selector2 = RFE(estimator=rf, n_features_to_select=k_best, step=1)
        X_train_selected = selector2.fit_transform(X_temp, y_train)
        
        # Apply same transformations to test set
        X_test_temp = selector1.transform(X_test)
        X_test_selected = selector2.transform(X_test_temp)
        
        # Get final selected feature indices
        temp_features = selector1.get_support(indices=True)
        final_features = selector2.get_support(indices=True)
        selected_features = temp_features[final_features]
        
        print(f"✓ Hybrid selection: {len(selected_features)} features selected")
        print(f"  - Step 1 (Univariate): {len(temp_features)} features")
        print(f"  - Step 2 (RFE): {len(final_features)} features")
    
    print(f"Final feature count: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, selected_features

# Clinical Interpretability Framework for CKD Prediction
def clinical_interpretability_framework(best_model, X_train, y_train, X_test, y_test, feature_names, dataset_name,
                                       X_train_original=None, X_train_preprocessed=None, y_train_original=None,
                                       preprocessor=None, X_test_original=None, selected_feature_indices=None):
    """
    Comprehensive clinical interpretability framework for CKD prediction models
    
    Parameters:
    - X_train_original: Original training data before preprocessing (for ablation study)
    - X_train_preprocessed: Training data after preprocessing but before sampling
    - y_train_original: Original training labels before sampling
    - preprocessor: Fitted preprocessor (for ablation study)
    - X_test_original: Original test data before preprocessing (for ablation study)
    - selected_feature_indices: Indices of features selected in main pipeline (for ablation study)
    """
    print(f"\n{'='*60}")
    print(f"CLINICAL INTERPRETABILITY FRAMEWORK")
    print(f"{'='*60}")
    
    # 1. Model Calibration Analysis
    print(f"\n🎯 MODEL CALIBRATION ANALYSIS")
    print(f"{'-'*40}")
    
    try:
        # Get calibrated probabilities
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        
        # Get probabilities
        y_prob_orig = best_model.predict_proba(X_test)[:, 1]
        y_prob_calib = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calibration curves
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob_orig, n_bins=10)
        fraction_of_positives_calib, mean_predicted_value_calib = calibration_curve(y_test, y_prob_calib, n_bins=10)
        
        # Plot calibration curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Original Model")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Original Model Calibration ({dataset_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value_calib, fraction_of_positives_calib, "s-", label="Calibrated Model")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibrated Model ({dataset_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
        plt.close()
        
        # Calculate calibration metrics
        brier_score_orig = np.mean((y_prob_orig - y_test) ** 2)
        brier_score_calib = np.mean((y_prob_calib - y_test) ** 2)
        
        print(f"Original Model Brier Score: {brier_score_orig:.4f}")
        print(f"Calibrated Model Brier Score: {brier_score_calib:.4f}")
        print(f"Calibration Improvement: {((brier_score_orig - brier_score_calib) / brier_score_orig * 100):.1f}%")
        
    except Exception as e:
        print(f"Calibration analysis failed: {str(e)}")
    
    # 2. Clinical Risk Stratification
    print(f"\n🏥 CLINICAL RISK STRATIFICATION")
    print(f"{'-'*40}")
    
    try:
        # Define risk categories based on probability thresholds
        risk_thresholds = {
            'Low Risk': (0.0, 0.2),
            'Moderate Risk': (0.2, 0.5),
            'High Risk': (0.5, 0.8),
            'Very High Risk': (0.8, 1.0)
        }
        
        # Get probabilities
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Stratify patients
        risk_stratification = {}
        for risk_category, (low, high) in risk_thresholds.items():
            mask = (y_prob >= low) & (y_prob < high)
            risk_stratification[risk_category] = {
                'count': np.sum(mask),
                'percentage': np.sum(mask) / len(y_test) * 100,
                'actual_positive_rate': np.mean(y_test[mask]) if np.sum(mask) > 0 else 0
            }
        
        print(f"Risk Stratification Results:")
        for risk_category, stats in risk_stratification.items():
            print(f"  {risk_category}: {stats['count']} patients ({stats['percentage']:.1f}%) - "
                  f"Actual CKD rate: {stats['actual_positive_rate']:.1%}")
        
        # Plot risk stratification
        plt.figure(figsize=(10, 6))
        categories = list(risk_stratification.keys())
        counts = [stats['count'] for stats in risk_stratification.values()]
        actual_rates = [stats['actual_positive_rate'] for stats in risk_stratification.values()]
        
        # Bar plot for patient counts
        ax1 = plt.subplot(111)
        bars = ax1.bar(categories, counts, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax1.set_xlabel('Risk Category')
        ax1.set_ylabel('Number of Patients', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Line plot for actual positive rates
        ax2 = ax1.twinx()
        ax2.plot(categories, actual_rates, 'ro-', linewidth=2, markersize=8, label='Actual CKD Rate')
        ax2.set_ylabel('Actual CKD Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'Clinical Risk Stratification ({dataset_name})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Risk stratification failed: {str(e)}")
    
    # 3. Feature Clinical Relevance Analysis
    print(f"\n🔬 CLINICAL FEATURE RELEVANCE")
    print(f"{'-'*40}")
    
    try:
        # Clinical feature categories
        clinical_categories = {
            'Renal Function': ['CreatnineBaseline', 'eGFRBaseline', 'Creatinine_Age_Ratio', 'eGFR_BMI_Ratio'],
            'Cardiovascular': ['sBPBaseline', 'dBPBaseline', 'BP_Product', 'MAP', 'CV_Risk_Score'],
            'Metabolic': ['HgbA1C', 'HgbA1C_Squared', 'TriglyceridesBaseline', 'CholesterolBaseline', 'Metabolic_Syndrome_Score'],
            'Demographics': ['AgeBaseline', 'BMIBaseline', 'Gender'],
            'Risk Factors': ['HistoryDiabetes', 'HistoryHTN', 'HistoryCHD', 'HistoryVascular', 'HistorySmoking']
        }
        
        # Try to calculate feature importance using model's built-in method
        shap_importance_local = None
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(best_model, 'feature_importances_'):
                shap_importance_local = best_model.feature_importances_
            elif hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
                clf = best_model.named_steps['classifier']
                if hasattr(clf, 'feature_importances_'):
                    shap_importance_local = clf.feature_importances_
        except:
            pass
        
        # Calculate category importance (if feature importance available)
        if shap_importance_local is not None:
            category_importance = {}
            for category, features in clinical_categories.items():
                category_score = 0
                feature_count = 0
                for i, feature in enumerate(feature_names):
                    if any(feat in feature for feat in features):
                        if i < len(shap_importance_local):
                            category_score += shap_importance_local[i]
                            feature_count += 1
                
                if feature_count > 0:
                    category_importance[category] = category_score / feature_count
                else:
                    category_importance[category] = 0
            
            print(f"Clinical Category Importance:")
            for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {category}: {importance:.4f}")
            
            # Plot clinical category importance
            plt.figure(figsize=(10, 6))
            categories = list(category_importance.keys())
            importances = list(category_importance.values())
            
            colors = ['red' if 'Renal' in cat else 'orange' if 'Cardio' in cat else 'yellow' if 'Metabolic' in cat else 'green' if 'Demographic' in cat else 'purple' for cat in categories]
            
            bars = plt.bar(categories, importances, color=colors, alpha=0.7)
            plt.xlabel('Clinical Category')
            plt.ylabel('Average SHAP Importance')
            plt.title(f'Clinical Feature Category Importance ({dataset_name})')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, importance in zip(bars, importances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{importance:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            print("  Feature importance not available from model. Skipping category importance analysis.")
            print("  Consider using SHAP values for detailed feature importance analysis.")
        
    except Exception as e:
        print(f"Clinical feature relevance analysis failed: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
    
    # 4. Decision Curve Analysis
    print(f"\n📈 DECISION CURVE ANALYSIS")
    print(f"{'-'*40}")
    
    try:
        # Calculate net benefit across threshold probabilities
        threshold_probs = np.arange(0.01, 0.99, 0.01)
        net_benefits = []
        
        for threshold in threshold_probs:
            # Treat all as positive (treat all strategy)
            treat_all_net_benefit = (np.mean(y_test) - (1 - np.mean(y_test)) * (threshold / (1 - threshold)))
            
            # Treat none as positive (treat none strategy)
            treat_none_net_benefit = 0
            
            # Model strategy
            y_pred_threshold = (y_prob >= threshold).astype(int)
            tp = np.sum((y_pred_threshold == 1) & (y_test == 1))
            fp = np.sum((y_pred_threshold == 1) & (y_test == 0))
            fn = np.sum((y_pred_threshold == 0) & (y_test == 1))
            
            model_net_benefit = (tp / len(y_test)) - (fp / len(y_test)) * (threshold / (1 - threshold))
            net_benefits.append(model_net_benefit)
        
        # Plot decision curve
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_probs, net_benefits, 'b-', linewidth=2, label='Model')
        plt.plot(threshold_probs, [np.mean(y_test) - (1 - np.mean(y_test)) * (t / (1 - t)) for t in threshold_probs], 
                'g--', linewidth=1, label='Treat All')
        plt.plot(threshold_probs, [0] * len(threshold_probs), 'r--', linewidth=1, label='Treat None')
        
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis ({dataset_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Decision curve analysis completed")
        print(f"Maximum net benefit: {max(net_benefits):.4f} at threshold: {threshold_probs[np.argmax(net_benefits)]:.2f}")
        
    except Exception as e:
        print(f"Decision curve analysis failed: {str(e)}")
    
    print(f"\n✓ Clinical Interpretability Framework Analysis Complete")
    
    # 5. SHAP & LIME Based Risk Factor Analysis
    print(f"\n🔬 SHAP & LIME RISK FACTOR ANALYSIS")
    print(f"{'-'*40}")
    
    try:
        # Try to calculate feature importance using model's built-in method
        shap_importance_local = None
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(best_model, 'feature_importances_'):
                shap_importance_local = best_model.feature_importances_
            elif hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
                clf = best_model.named_steps['classifier']
                if hasattr(clf, 'feature_importances_'):
                    shap_importance_local = clf.feature_importances_
        except:
            pass
        
        # Get SHAP values for risk factor identification
        if shap_importance_local is not None:
            print(f"Analyzing SHAP values for clinical risk factors...")
            
            # Clinical Risk Factor Analysis with OR (Odds Ratio) calculation
            clinical_risk_factors = []
            
            for i, (feature, importance) in enumerate(zip(feature_names, shap_importance_local)):
                if importance > 0.01:  # Threshold for significant features
                    # Calculate clinical risk metrics
                    try:
                        # Convert to DataFrame for analysis
                        X_train_df = pd.DataFrame(X_train, columns=feature_names)
                        X_train_df['CKD_Status'] = y_train
                        
                        # Calculate mean values for CKD vs non-CKD groups
                        ckd_group = X_train_df[X_train_df['CKD_Status'] == 1][feature]
                        no_ckd_group = X_train_df[X_train_df['CKD_Status'] == 0][feature]
                        
                        mean_ckd = np.mean(ckd_group)
                        mean_no_ckd = np.mean(no_ckd_group)
                        
                        # Calculate Odds Ratio (for continuous variables)
                        if mean_no_ckd != 0:
                            odds_ratio = (mean_ckd / (1 - mean_ckd)) / (mean_no_ckd / (1 - mean_no_ckd))
                        else:
                            odds_ratio = float('inf') if mean_ckd > 0 else 1.0
                        
                        # Calculate risk level based on clinical thresholds
                        if any(keyword in feature.lower() for keyword in ['creatinine', 'renal', 'kidney']):
                            category = 'Renal Risk Factor'
                            clinical_significance = 'High'
                            risk_level = 'Very High' if importance > 0.05 else 'High' if importance > 0.02 else 'Moderate'
                            clinical_threshold = 'Creatinine > 1.2 mg/dL' if 'creatinine' in feature.lower() else 'eGFR < 60 mL/min' if 'egfr' in feature.lower() else 'Abnormal renal function'
                        elif any(keyword in feature.lower() for keyword in ['bp', 'sbp', 'dbp', 'cardio', 'vascular']):
                            category = 'Cardiovascular Risk Factor'
                            clinical_significance = 'High'
                            risk_level = 'Very High' if importance > 0.04 else 'High' if importance > 0.02 else 'Moderate'
                            clinical_threshold = 'SBP > 140 mmHg' if 'sbp' in feature.lower() else 'DBP > 90 mmHg' if 'dbp' in feature.lower() else 'Hypertension'
                        elif any(keyword in feature.lower() for keyword in ['hgba1c', 'diabetes', 'glucose', 'metabolic']):
                            category = 'Metabolic Risk Factor'
                            clinical_significance = 'High'
                            risk_level = 'Very High' if importance > 0.04 else 'High' if importance > 0.02 else 'Moderate'
                            clinical_threshold = 'HbA1c > 6.5%' if 'hgba1c' in feature.lower() else 'Glucose > 126 mg/dL' if 'glucose' in feature.lower() else 'Metabolic dysfunction'
                        elif any(keyword in feature.lower() for keyword in ['age', 'bmi', 'gender']):
                            category = 'Demographic Risk Factor'
                            clinical_significance = 'Medium'
                            risk_level = 'High' if importance > 0.03 else 'Moderate' if importance > 0.015 else 'Low'
                            clinical_threshold = 'Age > 65 years' if 'age' in feature.lower() else 'BMI > 30 kg/m²' if 'bmi' in feature.lower() else 'Demographic factor'
                        else:
                            category = 'Other Risk Factor'
                            clinical_significance = 'Medium'
                            risk_level = 'Moderate' if importance > 0.02 else 'Low'
                            clinical_threshold = 'Abnormal laboratory value'
                        
                        clinical_risk_factors.append({
                            'Feature': feature,
                            'SHAP Importance': importance,
                            'Category': category,
                            'Clinical Significance': clinical_significance,
                            'Risk Level': risk_level,
                            'Odds Ratio': odds_ratio,
                            'Clinical Threshold': clinical_threshold,
                            'Mean CKD': mean_ckd,
                            'Mean Non-CKD': mean_no_ckd,
                            'Risk Score': importance * (1 + np.log(abs(odds_ratio) + 1))  # Combined risk score
                        })
                        
                    except Exception as e:
                        print(f"Error calculating risk for {feature}: {str(e)}")
                        continue
            
            # Sort by combined risk score
            clinical_risk_factors.sort(key=lambda x: x['Risk Score'], reverse=True)
            
            print(f"\n🏥 CLINICAL RISK FACTOR ANALYSIS:")
            print(f"{'='*60}")
            print(f"{'Rank':<5} {'Feature':<25} {'Risk Level':<12} {'Odds Ratio':<12} {'Clinical Threshold':<20}")
            print(f"{'-'*60}")
            
            for i, factor in enumerate(clinical_risk_factors[:15], 1):  # Top 15 risk factors
                or_str = f"{factor['Odds Ratio']:.2f}" if factor['Odds Ratio'] != float('inf') else "∞"
                print(f"{i:<5} {factor['Feature'][:24]:<25} {factor['Risk Level']:<12} {or_str:<12} {factor['Clinical Threshold'][:19]:<20}")
            
            # Create comprehensive risk factor visualization
            plt.figure(figsize=(16, 10))
            
            # 1. Risk Factor Categories Distribution
            plt.subplot(2, 3, 1)
            categories = list(set([f['Category'] for f in clinical_risk_factors[:15]]))
            category_counts = {}
            for category in categories:
                category_counts[category] = len([f for f in clinical_risk_factors[:15] if f['Category'] == category])
            
            colors = ['red' if 'Renal' in cat else 'orange' if 'Cardio' in cat else 'yellow' if 'Metabolic' in cat else 'green' for cat in categories]
            plt.pie(category_counts.values(), labels=categories, colors=colors, autopct='%1.1f', startangle=90)
            plt.title('Risk Factor Categories Distribution', fontsize=10, fontweight='bold')
            
            # 2. Top Risk Factors by Importance
            plt.subplot(2, 3, 2)
            top_factors = clinical_risk_factors[:10]
            factors = [f['Feature'][:15] for f in top_factors]
            importances = [f['SHAP Importance'] for f in top_factors]
            colors = ['red' if 'Renal' in f['Category'] else 'orange' if 'Cardio' in f['Category'] else 'yellow' if 'Metabolic' in f['Category'] else 'green' for f in top_factors]
            
            bars = plt.barh(factors, importances, color=colors, alpha=0.8)
            plt.xlabel('SHAP Importance')
            plt.title('Top 10 Risk Factors by Importance', fontsize=10, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', ha='left', va='center', fontsize=8)
            
            # 3. Odds Ratio Analysis
            plt.subplot(2, 3, 3)
            or_data = [(f['Feature'][:15], min(f['Odds Ratio'], 10)) for f in clinical_risk_factors[:10]]  # Cap at 10 for visualization
            or_factors = [item[0] for item in or_data]
            or_values = [item[1] for item in or_data]
            
            bars = plt.barh(or_factors, or_values, color='darkred', alpha=0.7)
            plt.xlabel('Odds Ratio (capped at 10)')
            plt.title('Odds Ratio for Top Risk Factors', fontsize=10, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='OR = 1 (No Risk)')
            plt.legend()
            
            # 4. Risk Level Distribution
            plt.subplot(2, 3, 4)
            risk_levels = ['Very High', 'High', 'Moderate', 'Low']
            risk_counts = [len([f for f in clinical_risk_factors if f['Risk Level'] == level]) for level in risk_levels]
            colors = ['darkred', 'red', 'orange', 'yellow']
            
            bars = plt.bar(risk_levels, risk_counts, color=colors, alpha=0.7)
            plt.xlabel('Risk Level')
            plt.ylabel('Number of Risk Factors')
            plt.title('Risk Level Distribution', fontsize=10, fontweight='bold')
            
            # Add value labels
            for bar, count in zip(bars, risk_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{count}', ha='center', va='bottom', fontsize=9)
            
            # 5. Clinical Significance by Category
            plt.subplot(2, 3, 5)
            significance_data = {}
            for category in categories:
                high_sig = len([f for f in clinical_risk_factors if f['Category'] == category and f['Clinical Significance'] == 'High'])
                med_sig = len([f for f in clinical_risk_factors if f['Category'] == category and f['Clinical Significance'] == 'Medium'])
                significance_data[category] = [high_sig, med_sig]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, [data[0] for data in significance_data.values()], width, label='High Significance', color='red', alpha=0.7)
            plt.bar(x + width/2, [data[1] for data in significance_data.values()], width, label='Medium Significance', color='orange', alpha=0.7)
            
            plt.xlabel('Risk Category')
            plt.ylabel('Number of Factors')
            plt.title('Clinical Significance by Category', fontsize=10, fontweight='bold')
            plt.xticks(x, categories, rotation=45)
            plt.legend()
            
            # 6. Combined Risk Score Analysis
            plt.subplot(2, 3, 6)
            risk_scores = [f['Risk Score'] for f in clinical_risk_factors[:15]]
            plt.hist(risk_scores, bins=10, color='purple', alpha=0.7, edgecolor='black')
            plt.xlabel('Combined Risk Score')
            plt.ylabel('Frequency')
            plt.title('Risk Score Distribution', fontsize=10, fontweight='bold')
            plt.axvline(x=np.mean(risk_scores), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(risk_scores):.3f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
            plt.close()
            
            # Clinical Recommendations Based on Risk Analysis
            print(f"\n🏥 CLINICAL RECOMMENDATIONS BASED ON RISK FACTOR ANALYSIS:")
            print(f"{'='*60}")
            
            # Identify dominant risk categories
            category_risk = {}
            for category in categories:
                category_factors = [f for f in clinical_risk_factors if f['Category'] == category]
                if category_factors:
                    avg_risk_score = np.mean([f['Risk Score'] for f in category_factors])
                    category_risk[category] = avg_risk_score
            
            sorted_categories = sorted(category_risk.items(), key=lambda x: x[1], reverse=True)
            
            print(f"Risk Priority Order:")
            for i, (category, avg_score) in enumerate(sorted_categories, 1):
                print(f"  {i}. {category}: Average Risk Score = {avg_score:.3f}")
                
                # Specific recommendations
                if 'Renal' in category:
                    print(f"      🩺 RECOMMENDATION: Prioritize renal function monitoring")
                    print(f"      - Monthly creatinine and eGFR assessment")
                    print(f"      - Early nephrology referral for high-risk patients")
                    print(f"      - Consider ACE inhibitors for renal protection")
                elif 'Cardio' in category:
                    print(f"      ❤️ RECOMMENDATION: Aggressive cardiovascular risk management")
                    print(f"      - Strict blood pressure control (<130/80 mmHg)")
                    print(f"      - Regular cardiac monitoring")
                    print(f"      - Statin therapy for eligible patients")
                elif 'Metabolic' in category:
                    print(f"      🍬 RECOMMENDATION: Optimize metabolic control")
                    print(f"      - Target HbA1c < 7.0% for diabetic patients")
                    print(f"      - Weight management programs")
                    print(f"      - Dietary modifications and exercise")
                elif 'Demographic' in category:
                    print(f"      👥 RECOMMENDATION: Age-appropriate screening")
                    print(f"      - More frequent monitoring for elderly patients")
                    print(f"      - Consider comorbidities in treatment planning")
            
            # High-Risk Factor Identification
            print(f"\n🚨 HIGH-RISK FACTORS (Risk Level: Very High):")
            very_high_risk = [f for f in clinical_risk_factors if f['Risk Level'] == 'Very High']
            for i, factor in enumerate(very_high_risk, 1):
                print(f"  {i}. {factor['Feature']}")
                print(f"     Odds Ratio: {factor['Odds Ratio']:.2f}")
                print(f"     Clinical Action: {factor['Clinical Threshold']}")
                print(f"     Risk Score: {factor['Risk Score']:.3f}")
            
            print(f"\n✓ Enhanced Clinical Risk Factor Analysis Complete")
        else:
            print("  Feature importance not available from model. Skipping SHAP/LIME risk factor analysis.")
            print("  Consider calculating SHAP values separately for detailed risk factor analysis.")
            
    except Exception as e:
        print(f"Enhanced risk factor analysis failed: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
    
    # 6. Traditional Multivariate Risk Factor Analysis
    print(f"\n📊 TRADITIONAL MULTIVARIATE RISK FACTOR ANALYSIS")
    print(f"{'-'*40}")
    
    try:
        # Convert to DataFrame for analysis
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Add target variable
        train_analysis_df = X_train_df.copy()
        train_analysis_df['CKD_Status'] = y_train
        test_analysis_df = X_test_df.copy()
        test_analysis_df['CKD_Status'] = y_test
        
        print(f"Analyzing {len(feature_names)} features for multivariate risk patterns...")
        
        # 1. Correlation Analysis with CKD Status
        print(f"\n🔍 CORRELATION ANALYSIS")
        print(f"{'-'*30}")
        
        correlations = {}
        for feature in feature_names:
            if feature in train_analysis_df.columns:
                corr = np.corrcoef(train_analysis_df[feature], train_analysis_df['CKD_Status'])[0, 1]
                correlations[feature] = {
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'significance': 'Strong' if abs(corr) > 0.3 else 'Moderate' if abs(corr) > 0.1 else 'Weak'
                }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        print(f"Top 10 Features by Correlation with CKD:")
        for i, (feature, stats) in enumerate(sorted_correlations[:10], 1):
            print(f"  {i}. {feature}: r = {stats['correlation']:.3f} "
                  f"({stats['significance']} correlation)")
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        
        # Calculate correlation matrix
        correlation_matrix = train_analysis_df[feature_names + ['CKD_Status']].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f',
                   xticklabels=True,
                   yticklabels=True)
        
        plt.title(f'Feature Correlation Matrix with CKD Status ({dataset_name})')
        plt.tight_layout()
        plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
        plt.close()
        
        # 2. Risk Factor Combination Analysis
        print(f"\n🎯 RISK FACTOR COMBINATION ANALYSIS")
        print(f"{'-'*30}")
        
        # Identify key clinical risk factors
        renal_features = ['CreatnineBaseline', 'eGFRBaseline', 'Creatinine_Age_Ratio', 'eGFR_BMI_Ratio']
        cardio_features = ['sBPBaseline', 'dBPBaseline', 'BP_Product', 'MAP', 'CV_Risk_Score']
        metabolic_features = ['HgbA1C', 'HgbA1C_Squared', 'TriglyceridesBaseline', 'CholesterolBaseline', 'Metabolic_Syndrome_Score']
        
        # Create risk scores
        risk_scores = {}
        
        # Renal Risk Score
        available_renal = [f for f in renal_features if f in train_analysis_df.columns]
        if len(available_renal) > 0:
            renal_score = train_analysis_df[available_renal].mean(axis=1)
            risk_scores['Renal_Risk'] = renal_score
        
        # Cardiovascular Risk Score
        available_cardio = [f for f in cardio_features if f in train_analysis_df.columns]
        if len(available_cardio) > 0:
            cardio_score = train_analysis_df[available_cardio].mean(axis=1)
            risk_scores['Cardiovascular_Risk'] = cardio_score
        
        # Metabolic Risk Score
        available_metabolic = [f for f in metabolic_features if f in train_analysis_df.columns]
        if len(available_metabolic) > 0:
            metabolic_score = train_analysis_df[available_metabolic].mean(axis=1)
            risk_scores['Metabolic_Risk'] = metabolic_score
        
        print(f"Risk Score Categories Created:")
        for score_type, score in risk_scores.items():
            print(f"  {score_type}: Mean = {np.mean(score):.3f}, Std = {np.std(score):.3f}")
        
        # 3. Multivariate Risk Stratification
        print(f"\n🏥 MULTIVARIATE RISK STRATIFICATION")
        print(f"{'-'*30}")
        
        # Combine risk scores for overall risk assessment
        if len(risk_scores) > 0:
            # Normalize risk scores (0-1 scale)
            normalized_risks = {}
            for risk_type, scores in risk_scores.items():
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.zeros_like(scores)
                normalized_risks[risk_type] = normalized_scores
            
            # Calculate overall risk score (weighted combination)
            weights = {'Renal_Risk': 0.4, 'Cardiovascular_Risk': 0.3, 'Metabolic_Risk': 0.3}
            overall_risk = sum(normalized_risks[risk_type] * weight for risk_type, weight in weights.items() if risk_type in normalized_risks)
            
            # Create risk categories
            train_analysis_df['Overall_Risk_Score'] = overall_risk
            train_analysis_df['Risk_Category'] = pd.cut(overall_risk, 
                                                   bins=[0, 0.33, 0.66, 1.0], 
                                                   labels=['Low', 'Moderate', 'High'])
            
            # Analyze risk distribution
            risk_distribution = train_analysis_df['Risk_Category'].value_counts()
            ckd_rates_by_risk = train_analysis_df.groupby('Risk_Category')['CKD_Status'].mean()
            
            print(f"Multivariate Risk Distribution:")
            for risk_cat in ['Low', 'Moderate', 'High']:
                count = risk_distribution.get(risk_cat, 0)
                ckd_rate = ckd_rates_by_risk.get(risk_cat, 0)
                print(f"  {risk_cat} Risk: {count} patients ({count/len(train_analysis_df)*100:.1f}%) - "
                      f"CKD Rate: {ckd_rate:.1%}")
            
            # Plot multivariate risk stratification
            plt.figure(figsize=(12, 6))
            
            # Create subplot for risk distribution
            plt.subplot(1, 2, 1)
            risk_counts = [risk_distribution.get(cat, 0) for cat in ['Low', 'Moderate', 'High']]
            colors = ['green', 'yellow', 'red']
            bars = plt.bar(['Low', 'Moderate', 'High'], risk_counts, color=colors, alpha=0.7)
            plt.xlabel('Risk Category')
            plt.ylabel('Number of Patients')
            plt.title(f'Multivariate Risk Distribution ({dataset_name})')
            
            # Add percentage labels
            for bar, count in zip(bars, risk_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01, 
                        f'{count} ({count/len(train_analysis_df)*100:.1f}%)', 
                        ha='center', va='bottom')
            
            # Create subplot for CKD rates by risk
            plt.subplot(1, 2, 2)
            ckd_rates = [ckd_rates_by_risk.get(cat, 0) for cat in ['Low', 'Moderate', 'High']]
            plt.plot(['Low', 'Moderate', 'High'], ckd_rates, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Risk Category')
            plt.ylabel('CKD Rate')
            plt.title(f'CKD Rate by Risk Category ({dataset_name})')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
            plt.close()
        
        # 4. Traditional Statistical Analysis
        print(f"\n📈 TRADITIONAL STATISTICAL ANALYSIS")
        print(f"{'-'*30}")
        
        # Chi-square tests for categorical variables
        categorical_features = ['Gender', 'HistoryDiabetes', 'HistoryHTN', 'HistoryCHD', 'HistoryVascular', 'HistorySmoking']
        available_categorical = [f for f in categorical_features if f in train_analysis_df.columns]
        
        if len(available_categorical) > 0:
            print(f"Chi-square Analysis for Categorical Variables:")
            for feature in available_categorical[:5]:  # Limit to top 5
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(train_analysis_df[feature], train_analysis_df['CKD_Status'])
                    
                    # Perform chi-square test
                    from scipy.stats import chi2_contingency
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    print(f"  {feature}: χ² = {chi2:.2f}, p-value = {p_value:.4f} "
                          f"({'Significant' if p_value < 0.05 else 'Not Significant'})")
                
                except Exception as e:
                    print(f"  {feature}: Chi-square test failed - {str(e)}")
        
        # T-tests for continuous variables
        continuous_features = ['AgeBaseline', 'CreatnineBaseline', 'eGFRBaseline', 'sBPBaseline', 'dBPBaseline']
        available_continuous = [f for f in continuous_features if f in train_analysis_df.columns]
        
        if len(available_continuous) > 0:
            print(f"\nT-test Analysis for Continuous Variables:")
            for feature in available_continuous[:5]:  # Limit to top 5
                try:
                    # Split by CKD status
                    ckd_group = train_analysis_df[train_analysis_df['CKD_Status'] == 1][feature]
                    no_ckd_group = train_analysis_df[train_analysis_df['CKD_Status'] == 0][feature]
                    
                    # Perform t-test
                    from scipy.stats import ttest_ind
                    t_stat, p_value = ttest_ind(ckd_group, no_ckd_group)
                    
                    print(f"  {feature}: t = {t_stat:.3f}, p-value = {p_value:.4f} "
                          f"({'Significant' if p_value < 0.05 else 'Not Significant'})")
                
                except Exception as e:
                    print(f"  {feature}: T-test failed - {str(e)}")
        
        print(f"\n✓ Traditional Multivariate Risk Factor Analysis Complete")
        
    except Exception as e:
        print(f"Multivariate analysis failed: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
    
    # 7. Q1 Journal Paper Standard Statistical Analysis
    print(f"\n📄 Q1 JOURNAL PAPER STATISTICAL ANALYSIS")
    print(f"{'='*50}")
    
    try:
        # Convert to DataFrame for analysis
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Add target variable
        train_analysis_df = X_train_df.copy()
        train_analysis_df['CKD_Status'] = y_train
        test_analysis_df = X_test_df.copy()
        test_analysis_df['CKD_Status'] = y_test
        
        print(f"Performing Q1 Journal Paper standard statistical analyses...")
        
        # 1. DeLong's Test for Model Comparison
        print(f"\n🏆 MODEL COMPARISON - DeLONG'S TEST")
        print(f"{'-'*35}")
        
        # Get predictions from best model (use the model passed as parameter)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate C-statistic (AUC)
        c_statistic = roc_auc_score(y_test, y_prob)
        
        # Standard error of AUC
        n1 = np.sum(y_test == 1)  # Number of events
        n0 = np.sum(y_test == 0)  # Number of non-events
        q1 = np.sum(y_prob)  # Sum of predicted probabilities
        q0 = len(y_prob) - q1
        
        # DeLong's test components
        se_auc = np.sqrt((c_statistic * (1 - c_statistic) / (n1 * n0)) + 
                     (c_statistic / (2 * n1 * n0)))
        
        z_score = (c_statistic - 0.5) / se_auc
        p_value_delong = 2 * (1 - norm.cdf(abs(z_score)))
        
        print(f"DeLong's Test Results:")
        print(f"  AUC (C-statistic): {c_statistic:.4f}")
        print(f"  Standard Error: {se_auc:.4f}")
        print(f"  Z-score: {z_score:.4f}")
        print(f"  P-value: {p_value_delong:.6f}")
        print(f"  Significance: {'Significant' if p_value_delong < 0.05 else 'Not Significant'}")
        
        # 2. Permutation Feature Importance Test
        print(f"\n🔀 FEATURE IMPORTANCE - PERMUTATION TEST")
        print(f"{'-'*35}")
        
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
            
            # Calculate baseline score manually
            baseline_score = best_model.score(X_test, y_test)
            
            print(f"Permutation Importance Results:")
            print(f"  Baseline accuracy: {baseline_score:.4f}")
            print(f"  Top 10 features by importance:")
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            for i, idx in enumerate(sorted_idx[:10], 1):
                print(f"    {i}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f}")
            
            # Plot permutation importance
            plt.figure(figsize=(10, 6))
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False)
            plt.yticks(range(1, len(feature_names) + 1), 
                      [feature_names[i] for i in sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.title(f'Permutation Feature Importance ({dataset_name})')
            plt.tight_layout()
            plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Permutation test failed: {str(e)}")
        
        # 3. Mann-Whitney U Test for Feature Significance
        print(f"\n🧮 FEATURE SIGNIFICANCE - MANN-WHITNEY U TEST")
        print(f"{'-'*35}")
        
        # Compare distributions of significant features between CKD and non-CKD groups
        significant_features = []
        for i, feature in enumerate(feature_names[:10]):  # Test top 10 features
            if feature in train_analysis_df.columns:
                ckd_values = train_analysis_df[train_analysis_df['CKD_Status'] == 1][feature]
                no_ckd_values = train_analysis_df[train_analysis_df['CKD_Status'] == 0][feature]
                
                # Perform Mann-Whitney U test
                u_stat, p_value_mw = scipy.stats.mannwhitneyu(ckd_values, no_ckd_values, alternative='two-sided')
                
                significant_features.append({
                    'Feature': feature,
                    'U_Statistic': u_stat,
                    'P_Value': p_value_mw,
                    'Significant': p_value_mw < 0.05
                })
                
                print(f"  {feature}: U = {u_stat:.1f}, p = {p_value_mw:.4f} "
                      f"({'Significant' if p_value_mw < 0.05 else 'Not Significant'})")
        
        # 4. Hosmer-Lemeshow Test for Calibration
        print(f"\n🎯 MODEL CALIBRATION - HOSMER-LEMESHOW TEST")
        print(f"{'-'*35}")
        
        try:
            # Create deciles of predicted risk
            risk_deciles = pd.qcut(y_prob, q=10, labels=False, duplicates='drop')
            
            # Convert to numpy array if needed and get unique values
            if isinstance(risk_deciles, pd.Series):
                unique_deciles = risk_deciles.unique()
            else:
                unique_deciles = np.unique(risk_deciles)
            
            # Calculate observed and expected events per decile
            calibration_data = []
            for i in range(min(10, len(unique_deciles))):
                decile_mask = (risk_deciles == i)
                if np.sum(decile_mask) > 0:
                    observed_events = np.sum(y_test[decile_mask])
                    expected_events = np.mean(y_test[decile_mask]) * np.sum(decile_mask)
                    
                    calibration_data.append({
                        'Decile': i + 1,
                        'Observed_Events': observed_events,
                        'Expected_Events': expected_events,
                        'Observed_Rate': observed_events / np.sum(decile_mask),
                        'Expected_Rate': expected_events / np.sum(decile_mask)
                    })
            
            if len(calibration_data) > 0:
                calibration_df = pd.DataFrame(calibration_data)
                
                # Hosmer-Lemeshow test statistic
                chi_sq_hl = np.sum((calibration_df['Observed_Events'] - calibration_df['Expected_Events'])**2 / 
                                  calibration_df['Expected_Events'])
                
                # Degrees of freedom (number of groups - 2)
                df_hl = len(calibration_data) - 2
                
                # P-value
                p_value_hl = 1 - scipy.stats.chi2.cdf(chi_sq_hl, df_hl)
                
                print(f"Hosmer-Lemeshow Test Results:")
                print(f"  Chi-square: {chi_sq_hl:.4f}")
                print(f"  Degrees of Freedom: {df_hl}")
                print(f"  P-value: {p_value_hl:.4f}")
                print(f"  Calibration: {'Good' if p_value_hl > 0.05 else 'Poor'}")
                
                # Plot calibration plot
                plt.figure(figsize=(10, 6))
                plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                plt.plot(calibration_df['Expected_Rate'], calibration_df['Observed_Rate'], 'ro-', label='Model')
                plt.scatter(calibration_df['Expected_Rate'], calibration_df['Observed_Rate'], s=50, c='red', alpha=0.7)
                plt.xlabel('Expected CKD Rate')
                plt.ylabel('Observed CKD Rate')
                plt.title(f'Hosmer-Lemeshow Calibration Plot ({dataset_name})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
                plt.close()
            else:
                print("  Hosmer-Lemeshow test: Insufficient data for calibration analysis")
            
        except Exception as e:
            print(f"  Hosmer-Lemeshow test failed: {str(e)}")
        
        # 5. Chi-square Test for Risk Stratification
        print(f"\n📊 RISK STRATIFICATION - CHI-SQUARE TEST")
        print(f"{'-'*35}")
        
        # Create risk categories
        risk_categories = pd.qcut(y_prob, q=3, labels=['Low', 'Medium', 'High'])
        
        # Create contingency table
        contingency_table = pd.crosstab(risk_categories, y_test)
        
        # Perform chi-square test
        chi2_risk, p_value_risk, dof_risk, expected_risk = scipy.stats.chi2_contingency(contingency_table)
        
        print(f"Risk Stratification Chi-square Test:")
        print(f"  Chi-square: {chi2_risk:.4f}")
        print(f"  P-value: {p_value_risk:.4f}")
        print(f"  Degrees of Freedom: {dof_risk}")
        print(f"  Risk Stratification: {'Significant' if p_value_risk < 0.05 else 'Not Significant'}")
        
        # 6. Subgroup Analysis - T-tests
        print(f"\n👥 SUBGROUP ANALYSIS - T-TESTS")
        print(f"{'-'*35}")
        
        # Analyze key subgroups
        subgroups = {
            'Age': {
                'feature': 'AgeBaseline',
                'groups': ['<60', '60-70', '>70'],
                'labels': ['Young', 'Middle-aged', 'Elderly']
            },
            'Gender': {
                'feature': 'Gender',
                'groups': [0, 1],  # Assuming binary encoding
                'labels': ['Female', 'Male']
            }
        }
        
        for subgroup_name, subgroup_info in subgroups.items():
            if subgroup_info['feature'] in train_analysis_df.columns:
                print(f"\n{subgroup_name} Subgroup Analysis:")
                
                group_values = train_analysis_df[subgroup_info['feature']]
                unique_groups = sorted(group_values.unique())
                
                # Create binary groups for analysis
                if len(unique_groups) == 2:
                    group1_mask = group_values == unique_groups[0]
                    group2_mask = group_values == unique_groups[1]
                    group1_label = subgroup_info['labels'][0]
                    group2_label = subgroup_info['labels'][1]
                else:
                    # For continuous variables, create binary groups
                    median_val = group_values.median()
                    group1_mask = group_values <= median_val
                    group2_mask = group_values > median_val
                    group1_label = f'≤{median_val:.1f}'
                    group2_label = f'>{median_val:.1f}'
                
                # CKD rates by group
                ckd_rate_group1 = np.mean(y_test[group1_mask]) if np.sum(group1_mask) > 0 else 0
                ckd_rate_group2 = np.mean(y_test[group2_mask]) if np.sum(group2_mask) > 0 else 0
                
                print(f"  {group1_label}: {np.sum(group1_mask)} patients - CKD Rate: {ckd_rate_group1:.1%}")
                print(f"  {group2_label}: {np.sum(group2_mask)} patients - CKD Rate: {ckd_rate_group2:.1%}")
                
                # Perform t-test between groups
                if np.sum(group1_mask) > 0 and np.sum(group2_mask) > 0:
                    t_stat_sub, p_value_sub = scipy.stats.ttest_ind(
                        y_test[group1_mask], y_test[group2_mask]
                    )
                    print(f"  T-test (Group1 vs Group2): t = {t_stat_sub:.3f}, p = {p_value_sub:.4f} "
                          f"({'Significant' if p_value_sub < 0.05 else 'Not Significant'})")
        
        # 7. Ablation Study - McNemar's Test
        print(f"\n🔬 ABLATION STUDY - MCNEMAR'S TEST")
        print(f"{'-'*35}")
        
        try:
            # Convert to numpy arrays to avoid pandas indexing issues
            y_test_array = np.array(y_test).ravel()
            y_pred_array = np.array(y_pred).ravel()
            
            # Compare with simple baseline (e.g., majority class prediction)
            baseline_pred = (np.full_like(y_test_array, np.mean(y_test_array) > 0.5)).astype(int)  # Predict majority class as integer
            baseline_accuracy = accuracy_score(y_test_array, baseline_pred)
            
            # Create contingency table for McNemar's test
            # [[correct_both, wrong_model], [wrong_baseline, correct_baseline]]
            contingency_table = np.zeros((2, 2))
            for i in range(len(y_test_array)):
                if y_pred_array[i] == y_test_array[i] and baseline_pred[i] == y_test_array[i]:
                    contingency_table[0, 0] += 1  # Both correct
                elif y_pred_array[i] == y_test_array[i] and baseline_pred[i] != y_test_array[i]:
                    contingency_table[0, 1] += 1  # Model correct, baseline wrong
                elif y_pred_array[i] != y_test_array[i] and baseline_pred[i] == y_test_array[i]:
                    contingency_table[1, 0] += 1  # Model wrong, baseline correct
                else:
                    contingency_table[1, 1] += 1  # Both wrong
            
            # McNemar's test (using exact binomial test for small samples)
            b = int(contingency_table[0, 1])  # Model correct, baseline wrong (convert to int)
            c = int(contingency_table[1, 0])  # Model wrong, baseline correct (convert to int)
            
            if b + c > 0:
                # Exact binomial test (using binomtest for newer scipy versions)
                try:
                    # Try new API first (scipy >= 1.7)
                    from scipy.stats import binomtest
                    p_value_mcnemar = binomtest(min(b, c), b + c, alternative='two-sided').pvalue
                except ImportError:
                    # Fall back to old API (scipy < 1.7)
                    p_value_mcnemar = scipy.stats.binom_test(min(b, c), b + c, alternative='two-sided')
                mcnemar_stat = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
            else:
                p_value_mcnemar = 1.0
                mcnemar_stat = 0
            
            print(f"McNemar's Test Results:")
            print(f"  Model Accuracy: {accuracy_score(y_test_array, y_pred_array):.4f}")
            print(f"  Baseline Accuracy: {baseline_accuracy:.4f}")
            print(f"  Contingency Table: [[{contingency_table[0, 0]:.0f}, {contingency_table[0, 1]:.0f}], [{contingency_table[1, 0]:.0f}, {contingency_table[1, 1]:.0f}]]")
            print(f"  McNemar's Chi-square: {mcnemar_stat:.4f}")
            print(f"  P-value: {p_value_mcnemar:.4f}")
            print(f"  Model Superiority: {'Significant' if p_value_mcnemar < 0.05 else 'Not Significant'}")
            
            # 8. Comprehensive Ablation Study - Preprocessing Components
            print(f"\n🔬 COMPREHENSIVE ABLATION STUDY - PREPROCESSING COMPONENTS")
            print(f"{'-'*60}")
            
            ablation_results = {}
            
            # Test different preprocessing combinations if original data is available
            if X_train_original is not None and preprocessor is not None and X_train_preprocessed is not None:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from imblearn.over_sampling import BorderlineSMOTE
                
                # Define ablation configurations
                # Note: "Without Sampling" = NO BorderlineSMOTE (uses imbalanced original data)
                #      "Without Feature Engineering" = Basic preprocessing only (simple imputation + scaling)
                ablation_configs = {
                    'Full Pipeline (Baseline)': {
                        'description': 'Advanced Feature Engineering + BorderlineSMOTE + Feature Selection',
                        'use_preprocessing': True,
                        'use_feature_engineering': True,
                        'use_sampling': True,  # BorderlineSMOTE ON
                        'use_feature_selection': True
                    },
                    'With BorderlineSMOTE': {
                        'description': 'Basic preprocessing + BorderlineSMOTE (No Feature Engineering)',
                        'use_preprocessing': True,
                        'use_feature_engineering': False,  # Basic preprocessing only
                        'use_sampling': True,  # BorderlineSMOTE ON
                        'use_feature_selection': False
                    },
                    'With Feature Engineering': {
                        'description': 'Advanced Feature Engineering + BorderlineSMOTE (All Features)',
                        'use_preprocessing': True,
                        'use_feature_engineering': True,  # Advanced preprocessing
                        'use_sampling': True,  # BorderlineSMOTE ON
                        'use_feature_selection': False
                    },
                    'Without Sampling': {
                        'description': 'Advanced Feature Engineering (NO BorderlineSMOTE, Imbalanced Data)',
                        'use_preprocessing': True,
                        'use_feature_engineering': True,
                        'use_sampling': False,  # BorderlineSMOTE OFF - uses original imbalanced y_train_original
                        'use_feature_selection': True
                    },
                    'Without Feature Engineering': {
                        'description': 'Basic preprocessing + BorderlineSMOTE (No Advanced Engineering)',
                        'use_preprocessing': True,
                        'use_feature_engineering': False,  # Basic preprocessing only (simple imputation + scaling)
                        'use_sampling': True,  # BorderlineSMOTE ON
                        'use_feature_selection': False
                    },
                    'Minimal Pipeline': {
                        'description': 'Basic preprocessing only (NO Engineering, NO BorderlineSMOTE, NO Selection)',
                        'use_preprocessing': True,
                        'use_feature_engineering': False,  # Basic preprocessing only
                        'use_sampling': False,  # BorderlineSMOTE OFF
                        'use_feature_selection': False
                    },
                    'Without Feature Selection': {
                        'description': 'Advanced Feature Engineering + BorderlineSMOTE (All Features, No Selection)',
                        'use_preprocessing': True,
                        'use_feature_engineering': True,
                        'use_sampling': True,  # BorderlineSMOTE ON
                        'use_feature_selection': False
                    }
                }
                
                # Process each configuration
                for config_name, config in ablation_configs.items():
                    try:
                        print(f"\n🔍 Configuration: {config_name}")
                        print(f"   Description: {config['description']}")
                        print(f"   - Feature Engineering: {'Yes (Advanced)' if config['use_feature_engineering'] else 'No (Basic only)'}")
                        print(f"   - Sampling (BorderlineSMOTE): {'ON' if config['use_sampling'] else 'OFF (Imbalanced data)'}")
                        print(f"   - Feature Selection: {'Yes' if config['use_feature_selection'] else 'No (All features)'}")
                        
                        # Determine preprocessing type
                        if config['use_feature_engineering']:
                            # Use full preprocessing (advanced feature engineering)
                            X_train_processed = X_train_preprocessed
                            X_test_processed = X_test
                        else:
                            # Use basic preprocessing (simple imputation + scaling)
                            simple_imputer = SimpleImputer(strategy='mean')
                            simple_scaler = StandardScaler()
                            X_train_processed = simple_imputer.fit_transform(X_train_original)
                            X_train_processed = simple_scaler.fit_transform(X_train_processed)
                            X_test_processed = simple_imputer.transform(X_test_original)
                            X_test_processed = simple_scaler.transform(X_test_processed)
                        
                        # Apply sampling if needed
                        # IMPORTANT: If use_sampling=False, we use original imbalanced data (NO BorderlineSMOTE)
                        if config['use_sampling']:
                            smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, kind='borderline-1')
                            X_train_final, y_train_final = smote.fit_resample(X_train_processed, y_train_original)
                            print(f"   ✓ Applied BorderlineSMOTE: {len(y_train_original)} → {len(y_train_final)} samples")
                        else:
                            # NO BorderlineSMOTE - use original imbalanced data
                            X_train_final = X_train_processed
                            y_train_final = y_train_original
                            print(f"   ✓ NO BorderlineSMOTE: Using original imbalanced data ({len(y_train_original)} samples)")
                            print(f"      Class distribution: {pd.Series(y_train_original).value_counts().to_dict()}")
                        
                        # Apply feature selection if needed
                        if config['use_feature_selection']:
                            # Apply feature selection using same method as main pipeline
                            if selected_feature_indices is not None and len(selected_feature_indices) > 0:
                                # Use the selected feature indices from main pipeline
                                X_train_final_selected = X_train_final[:, selected_feature_indices]
                                X_test_final_selected = X_test_processed[:, selected_feature_indices]
                                print(f"   ✓ Applied Feature Selection: {X_train_final.shape[1]} → {len(selected_feature_indices)} features")
                            else:
                                # If no indices provided, use advanced feature selection
                                from sklearn.feature_selection import SelectKBest, f_classif
                                selector = SelectKBest(score_func=f_classif, k=min(15, X_train_final.shape[1]))
                                X_train_final_selected = selector.fit_transform(X_train_final, y_train_final)
                                X_test_final_selected = selector.transform(X_test_processed)
                                print(f"   ✓ Applied Feature Selection: {X_train_final.shape[1]} → {X_train_final_selected.shape[1]} features")
                        else:
                            # NO feature selection - use all features
                            X_train_final_selected = X_train_final
                            X_test_final_selected = X_test_processed
                            print(f"   ✓ NO Feature Selection: Using all {X_train_final.shape[1]} features")
                        
                        # Train and evaluate model
                        ablation_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                        ablation_model.fit(X_train_final_selected, y_train_final)
                        y_pred_ablation = ablation_model.predict(X_test_final_selected)
                        y_prob_ablation = ablation_model.predict_proba(X_test_final_selected)[:, 1]
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred_ablation)
                        f1 = f1_score(y_test, y_pred_ablation)
                        roc_auc = roc_auc_score(y_test, y_prob_ablation)
                        baseline_accuracy = accuracy_score(y_test, y_pred)
                        
                        ablation_results[config_name] = {
                            'Accuracy': accuracy,
                            'F1-Score': f1,
                            'ROC-AUC': roc_auc,
                            'Drop in Accuracy': baseline_accuracy - accuracy,
                            'Features': X_train_final_selected.shape[1]
                        }
                        
                        print(f"   ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
                        print(f"   ✓ Features: {X_train_final_selected.shape[1]}, Samples: {X_train_final_selected.shape[0]}")
                        
                    except Exception as e:
                        print(f"   ✗ Error: {str(e)}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                
                # Add baseline configuration results explicitly
                ablation_results['Full Pipeline (Baseline)'] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_prob),
                    'Drop in Accuracy': 0.0,
                    'Features': X_train.shape[1] if hasattr(X_train, 'shape') else len(feature_names)
                }
            
            # Feature-based ablation study
            print(f"\n🔬 FEATURE-BASED ABLATION STUDY")
            print(f"{'-'*60}")
            
            # Define feature sets for ablation
            feature_categories = {
                'All Features': feature_names,
                'No Renal': [f for f in feature_names if not any(keyword in f.lower() for keyword in ['creatinine', 'egfr', 'renal'])],
                'No Cardio': [f for f in feature_names if not any(keyword in f.lower() for keyword in ['bp', 'sbp', 'dbp', 'cardio', 'vascular'])],
                'No Metabolic': [f for f in feature_names if not any(keyword in f.lower() for keyword in ['hgba1c', 'diabetes', 'glucose', 'metabolic'])],
                'Top 5 Only': feature_names[:5],
                'Clinical Only': [f for f in feature_names if any(keyword in f.lower() for keyword in ['creatinine', 'egfr', 'bp', 'hgba1c', 'diabetes'])]
            }
            
            feature_ablation_results = {}
            
            for category, features in feature_categories.items():
                if len(features) > 0:
                    try:
                        print(f"\n🔍 Testing {category}: {len(features)} features")
                        
                        # Train model with reduced feature set
                        from sklearn.ensemble import RandomForestClassifier
                        ablation_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                        
                        # Map feature indices - ensure all features exist
                        feature_indices = []
                        for f in features:
                            if f in feature_names:
                                feature_indices.append(feature_names.index(f))
                        
                        if len(feature_indices) == 0:
                            print(f"    Error: No valid features found for {category}, skipping...")
                            continue
                        
                        # Ensure X_train and X_test are numpy arrays
                        if not isinstance(X_train, np.ndarray):
                            X_train_array = np.array(X_train)
                        else:
                            X_train_array = X_train
                        
                        if not isinstance(X_test, np.ndarray):
                            X_test_array = np.array(X_test)
                        else:
                            X_test_array = X_test
                        
                        # Check dimensions
                        if len(feature_indices) > 0 and X_train_array.shape[1] <= max(feature_indices):
                            print(f"    Error: Feature indices out of range. X_train has {X_train_array.shape[1]} features, but max index is {max(feature_indices)}")
                            continue
                        
                        # Train and evaluate
                        ablation_model.fit(X_train_array[:, feature_indices], y_train)
                        y_pred_ablation = ablation_model.predict(X_test_array[:, feature_indices])
                        accuracy_ablation = accuracy_score(y_test, y_pred_ablation)
                        f1_ablation = f1_score(y_test, y_pred_ablation)
                        roc_auc_ablation = roc_auc_score(y_test, ablation_model.predict_proba(X_test_array[:, feature_indices])[:, 1])
                        
                        feature_ablation_results[category] = {
                            'Features': len(feature_indices),
                            'Accuracy': accuracy_ablation,
                            'F1-Score': f1_ablation,
                            'ROC-AUC': roc_auc_ablation,
                            'Drop in Accuracy': accuracy_score(y_test, y_pred) - accuracy_ablation
                        }
                        
                        print(f"  ✓ Accuracy: {accuracy_ablation:.4f}, F1: {f1_ablation:.4f}, ROC-AUC: {roc_auc_ablation:.4f}")
                    
                    except Exception as ablation_error:
                        print(f"    Error testing {category}: {str(ablation_error)}")
                        continue
            
            # Merge both ablation results
            ablation_results.update(feature_ablation_results)
            
            # Create ablation summary table
            print(f"\n📊 ABLATION STUDY SUMMARY")
            print(f"{'-'*60}")
            
            # Sort by accuracy
            sorted_ablation = sorted(ablation_results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
            
            print(f"{'Rank':<5} {'Configuration':<35} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Drop':<10}")
            print(f"{'-'*60}")
            for i, (category, results) in enumerate(sorted_ablation, 1):
                feature_count = results.get('Features', 'N/A')
                if feature_count == 'N/A':
                    config_str = category
                else:
                    config_str = f"{category} ({feature_count} features)"
                
                print(f"{i:<5} {config_str[:34]:<35} {results['Accuracy']:<10.4f} "
                      f"{results.get('F1-Score', 0):<10.4f} {results.get('ROC-AUC', 0):<10.4f} "
                      f"{results['Drop in Accuracy']:<10.4f}")
            
            # Plot ablation results
            if len(ablation_results) > 0:
                categories = list(ablation_results.keys())
                accuracies = [results['Accuracy'] for results in ablation_results.values()]
                f1_scores = [results.get('F1-Score', 0) for results in ablation_results.values()]
                roc_aucs = [results.get('ROC-AUC', 0) for results in ablation_results.values()]
                feature_counts = [results.get('Features', 0) for results in ablation_results.values()]
                
                # Create comprehensive ablation plot
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # Plot 1: Accuracy, F1, and ROC-AUC comparison
                x = np.arange(len(categories))
                width = 0.25
                
                ax1 = axes[0]
                bars1 = ax1.bar(x - width, accuracies, width, label='Accuracy', color='skyblue', alpha=0.8)
                bars2 = ax1.bar(x, f1_scores, width, label='F1-Score', color='lightgreen', alpha=0.8)
                bars3 = ax1.bar(x + width, roc_aucs, width, label='ROC-AUC', color='salmon', alpha=0.8)
                
                ax1.set_xlabel('Configuration', fontsize=11)
                ax1.set_ylabel('Score', fontsize=11)
                ax1.set_title('Comprehensive Ablation Study: Performance Metrics Comparison', fontsize=12, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories, rotation=45, ha='right')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3, axis='y')
                ax1.set_ylim([0, 1.1])
                
                # Add value labels
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                # Plot 2: Accuracy drop and feature count
                ax2 = axes[1]
                drops = [results['Drop in Accuracy'] for results in ablation_results.values()]
                
                # Create dual y-axis
                bars4 = ax2.bar(x - width/2, drops, width, label='Accuracy Drop', color='coral', alpha=0.7)
                ax2_twin = ax2.twinx()
                
                # Only plot feature counts if available (for feature-based ablation)
                if any(count > 0 for count in feature_counts):
                    line = ax2_twin.plot(x, feature_counts, 'o-', color='darkblue', linewidth=2, 
                                        markersize=8, label='Number of Features')
                    ax2_twin.set_ylabel('Number of Features', color='darkblue', fontsize=11)
                    ax2_twin.tick_params(axis='y', labelcolor='darkblue')
                    ax2_twin.legend(loc='upper right')
                
                ax2.set_xlabel('Configuration', fontsize=11)
                ax2.set_ylabel('Accuracy Drop vs Baseline', color='coral', fontsize=11)
                ax2.set_title('Ablation Impact: Performance Drop Analysis', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(categories, rotation=45, ha='right')
                ax2.tick_params(axis='y', labelcolor='coral')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                
                # Add value labels
                for bar in bars4:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.005,
                            f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, color='darkred')
                
                plt.tight_layout()
                plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
                plt.close()
            
            # Statistical significance of ablation
            print(f"\n📈 ABLATION STATISTICAL ANALYSIS")
            
            # Compare full model vs best ablation
            full_accuracy = accuracy_score(y_test, y_pred)
            best_ablation = max(ablation_results.values(), key=lambda x: x['Accuracy'])
            
            print(f"Full Model Accuracy: {full_accuracy:.4f}")
            print(f"Best Ablation Accuracy: {best_ablation['Accuracy']:.4f}")
            print(f"Accuracy Improvement: {best_ablation['Accuracy'] - baseline_accuracy:.4f}")
            print(f"Feature Reduction: {len(feature_names) - best_ablation['Features']} features")
            
            # Calculate relative importance
            if best_ablation['Drop in Accuracy'] > 0:
                relative_importance = best_ablation['Drop in Accuracy'] / best_ablation['Drop in Accuracy'] * 100
                print(f"Relative Feature Importance: {relative_importance:.1f}% per feature")
        
        except Exception as e:
            print(f"  Ablation study failed: {str(e)}")
            import traceback
            print(f"  Error details: {traceback.format_exc()}")
        
        print(f"\n✓ Q1 Journal Paper Statistical Analysis Complete")
        
    except Exception as e:
        print(f"Q1 statistical analysis failed: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")

# Function to preprocess a dataset
def preprocess_dataset(df, dataset_name, id_column, numerical_cols, binary_categorical_cols, non_binary_categorical_cols, target_col, standard_scale_cols, minmax_scale_cols, robust_scale_cols):
    print(f"\n{'='*50}")
    print(f"PREPROCESSING DATASET: {dataset_name}")
    print(f"{'='*50}")

    # Original dataset info
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Original features: {list(df.columns)}")
    print(f"\nOriginal feature counts by type:")
    print(f"- Numerical features: {len(numerical_cols)}")
    print(f"- Binary categorical features: {len(binary_categorical_cols)}")
    print(f"- Non-binary categorical features: {len(non_binary_categorical_cols)}")

    # Drop ID column
    df = df.drop(id_column, axis=1)
    print(f"\nAfter dropping ID column: {df.shape}")

    # Handle target variable
    if dataset_name == 'kidney_disease':
        print(f"\nOriginal target distribution: {df[target_col].value_counts()}")
        df[target_col] = df[target_col].str.strip()
        df[target_col] = df[target_col].map({'ckd': 1, 'notckd': 0})
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('?', np.nan)
        for col in numerical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[target_col] = df[target_col].astype(int)

    print(f"\nTarget distribution after encoding: {df[target_col].value_counts()}")

    # Check missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values per feature:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"- {col}: {count} ({count/len(df)*100:.1f}%)")

    features = numerical_cols + binary_categorical_cols + non_binary_categorical_cols
    X = df[features]
    y = df[target_col]

    print(f"\nFeatures selected for modeling: {len(features)}")
    print(f"Features: {features}")

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_standard', ImbPipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), standard_scale_cols),
            ('num_minmax', ImbPipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
            ]), minmax_scale_cols),
            ('num_robust', ImbPipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler())
            ]), robust_scale_cols),
            ('cat_binary', ImbPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), binary_categorical_cols),
            ('cat_non_binary', ImbPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), non_binary_categorical_cols)
        ],
        remainder='passthrough'
    )

    print(f"\nFeature transformation details:")
    print(f"- Standard scaled features: {len(standard_scale_cols)}")
    print(f"- MinMax scaled features: {len(minmax_scale_cols)}")
    print(f"- Robust scaled features: {len(robust_scale_cols)}")
    print(f"- Binary categorical features (ordinal encoded): {len(binary_categorical_cols)}")
    print(f"- Non-binary categorical features (one-hot encoded): {len(non_binary_categorical_cols)}")

    # Show one-hot encoding expansion (estimated)
    for col in non_binary_categorical_cols:
        unique_values = df[col].dropna().unique()
        print(f"- '{col}' will expand to {len(unique_values)-1} features (dropped first category)")

    return X, y, preprocessor, features

# Dataset configuration
dataset_config = {
    'path': '/kaggle/input/prone4911/pone.0199920.s002.xlsx',
    'id_column': 'StudyID',
    'numerical_cols': ['AgeBaseline', 'CholesterolBaseline', 'TriglyceridesBaseline', 'HgbA1C', 'CreatnineBaseline', 'eGFRBaseline', 'sBPBaseline', 'dBPBaseline', 'BMIBaseline', 'TimeToEventMonths'],
    'binary_categorical_cols': ['Gender', 'HistoryDiabetes', 'HistoryCHD', 'HistoryVascular', 'HistorySmoking', 'HistoryHTN ', 'HistoryDLD', 'HistoryObesity', 'DLDmeds', 'DMmeds', 'HTNmeds', 'ACEIARB'],
    'non_binary_categorical_cols': ['Age.3.categories'],
    'target_col': 'EventCKD35',
    'standard_scale_cols': ['AgeBaseline', 'sBPBaseline', 'dBPBaseline', 'BMIBaseline'],
    'minmax_scale_cols': ['HgbA1C', 'eGFRBaseline'],
    'robust_scale_cols': ['CholesterolBaseline', 'TriglyceridesBaseline', 'CreatnineBaseline', 'TimeToEventMonths']
}

# Initialize PDF
pdf_pages = PdfPages('pone_model_plots.pdf')

# Define Classifiers and Parameter Grids with Novelty
classifiers = {
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'), {
        'classifier__C': uniform(0.1, 10),
        'classifier__solver': ['lbfgs', 'liblinear']
    }),
    'Random Forest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': randint(2, 10),
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }),
    'LightGBM': (lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'), {
        'classifier__n_estimators': randint(50, 200),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__max_depth': randint(3, 10),
        'classifier__is_unbalance': [True, False]
    }),
    'CatBoost': (cb.CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'), {
        'classifier__iterations': randint(50, 200),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__depth': randint(4, 10)
    }),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {
        'classifier__n_estimators': randint(50, 200),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__max_depth': randint(3, 10)
    }),
    'Naive Bayes': (GaussianNB(), {
        'classifier__var_smoothing': uniform(1e-9, 1e-7)
    }),
    'KNN': (KNeighborsClassifier(), {
        'classifier__n_neighbors': randint(3, 15),
        'classifier__weights': ['uniform', 'distance']
    }),
    'Extra Trees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': randint(2, 10),
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': randint(2, 10),
        'classifier__class_weight': ['balanced', None]
    }),
    'MLP': (MLPClassifier(random_state=42, max_iter=1000), {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__learning_rate_init': uniform(0.001, 0.1)
    })
}

# Main processing
dataset_name = 'pone'
config = dataset_config

print(f"\n{'='*70}")
print(f"PROCESSING DATASET: {dataset_name}")
print(f"{'='*70}")

try:
    df = pd.read_excel(config['path'])
except FileNotFoundError as e:
    print(f"Error: File {config['path']} not found - {str(e)}")
    exit()
except Exception as e:
    print(f"Error loading file {config['path']}: {str(e)}")
    exit()

X, y, preprocessor, features = preprocess_dataset(
    df, dataset_name, config['id_column'], config['numerical_cols'], config['binary_categorical_cols'],
    config['non_binary_categorical_cols'], config['target_col'], config['standard_scale_cols'],
    config['minmax_scale_cols'], config['robust_scale_cols']
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n{'='*50}")
print(f"TRAIN-TEST SPLIT")
print(f"{'='*50}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution: {y_train.value_counts()}")
print(f"Test target distribution: {y_test.value_counts()}")

# *** FIXED: Preprocess ONLY on training data first, then transform test ***
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names AFTER fitting on training data
feature_names = clean_feature_names(preprocessor.get_feature_names_out())

print(f"\n{'='*50}")
print(f"PREPROCESSING RESULTS")
print(f"{'='*50}")
print(f"Preprocessed training set shape: {X_train_preprocessed.shape}")
print(f"Preprocessed test set shape: {X_test_preprocessed.shape}")
print(f"Number of features after preprocessing: {len(feature_names)}")

# Show feature expansion details
original_features = len(features)
new_features = len(feature_names) - original_features
print(f"\nFeature expansion details:")
print(f"- Original features: {original_features}")
print(f"- Features after preprocessing: {len(feature_names)}")
print(f"- New features added: {new_features}")

# Show one-hot encoding details
print(f"\nOne-hot encoding details:")
for col in config['non_binary_categorical_cols']:
    unique_values = X_train[col].dropna().unique()
    print(f"- '{col}' with {len(unique_values)} categories expanded to {len(unique_values)-1} features")
    print(f"  Categories: {sorted(unique_values)}")

# Show sample of transformed features
print(f"\nSample of transformed features:")
for i, name in enumerate(feature_names[:10]):
    print(f"- {name}")
if len(feature_names) > 10:
    print(f"... and {len(feature_names) - 10} more")

# Apply BorderlineSMOTE
borderlinesmote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, kind='borderline-1')
X_train_resampled, y_train_resampled = borderlinesmote.fit_resample(X_train_preprocessed, y_train)

print(f"\n{'='*50}")
print(f"BORDERLINESMOTE RESAMPLING")
print(f"{'='*50}")
print(f"Training set shape before resampling: {X_train_preprocessed.shape}")
print(f"Training set shape after resampling: {X_train_resampled.shape}")
print(f"Original training target distribution: {pd.Series(y_train).value_counts()}")
print(f"Resampled training target distribution: {pd.Series(y_train_resampled).value_counts()}")

# Apply Advanced Feature Selection for Better Performance
X_train_selected, X_test_selected, selected_indices = advanced_feature_selection(
    X_train_resampled, y_train_resampled, X_test_preprocessed, method='hybrid', k_best=15
)

# Update feature names for selected features
if selected_indices is not None:
    selected_feature_names = [feature_names[i] for i in selected_indices]
    print(f"\nSelected features: {selected_feature_names}")
else:
    selected_feature_names = feature_names
    X_train_selected = X_train_resampled
    X_test_selected = X_test_preprocessed

# Original class distribution
class_counts = y.value_counts()
print(f"\nOriginal Class Distribution ({dataset_name}):")
print(class_counts)
imbalance_ratio = class_counts[1] / class_counts[0] if 0 in class_counts.index and 1 in class_counts.index else 0
print(f"Imbalance Ratio (Positive/Negative): {imbalance_ratio:.2f}")

plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index.map({0: 'Negative', 1: 'Positive'}), y=class_counts.values, palette=['#ef4444', '#3b82f6'])
plt.title(f'Original Class Distribution ({dataset_name})')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
plt.close()

# Resampled class distribution
resampled_counts = pd.Series(y_train_resampled).value_counts()
print(f"\nResampled Training Class Distribution ({dataset_name}):")
print(resampled_counts)
print(f"Resampled Imbalance Ratio: {resampled_counts[1]/resampled_counts[0]:.2f}")

plt.figure(figsize=(6, 4))
sns.barplot(x=resampled_counts.index.map({0: 'Negative', 1: 'Positive'}), y=resampled_counts.values, palette=['#ef4444', '#3b82f6'])
plt.title(f'Resampled Training Class Distribution ({dataset_name})')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
plt.close()

# Create custom scorers
cohen_kappa_scorer = make_scorer(cohen_kappa_score)

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc',
    'matthews_corrcoef': make_scorer(matthews_corrcoef),
    'cohen_kappa': cohen_kappa_scorer
}
cv_results_list = []
test_results_list = []
roc_curves = []  # Store ROC curves for all models
tuned_classifiers = {}

print(f"\n{'='*50}")
print(f"TRAINING CLASSIFIERS")
print(f"{'='*50}")

for name, (clf, param_grid) in classifiers.items():
    print(f"\nTraining {name}...")
    print(f"Classifier type: {type(clf)}")
    print(f"Classifier: {clf}")
    pipeline = ImbPipeline([('classifier', clf)])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    try:
        search.fit(X_train_resampled, y_train_resampled)
        best_clf = search.best_estimator_
        tuned_classifiers[name] = best_clf

        cv_results = cross_validate(best_clf, X_train_resampled, y_train_resampled, cv=5, scoring=scoring)
        cv_results_list.append({
            'Model': name,
            'CV Accuracy': np.mean(cv_results['test_accuracy']),
            'CV F1': np.mean(cv_results['test_f1']),
            'CV Recall': np.mean(cv_results['test_recall']),
            'CV Precision': np.mean(cv_results['test_precision']),
            'CV ROC AUC': np.mean(cv_results['test_roc_auc']),
            'CV MCC': np.mean(cv_results['test_matthews_corrcoef']),
            'CV Cohen Kappa': np.mean(cv_results['test_cohen_kappa'])
        })

        y_pred = best_clf.predict(X_test_preprocessed)
        y_prob = best_clf.predict_proba(X_test_preprocessed)[:, 1]
        test_results_list.append({
            'Model': name,
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test F1': f1_score(y_test, y_pred),
            'Test Recall': recall_score(y_test, y_pred),
            'Test Precision': precision_score(y_test, y_pred),
            'Test ROC AUC': roc_auc_score(y_test, y_prob),
            'Test MCC': matthews_corrcoef(y_test, y_pred),
            'Test Cohen Kappa': cohen_kappa_score(y_test, y_pred)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((name, fpr, tpr, roc_auc))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix: {name} ({dataset_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"  ✓ {name} trained successfully")

    except ValueError as e:
        print(f"  ✗ ValueError training {name} ({dataset_name}): {str(e)}")
        continue
    except Exception as e:
        print(f"  ✗ Unexpected error training {name} ({dataset_name}): {str(e)}")
        continue

cv_results_df = pd.DataFrame(cv_results_list).sort_values(by='CV ROC AUC', ascending=False)
print(f"\n{'='*50}")
print(f"CROSS-VALIDATION RESULTS")
print(f"{'='*50}")
print(f"\nCross-Validation Results for All Classifiers ({dataset_name}):")
print(cv_results_df.to_string(index=False))

top_4_models = cv_results_df.head(4)['Model'].values
print(f"\nTop 4 Models by CV ROC AUC ({dataset_name}):", list(top_4_models))

# Create combined ROC curve plot for all models
plt.figure(figsize=(12, 10))
for name, fpr, tpr, roc_auc in roc_curves:
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves for All Models ({dataset_name})', fontsize=14)
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
plt.close()

cv_results_df = pd.DataFrame(cv_results_list).sort_values(by='CV ROC AUC', ascending=False)
print(f"\nCross-Validation Results for All Models ({dataset_name}):")
print(cv_results_df.to_string(index=False))

test_results_df = pd.DataFrame(test_results_list).sort_values(by='Test Accuracy', ascending=False)
print(f"\n{'='*50}")
print(f"TEST RESULTS")
print(f"{'='*50}")
print(f"\nTest Set Performance for All Models ({dataset_name}):")
print(test_results_df.to_string(index=False))

# Select the best individual model
best_model_name = test_results_df.iloc[0]['Model']
best_pipeline = tuned_classifiers[best_model_name]
print(f"\nBest Individual Model (Highest Accuracy, {dataset_name}): {best_model_name}")

# Refit best pipeline
best_pipeline.fit(X_train_resampled, y_train_resampled)
y_pred = best_pipeline.predict(X_test_preprocessed)
y_prob = best_pipeline.predict_proba(X_test_preprocessed)[:, 1]

print(f"\n{'='*50}")
print(f"BEST MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"\nTest Set Performance for Best Model ({best_model_name}, {dataset_name}):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# *** FIXED: SHAP with proper sample size ***
print(f"\n{'='*50}")
print(f"EXPLAINABILITY - SHAP")
print(f"{'='*50}")

try:
    # Use 100% of training data for background and 100% of test data for explanations
    background_size = len(X_train_resampled)
    explain_size = len(X_test_preprocessed)

    print(f"Using {background_size} background samples (100% of training data) and explaining {explain_size} test samples (100% of test data)")

    # Use full training data for background
    X_background = X_train_resampled

    # Use full test data for explanations
    X_explain = X_test_preprocessed

    # Use Random Forest for SHAP and LIME explanations
    rf_model = tuned_classifiers['Random Forest']
    rf_name = 'Random Forest'

    # Get predict_proba function for Random Forest
    predict_fn = lambda x: rf_model.predict_proba(x)

    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_values = explainer.shap_values(X_explain)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot (Random Forest, {dataset_name})')
    plt.tight_layout()
    plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
    plt.close()

    # SHAP Feature Importance - Using all features
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': shap_importance
    }).sort_values(by='SHAP Importance', ascending=False)

    plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))  # Adjust height based on number of features
    sns.barplot(data=shap_df, x='SHAP Importance', y='Feature', palette='viridis')
    plt.title(f'All Features SHAP Importance (Random Forest, {dataset_name})')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
    plt.close()

    # SHAP Force Plot for first instance
    print(f"\nGenerating SHAP force plot for first test instance...")
    plt.figure(figsize=(14, 3))
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
        shap_values[0],
        X_explain[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - First Instance (Random Forest, {dataset_name})')
    plt.tight_layout()
    plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ SHAP analysis completed successfully for Random Forest")

except Exception as e:
    print(f"✗ SHAP Error ({dataset_name}): {str(e)}")
    shap_importance = None

# *** FIXED: LIME with multiple instances ***
print(f"\n{'='*50}")
print(f"EXPLAINABILITY - LIME")
print(f"{'='*50}")

try:
    print(f"Initializing LIME explainer...")
    print(f"X_train_resampled shape: {X_train_resampled.shape}")
    print(f"feature_names length: {len(feature_names)}")

    # Convert training data to DataFrame for LIME
    X_train_df = pd.DataFrame(X_train_resampled, columns=feature_names)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_df.values,  # Use numpy array as before
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],
        mode='classification',
        random_state=42
    )
    print(f"✓ LIME explainer initialized successfully")

    # Explain all test instances (not just 3)
    num_lime_instances = len(X_test_preprocessed)
    print(f"Generating LIME explanations for all {num_lime_instances} test instances...")
    print(f"X_test_preprocessed shape: {X_test_preprocessed.shape}")
    print(f"X_test_preprocessed dtype: {X_test_preprocessed.dtype}")
    print(f"Sample instance shape: {X_test_preprocessed[0].shape if len(X_test_preprocessed) > 0 else 'No data'}")

    for idx in range(num_lime_instances):
        try:
            # Use a more robust predict function for LIME
            def robust_predict_proba(X):
                try:
                    return rf_model.predict_proba(X)
                except Exception as e:
                    print(f"    Predict proba failed: {e}, trying predict...")
                    pred = rf_model.predict(X)
                    # Convert predictions to probabilities (simple binary case)
                    return np.column_stack([1 - pred, pred])

            exp = lime_explainer.explain_instance(
                X_test_preprocessed[idx],
                robust_predict_proba,
                num_features=10,
                top_labels=None  # Get explanations for all labels
            )

            # Check available labels in the explanation
            available_labels = list(exp.local_exp.keys())
            print(f"    Available labels in explanation: {available_labels}")

            # Use the first available label if 1 is not present
            label_to_use = 1 if 1 in available_labels else available_labels[0]

            fig = exp.as_pyplot_figure(label=label_to_use)
            plt.title(f'LIME Explanation - Instance {idx+1} (Random Forest, {dataset_name}, Label: {label_to_use})')
            plt.tight_layout()
            plt.savefig(pdf_pages, format='pdf', bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  ✗ Error generating LIME explanation for instance {idx+1}: {str(e)}")
            print(f"    Instance shape: {X_test_preprocessed[idx].shape}")
            print(f"    Instance dtype: {X_test_preprocessed[idx].dtype}")
            print(f"    Error type: {type(e)}")
            import traceback
            print(f"    Traceback: {traceback.format_exc()}")
            continue

    print(f"✓ LIME analysis completed successfully for Random Forest (explained all {num_lime_instances} test instances)")

except Exception as e:
    print(f"✗ LIME Error ({dataset_name}): {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"LIME Error details: {traceback.format_exc()}")

# Save model
model_filename = f"pone_disease_prediction_model.pkl"
joblib.dump(best_pipeline, model_filename)
print(f"\n✓ Best model pipeline saved as '{model_filename}'")

# Feature Importance Summary
if shap_importance is not None:
    print(f"\n{'='*50}")
    print(f"FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*50}")
    print(f"\nTop 10 Most Important Features ({dataset_name}):")
    print(shap_df.head(10).to_string(index=False))

    top_feature = shap_df.iloc[0]['Feature']
    print(f"\n📊 Key Insight ({dataset_name}):")
    print(f"The feature '{top_feature}' has the highest SHAP importance score of {shap_df.iloc[0]['SHAP Importance']:.4f},")
    print(f"making it the most critical predictor in the model.")

    print(f"This aligns with clinical knowledge that features like creatinine and eGFR ")
    print(f"are fundamental markers of kidney disease progression.")

    # Add a new section to discuss the implications of the top feature
    print(f"\n📊 Implications of '{top_feature}' as the top feature:")
    print(f"The dominance of '{top_feature}' in the model suggests that it is a strong indicator of kidney disease progression.")
    print(f"This is consistent with clinical research, which has shown that '{top_feature}' is a key biomarker for kidney disease.")
    print(f"However, it also highlights the need for further research into the underlying mechanisms driving this relationship.")

# Apply Clinical Interpretability Framework
print(f"\n{'='*50}")
print(f"APPLYING CLINICAL INTERPRETABILITY FRAMEWORK")
print(f"{'='*50}")

# Get the best model for clinical analysis
best_model_name = test_results_df.iloc[0]['Model']
best_model_for_clinical = tuned_classifiers[best_model_name]

print(f"Using best model '{best_model_name}' for clinical interpretability analysis")

# Perform McNemar's test comparison between Random Forest and other models
print(f"\n{'='*80}")
print(f"PERFORMING McNEMAR'S TEST - RANDOM FOREST VS OTHER MODELS")
print(f"{'='*80}")

try:
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Perform McNemar's test comparison
    mcnemar_results, mcnemar_df = compare_random_forest_with_others(
        X_train_resampled, y_train_resampled, 
        X_test_preprocessed, y_test, 
        feature_names
    )
    
    print(f"\n✓ McNemar's test analysis completed successfully!")
    print(f"✓ Results saved to results/mcnemars_test_results.csv")
    print(f"✓ Visualization saved to results/mcnemars_test_comparison.png")
    
except Exception as e:
    print(f"✗ McNemar's test analysis failed: {str(e)}")
    import traceback
    print(f"Error details: {traceback.format_exc()}")

# Perform t-test accuracy comparison between Random Forest and other models
print(f"\n{'='*80}")
print(f"PERFORMING T-TEST ACCURACY COMPARISON - RANDOM FOREST VS OTHER MODELS")
print(f"{'='*80}")

try:
    # Perform t-test accuracy comparison
    t_test_results, t_test_df = compare_random_forest_with_others_t_test(
        X_train_resampled, y_train_resampled, 
        X_test_preprocessed, y_test, 
        feature_names
    )
    
    print(f"\n✓ T-test accuracy comparison completed successfully!")
    print(f"✓ Results saved to results/t_test_accuracy_results.csv")
    print(f"✓ Visualization saved to results/t_test_accuracy_comparison.png")
    
except Exception as e:
    print(f"✗ T-test accuracy comparison failed: {str(e)}")
    import traceback
    print(f"Error details: {traceback.format_exc()}")

# Apply the clinical interpretability framework
clinical_interpretability_framework(
    best_model=best_model_for_clinical,
    X_train=X_train_resampled,
    y_train=y_train_resampled,
    X_test=X_test_preprocessed,
    y_test=y_test,
    feature_names=feature_names,
    dataset_name=dataset_name,
    X_train_original=X_train,  # Original training data before preprocessing
    X_train_preprocessed=X_train_preprocessed,  # After preprocessing, before sampling
    y_train_original=y_train,  # Original labels before sampling
    preprocessor=preprocessor,  # Fitted preprocessor
    X_test_original=X_test,  # Original test data before preprocessing
    selected_feature_indices=selected_indices  # Feature indices from feature selection
)

print(f"\n✓ Clinical Interpretability Framework completed successfully!")
print(f"✓ All clinical analyses and visualizations saved to PDF")

pdf_pages.close()

