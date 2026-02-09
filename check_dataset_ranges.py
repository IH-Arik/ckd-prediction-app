import pandas as pd
import numpy as np

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

print('=== ACTUAL DATASET VALUE RANGES ===')

# CholesterolBaseline
print('CholesterolBaseline:')
print(f'  Min: {df["CholesterolBaseline"].min():.2f}')
print(f'  Max: {df["CholesterolBaseline"].max():.2f}')
print(f'  Mean: {df["CholesterolBaseline"].mean():.2f}')
print(f'  Sample values: {df["CholesterolBaseline"].head(10).tolist()}')

# Age.3.categories
print('\nAge.3.categories:')
print(f'  Unique values: {sorted(df["Age.3.categories"].unique())}')
print(f'  Value counts: {df["Age.3.categories"].value_counts().sort_index()}')

# AgeBaseline for comparison
print('\nAgeBaseline:')
print(f'  Min: {df["AgeBaseline"].min()}')
print(f'  Max: {df["AgeBaseline"].max()}')
print(f'  Mean: {df["AgeBaseline"].mean():.1f}')

# Check if Age.3.categories is derived from AgeBaseline
print('\nAge vs Age.3.categories relationship:')
sample_ages = df[['AgeBaseline', 'Age.3.categories']].head(20)
for idx, row in sample_ages.iterrows():
    age = row['AgeBaseline']
    cat = row['Age.3.categories']
    calculated_cat = age // 10
    match = 'MATCH' if cat == calculated_cat else 'MISMATCH'
    print(f'  Age: {age}, Cat: {cat}, Calc: {calculated_cat} - {match}')

# Show all unique Age.3.categories values with corresponding age ranges
print('\nAge.3.categories analysis:')
for cat in sorted(df['Age.3.categories'].unique()):
    ages_in_cat = df[df['Age.3.categories'] == cat]['AgeBaseline']
    min_age = ages_in_cat.min()
    max_age = ages_in_cat.max()
    count = len(ages_in_cat)
    print(f'  Category {cat}: Age {min_age}-{max_age} (count: {count})')

# Check other important features
print('\nOther important features:')
features_to_check = ['TriglyceridesBaseline', 'HgbA1C', 'CreatnineBaseline', 'eGFRBaseline', 'sBPBaseline', 'dBPBaseline', 'BMIBaseline']

for feature in features_to_check:
    if feature in df.columns:
        print(f'{feature}:')
        print(f'  Min: {df[feature].min():.2f}')
        print(f'  Max: {df[feature].max():.2f}')
        print(f'  Mean: {df[feature].mean():.2f}')
        print()
