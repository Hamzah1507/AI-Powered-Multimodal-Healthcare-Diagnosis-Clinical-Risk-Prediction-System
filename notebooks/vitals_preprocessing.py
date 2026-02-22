import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load dataset
df = pd.read_csv('data/vitals/diabetes.csv')
print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing/zero values (in medical data 0 means missing)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nZero values count:")
print((df[cols_with_zeros] == 0).sum())

# Replace 0s with column mean
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].mean())

# Separate features and labels
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use
os.makedirs('models', exist_ok=True)
with open('models/vitals_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nAfter preprocessing:")
print("Features shape:", X_scaled.shape)
print("Labels shape:", y.shape)
print("Mean of scaled data:", X_scaled.mean().round(4))
print("Std of scaled data:", X_scaled.std().round(4))
print("\nVitals preprocessing working! ✅")