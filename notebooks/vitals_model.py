import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load data
df = pd.read_csv('data/vitals/diabetes.csv')

# Fix zero values
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

# Split features and labels
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
with open('models/vitals_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)

# Build MLP model
class VitalsModel(nn.Module):
    def __init__(self):
        super(VitalsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

model = VitalsModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
print("Starting vitals model training...")
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_out = model(X_test)
            pred = torch.argmax(test_out, dim=1)
            acc = (pred == y_test).float().mean() * 100
        print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f} | Test Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), 'models/vitals_model.pth')
print("\nVitals model saved! ✅")