import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

print("=== Training Gesture Model ===\n")

data_dir = "data/raw"
all_data = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path, header=None)
        all_data.append(df)
        print(f"Loaded: {file} - {len(df)} samples")

data = pd.concat(all_data, ignore_index=True)
print(f"\nTotal samples: {len(data)}")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype(str).values

print("\nClass distribution:")
for label, count in sorted(Counter(y).items(), key=lambda x: str(x[0])):
    print(f"  {label}: {count}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nReport:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
with open("models/gesture_model.pkl", 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved!")