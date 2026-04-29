import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "data/live_processed"

X_all = []
y_all = []

for file in os.listdir(DATA_DIR):
    if file.endswith("_X.npy"):
        label = file.replace("_X.npy", "")
        X = np.load(os.path.join(DATA_DIR, file))

        X_all.append(X)
        y_all.extend([label] * len(X))

X_all = np.vstack(X_all)
y_all = np.array(y_all)

print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
print("Labels:", set(y_all))

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

joblib.dump(model, "models/gesture_model.pkl")

print("Saved live model to models/gesture_model.pkl")