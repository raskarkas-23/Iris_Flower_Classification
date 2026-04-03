import numpy as np
import pandas as pd
import pickle   

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Kasturi\OneDrive\Desktop\Iris.csv")

print("Dataset Loaded Successfully")

# Select Features
X = df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]

# Target variable
y = df['Species']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("Model Training Completed")

# Test Accuracy
y_pred = model.predict(X_test)
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved as model.pkl")