import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
DATA_PATH = "data/customer_data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Target
y = df["Churn"]

# Drop target and keep all other columns as features
X = df.drop("Churn", axis=1)

print("📊 Churn distribution:")
print(y.value_counts())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Model trained successfully!")
print("📈 Accuracy:", accuracy_score(y_test, y_pred))
print("📝 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
MODEL_PATH = "models/logistic_model.pkl"
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved at {MODEL_PATH}")