import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === 1️⃣ Load your dataset ===
df = pd.read_csv("blood_pressure_data.csv")

# Display columns to verify
print("Columns in dataset:", df.columns)

# === 2️⃣ Clean missing data ===
# Use actual column names: 'BMI', 'BP_Sy', 'BP_Di'
df = df.dropna(subset=["BP_Sy", "BP_Di"])

# === 3️⃣ Create a target column if not present ===
# If dataset doesn't have a label column, categorize manually
if 'Category' not in df.columns:
    def categorize_bp(sy, di):
        if sy < 90 or di < 60:
            return "Low"
        elif sy > 140 or di > 90:
            return "High"
        else:
            return "Normal"
    df["Category"] = df.apply(lambda row: categorize_bp(row["BP_Sy"], row["BP_Di"]), axis=1)

# === 4️⃣ Define features (X) and target (y) ===
X = df[["BMI", "BP_Sy", "BP_Di"]]
y = df["Category"]

# === 5️⃣ Split into training/testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 6️⃣ Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 7️⃣ Evaluate performance ===
y_pred = model.predict(X_test)

print("\n✅ Model Training Complete!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 8️⃣ Save trained model ===
joblib.dump(model, "bp_classifier.pkl")
print("\n✅ Model saved successfully as 'bp_classifier.pkl'")
