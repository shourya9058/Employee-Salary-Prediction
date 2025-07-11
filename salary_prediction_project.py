# employee_salary_prediction.py

# ✅ 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 2. Load the dataset
df = pd.read_csv("adult 3.csv")  # Ensure this CSV is in the same folder
print("Initial Dataset:")
print(df.head())

# ✅ 3. Data Cleaning
# Replace '?' with NaN and drop rows with missing values
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ 4. Label Encoding (for categorical columns)
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nAfter Label Encoding:")
print(df.head())

# ✅ 5. Feature Selection
X = df.drop("income", axis=1)
y = df["income"]  # Target column

# ✅ 6. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ✅ 7. Model Training: Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ 8. Prediction and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ 9. Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ✅ 10. Feature Importance Graph
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
