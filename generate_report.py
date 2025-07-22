"""
Salary Prediction - Report Generation Script
This script generates visualizations and metrics for project submission.
It doesn't modify any existing files or models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Set style for better looking plots
plt.style.use('seaborn')
sns.set_palette("husl")

print("="*80)
print("SALARY PREDICTION - PROJECT REPORT")
print("="*80)

# Load the dataset
print("\n📊 Loading dataset...")
df = pd.read_csv("adult 3.csv")

# Basic dataset info
print("\n📋 Dataset Overview:")
print("-"*40)
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print("\n📝 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# Data Cleaning
print("\n🧹 Data Cleaning:")
print("-"*40)
print("Handling missing values...")
df_clean = df.copy()
df_clean.replace(' ?', np.nan, inplace=True)
df_clean.replace('?', np.nan, inplace=True)
initial_rows = len(df_clean)
df_clean.dropna(inplace=True)
dropped_rows = initial_rows - len(df_clean)
print(f"Dropped {dropped_rows} rows with missing values")

# Target distribution
plt.figure(figsize=(10, 6))
target_counts = df_clean['income'].value_counts()
plt.pie(target_counts, 
        labels=target_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=['#66b3ff','#ff9999'])
plt.title('Salary Distribution (<=50K vs >50K)')
plt.tight_layout()
plt.savefig('salary_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: salary_distribution.png")

# Feature Engineering
def create_features(df):
    df = df.copy()
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 25, 35, 45, 55, 65, 100],
                           labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    return df

df_clean = create_features(df_clean)

# Data Preparation
print("\n🔧 Preparing data for modeling...")

# Convert target variable
y = df_clean['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

# Select features
X = df_clean.drop('income', axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Convert all columns to numeric to avoid any string values
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values that might have been created during conversion
valid_rows = ~X_encoded.isnull().any(axis=1)
X_encoded = X_encoded[valid_rows]
y = y[valid_rows]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train a model for visualization
print("\n🤖 Training model for visualization...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

print("Training model with optimized parameters...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Model Evaluation
print("\n📈 Model Performance:")
print("-"*40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2%}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'],
            annot_kws={"size": 14},  # Larger font for annotations
            cbar=False)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('Top 15 Important Features', fontsize=16, pad=20)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on the bars
for i, v in enumerate(feature_importance['importance']):
    ax.text(v + 0.005, i, f"{v:.3f}", color='black', va='center', fontsize=10)
plt.title('Top 15 Important Features')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: feature_importance.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#2ecc71', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ Saved: roc_curve.png")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='#e74c3c', lw=3, label='Precision-Recall curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✅ Saved: precision_recall_curve.png")

# Cross-validation
print("\n🔍 Performing Cross-Validation...")
try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_encoded, y, cv=cv, scoring='accuracy')
    print(f"\n✅ Cross-Validation Results:")
    print("-" * 40)
    print(f"Fold Accuracies: {', '.join([f'{score:.2%}' for score in cv_scores])}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std() * 2:.2f})")
except Exception as e:
    print(f"\n⚠️ Cross-validation skipped due to error: {str(e)}")
    print("Proceeding with model evaluation on test set...")

# Save summary to text file
with open('model_summary.txt', 'w') as f:
    f.write("SALARY PREDICTION - MODEL SUMMARY\n")
    f.write("="*40 + "\n\n")
    f.write(f"Dataset Size: {len(df)} rows, {len(df.columns)} columns\n")
    f.write(f"After Cleaning: {len(df_clean)} rows\n")
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    f.write(f"\nCross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2f}%)\n")
    f.write("\nTop 10 Features:\n")
    f.write(feature_importance.head(10).to_string())

# Create a summary of key metrics for the report
with open('project_summary.txt', 'w') as f:
    f.write("SALARY PREDICTION - PROJECT SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset Size: {len(df):,} rows, {len(df.columns)} columns\n")
    f.write(f"After Cleaning: {len(df_clean):,} rows ({len(df_clean)/len(df):.1%} of original)\n\n")
    f.write("Key Metrics:\n")
    f.write("-"*50 + "\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred):.2%}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred):.2%}\n")
    f.write(f"F1-Score: {f1_score(y_test, y_pred):.2%}\n")
    f.write(f"ROC AUC: {roc_auc:.3f}\n\n")
    f.write("Top 5 Most Important Features:\n")
    f.write("-"*50 + "\n")
    for i, (feature, imp) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):
        if i <= 5:  # Only show top 5
            f.write(f"{i}. {feature}: {imp:.4f}\n")

print("\n✅ Saved: project_summary.txt")
print("\n🎉 REPORT GENERATION COMPLETE! 🎉\n")
print("📊 VISUALIZATIONS SAVED:")
print("1. salary_distribution.png    - Income distribution pie chart")
print("2. confusion_matrix.png       - Model performance matrix")
print("3. feature_importance.png     - Top 15 influential features")
print("4. roc_curve.png             - ROC curve with AUC score")
print("5. precision_recall_curve.png - Precision-Recall curve\n")
print("📝 TEXT FILES SAVED:")
print("6. model_summary.txt         - Detailed model metrics")
print("7. project_summary.txt       - Key project metrics\n")
print("💡 TIP: Use these files for your project submission presentation!")

# Show all plots
plt.show()
