# employee_salary_prediction.py

# ✅ 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings('ignore')

# ✅ 2. Load the dataset
df = pd.read_csv("adult 3.csv")  # Ensure this CSV is in the same folder
print("Initial Dataset:")
print(df.head())

# ✅ 3. Data Cleaning and Feature Engineering
# Replace '?' with NaN and drop rows with missing values
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert target to binary (0/1)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

# Feature Engineering
def create_new_features(df):
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 45, 55, 65, 100],
                            labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    # Create work hours category
    df['hours_category'] = pd.cut(df['hours-per-week'],
                                bins=[0, 30, 40, 50, 100],
                                labels=['part-time', 'full-time', 'overtime', 'double-time'])
    
    # Create education level grouping
    education_map = {
        'Preschool': 'dropout',
        '10th': 'dropout',
        '11th': 'dropout',
        '12th': 'dropout',
        '1st-4th': 'dropout',
        '5th-6th': 'dropout',
        '7th-8th': 'dropout',
        '9th': 'dropout',
        'HS-Grad': 'HighGrad',
        'HS-grad': 'HighGrad',
        'Some-college': 'CommunityCollege',
        'Assoc-acdm': 'CommunityCollege',
        'Assoc-voc': 'CommunityCollege',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters',
        'Prof-school': 'Doctorate',
        'Doctorate': 'Doctorate'
    }
    df['education_group'] = df['education'].map(education_map)
    
    return df

df = create_new_features(df)

# ✅ 4. Prepare data for modeling
# Separate features and target
y = df['income']
X = df.drop('income', axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ✅ 5. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 6. Model Pipeline with Hyperparameter Tuning
# Base model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# Hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with cross-validation
print("Starting GridSearchCV...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Cross-validation scores
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save the label encoders for inference
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(X[col])
    label_encoders[col] = le

# ✅ 7. Prediction and Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🎯 Test Set Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Feature importance (for numeric features)
try:
    # For tree-based models with feature_importances_
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after one-hot encoding
        ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])
        
        # Get feature importances
        importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print top 20 most important features
        print("\nTop 20 Most Important Features:")
        for f in range(min(20, len(all_feature_names))):
            print(f"{f + 1}. {all_feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
except Exception as e:
    print(f"\nCould not compute feature importances: {str(e)}")

# Save the model and preprocessors
import joblib
joblib.dump(model, 'model.joblib')
joblib.dump(label_encoders, 'encoders.joblib')
print("\n✅ Model and encoders saved to disk")

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
