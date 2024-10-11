# Install necessary libraries (if not already installed)
# !pip install pandas scikit-learn shap matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COMPAS dataset
data = pd.read_csv('datasets/compas.csv')

# Data preprocessing
# Selecting relevant features for the analysis
features = [
    'age', 'race', 'sex', 'juv_fel_count', 'juv_misd_count',
    'priors_count', 'c_charge_degree'
]
target = 'two_year_recid'

# Filter the dataset and drop rows with missing values
data = data[features + [target]].dropna()

# Encode categorical features
data['c_charge_degree'] = pd.Categorical(data['c_charge_degree']).codes
data['race'] = pd.Categorical(data['race']).codes
data['sex'] = pd.Categorical(data['sex']).codes

# Split the dataset into features and target
X = data.drop('two_year_recid', axis=1)
y = data['two_year_recid']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression (for interpretability)
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train_scaled, y_train)

# Train Random Forest (for complex pattern detection)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_logreg = logreg_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Evaluate Random Forest
y_pred_rf = random_forest_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Hybrid model: Take the average of both predictions for a balanced prediction
hybrid_pred = (y_pred_logreg + y_pred_rf) / 2
hybrid_pred = np.round(hybrid_pred).astype(int)
print("\nHybrid Model Accuracy:", accuracy_score(y_test, hybrid_pred))
print("Hybrid Model Classification Report:\n", classification_report(y_test, hybrid_pred))

# Confusion Matrix for the Hybrid Model
conf_matrix = confusion_matrix(y_test, hybrid_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Hybrid Model')
plt.show()

# SHAP Explainability for Random Forest
explainer = shap.TreeExplainer(random_forest_model)
shap_values = explainer.shap_values(X_test)

# Plot summary of feature importance using SHAP
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)

# Waterfall plot for an individual prediction
i = 0  # Index of the sample to explain
# Create a SHAP Explanation object for the selected prediction
shap_values_explanation = shap.Explanation(values=shap_values[1][i],
                                             base_values=explainer.expected_value[1],
                                             data=X_test.iloc[i],
                                             feature_names=X.columns)

# Plot the waterfall
shap.waterfall_plot(shap_values_explanation)