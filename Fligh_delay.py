# Flight Delay Analysis and Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
flight_data = pd.read_csv('C:/Users/jayde/Downloads/Flight_Delay_analysis/Airline_Delay_Cause.csv')
print("Loaded data with shape:", flight_data.shape)

# 2. Clean & Prepare Data
flight_data.dropna(inplace=True)
flight_data['Delayed'] = np.where(flight_data['arr_del15'] > 0, 1, 0)

# 3. Full EDA Report
plt.figure(figsize=(8, 5))
sns.histplot(flight_data['arr_delay'], bins=50, kde=True)
plt.title("Arrival Delay Distribution")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='arr_delay', data=flight_data)
plt.title("Delay by Month")
plt.xlabel("Month")
plt.ylabel("Arrival Delay")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='carrier', y='arr_delay', data=flight_data)
plt.title("Delay by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Arrival Delay")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(flight_data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

flight_data[['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']].mean().plot(kind='bar', figsize=(10,5))
plt.title("Average Delay by Cause")
plt.ylabel("Minutes")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 4. Classification Model
features = ['month', 'arr_flights', 'carrier']
X = pd.get_dummies(flight_data[features], drop_first=True)
y = flight_data['Delayed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- Classification Metrics (Delay Prediction) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.2f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred):.2f}")

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# 5. Regression Model (Improved: filtering out extreme outliers for better fit)
reg = LinearRegression()
y_delay = flight_data['arr_delay']
valid_indices = y_delay[(y_delay >= 0) & (y_delay <= 500)].index
X_reg = X.loc[valid_indices]
y_reg = y_delay[valid_indices]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_reg_pred = reg.predict(X_test_reg)

print("\n--- Regression Metrics (Delay Duration) ---")
print(f"MAE:  {mean_absolute_error(y_test_reg, y_reg_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_reg_pred)):.2f}")

# 6. SHAP Explainability
print("\n--- SHAP started---")
X_shap = X_test.sample(n=100, random_state=42) if len(X_test) > 100 else X_test
X_shap_numeric = X_shap.astype(float)
explainer = shap.Explainer(clf, X_shap_numeric)
shap_values = explainer(X_shap_numeric)
shap.summary_plot(shap_values, X_shap_numeric, show=True)
print("SHAP summary plot generated.")
print("\n--- SHAP Analysis Complete ---")

# 7. Operational Adjustability Index (OAI)
weights = {
    'carrier_delay': 3,
    'weather_delay': 1,
    'nas_delay': 2,
    'security_delay': 1,
    'late_aircraft_delay': 3
}
oai_components = pd.DataFrame({col: flight_data[col] * weight for col, weight in weights.items()})
oai_score = oai_components.sum(axis=1)
oai_normalized = oai_score / oai_score.max() * 100
print("\n--- Operational Adjustability Index (OAI) ---")
print(f"Average OAI Score (Raw): {oai_score.mean():.2f}")
print(f"Normalized OAI Score (0–100): {oai_normalized.mean():.2f} (lower is better)")

# 8. Root Cause Analysis
# 8. Root Cause Analysis
print("\n--- Root Cause Analysis ---")
print("Top delay causes (mean minutes):")
delay_causes = flight_data[['carrier_delay','late_aircraft_delay','nas_delay','weather_delay','security_delay']]
mean_delays = delay_causes.mean().sort_values(ascending=False)
print(mean_delays)

# Add bar chart for root cause analysis
plt.figure(figsize=(10, 5))
sns.barplot(x=mean_delays.values, y=mean_delays.index, palette="Reds_r")
plt.title("Root Cause Analysis: Mean Delay by Cause")
plt.xlabel("Average Delay (minutes)")
plt.ylabel("Delay Cause")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

print("\n--- Flight Delay Analysis and Prediction Project Completed ---")