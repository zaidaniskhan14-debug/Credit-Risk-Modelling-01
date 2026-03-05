# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("credit_risk_modelling_dataset.csv")
df.head()

# %%
print("Shape of dataset:", df.shape)
print("\nData Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
df.describe()

# %%
df.fillna(df.median(), inplace=True)

# %%
sns.countplot(x='default', data=df)
plt.title("Default vs Non-Default Distribution")
plt.show()

# %%
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %%
X = df.drop("default", axis=1)
y = df["default"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# %%
y_pred_log = log_model.predict(X_test)

# %%
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# %%
y_prob_log = log_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob_log)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_prob_log))

# %%
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# %%
y_pred_rf = rf_model.predict(X_test)

# %%
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# %%
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance - Random Forest")
plt.show()

# %%
def predict_credit_risk(input_data):
    
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    
    prediction = rf_model.predict(input_data)
    
    if prediction[0] == 1:
        return "High Risk - Loan Likely to Default"
    else:
        return "Low Risk - Loan Safe"

# %%
print("\nEnter Customer Details For Credit Risk Prediction")

income = float(input("Enter Income: "))
loan_amount = float(input("Enter Loan Amount: "))
credit_score = float(input("Enter Credit Score: "))
missed_payments = float(input("Enter Missed Payments: "))

user_data = [income, loan_amount, credit_score, missed_payments]

result = predict_credit_risk(user_data)

print("\nPrediction Result:")
print(result)


