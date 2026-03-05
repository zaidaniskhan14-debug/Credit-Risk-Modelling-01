# 💳 Credit Risk Modelling using Machine Learning

## 📌 Project Overview
This project focuses on predicting whether a customer is likely to default on a loan or not using Machine Learning techniques.

The model analyzes customer financial details such as income, loan amount, credit score, and employment years to classify credit risk as:

- Low Risk (Loan Safe)
- High Risk (Likely to Default)

---

## 🎯 Objective
To build a predictive classification model that helps financial institutions reduce loan default risk by identifying high-risk applicants before loan approval.

---

## 🛠 Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- VS Code

---

## 📊 Machine Learning Models Used
1. Logistic Regression
2. Random Forest Classifier

Random Forest was selected as the final model due to better accuracy and feature importance analysis.

---

## 📂 Project Structure

```
CreditProject/
│
├── credit_risk_model.py
├── credit_data.csv
├── README.md
```

---

## ⚙ Installation & Setup

### Step 1: Clone Repository
```
git clone <your-repo-link>
```

### Step 2: Install Required Libraries
```
pip install pandas numpy scikit-learn joblib
```

### Step 3: Run the Project
```
python credit_risk_model.py
```

---

## 🧠 How It Works

1. Load dataset
2. Handle missing values
3. Split dataset into training & testing
4. Apply feature scaling
5. Train Random Forest model
6. Take user input
7. Predict credit risk

---

## 📈 Model Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Cross Validation Score

---

## 🔍 Key Features

- Binary Classification (Default / No Default)
- Feature Scaling using StandardScaler
- Model Persistence using Joblib
- Console-based User Input Prediction
- Professional deployment-ready Python script

---

## 🚀 Future Improvements

- Web App deployment using Streamlit
- API deployment using Flask
- Cloud deployment (Render / Railway / AWS)
- Hyperparameter tuning
- Handling imbalanced datasets using SMOTE

---

## 👨‍💻 Developed By
Zaid Anis Khan  
BCA – Data Science & Machine Learning Enthusiast  

---

## 📌 Conclusion
This project demonstrates practical implementation of Credit Risk Modelling using supervised machine learning techniques and can be extended into a real-world financial risk assessment system.
