# 🏥 Health Insurance Premium Prediction

A Machine Learning web application that predicts **health insurance premiums** based on user information such as age, income, dependants, BMI, smoking status, medical history, insurance plan, and other health-related factors.

The application provides an interactive web interface built using **Streamlit**.

---

## 🚀 Live Application

You can run the application locally using Streamlit.

```bash
streamlit run app.py


📌 Project Overview

The goal of this project is to build a machine learning system that predicts the expected health insurance premium for a customer.

The application takes user information through a Streamlit UI, preprocesses the input in the same way as the training data, and passes it to the appropriate trained machine learning model.

The project uses two separate models:

model_young → for users aged 25 or below
model_rest → for users above 25

Separate scalers are also used for the two age groups.

🖥️ User Interface

The Streamlit application allows the user to enter:

Age
Number of Dependants
Income
Genetical Risk
Insurance Plan
Employment Status
Gender
Marital Status
BMI Category
Smoking Status
Region
Medical History

After entering the information, the user clicks Predict and the application displays the predicted health insurance premium.

🧠 Machine Learning Pipeline

The prediction pipeline follows these steps:

User Input
    ↓
Streamlit UI
    ↓
Input Dictionary
    ↓
Feature Preprocessing
    ↓
Categorical Encoding
    ↓
Medical Risk Score Calculation
    ↓
Feature Scaling
    ↓
Age-based Model Selection
    ↓
Prediction
    ↓
Predicted Insurance Premium
🔧 Feature Engineering

Several preprocessing techniques are used before prediction.

Insurance Plan Encoding

Insurance plans are converted into numerical values:

Bronze → 1
Silver → 2
Gold   → 3
One-Hot Encoding

Categorical variables such as:

Gender
Region
Marital Status
BMI Category
Smoking Status
Employment Status

are converted into numerical dummy variables.

Medical Risk Score

Medical history is converted into a numerical risk score.

For example:

Diabetes            → 6
Heart disease       → 8
High blood pressure → 6
Thyroid             → 5
No Disease          → 0

For multiple diseases, the individual scores are combined and normalized.

🤖 Model Selection

The application uses different models depending on the customer's age.

Age ≤ 25
   ↓
Young Model

Age > 25
   ↓
Rest Model

This allows the model to capture different relationships between customer characteristics and insurance premiums for different age groups.

📁 Project Structure
Health_Primum_Prediction/
│
├── Premium_prediction_app/
│   │
│   ├── app.py
│   ├── prediction_helper.py
│   │
│   └── artifacts/
│       ├── model_young.joblib
│       ├── model_rest.joblib
│       ├── scaler_young.joblib
│       └── scaler_rest.joblib
│
├── requirements.txt
└── README.md
