import pandas as pd
import joblib
from pathlib import Path

#path
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR.parent / "artifacts"

#load models and scalers files 
model_young = joblib.load(ARTIFACTS / "model_young.joblib")
model_rest = joblib.load(ARTIFACTS / "model_rest.joblib")

scaler_young = joblib.load(ARTIFACTS / "scaler_young.joblib")
scaler_rest = joblib.load(ARTIFACTS / "scaler_rest.joblib")


# Constants

INSURANCE_PLAN_ENCODING = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3
}


RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0
}


EXPECTED_COLUMNS = [
    "age",
    "number_of_dependants",
    "income_lakhs",
    "insurance_plan",
    "genetical_risk",
    "normalized_risk_score",
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity",
    "bmi_category_Overweight",
    "bmi_category_Underweight",
    "smoking_status_Occasional",
    "smoking_status_Regular",
    "employment_status_Salaried",
    "employment_status_Self-Employed"
]

# Medical Risk


def calculate_normalized_risk(medical_history):

    diseases = medical_history.lower().split(" & ")

    total_risk = sum(
        RISK_SCORES.get(disease.strip(), 0)
        for disease in diseases
    )

    return total_risk / 14



# One-Hot Encoding


def encode_categories(df, input_dict):

    if input_dict["Gender"] == "Male":
        df["gender_Male"] = 1

    region_column = {
        "Northwest": "region_Northwest",
        "Southeast": "region_Southeast",
        "Southwest": "region_Southwest"
    }.get(input_dict["Region"])

    if region_column:
        df[region_column] = 1

    if input_dict["Marital Status"] == "Unmarried":
        df["marital_status_Unmarried"] = 1

    bmi_column = {
        "Obesity": "bmi_category_Obesity",
        "Overweight": "bmi_category_Overweight",
        "Underweight": "bmi_category_Underweight"
    }.get(input_dict["BMI Category"])

    if bmi_column:
        df[bmi_column] = 1

    smoking_column = {
        "Occasional": "smoking_status_Occasional",
        "Regular": "smoking_status_Regular"
    }.get(input_dict["Smoking Status"])

    if smoking_column:
        df[smoking_column] = 1

    employment_column = {
        "Salaried": "employment_status_Salaried",
        "Self-Employed": "employment_status_Self-Employed"
    }.get(input_dict["Employment Status"])

    if employment_column:
        df[employment_column] = 1

    return df



# Scaling

def handle_scaling(age, df):

    scaler_object = (
        scaler_young
        if age <= 25
        else scaler_rest
    )

    scaler = scaler_object["scaler"]
    cols_to_scale = scaler_object["cols_to_scale"]

    if "income_level" in cols_to_scale:
        df["income_level"] = 0

    df[cols_to_scale] = scaler.transform(
        df[cols_to_scale]
    )

    if "income_level" in df.columns:
        df.drop(
            columns="income_level",
            inplace=True
        )

    return df



# Preprocessing


def preprocess_input(input_dict):

    df = pd.DataFrame(
        0,
        index=[0],
        columns=EXPECTED_COLUMNS
    )

    # Numeric features
    df.loc[0, "age"] = input_dict["Age"]

    df.loc[0, "number_of_dependants"] = (
        input_dict["Number of Dependants"]
    )

    df.loc[0, "income_lakhs"] = (
        input_dict["Income in Lakhs"]
    )

    df.loc[0, "genetical_risk"] = (
        input_dict["Genetical Risk"]
    )

    # Insurance plan
    df.loc[0, "insurance_plan"] = (
        INSURANCE_PLAN_ENCODING.get(
            input_dict["Insurance Plan"],
            1
        )
    )

    # Medical risk
    df.loc[0, "normalized_risk_score"] = (
        calculate_normalized_risk(
            input_dict["Medical History"]
        )
    )

    # Categorical features
    df = encode_categories(
        df,
        input_dict
    )

    # Scaling
    df = handle_scaling(
        input_dict["Age"],
        df
    )

    # Ensure exact feature order
    df = df[EXPECTED_COLUMNS]

    return df



# Prediction

def predict(input_dict):

    input_df = preprocess_input(input_dict)

    if input_df.isna().any().any():
        raise ValueError(
            "NaN detected in input dataframe"
        )

    model = (
        model_young
        if input_dict["Age"] <= 25
        else model_rest
    )

    prediction = model.predict(input_df)

    return int(prediction[0])