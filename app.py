
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# Dashboard title
st.title("📊 Customer Churn Prediction Dashboard")
st.write(
    "Predict whether a customer is likely to leave "
    "the telecom company."
)

st.divider()

# Customer details
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

with col2:
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0,
        max_value=200.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0,
        max_value=10000.0, value=840.0
    )
    if tenure == 0:
        total_charges = 0.0

st.divider()

# Predict churn
if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Churn Probability", f"{probability:.1%}")

    with metric_col2:
        st.metric(
            "Prediction",
            "Likely to Leave" if prediction == 1 else "Likely to Stay"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.warning(
            "This customer is predicted to churn. "
            "Consider offering support or a suitable retention plan."
        )
    else:
        st.success(
            "This customer is predicted to stay."
        )

st.caption(
    "This prediction is an estimate from a machine-learning model, "
    "not a guarantee of future customer behavior."
)
