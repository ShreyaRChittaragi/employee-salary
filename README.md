# employee_salary_predictor/app.py

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_predictor_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🧑‍🏋 Employee Salary Predictor")

# Inputs for all features
age = st.number_input("Age", 17, 90, 30)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.number_input("fnlwgt", 10000, 1000000, 150000)
education = st.selectbox("Education", encoders['education'].classes_)
educational_num = st.slider("Education Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
race = st.selectbox("Race", encoders['race'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Country", encoders['native-country'].classes_)

# Prepare input for prediction
input_df = pd.DataFrame({
    "age": [age],
    "workclass": [encoders['workclass'].transform([workclass])[0]],
    "fnlwgt": [fnlwgt],
    "education": [encoders['education'].transform([education])[0]],
    "educational-num": [educational_num],
    "marital-status": [encoders['marital-status'].transform([marital_status])[0]],
    "occupation": [encoders['occupation'].transform([occupation])[0]],
    "relationship": [encoders['relationship'].transform([relationship])[0]],
    "race": [encoders['race'].transform([race])[0]],
    "gender": [encoders['gender'].transform([gender])[0]],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss],
    "hours-per-week": [hours_per_week],
    "native-country": [encoders['native-country'].transform([native_country])[0]]
})

# Predict
if st.button("Predict Salary Category"):
    result = model.predict(input_df)[0]
    prediction = encoders['income'].inverse_transform([result])[0]
    st.success(f"💰 Predicted Salary Category: {prediction}")


# employee_salary_predictor/README.md

"""
# 🧑‍🏋 Employee Salary Predictor

> A smart, interactive web app that predicts whether a person's income is **above or below $50K**, based on demographic and employment details.  
> Built using **Streamlit** + **Machine Learning**, with a sleek UI and real-time predictions.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://7b3b0b5a3688.ngrok-free.app/)  

---

## 💡 What It Does

- Takes user input like age, education, job, hours worked, etc.
- Predicts if the person earns **> $50K or ≤ $50K**.
- Runs instantly in the browser using a clean, fast Streamlit interface.

---

## 💠 Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Ngrok** (for public access)

---

## 💻 Features

👉 Clean, interactive frontend  
👉 Fast and accurate predictions  
👉 Fully ML-driven  
👉 Secure public deployment  
👉 Beginner-friendly UI

---

## 📆 How to Run Locally

1. **Clone this repo:**

   ```bash
   git clone https://github.com/your-username/salary-predictor.git
   cd salary-predictor
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

4. *(Optional)* Use Ngrok to share:

   ```bash
   ngrok authtoken your_token_here
   ngrok http 8501
   ```

---

## 👩‍💻 Author

Made with 💜 by **[Shreya R Chittaragi](https://github.com/ShreyaRChittaragi)**  
> _“Empowering machines to make meaningful predictions — one model at a time.”_

---

## ⭐️ Show Some Love

If this helped or inspired you, drop a ⭐ and share the repo!  
Let’s grow and learn together ⭐
"""
