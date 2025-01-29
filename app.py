import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
import os

model= joblib.load('rf_classifier.pkl')
scaler=joblib.load('rf_scaler.pkl')

# Feature names (update based on dataset)
feature_names = ["malic_acid", "magnesium", "flavanoids", "color_intensity", "proline"]

# Streamlit UI
st.title("🍷 Wine Classification App")
st.write("Enter feature values below to predict the wine class.")

# Create input fields for each feature
user_input = []
for feature in feature_names:
    value = st.text_input(f"{feature}:", "")
    user_input.append(value)

# Convert input to NumPy array & preprocess
if st.button("Predict"):
    try:
        # Convert inputs to float & reshape for model
        input_array = np.array(user_input, dtype=float).reshape(1, -1)

        # Apply scaling
        input_scaled = scaler.transform(input_array)

        # Predict wine class
        prediction = model.predict(input_scaled)[0]

        # Display prediction
        st.success(f"🍷 Predicted Wine Class: **{prediction}**")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values for all fields.")
