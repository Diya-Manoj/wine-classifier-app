
import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
import os

model_path = os.path.join(os.getcwd(), "rf_classifier.pkl")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)


# Load the scaler
with open("rf_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature names (update based on dataset)
feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]

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
