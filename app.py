import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Check and load model and scaler files
if not os.path.exists('rf_classifier.pkl') or not os.path.exists('rf_scaler.pkl'):
    st.error("⚠️ Model or scaler file not found. Please upload the necessary files.")
else:
    model = joblib.load('rf_classifier.pkl')
    scaler = joblib.load('rf_scaler.pkl')

st.title("🍷 Wine Classifier App")
st.write("Enter the wine features below to classify the wine category.")

# Feature Names
feature_names = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium"]

# User Input Section
with st.expander("🔧 Enter Wine Features"):
    user_input = [st.number_input(f"🔹 {feature}", min_value=0.0, step=0.1, format="%.2f") for feature in feature_names]

# Initialize Prediction History in session state if not exists
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Sidebar for Prediction History
with st.sidebar:
    st.header("📜 Prediction History")
    if st.session_state['history']:
        for idx, (inputs, pred) in enumerate(st.session_state['history'][-5:]):
            input_details = ", ".join([f"{feature}: {value}" for feature, value in zip(feature_names, inputs)])
            st.write(f"**{idx + 1}.** {input_details}, Predicted Wine Category: **{pred}**")
    else:
        st.write("No predictions yet.")

# Predict Button
if st.button("🔍 Predict Wine Category"):
    if not all(value > 0 for value in user_input):
        st.warning("⚠️ Please enter all values greater than zero before predicting.")
    else:
        # Convert input to 2D array and scale it
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display Result
        st.success(f"🍷 Predicted Wine Category: **{prediction[0]}**")
        st.write(f"📊 Prediction Confidence: {max(prediction_proba[0]) * 100:.2f}%")

        # Handle probability visualization
        try:
            categories = model.classes_
        except AttributeError:
            categories = [f"Category {i}" for i in range(len(prediction_proba[0]))]

        prob_df = pd.DataFrame(prediction_proba, columns=categories)
        st.bar_chart(prob_df.T)

        # Save Prediction History
        st.session_state['history'].append((user_input.copy(), prediction[0]))

        # Download Option
        st.download_button("📥 Download Prediction", f"Wine Category: {prediction[0]}", file_name="prediction.txt")

# Reset Button
if st.button("🔄 Reset"):
    # Clear only inputs, retain history
    st.session_state['last_inputs'] = []
    st.rerun()

# Ensure history is retained without resetting it
if 'last_inputs' not in st.session_state:
    st.session_state['last_inputs'] = []

# Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)
