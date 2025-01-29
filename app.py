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

# Prediction History from session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Sidebar for Prediction History
with st.sidebar:
    st.header("📜 Prediction History")
    if st.session_state['history']:
        for idx, (inputs, pred) in enumerate(st.session_state['history'][-5:]):
            st.write(f"**{idx + 1}.** Inputs: {inputs}, Predicted: **{pred}**")
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

        # Visualize prediction probabilities
        prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.bar_chart(prob_df.T)

        # Save Prediction History
        st.session_state['history'].append((user_input, prediction[0]))

        # Download Option
        st.download_button("📥 Download Prediction", f"Wine Category: {prediction[0]}", file_name="prediction.txt")

# Reset Button
if st.button("🔄 Reset"):
    st.session_state['history'] = []
    st.rerun()

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
