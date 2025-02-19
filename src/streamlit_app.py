import streamlit as st
import requests

# Define the API endpoint
API_URL = "http://localhost:8000/predict"

# Streamlit app
st.title("SMS Spam Detection")
st.write("Enter a message to check if it's spam or not.")

# Input text box
text = st.text_area("Message")

# Predict button
if st.button("Predict"):
    if text:
        # Send a POST request to the FastAPI backend
        response = requests.post(API_URL, json={"text": text})
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: **{result['prediction']}**")
            st.write(f"Probability: {result['probability']:.4f}")
        else:
            st.write("Error: Unable to get a prediction.")
    else:
        st.write("Please enter a message.")