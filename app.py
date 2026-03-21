import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_features

model = load_model("deepfake_detector.h5")

st.title("🎤 AI Deepfake Audio Detector")


uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    features = extract_features("temp.wav")
    features = features[np.newaxis, ..., np.newaxis]


    prediction = model.predict(features)[0][0]


    if prediction > 0.5:
        st.error(f"Fake Voice ({prediction*100:.2f}%)")
    else:
        st.success(f"Real Voice ({(1-prediction)*100:.2f}%)")