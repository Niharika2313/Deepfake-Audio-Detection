import streamlit as st
import numpy as np
import os
from keras.models import load_model
from utils import extract_features
from pydub import AudioSegment

model = load_model("deepfake_detector.h5")

st.title("AI Deepfake Audio Detector")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "mp4"])

if uploaded_file is not None:

    input_path = "input_audio"
    
    file_ext = uploaded_file.name.split(".")[-1].lower()
    input_file = f"{input_path}.{file_ext}"

    with open(input_file, "wb") as f:
        f.write(uploaded_file.read())

    if file_ext in ["mp3", "mp4"]:
        audio = AudioSegment.from_file(input_file)
        wav_path = "converted_audio.wav"
        audio.export(wav_path, format="wav")
        final_path = wav_path
    else:
        final_path = input_file

    st.audio(final_path)

    try:
        features = extract_features(final_path)

        if features is None:
            st.error("Could not process audio")
        else:
            features = features[np.newaxis, ..., np.newaxis]

            prediction = model.predict(features)[0][0]

            st.write(f"Confidence Score: {prediction:.4f}")

            if prediction > 0.55:
                st.error("Fake Voice Detected")
            else:
                st.success("Real Voice Detected")

    except Exception as e:
        st.error(f"Error: {e}")

    try:
        os.remove(input_file)
        if os.path.exists("converted_audio.wav"):
            os.remove("converted_audio.wav")
    except:
        pass