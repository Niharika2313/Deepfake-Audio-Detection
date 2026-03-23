import streamlit as st
import numpy as np
import os
from keras.models import load_model
from utils import extract_features
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")

model = load_model("deepfake_detector.h5")

st.title("AI Deepfake Audio Detector")
st.markdown("Upload an audio file to check whether it's **Real or AI-generated**.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "mp4"])

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

    st.markdown("---")

    st.subheader("Uploaded Audio")
    st.audio(final_path)

    try:

        features = extract_features(final_path)

        if features is None:
            st.error("Could not process audio")
        else:
            features = features[np.newaxis, ...]

            prediction = model.predict(features)[0][0]

            st.markdown("---")

            st.subheader("Prediction Confidence")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Fake Probability", f"{prediction:.2f}")

            with col2:
                st.metric("Real Probability", f"{1 - prediction:.2f}")

            st.progress(float(prediction))

            st.markdown("---")
            st.subheader("Result")

            if prediction > 0.6:
                st.error("FAKE VOICE DETECTED")
            else:
                st.success("REAL VOICE DETECTED")

            y, sr = librosa.load(final_path)
            duration = len(y) / sr

            st.markdown("---")
            st.subheader("Audio Information")

            st.write(f"**Duration:** {duration:.2f} sec")
            st.write(f"**Sample Rate:** {sr}")

            st.markdown("---")
            st.subheader("Spectrogram Visualization")

            plt.figure(figsize=(6,4))
            plt.imshow(features[0], aspect='auto', origin='lower')
            plt.title("Mel Spectrogram")
            plt.colorbar()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {e}")

    try:
        os.remove(input_file)
        if os.path.exists("converted_audio.wav"):
            os.remove("converted_audio.wav")
    except:
        pass