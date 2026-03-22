import librosa
import numpy as np
import cv2

def extract_features(file_path, max_len=128):
    try:
        y, sr = librosa.load(file_path)

        y, _ = librosa.effects.trim(y)

        target_len = sr * 3

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        mel_db = cv2.resize(mel_db, (128, 128))

        return mel_db

    except Exception as e:
        print("Error processing:", file_path, e)
        return None