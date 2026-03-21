import librosa
import numpy as np

def extract_features(file_path, max_len=128):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        audio = librosa.util.normalize(audio)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        if mel_db.shape[1] < max_len:
            mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :max_len]

        return mel_db

    except:
        return None