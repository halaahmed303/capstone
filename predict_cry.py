import os
import pickle
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow import keras

# تحديد مسار الفولدر الحالي عشان ميحصلش مشاكل في المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("جاري تحميل الموديل...")
model = keras.models.load_model(os.path.join(BASE_DIR, "infant_cry_dnn.keras"))
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
print("تم تحميل الموديل بنجاح!")


def record_from_mic(output_path="live_cry.wav", duration=5, sr=22050):
    full_path = os.path.join(BASE_DIR, output_path)
    print(f"جاري التسجيل من المايك لمدة {duration} ثواني...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    write(full_path, sr, (audio * 32767).astype(np.int16))
    return full_path


def extract_features(y, sr, n_mfcc=40):
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(mfcc_delta2, axis=1))
    features.extend(np.std(mfcc_delta2, axis=1))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    fmin_sc = 200.0
    safe_bands = max(1, min(6, int(np.floor(np.log2((sr / 2) / fmin_sc)))))
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, fmin=fmin_sc, n_bands=safe_bands
    )
    pad = np.zeros((7 - contrast.shape[0], contrast.shape[1]))
    contrast = np.vstack([contrast, pad])
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz, axis=1))

    return np.array(features, dtype=np.float32)


def predict_cry(audio_path):
    signal, sr = librosa.load(audio_path, sr=22050, mono=True)
    feat = extract_features(signal, sr).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    probs = model.predict(feat_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return label, confidence


if __name__ == "__main__":
    wav_file = record_from_mic(duration=5, sr=22050)
    label, conf = predict_cry(wav_file)
    print(f"النتيجة: {label} (نسبة الثقة: {conf:.2%})")