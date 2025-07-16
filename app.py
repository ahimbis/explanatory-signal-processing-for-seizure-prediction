from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.signal as signal
import os

app = Flask(__name__)

# Load pre-trained CNN model
model = tf.keras.models.load_model('cnn_seizure_prediction_model.h5')

# Band ranges in Hz
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

SAMPLING_RATE = 256  # Hz
WINDOW_DURATION = 5  # seconds
WINDOW_SIZE = SAMPLING_RATE * WINDOW_DURATION  # 1280 samples


def compute_band_powers(segment, fs):
    freqs, psd = signal.welch(segment, fs=fs, nperseg=min(256, len(segment)))
    powers = []
    for band in bands.values():
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        powers.append(band_power)
    return powers


def extract_features(df):
    eeg_channels = df.select_dtypes(include=[np.number]).columns.tolist()
    segments = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        feature_vector = []
        for ch in eeg_channels:
            segment = window[ch].values
            powers = compute_band_powers(segment, SAMPLING_RATE)
            feature_vector.extend(powers)  # 5 values per channel
        segments.append(feature_vector)
    return np.array(segments)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv'}), 400

    try:
        df = pd.read_csv(file)
        features = extract_features(df)

        # Reshape for CNN: (samples, features, 1)
        features_reshaped = features.reshape((features.shape[0], features.shape[1], 1))

        predictions = model.predict(features_reshaped)
        predicted_labels = (predictions > 0.5).astype(int).flatten().tolist()

        return jsonify({'predictions': predicted_labels})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
