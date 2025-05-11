import wfdb                          # To read the ECG files
from wfdb import processing          # For QRS detection
import numpy as np                   # Numerical operations
import joblib                        # To load the saved model
import pywt                          # For wavelet feature extraction

def extract_features_from_signal(signal):
    features = []
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.median(signal))
    features.append(np.min(signal))
    features.append(np.max(signal))
    features.append(np.percentile(signal, 25))
    features.append(np.percentile(signal, 75))
    features.append(np.mean(np.diff(signal)))

    coeffs = pywt.wavedec(signal, 'db4', level=5)
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.min(coeff))
        features.append(np.max(coeff))

    return features


def load_dat_signal(file_path, n_leads=12, n_samples=5000, dtype=np.int16):
    raw = np.fromfile(file_path + '.dat', dtype=dtype)
    if raw.size != n_leads * n_samples:
        raise ValueError(f"Unexpected size: {raw.size}, expected {n_leads * n_samples}")
    signal = raw.reshape(n_samples, n_leads)
    return signal, 500  # Signal + sampling frequency

def classify_new_ecg(file_path, model):
    try:
        signal_all_leads, fs = load_dat_signal(file_path)
        lead_priority = [1, 0]  # Try Lead II (index 1), then I (index 0)
        lead_index = next((i for i in lead_priority if i < signal_all_leads.shape[1]), None)
        if lead_index is None:
            return "No suitable lead found"

        signal = signal_all_leads[:, lead_index]
        signal = (signal - np.mean(signal)) / np.std(signal)

        try:
            xqrs = processing.XQRS(sig=signal, fs=fs)
            xqrs.detect()
            r_peaks = xqrs.qrs_inds
        except:
            r_peaks = processing.gqrs_detect(sig=signal, fs=fs)

        if len(r_peaks) < 5:
            return "Insufficient beats"

        rr_intervals = np.diff(r_peaks) / fs
        qrs_durations = np.array([r_peaks[i] - r_peaks[i - 1] for i in range(1, len(r_peaks))])

        features = extract_features_from_signal(signal)
        features.extend([
            len(r_peaks),
            np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
            np.median(rr_intervals) if len(rr_intervals) > 0 else 0,
            np.mean(qrs_durations) if len(qrs_durations) > 0 else 0,
            np.std(qrs_durations) if len(qrs_durations) > 0 else 0
        ])

        prediction = model.predict([features])[0]
        return "Abnormal" if prediction == 1 else "Normal"

    except Exception as e:
        return f"Error: {str(e)}"


# Load the saved model
voting_loaded = joblib.load('voting_classifier.pkl')

file_path = "00008_hr"
result = classify_new_ecg(file_path, voting_loaded)
print(result)
