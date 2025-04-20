import os
import wfdb
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class PhysioNetDataset:
    def __init__(self, data_dir=None, target_fs=200, window_size=2000, stride=2000, save_path=None):
        # Automatically detect whether running on Kaggle or locally
        if os.path.exists("/kaggle/input/physionet-processed"):
            self.data_dir = "/kaggle/input/physionet-processed"
            self.save_path = "/kaggle/working/processed_data.npz"
        else:
            self.data_dir = "C:/Users/ranaa/OneDrive/Desktop/GP-Ya Rab Nekhlas/El7a2ny/ECG/Data/PhysioNet"
            self.save_path = "C:/Users/ranaa/OneDrive/Desktop/GP-Ya Rab Nekhlas/El7a2ny/ECG/Data/PhysioNet_Processed"
        
        self.target_fs = target_fs
        self.window_size = window_size
        self.stride = stride
        self.labels_df = pd.read_csv(f"{self.data_dir}/REFERENCE.csv", header=None, names=["record", "label"])
        self.label_mapping = {"N": 0, "A": 1, "O": 2, "~": 3}
        self.labels_df["label_num"] = self.labels_df["label"].map(self.label_mapping)

    def load_ecg(self, record_name):
        """Loads the ECG signal (Lead I) from a .mat file."""
        mat_data = scipy.io.loadmat(f"{self.data_dir}/{record_name}.mat")
        return mat_data['val'][0]  # Lead I is the first row

    def normalize_signal(self, ecg_signal):
        """Normalizes the ECG signal to the range [-1, 1]."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(ecg_signal.reshape(-1, 1)).flatten()

    def resample_signal(self, ecg_signal, original_fs=300):
        """Resamples the ECG signal to the target frequency."""
        target_length = int(len(ecg_signal) * (self.target_fs / original_fs))
        return resample(ecg_signal, target_length)

    def segment_signal(self, ecg_signal):
        """Segments the ECG signal into fixed-size windows."""
        return np.array([ecg_signal[i:i + self.window_size] 
                         for i in range(0, len(ecg_signal) - self.window_size + 1, self.stride)])

    def preprocess_record(self, record_name):
        """Load, normalize, resample, and segment a single ECG record."""
        ecg_signal = self.load_ecg(record_name)  # Load raw signal
        ecg_signal = self.normalize_signal(ecg_signal)  # Normalize
        ecg_signal = self.resample_signal(ecg_signal)  # Resample to 200 Hz
        segments = self.segment_signal(ecg_signal)  # Segment into 10s windows
        label = self.labels_df[self.labels_df["record"] == record_name]["label_num"].values[0]
        labels = [label] * len(segments)  # Assign label to each segment
        return segments, labels

    def save_data(self, X, y):
        """Saves the processed data to a compressed .npz file."""
        np.savez_compressed(self.save_path, X=X, y=y)
        print(f"✅ Processed data saved to {self.save_path}")

    def load_data(self):
        """Loads the preprocessed data if available."""
        if os.path.exists(self.save_path):
            print(f"✅ Loading preprocessed data from {self.save_path}")
            data = np.load(self.save_path)
            return data["X"], data["y"]
        return None, None

    def load_or_process_dataset(self):
        """Loads preprocessed data or processes it if not available."""
        X, y = self.load_data()
        if X is not None and y is not None:
            return train_test_split(X, y, test_size=0.2, random_state=42)

        all_segments, all_labels = [], []
        for record_name in self.labels_df["record"]:
            segments, labels = self.preprocess_record(record_name)
            all_segments.extend(segments)
            all_labels.extend(labels)

        X = np.array(all_segments).reshape(-1, self.window_size, 1)
        y = to_categorical(np.array(all_labels), num_classes=4)

        self.save_data(X, y)  # Save after processing
        return train_test_split(X, y, test_size=0.2, random_state=42)
