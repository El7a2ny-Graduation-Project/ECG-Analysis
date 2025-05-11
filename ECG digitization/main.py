import numpy as np
import os
import tensorflow as tf
import pickle
from scipy.interpolate import interp1d

def read_lead_i_dat_file(dat_file_path, sampling_rate=500, duration_seconds=10, data_format='16'):
    """
    Read a .dat file directly without requiring a .hea file, focusing only on Lead I
    
    Parameters:
    -----------
    dat_file_path : str
        Path to the .dat file (with or without .dat extension)
    sampling_rate : int
        Sampling rate in Hz (default 500Hz)
    duration_seconds : int
        Duration of the ECG recording in seconds (default 10s, resulting in 5000 samples at 500Hz)
    data_format : str
        Data format of the binary file: '16' for 16-bit integers, '32' for 32-bit floats
        
    Returns:
    --------
    numpy.ndarray
        ECG signal data for Lead I with shape (samples,)
    """
    # Ensure the path ends with .dat
    if not dat_file_path.endswith('.dat'):
        dat_file_path += '.dat'
    
    # Calculate expected samples
    expected_samples = sampling_rate * duration_seconds
    
    # Read the binary data
    if data_format == '16':
        # 16-bit signed integers (common format for ECG)
        data = np.fromfile(dat_file_path, dtype=np.int16)
    elif data_format == '32':
        # 32-bit floating point (less common)
        data = np.fromfile(dat_file_path, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    # Assuming standard 12-lead ECG with interleaved data
    # Lead I is typically the first lead (index 0)
    try:
        # Try to extract Lead I (first lead)
        signal = data.reshape(-1, 12)[:, 0]  # Extract just Lead I
        
        # Handle case where the signal is not exactly the expected length
        if len(signal) != expected_samples:
            print(f"Warning: Signal length {len(signal)} doesn't match expected {expected_samples} samples")
            
            # Resample to expected length
            x = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, expected_samples)
            f = interp1d(x, signal, kind='linear', bounds_error=False, fill_value="extrapolate")
            signal = f(x_new)
    
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        print(f"Data length: {len(data)}")
        
        # If reshaping fails, try to adapt
        # For Lead I, we'll take every 12th sample starting from the first
        if len(data) >= 12:
            signal = data[::12]  # Take every 12th sample (Lead I)
            
            # Resample if needed
            if len(signal) != expected_samples:
                x = np.linspace(0, 1, len(signal))
                x_new = np.linspace(0, 1, expected_samples)
                f = interp1d(x, signal, kind='linear', bounds_error=False, fill_value="extrapolate")
                signal = f(x_new)
        else:
            # If file is too small, pad with zeros
            signal = np.zeros(expected_samples)
            signal[:len(data)] = data
    
    return signal

def predict_ecg_lead_i(dat_file_path, model_path, mlb_path=None, sampling_rate=500):
    """
    Process a .dat file containing ECG data, extract Lead I, and make predictions.
    
    Parameters:
    -----------
    dat_file_path : str
        Path to the .dat file
    model_path : str
        Path to the saved model (.h5 file)
    mlb_path : str, optional
        Path to the saved MultiLabelBinarizer pickle file for label decoding
    sampling_rate : int
        Sampling rate in Hz (default 500Hz)
        
    Returns:
    --------
    dict
        Dictionary containing prediction probabilities and class names if mlb_path is provided
    """
    try:
        # Step 1: Read the raw ECG data for Lead I
        signal = read_lead_i_dat_file(dat_file_path, sampling_rate=sampling_rate)
        print(f"Loaded Lead I signal with {len(signal)} samples")
        
        # Ensure we have 5000 samples (10 seconds at 500Hz)
        if len(signal) != 5000:
            # Resample to 5000 samples
            x = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, 5000)
            f = interp1d(x, signal, kind='linear', bounds_error=False, fill_value="extrapolate")
            signal = f(x_new)
        
        # Step 2: Reshape for model input (batch, time, channels)
        # Your model expects shape (batch, 5000, 1)
        X = signal.reshape(1, 5000, 1)
        print(f"Input shape for model: {X.shape}")
        
        # Step 3: Load the model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Step 4: Make predictions
        predictions = model.predict(X)
        
        # Step 5: Process results
        result = {"raw_predictions": predictions[0].tolist()}
        
        # If mlb_path is provided, decode the labels
        if mlb_path and os.path.exists(mlb_path):
            with open(mlb_path, 'rb') as f:
                mlb = pickle.load(f)
            
            # Get classes with probability > 0.5
            binary_preds = (predictions[0] >= 0.5).astype(int)
            predicted_classes = mlb.inverse_transform(binary_preds.reshape(1, -1))[0]
            
            # Add class probabilities to result
            class_probs = {}
            for i, class_name in enumerate(mlb.classes_):
                class_probs[class_name] = float(predictions[0][i])
            
            result["predicted_classes"] = list(predicted_classes)
            result["class_probabilities"] = class_probs
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual paths
    dat_file_path = "00001hr.dat"
    model_path = "binary_deep_classifier.h5"
    mlb_path = "binary.pkl"
    
    # Get predictions
    results = predict_ecg_lead_i(dat_file_path, model_path, mlb_path)
    
    # Print results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("Predicted classes:")
        if "predicted_classes" in results:
            for cls in results["predicted_classes"]:
                print(f"- {cls}")
            
            print("\nClass probabilities (sorted):")
            for cls, prob in sorted(results["class_probabilities"].items(), 
                                  key=lambda x: x[1], reverse=True):
                print(f"- {cls}: {prob:.4f}")
        else:
            print("Raw predictions:", results["raw_predictions"])
