from ECG.src.Preprocessing.preprocessing import PhysioNetDataset
# Usage example
dataset = PhysioNetDataset()
X_train, X_test, y_train, y_test = dataset.load_or_process_dataset()
print("Training data shape:", X_train.shape, y_train.shape)
print("Test data shape:", X_test.shape, y_test.shape)