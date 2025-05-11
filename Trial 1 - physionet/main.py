from src.data_preprocessing.preprocessing import PhysioNetDataset
# Usage example
dataset = PhysioNetDataset()
X,Y = dataset.load_or_process_dataset()
print(X.shape, Y.shape)

import numpy as np
import collections


# Convert one-hot encoding back to class labels
y_labels = np.argmax(Y, axis=1)

# Count occurrences of each class
class_counts = collections.Counter(y_labels)

print("Class Distribution:", class_counts)

segment_counts = []
for record_name in dataset.labels_df["record"]:
    segments, labels = dataset.preprocess_record(record_name)
    segment_counts.append(len(segments))

print("Average Segments per Record:", np.mean(segment_counts))
print("Min:", np.min(segment_counts), "| Max:", np.max(segment_counts))

