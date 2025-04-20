import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from se_resnet import build_se_resnet

# Step 1: Load Training and Test Data
data = np.load("/kaggle/input/physionet-processed/PhysioNet_Processed.npz")
X, y = data["X"], data["y"]

# Perform train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

print(" Training Data Loaded. Shape:", X_train.shape, y_train.shape)
print(" Validation Data Loaded. Shape:", X_val.shape, y_val.shape)
print(" Test Data Loaded. Shape:", X_test.shape, y_test.shape)

#  Step 2: Build the SE-ResNet Model with Optimized Architecture
model = build_se_resnet((2000, 1), 4)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#  Step 3: Set Up Callbacks for Hyperparameter Optimization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)

#  Step 4: Train the Model with Optimized Hyperparameters
history = model.fit(
    X_train, y_train,
    epochs=150, batch_size=16, validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler]
)

#  Step 5: Save the Newly Trained Model
model.save("/kaggle/working/se_resnet_physionet_new.h5")
print(" Model trained from scratch and saved successfully!")

#  Step 6: Evaluate Model
# Compute test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f" Test Accuracy: {test_acc * 100:.2f}%")

#  Step 7: Generate Predictions
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_test_classes = tf.argmax(y_test, axis=1).numpy()

#  Step 8: Generate Classification Report
print(" Classification Report:\n")
print(classification_report(y_test_classes, y_pred_classes, target_names=["Normal", "AFib", "Other", "Noisy"]))

#  Step 9: Plot Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "AFib", "Other", "Noisy"], 
            yticklabels=["Normal", "AFib", "Other", "Noisy"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#  Step 10: Download the New Model
from IPython.display import FileLink
!zip -r /kaggle/working/se_resnet_new.zip /kaggle/working/se_resnet_physionet_new.h5
FileLink(r'/kaggle/working/se_resnet_new.zip')
