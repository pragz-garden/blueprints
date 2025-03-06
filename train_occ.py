import os
import numpy as np
import pickle
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models

# Load & Preprocess Dataset (All Normal Images for Training)
with open("preprocessed_normal.pkl", "rb") as f:
    data = pickle.load(f)

#  Ensure correct extraction from pickle
if isinstance(data, tuple):  
    X_data, _ = data  # Extract images only if labels exist
else:
    X_data = data  

#  Convert X_data into a properly formatted NumPy array
X_data_fixed = []  # Empty list to store fixed images

for img in X_data:
    if isinstance(img, np.ndarray):  # Check if it's a valid image array
        resized_img = cv2.resize(img, (128, 128))  # Resize if needed
        X_data_fixed.append(resized_img)
    else:
        print("⚠ Warning: Skipping invalid image data")

#  Convert list into a NumPy array
X_train = np.array(X_data_fixed, dtype=np.float32) / 255.0  # Normalize

print(f" Training on Normal Images Only: {X_train.shape}")

# Define the Autoencoder Model
autoencoder = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # RGB Input
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),

    layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),

    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),

    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output RGB Image
])

# Compile Model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])

#Train Model
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

#  Save the Trained Autoencoder
autoencoder.save("one_class_autoencoder.h5")
print(" Model training complete and saved as 'one_class_autoencoder.h5'")


# Anomaly Detection - Load Test Data

with open("preprocessed_normal.pkl", "rb") as f:
    data = pickle.load(f)

#  Ensure correct extraction and formatting
if isinstance(data, tuple):
    X_test = data[0]  # Extract images only
else:
    X_test = data

#  Convert X_test into a properly formatted NumPy array
X_test_fixed = []
for img in X_test:
    if isinstance(img, np.ndarray):
        resized_img = cv2.resize(img, (128, 128))
        X_test_fixed.append(resized_img)
    else:
        print("⚠ Warning: Skipping invalid image data")

#  Convert list into a NumPy array & Normalize
X_test = np.array(X_test_fixed, dtype=np.float32) / 255.0


#   Compute Reconstruction Errors for Test Images

reconstructed = autoencoder.predict(X_test)
mse_errors = np.mean(np.square(X_test - reconstructed), axis=(1, 2, 3))

# Set Anomaly Threshold (95th Percentile of Normal Data)
threshold = np.percentile(mse_errors, 60)

# Detect Anomalies (If Error > Threshold, It's Defective)
predictions = mse_errors > threshold  # True = Defective, False = Normal


# Print & Save Results

print(" Reconstruction Errors:", mse_errors)
print(" Anomaly Detection Threshold:", threshold)
print(f" Predicted Defective Images: {np.sum(predictions)}")
print(f" Predicted Normal Images: {np.sum(~predictions)}")

#  Save threshold for Web App Integration
np.save("anomaly_threshold.npy", threshold)
print(" Anomaly detection threshold saved as 'anomaly_threshold.npy'")
