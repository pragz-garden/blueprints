import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Define dataset path
data_dir = "C:/Users/pragp/OneDrive/Desktop/project/structured_dataset/"
categories = ["normal", "defective"]
img_size = 128  # Resize images to 128x128

# Load dataset (Only Normal Images)
X = []
for img_name in os.listdir(os.path.join(data_dir, "normal")):
    img_path = os.path.join(data_dir, "normal", img_name)
    
    # Load image in RGB format
    img = cv2.imread(img_path)  
    if img is None:
        print(f"⚠ Warning: Skipping unreadable file {img_path}")
        continue

    img = cv2.resize(img, (img_size, img_size))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values (0 to 1)
    
    X.append(img)

#  Convert lists to NumPy arrays
X = np.array(X)

#  Split into train & test sets (Only using Normal images)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

#  Save preprocessed data for training autoencoder
with open("preprocessed_normal.pkl", "wb") as f:
    pickle.dump((X_train, X_test), f)

print(f"✅ Dataset Preprocessed: {len(X_train)} training images, {len(X_test)} testing images")
