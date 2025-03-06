*** Microscopic Surface Anomaly Detection ***

** Project Overview **

This project implements an AI-powered anomaly detection system that analyzes microscopic surface images for defects. It consists of:

Preprocessing script (preprocess_norm.py) - Prepares the dataset (only normal images) for training.

Autoencoder Training (train_occ.py) - Trains a deep learning autoencoder on normal images to learn their structure.

Web App (app.py) - Deploys a Streamlit-based UI to classify images as "Normal" or "Defective."

** Project Structure **

Microscopic-Surface-Anomaly-Detection/
│── preprocess_norm.py  # Preprocessing dataset

│── train_occ.py        # Training the autoencoder model

│── app.py              # Streamlit Web App for anomaly detection

│── preprocessed_normal.pkl # Preprocessed dataset

│── one_class_autoencoder.h5 # Trained model

│── anomaly_threshold.npy # Precomputed anomaly threshold

│── requirements.txt    # Dependencies

│── README.md           # Documentation (this file)

** Dependencies **

Ensure you have the following Python packages installed:

* numpy

* opencv-python

* tensorflow

* streamlit

* pandas

* scikit-learn

* pickle-mixin

Install them using:

pip install -r requirements.txt

** Setup and Execution **

🔹 1. Preprocess the Dataset

This script extracts only normal images, resizes, normalizes them, and saves them for training.

python preeprocess_norm.py

Output: preprocessed_normal.pkl

🔹 2. Train the Autoencoder

Run the training script to learn the normal image structure.

python train_occ.py

Output:

one_class_autoencoder.h5 - Trained model

anomaly_threshold.npy - Computed anomaly detection threshold

🔹 3. Start the Web App

Launch the Streamlit-based UI:

streamlit run app.py

** Expected Output **

The system will:

* Identify normal images (No defects found)

* Detect defective images (Anomaly detected)

* Show uploaded and processed images

* Provide detection statistics via a sidebar

* Technical Details

* Data Preprocessing (preeprocess_norm.py)

Reads images from structured_dataset/normal.

Resizes them to 128x128.

Normalizes pixel values to [0,1].

Splits into train & test sets.

Saves to preprocessed_normal.pkl.

**  Autoencoder Model Training (train_occ.py) **

Uses a convolutional autoencoder to learn normal images.

Trains on only normal images.

Saves a learned threshold for detecting anomalies.

Uses MSE loss for reconstruction error.

Threshold is set at 60th percentile.

**  Web App (app.py)**

Loads a trained model to classify new images.

Calculates MSE reconstruction error.

Displays before & processed images.

Shows detection stats in the sidebar.

Interactive UI using Streamlit.

** Final Notes **

The MSE threshold is configurable in train_occ.py.


UI is designed to be intuitive and visually appealing.
