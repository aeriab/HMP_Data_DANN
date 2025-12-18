import sys
import os
import numpy as np
# We need tensorflow to load the model, even if it's via a custom function
import tensorflow as tf 

# --- 1. Set up path to load your custom function ---
# This is copied directly from your script
Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)
from CNN_multiclass_data_mergeSims_A100 import load_cnn_model

# --- 2. Define model and image paths ---
# Use the model name from your prompt
model_name = 'CNN_color_multiclass_sims_trained' 
# This is the npy file we created in the previous step
image_path = '/u/project/ngarud/baeria/ProcessHMPData/ModifyDataForInference/data_npy_file.npy'

# --- 3. Load the model ---
print(f"Loading model: {model_name}...")
# This uses your custom function to load the .json and .weights.h5
model = load_cnn_model(model_name)

# You can uncomment this line to see the model's architecture
# and confirm its expected input shape
# model.summary()

# --- 4. Load and prepare the single image ---
print(f"Loading image batch: {image_path}...")
# Load the entire (20773, 100, 201, 2) array
all_images = np.load(image_path)

single_image = all_images[1]
# single_image = all_images

# Keras .predict() expects a BATCH of images.
# We must add a new dimension at the beginning (axis=0) to create a batch of 1.
# This changes the image shape from (100, 201, 2) -> (1, 100, 201, 2)
image_batch = np.expand_dims(single_image, axis=0)
print(f"Image shape for prediction: {image_batch.shape}")

# --- 5. Make the prediction ---
print("Running prediction...")
# This returns the probabilities for each class
prediction_probabilities = model.predict(image_batch)

# --- 6. Interpret the results ---
# The output shape is (1, 3) for [batch_size, num_classes]
# Get the probabilities for our single image (the first item in the batch)
probabilities = prediction_probabilities[0]

# Define the class labels (based on your PR script's plotting)
class_labels = ["Neutral", "Hard sweep", "Soft sweep"]

# Find the index of the class with the highest probability
predicted_index = np.argmax(probabilities)
predicted_label = class_labels[predicted_index]
confidence = probabilities[predicted_index]

print("\n--- Prediction Complete ---")
print(f"Predicted class: {predicted_label} (Index: {predicted_index})")
print(f"Confidence: {confidence:.4f}")
print("All probabilities:")
for i, label in enumerate(class_labels):
    print(f"  {label}: {probabilities[i]:.4f}")