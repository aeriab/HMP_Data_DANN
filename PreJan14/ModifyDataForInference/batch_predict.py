### --------- load modules -------------------#
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow.keras.backend as K
import os

# Define a batch size for prediction to manage memory
# The model will process this many images at a time.
PRED_BATCH_SIZE = 1468

### --------- Custom Layer Definition -------------------#
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return grad_reverse(x)

### --------- Custom Loss/Metric Definitions -------------------#
def custom_bce(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)

def custom_categorical_ce(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
    y_true = tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
    return categorical_crossentropy(y_true, y_pred)

def custom_binary_accuracy(y_true, y_pred):
     y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
     y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
     return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def custom_categorical_accuracy(y_true, y_pred):
     y_pred = tf.boolean_mask(y_pred, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
     y_true =  tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
     return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

### --------- Model Loading Function -------------------#
def load_cnn_model_weights(path_model, path_weights):
    # Register all custom objects
    custom_objects = {
        'GradReverse': GradReverse,
        'custom_bce': custom_bce,
        'custom_categorical_ce': custom_categorical_ce,
        'custom_binary_accuracy': custom_binary_accuracy,
        'custom_categorical_accuracy': custom_categorical_accuracy
    }
    
    # Load model architecture from JSON file
    if not os.path.exists(path_model):
        print(f"Error: Model JSON file not found at {path_model}")
        sys.exit(1)
    if not os.path.exists(path_weights):
        print(f"Error: Model weights file not found at {path_weights}")
        sys.exit(1)

    with open(path_model, 'r') as f:
      model = model_from_json(f.read(), custom_objects=custom_objects)
    
    # Load model weights from HDF5 file
    model.load_weights(path_weights)
    return model

### --------- Main Inference Block -------------------#
if __name__ == "__main__":
    # Updated to accept 5 arguments
    if len(sys.argv) != 6:
        print("Usage: python batch_predict.py <model.json> <model.weights.h5> <images.npy> <site_map.npy> <results.txt>")
        sys.exit(1)

    model_json_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    image_npy_path = sys.argv[3]
    site_map_path = sys.argv[4]  # New argument
    output_results_path = sys.argv[5]

    # 1. Load the trained model
    print(f"Loading model...")
    model = load_cnn_model_weights(model_json_path, model_weights_path)
    expected_shape_hw = model.input_shape[1:3]

    # 2. Load Data
    print(f"Loading images: {image_npy_path}")
    img_batch = np.load(image_npy_path)
    
    print(f"Loading site map: {site_map_path}")
    site_map = np.load(site_map_path)
    
    num_images = img_batch.shape[0]
    
    # Validation: Ensure site map matches the number of images
    if site_map.shape[0] != num_images:
        print(f"Error: Image count ({num_images}) doesn't match site map count ({site_map.shape[0]})")
        sys.exit(1)

    # 3. Preprocess and Predict
    img_processed = np.expand_dims(img_batch, axis=-1)
    print(f"Running predictions on {num_images} windows...")
    prediction = model.predict(img_processed, batch_size=PRED_BATCH_SIZE)
    
    # Extract classifier output (index 0)
    classifier_output = prediction[0]

    # 4. Interpret and save results with Site Info
    print(f"Saving results to {output_results_path}...")
    threshold = 0.5
    
    with open(output_results_path, 'w') as f:
        # We add "Start_Site", "End_Site", and "Center_Site" for genomic context
        f.write("Window_Index\tStart_Site\tEnd_Site\tCenter_Site\tLabel\tScore\n")
        
        for i in range(num_images):
            score = classifier_output[i][0]
            label = "SWEEP" if score > threshold else "NEUTRAL"
            
            # Extract genomic coordinates from the site_map
            # Assuming site_map[i] is a row of site positions for that window
            window_sites = site_map[i]
            start_s = window_sites[0]
            end_s = window_sites[-1]
            center_s = window_sites[len(window_sites) // 2] # The middle SNP
            
            f.write(f"{i}\t{start_s}\t{end_s}\t{center_s}\t{label}\t{score:.4f}\n")

    print("Done.")