### --------- load modules -------------------#
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow.keras.backend as K
import os

# Define a batch size for prediction to manage memory
PRED_BATCH_SIZE = 1468

### --------- Custom Layer Definition -------------------#
# These must remain to successfully load the architecture from .json
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
    custom_objects = {
        'GradReverse': GradReverse,
        'custom_bce': custom_bce,
        'custom_categorical_ce': custom_categorical_ce,
        'custom_binary_accuracy': custom_binary_accuracy,
        'custom_categorical_accuracy': custom_categorical_accuracy
    }
    
    if not os.path.exists(path_model) or not os.path.exists(path_weights):
        print(f"Error: Model files not found at {path_model} or {path_weights}")
        sys.exit(1)

    print("Loading model architecture...")
    with open(path_model, 'r') as f:
        model = model_from_json(f.read(), custom_objects=custom_objects)
    
    print("Loading model weights...")
    model.load_weights(path_weights)
    return model

### --------- Main Inference Block -------------------#
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python batch_predict_multiclass.py <model.json> <model.weights.h5> <images.npy> <site_map.npy> <results.txt>")
        sys.exit(1)

    model_json_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    image_npy_path = sys.argv[3]
    site_map_path = sys.argv[4]
    output_results_path = sys.argv[5]

    # 1. Load the trained model
    model = load_cnn_model_weights(model_json_path, model_weights_path)
    
    expected_shape_hw = model.input_shape[1:3] 
    print(f"Model expects input shape (Rows, Cols): {expected_shape_hw}")

    # 2. Load Data
    print(f"Loading images from {image_npy_path}...")
    img_batch = np.load(image_npy_path)
    
    print(f"Loading site map from {site_map_path}...")
    site_map = np.load(site_map_path)
    
    # --- Shape Adaptation Logic ---
    # Slice chromosomes (rows) if the input simulates more than the model was trained on
    if img_batch.shape[1] != expected_shape_hw[0]:
        print(f"Warning: Input has {img_batch.shape[1]} chromosomes, model expects {expected_shape_hw[0]}.")
        print(f"Slicing input to use only the first {expected_shape_hw[0]} chromosomes...")
        img_batch = img_batch[:, :expected_shape_hw[0], :]

    num_images = img_batch.shape[0]

    # 3. Preprocess and Predict
    # Add channel dimension: (N, H, W) -> (N, H, W, 1)
    img_processed = np.expand_dims(img_batch, axis=-1)
    
    print(f"Final input shape for prediction: {img_processed.shape}")
    print(f"Running predictions on {num_images} windows...")
    
    prediction = model.predict(img_processed, batch_size=PRED_BATCH_SIZE, verbose=1)
    
    # Handle DANN/Multi-output models
    # If the model returns [classifier_output, domain_output], take the first one
    if isinstance(prediction, list):
        classifier_output = prediction[0]
    else:
        classifier_output = prediction

    # 4. Interpret and save results
    print(f"Saving results to {output_results_path}...")
    
    # Class mapping based on your training scripts:
    # 0 = Neutral, 1 = Hard Sweep, 2 = Soft Sweep
    LABELS = {0: "NEUTRAL", 1: "HARD", 2: "SOFT"}

    with open(output_results_path, 'w') as f:
        # Header updated for multiclass probabilities
        f.write("Window_Idx\tStart\tEnd\tCenter\tPred_Label\tProb_Neu\tProb_Hard\tProb_Soft\n")
        
        for i in range(num_images):
            probs = classifier_output[i]
            
            # Determine the winner
            pred_idx = np.argmax(probs)
            pred_label = LABELS.get(pred_idx, "UNKNOWN")
            
            # Extract individual probabilities safely
            # Assuming output is shape (3,)
            p_neu = probs[0] if len(probs) > 0 else 0.0
            p_hard = probs[1] if len(probs) > 1 else 0.0
            p_soft = probs[2] if len(probs) > 2 else 0.0
            
            # Map back to site information
            # Handle edge case where site_map might be shorter than images (unlikely but safe)
            if i < len(site_map):
                window_sites = site_map[i]
                start_s = int(window_sites[0])
                end_s = int(window_sites[-1])
                center_s = int(window_sites[len(window_sites) // 2])
            else:
                start_s, end_s, center_s = -1, -1, -1
            
            # Write row
            f.write(f"{i}\t{start_s}\t{end_s}\t{center_s}\t{pred_label}\t{p_neu:.4f}\t{p_hard:.4f}\t{p_soft:.4f}\n")

    print("Successfully completed inference.")