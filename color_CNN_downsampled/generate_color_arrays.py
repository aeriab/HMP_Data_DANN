import numpy as np
import pandas as pd

# --- Parameters ---
input_csv_file = 'cropped_r_bromii_data.csv'
output_npy_file = 'r_bromii_sliding_color.npy' # Updated filename for clarity

# Dimensions for the final "image"
window_height = 201     # (sites_per_image)
slide_step = 10         # The number of sites to move the window by

# --- Script ---

print(f"Loading data from {input_csv_file}...")

# 1. Load Data
# We use pandas because the CSV contains mixed types (strings and numbers).
try:
    df = pd.read_csv(input_csv_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Parse Columns
# Site positions (Col 0)
site_pos_col = df['site_pos'].values.astype(np.int32)

# Site types (Col 1) -> Map to Integers for Channel 2
# 'syn' = 0, 'nonsyn' = 1
# We use .fillna(0) to handle any unexpected blank values safely
site_type_map = {'syn': 0, 'nonsyn': 1}
site_types = df['site_type'].map(site_type_map).fillna(0).values.astype(np.int8)

# Genotype Data (Col 2 onwards) -> Channel 1
# We cast to int8 immediately to save memory (0, 1, -1)
genotype_data = df.iloc[:, 2:].values.astype(np.int8)

# Get dimensions
total_sites, num_samples = genotype_data.shape
print(f"Data Loaded: {total_sites} sites x {num_samples} samples")

# --- Sliding Window Logic ---

# 3. Calculate Number of Images
num_images = int(np.floor((total_sites - window_height) / slide_step) + 1)
print(f"Calculated {num_images} images based on window {window_height} and slide {slide_step}.")

# 4. Pre-allocate Arrays
print("Creating sliding windows...")

# Shape: (Num Images, Num Samples, Window Height, 2 Channels)
final_data = np.zeros((num_images, num_samples, window_height, 2), dtype=np.int8)
final_site_indices = np.zeros((num_images, window_height), dtype=np.int32)

for i in range(num_images):
    start_idx = i * slide_step
    end_idx = start_idx + window_height
    
    # --- Meta: Site Indices ---
    final_site_indices[i] = site_pos_col[start_idx:end_idx]
    
    # --- Channel 0: Genotype (B/W) ---
    # Slice (Site x Sample) -> Transpose to (Sample x Site)
    # The model expects rows to be samples, columns to be sites
    final_data[i, :, :, 0] = genotype_data[start_idx:end_idx, :].T
    
    # --- Channel 1: Site Type (Color) ---
    # The site_type array is 1D (just sites).
    # We must broadcast/tile this value across every sample row 
    # so the whole column has the same "color" (syn or nonsyn).
    current_site_types = site_types[start_idx:end_idx]
    final_data[i, :, :, 1] = np.tile(current_site_types, (num_samples, 1))

# 5. Save Files
np.save(output_npy_file, final_data)
np.save('r_bromii_site_map.npy', final_site_indices)

print("---")
print(f"Successfully saved data with final shape: {final_data.shape}")
print(f"Output file: {output_npy_file}")