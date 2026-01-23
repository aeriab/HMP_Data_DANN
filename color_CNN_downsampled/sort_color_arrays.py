import numpy as np
import sys
from tqdm import tqdm

# --- Configuration ---
INPUT_NPY_FILE = 'r_bromii_sliding_color.npy'   # The 2-channel file you just created
OUTPUT_NPY_FILE = 'sorted_r_bromii_color.npy'
SORT_ORDERING = 'rows_dist'                      # 'rows_dist' sorts the samples (haplotypes)
# ---------------------

# Import the sorting function
try:
    import haplotype_sorter
except ImportError:
    print("Error: Could not find 'haplotype_sorter.py'.")
    print("Please make sure your sorting logic script is in this folder.")
    sys.exit(1)

print(f"Loading data from {INPUT_NPY_FILE}...")
# Expected shape: (Num_Images, Num_Samples, Num_Sites, 2)
all_data = np.load(INPUT_NPY_FILE)

# Get dimensions
try:
    num_images, num_samples, num_sites, num_channels = all_data.shape
except ValueError:
    print(f"Error: Input file has shape {all_data.shape}, but we expect 4 dimensions (Images, Samples, Sites, 2).")
    sys.exit(1)

print(f"Loaded {num_images} images.")
print(f"Shape per image: {num_samples} samples x {num_sites} sites x {num_channels} channels")

# Pre-allocate the sorted array
# We copy the original data structure exactly
sorted_all_data = np.zeros_like(all_data, dtype=np.int8)

print(f"Sorting images by '{SORT_ORDERING}'...")

for i in tqdm(range(num_images)):
    # 1. Get the current image (Samples x Sites x 2)
    # We use .copy() to ensure we have a writable array that doesn't modify the original 'all_data' in memory yet
    current_image = all_data[i].copy()
    
    # 2. Sort the image
    # 'haplotype_sorter' modifies the array in-place.
    # Because we pass the full (Samples, Sites, 2) array:
    # - The distance calculation usually uses Channel 0 (Genotype).
    # - Channel 1 (Color) adds 0 distance (since it's identical down the columns).
    # - When rows are swapped, BOTH channels move together.
    try:
        haplotype_sorter.sort_haplotypes(current_image, ordering=SORT_ORDERING)
    except Exception as e:
        print(f"\nError sorting image {i}: {e}")
        # If sorting fails, just save the unsorted version
        sorted_all_data[i] = all_data[i]
        continue

    # 3. Store the result
    sorted_all_data[i] = current_image

# Save final output
np.save(OUTPUT_NPY_FILE, sorted_all_data)

print("---")
print(f"Success! Saved sorted data to: {OUTPUT_NPY_FILE}")
print(f"Final shape: {sorted_all_data.shape}")