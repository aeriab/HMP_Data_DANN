import numpy as np
import sys
import os


# python downsample_sims.py <input_npy_file> <output_npy_file>

# ================= Configuration =================
# Set the target number of samples you want to keep
TARGET_SAMPLES = 100 

# Define what counts as "missing" in the numpy arrays.
MISSING_VAL = -1
# =================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python downsample_sims.py <input_npy_file> <output_npy_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    print(f"Loading {input_path}...")
    
    # Load in read-only memmap mode to handle large files without RAM explosion
    try:
        in_array = np.load(input_path, mmap_mode='r')
    except ValueError:
        # Fallback if the file wasn't saved with np.save
        in_array = np.lib.format.open_memmap(input_path, mode='r')

    # Get original shape
    # Expected shape: (NUM_SIMS, NUM_SAMPS, WINDOW_SIZE, CHANNELS)
    num_sims, num_samps, win_size, channels = in_array.shape

    if num_samps <= TARGET_SAMPLES:
        print(f"Warning: Original sample count ({num_samps}) is <= target ({TARGET_SAMPLES}).")
        print("Copying file directly...")
        # In this case, just copy the file logic or exit
        # For this script, we will just proceed, effectively doing nothing but copying
    
    print(f"Processing {num_sims} simulations.")
    print(f"Downsampling from {num_samps} to {TARGET_SAMPLES} samples per image based on data density.")

    # Prepare output array
    out_shape = (num_sims, TARGET_SAMPLES, win_size, channels)
    
    # Create the output file on disk
    out_array = np.lib.format.open_memmap(
        output_path, 
        mode='w+', 
        dtype=in_array.dtype, 
        shape=out_shape
    )

    # Process loop
    for i in range(num_sims):
        if i % 100 == 0:
            print(f"Processing simulation {i}/{num_sims}...", end='\r')

        # Extract the current simulation image: Shape (NUM_SAMPS, WIN_SIZE, 2)
        sim_img = in_array[i]

        # ---------------- LOGIC: CALCULATE MISSINGNESS ----------------
        # We check channel 0 (allelic state) for missing values across the window
        
        if np.isnan(MISSING_VAL):
            # Count NaNs per row (axis 1 is the window size)
            missing_counts = np.isnan(sim_img[:, :, 0]).sum(axis=1)
        else:
            # Count occurrences of specific integer flag
            missing_counts = (sim_img[:, :, 0] == MISSING_VAL).sum(axis=1)

        # ---------------- LOGIC: SORT & SELECT ----------------
        # argsort returns indices that would sort the array (ascending)
        # We want the rows with the LEAST missing data (lowest counts) first.
        sorted_indices_by_quality = np.argsort(missing_counts)

        # Select the top N best indices
        best_indices = sorted_indices_by_quality[:TARGET_SAMPLES]

        # IMPORTANT: Re-sort the indices numerically.
        # This preserves the original relative ordering (e.g., if your data 
        # was already sorted by haplotype frequency, this keeps that sort 
        # intact for the retained samples).
        best_indices.sort()

        # Write to output
        out_array[i] = sim_img[best_indices]

    # Flush changes to disk
    del out_array
    del in_array

    print(f"\nSuccess! Downsampled data saved to: {output_path}")
    print(f"New shape: {out_shape}")

if __name__ == "__main__":
    main()