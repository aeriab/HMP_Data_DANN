import numpy as np
import sys
import os

# ================= Configuration =================
# Set the target number of samples you want to keep
TARGET_SAMPLES = 100 

# Define what counts as "missing" in the numpy arrays.
# Set to np.nan if your data uses NaNs, or -1 if it uses integer flags.
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

    # ---------------- DYNAMIC SHAPE HANDLING ----------------
    input_shape = in_array.shape
    ndim = in_array.ndim

    if ndim == 4:
        # Shape: (NUM_SIMS, NUM_SAMPS, WINDOW_SIZE, CHANNELS)
        num_sims, num_samps, win_size, channels = input_shape
        out_shape = (num_sims, TARGET_SAMPLES, win_size, channels)
        print(f"Detected 4D array: {input_shape}")
    elif ndim == 3:
        # Shape: (NUM_SIMS, NUM_SAMPS, WINDOW_SIZE)
        num_sims, num_samps, win_size = input_shape
        out_shape = (num_sims, TARGET_SAMPLES, win_size)
        print(f"Detected 3D array: {input_shape}")
    else:
        print(f"Error: Unexpected array dimensionality: {ndim}. Expected 3 or 4.")
        sys.exit(1)
    # --------------------------------------------------------

    if num_samps <= TARGET_SAMPLES:
        print(f"Warning: Original sample count ({num_samps}) is <= target ({TARGET_SAMPLES}).")
        print("Copying file directly...")
        # Effectively just copies the file structure
    else:
        print(f"Downsampling from {num_samps} to {TARGET_SAMPLES} samples based on data density.")

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

        # Extract the current simulation image
        sim_img = in_array[i]

        # ---------------- LOGIC: CALCULATE MISSINGNESS ----------------
        # Define which data to check for missing values
        if ndim == 4:
            # For 4D (N, S, W, C), check channel 0
            data_to_check = sim_img[:, :, 0]
        else:
            # For 3D (N, S, W), check the whole 2D window directly
            data_to_check = sim_img

        # Calculate missing counts per row (axis 1 is the window size)
        if np.isnan(MISSING_VAL):
            missing_counts = np.isnan(data_to_check).sum(axis=1)
        else:
            missing_counts = (data_to_check == MISSING_VAL).sum(axis=1)

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