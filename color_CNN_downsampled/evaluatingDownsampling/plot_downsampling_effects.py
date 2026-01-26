import numpy as np
import matplotlib.pyplot as plt
import os

# ================= Configuration =================
INPUT_FILE = "/u/home/b/baeria/project-ngarud/Research/ProcessHMPData/color_CNN_downsampled/sorted_r_bromii_color.npy"
MISSING_VAL = -1
USABILITY_THRESHOLD = 0.05 
# =================================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE} in memmap mode...")
    in_array = np.load(INPUT_FILE, mmap_mode='r')
    num_sims, orig_n, win_size, _ = in_array.shape
    
    # Range of N values to test: 1 to 154
    n_range = np.arange(1, orig_n + 1)
    # This will store how many images are "usable" at each N
    usability_counts = np.zeros(len(n_range))

    threshold_val = win_size * USABILITY_THRESHOLD
    print(f"Evaluating usability for N in range 1 to {orig_n}...")

    for i in range(num_sims):
        if i % 1000 == 0:
            print(f"Processing simulation {i}/{num_sims}...", end='\r')
            
        sim_img = in_array[i]
        
        # 1. Calculate missingness for all samples in this image once
        missing_counts = (sim_img[:, :, 0] == MISSING_VAL).sum(axis=1)
        
        # 2. Sort missingness (ascending: best quality first)
        sorted_missing = np.sort(missing_counts)

        # 3. Vectorized check: 
        # For a given N, the image is usable if the N-th best sample 
        # is still below the threshold.
        is_usable_at_n = (sorted_missing <= threshold_val)
        
        # Add to cumulative count
        usability_counts += is_usable_at_n

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    plt.plot(n_range, usability_counts, color='darkorange', linewidth=2.5)
    
    # Highlight specific points (like your original targets) for clarity
    targets = [20, 40, 60, 80, 100, 120, 140, 154]
    for t in targets:
        if t <= orig_n:
            plt.scatter(t, usability_counts[t-1], color='black', zorder=5)
            plt.text(t, usability_counts[t-1], f' N={t}', verticalalignment='bottom')

    plt.title(f"Number of 'Usable' Haplotype Images vs. Downsampling Size (Threshold: {USABILITY_THRESHOLD*100}% missing/sample)")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel(f"Total Usable Images (out of {num_sims})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_plot = "usability_plot.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\nSuccess! Plot saved as {output_plot}")

if __name__ == "__main__":
    main()