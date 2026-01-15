import matplotlib.pyplot as plt
import pandas as pd
import sys

# 1. Read the file
# sep=r'\s+' handles both tabs and spaces which is common in these result files
filename = 'results.txt'

try:
    df = pd.read_csv(filename, sep=r'\s+', engine='python')
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    sys.exit(1)

# Ensure data is sorted by genomic position
df = df.sort_values('Center')

# 2. Setup the plot
plt.figure(figsize=(12, 6))

# 3. Plot the scores
# We plot all three probabilities to see the full picture.
# Neutral is usually high, while sweeps might be low signal spikes.
plt.plot(df['Center'], df['Prob_Neu'], label='Neutral', color='gray', linewidth=1, alpha=0.6)
plt.plot(df['Center'], df['Prob_Soft'], label='Soft Sweep', color='blue', linewidth=1.0, alpha=0.6)
plt.plot(df['Center'], df['Prob_Hard'], label='Hard Sweep', color='red', linewidth=1.0, alpha=0.6)

# 4. Formatting
plt.title('Predicted Probabilities across Genomic Position')
plt.xlabel('Genomic Position (Center BP)')
plt.ylabel('Probability Score')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)

# Set Y-axis limit
# Standard probability plot is 0 to 1.
plt.ylim(-0.05, 1.05)

# OPTIONAL: If you want to zoom in on very small sweep probabilities 
# (similar to the ylim(top=0.02) in your example), uncomment the line below.
# Note: This will cut off the Neutral line which is near 1.0.
# plt.ylim(top=0.05) 

# 5. Save output
output_file = 'genome_scan_scores.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")