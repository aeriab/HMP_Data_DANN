import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

# 1. Read the file
# sep=r'\s+' handles both tabs and spaces which is common in these result files
filename = 'results.txt'
df = pd.read_csv(filename, sep=r'\s+', engine='python')

# d = df.sort_values('Center').reset_index(drop=True)
df_binned = df.groupby(df.index // 5).mean(numeric_only=True)

highest_prob = df_binned[['Prob_Neu', 'Prob_Soft', 'Prob_Hard']].idxmax(axis=1)

color_map = {'Prob_Neu': 'grey', 'Prob_Soft': 'skyblue', 'Prob_Hard': 'red'}

point_colors = highest_prob.map(color_map)

# 2. Setup the plot
plt.figure(figsize=(12, 6))

# 3. Plot the scores
# We plot -log10(Prob_Neu) as a single dot, colored by classification
y_values = -np.log10(df_binned['Prob_Neu'] + 1e-9)

plt.scatter(df_binned['Center'], y_values, c=point_colors, s=15, alpha=0.8)

# Formatting
plt.title('R. Bromii Genomic Scan (5-window average)')
plt.xlabel('Genomic Position (Center BP)')
plt.ylabel('-log10(P_Neutral)')
plt.grid(True, alpha=0.2)

# Setting a limit to the y-axis
# -log10(1) = 0 corresponds to high P_neu, -log10(0.01)=2 corresponds to low P_neu
plt.autoscale(enable=True, axis='y')

# 5. Save output
output_file = 'genome_scan_scores.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")
