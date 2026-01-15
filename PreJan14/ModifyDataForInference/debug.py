import sys
import os
import numpy as np
import pandas as pd

# Load the CSV file
df = pd.read_csv('cropped_r_bromii_data.csv')

# Drop the second column (index 1)
# axis=1 tells pandas to look for a column, not a row
df.drop(df.columns[1], axis=1, inplace=True)

# Save the result to a new file
df.to_csv('bw_cropped_r_bromii_data.csv', index=False)