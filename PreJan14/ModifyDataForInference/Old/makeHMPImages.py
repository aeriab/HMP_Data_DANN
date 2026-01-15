import numpy as np
import csv
import pandas as pd

HMP_modified = pd.read_csv("data_csv_file.csv")
# --- Start of new NPY object creation ---

# 1. Define the slice boundaries
num_cols_to_take = 100
num_rows_to_take = 201

npy_object = np.array([[1,2,3],[4,5,6],[7,8,9]])


np.save('data_npy_file.npy', npy_object)

# --- End of new NPY object creation ---