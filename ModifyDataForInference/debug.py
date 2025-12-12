import sys
import os
import numpy as np
import pandas as pd


image_path = './first_test_npy_file.npy'

single_image = np.load(image_path)

# single_image_array = np.expand_dims(single_image, axis=0)

print(single_image.shape)

# HMP_modified = pd.read_csv("data_csv_file.csv")

# label_column = HMP_modified.iloc[0:, 1]
# print(label_column)