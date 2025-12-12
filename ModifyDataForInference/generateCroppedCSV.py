import numpy as np
import csv
import pandas as pd


good_samples_list = pd.read_csv("good_samples.txt", header=None)[0].tolist()
HMP_haplos = pd.read_csv("/u/project/ngarud/Garud_lab/HMP_haplos_proc/haplotypes/Ruminococcus_bromii_62047/FP929051_haplotypes.csv")



HMP_modified = HMP_haplos.loc[:, HMP_haplos.columns.isin(good_samples_list)]

# Keep only rows where the value in that first column is NOT "NC"
HMP_modified = HMP_modified[HMP_modified["site_type"] != "NC"]

HMP_modified = HMP_modified.fillna(-1)


HMP_modified.to_csv("data_csv_file.csv", index=False)