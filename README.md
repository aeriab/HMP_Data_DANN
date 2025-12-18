This repo is for converting Human Microbiome Project csv files into npy objects that the DANN model can use.

The csv file of choice is of R. Bromii, and is found at:
"/u/project/ngarud/Garud_lab/HMP_haplos_proc/haplotypes/Ruminococcus_bromii_62047/FP929051_haplotypes.csv"

This genetic data is not controlled for population structure, so we only observe the 154 strains that are grouped by population structure. We delete columns from the original csv unless they match one of the column names from "good_samples.txt".

The resulting cropped csv file is under "ModifyDataForInference/cropped_r_bromii_data.csv"
