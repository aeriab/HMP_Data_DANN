This repo is for converting Human Microbiome Project csv files into npy objects that the DANN model can use.

The csv file of choice is of R. Bromii, and is found at:
"/u/project/ngarud/Garud_lab/HMP_haplos_proc/haplotypes/Ruminococcus_bromii_62047/FP929051_haplotypes.csv"

This genetic data is not controllled for population structure, so we only observe the 153 strains that are grouped by population structure. We further shrink this to 100 using column names from "good_samples.txt", purely for testing purposes so that this data fits with our pretrained DANN.

