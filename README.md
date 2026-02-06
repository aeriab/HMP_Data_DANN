Official details of a color haplotype image.

Numpy shape: (# of images, # of samples, #sites per window, 2)

For the first "black and white" channel: 
a '-1' represents the major allele, 
a '0' represents missing data,
a '1' represents the minor allele.

For the second "color" channel:
a '-1' represents a synonymous mutation, 
a '0' represents major allele or missing data, 
a '1' represents a non-synonymous mutation.