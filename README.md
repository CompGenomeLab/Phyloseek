# Phyloseek

This repository contains code for computing Pdiff matrices for every residue across the extensive set of 10.2 million proteins covered by PHACT (Kuru et al., 2022) trees. 

These matrices are then fed into a VQ-VAE (Oord et al., 2018) in a per-residue fashion to obtain a lower dimensional representation of this data. This results in a k-letter alphabet similar to 20-letter Foldseek (van Kempen et al., 2023) alphabet.

Note: In case of available resources, you can run the PHACT pipeline to compute Pdiff matrices for rest of the UniRef50.
