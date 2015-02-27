#!/bin/bash 

# Steps to regenerate the data-fitting calculations presented in the
# text. The commands are collected in this file for explanatory
# purposes, but they were run separately to generate the original
# results, so performance of this script is not guaranteed, and it is
# probably not a good idea to run it as is, unless you are content to 
# wait several days for everything to finish (assuming ~60 available
# processors.)

# Extract averages and standard deviations from the 15-segment RNA-seq data
cd data
python process_fpkm.py

# Combine 15-segment data with 3-segment LCM data, interpolated, 
# to produce 15-segments of cell-type-specific data
python combine_data.py
cd ..

# Regenerate a cache of FVA results for the two-cell model
# (allowing, e.g., rapid determination of which reactions can 
# actually take positive and negative signs.) Uses 50 processes.
python update_fva_cache.py 

# Do some short calculations:
# 1. Generate a parsimonious FBA solution for comparison to the 
# fitting solution
python fba_comparison.py

# 2. Count genes in the model associated with 
# the effective Vpr parameter 
python vpr_table.py > vpr_table.txt

# 3. Generate data for figure 2
python figure2.py

# Perform the basic fit.
# Saves results as gradient_fit.pickle and gradient_fit.csv
python gradient_fit.py

# Use the preceding result to do FVA calculations for selected variables,
# writing key_fva_broad.pickle and key_fva.csv
python key_fva.py

# Perform variant fits, in parallel:
# ... with scale factors fixed to zero:
python fit_zero_scales.py & # fit_zero_scales.pickle
# ... with a fixed biomass composition:
python fit_fixed_biomass.py & # fit_fixed_biomass.pickle
# ... using the E-flux method:
python fit_eflux.py & # fit_eflux.pickle
# ... using the E-flux method with fixed biomass composition:
python fit_eflux_fixed.py & # fit_eflux_fixed.pickle

# After all the above finish, copy all the .pickle files from this directory
# to ../figures before redrawing the figures (except the intermediate
# key_fva_nnn.pickle partial saves from the FVA calculation.)

