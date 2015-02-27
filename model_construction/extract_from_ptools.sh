#!/bin/bash
# This script handles all steps in the 
# model creation process which require an available
# pathway tools session (except the FBA model construction
# itself, which can't be done from the command line 
# as far as I know.)

#pathway-tools -api &

# Extract FBA model
python parse_ptools_fba_solution.py ptools_fba_problem/corn_simple.sol corn_table.txt corn > corn_exclusions.txt

# Extract name strings for reaction and compound frames
python names_from_pathway_tools.py corn > corn_name_hints.txt

# Extract gene annotations
python gra_from_corncyc.py
# That produces two tables of GRA, one with genes directly associated to 
# reactions, the other showing genes associated with generic forms, etc., of reactions;
# combine them, discarding duplicates:
cat gene_to_corncyc_reaction_indirect.txt gene_to_corncyc_reaction_direct.txt | sort -u > corncyc_global_gra.txt

