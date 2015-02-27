#!/bin/bash
# This script handles all steps in the model
# creation process after the initial export of the Pathway Tools
# FBA problem. Most of these steps do not require 
# a Pathway Tools session (some exported flat files
# are required,) but building the reduced model does.

# Remove all ATPases
python purge_stoichiometries.py atpase_stoichiometry.txt corn_table.txt corn_table_clean_1.txt > corn_excluded_atpases.txt

# Remove various other problematic reactions
python purge_reactions.py reactions_to_remove.txt corn_table_clean_1.txt corn_table_clean_2.txt

# Remove all reactions involving certain species
python purge_species.py species_to_remove.txt corn_table_clean_2.txt corn_table_clean_3.txt

# Combine the (translated and reformatted) Pathway Tools FBA model, the special cases, and
# the biomass and external exchange reactions, producing corn_model_table.txt 
python assemble_model.py corn_model_table_step1.txt corn_table_clean_3.txt special_cases.txt --biomass biomass_components.txt --exchange exchange_reactions.txt

# Add complete biomass reaction, adapted from iRS1563
python assemble_model.py corn_model_table_step2.txt corn_model_table_step1.txt adapted_irs1563_biomass.txt

# Relocate reactions to subcellular comparments following corn_compartments.txt ,
# writing corn_compartmentalized.txt. 
# The '_in' compartment is an artifact, but treating it as a real compartment
# avoids obscuring its erroneousness
python compartmentalize_table.py corn_model_table_step2.txt corn_compartments.txt corn_compartmentalized.txt --default_compartment cytoplasm --respect_compartments thylakoid_lumen mitochondrial_intermembrane_space mitochondrion in

# Add the intracellular transport reactions and write corn_full.txt
python assemble_model.py corn_full.txt corn_intracellular_transport.txt corn_compartmentalized.txt

# Run the cheap diagnostics
python diagnostics.py corn_full.txt

# 1. Write the full model

# Write an SBML version, corn_fba.xml
SUBCOMPARTMENTS="mitochondrial_intermembrane_space mitochondrion chloroplast thylakoid_lumen biomass peroxisome external intercellular_air_space xylem phloem fixed_biomass"
python table_to_sbml.py corn_full.txt corn_fba --name_hints corn_name_hints.txt --compartments $SUBCOMPARTMENTS > corn_fba.xml  

# Write a report on reactions which occur in more than one compartment,
# listing their associated genes and whether they have been listed
# in gra_overrides.txt yet
python compartment_check.py 

# Store GRA data in the SBML version. 
python add_gra.py corncyc_global_gra.txt corn_fba.xml PARENT_FRAME gra_overrides.txt > corn_fba_with_gra.xml

# Remove singletons, blocked reactions, etc., and write a new SBML version
python deblock_sbml.py corn_fba_with_gra.xml --protect_compartments biomass fixed_biomass external intercellular_air_space xylem phloem --blocked_reaction_file onecell_blocked.txt > iEB5204.xml

# 2. Write a reduced model with better-supported reactions, as many
#  reactions without genes or pathway assignments as possible.  
#  (Branches off from the above process at corn_full.txt).

# Reduce the model, producing corn_reduced.txt
python try_removal.py

# Write an SBML version, corn_fba_reduced.xml
SUBCOMPARTMENTS="mitochondrial_intermembrane_space mitochondrion chloroplast thylakoid_lumen biomass peroxisome external intercellular_air_space xylem phloem fixed_biomass"
python table_to_sbml.py corn_reduced.txt corn_fba --name_hints corn_name_hints.txt --compartments $SUBCOMPARTMENTS > corn_fba_reduced.xml  

# Store GRA data in the SBML version of the reduced model 
python add_gra.py corncyc_global_gra.txt corn_fba_reduced.xml PARENT_FRAME gra_overrides.txt > corn_fba_with_gra_reduced.xml

# Remove singletons, blocked reactions, etc., and write a new SBML version
# Note all blocked reactions should already have been removed from the
# reduced model-- the main thing this will do here is discard unneeded species
python deblock_sbml.py corn_fba_with_gra_reduced.xml --protect_compartments biomass fixed_biomass external intercellular_air_space xylem phloem --blocked_reaction_file onecell_blocked_reduced.txt > iEB2140.xml

# Create a two-cell version from the reduced version
# splitting genes across the two cells, eg 
# 'GRMZM2G000001' -> 'bs_GRMZM2G000001, ms_...' 
# and allowing transport of only a specified list of species between
# the two cells. To reflect leaf physiology, do not allow bundle
# sheath tissue to exchange CO2 and O2 directly with the intercellular
# airspace, nor mesophyll tissue to exchange nutrients with the
# vascular system
python clone.py iEB2140.xml --transport_whitelist transport_whitelist.txt --join_compartments biomass external intercellular_air_space xylem phloem --ms_only_compartments intercellular_air_space --bs_only_compartments xylem phloem --split_genes > iEB2140x2.xml

cp iEB*.xml ..
cp iEB2140*.xml ../fitting/models
