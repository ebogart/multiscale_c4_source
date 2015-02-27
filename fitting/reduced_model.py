# 10 October 2014: add pmd proton transport

""" Add nonlinear constraints to CornCyc-based two-tissue FBA model.

model(): return an nlcm.NonlinearModel instance representing the full
    two-cell model with sensible defaults
reactions: KeyedList of reaction objects
reaction_ids: list of reactions in the model
exchanges: list of tx/sink reactions in the model
decoupled: set of reactions which may act in closed loops
blocked: set of blocked reactions
fva_cache: dict of fva results for (some) of the (reversible)
    reactions in the model, given the bounds applied by default

"""
import pickle
import numpy as np

import fluxtools.sbml_interface as si
import fluxtools.nlcm as nlcm
import fluxtools.stoichiometry_matrix as st
from fluxtools.functions import Linear

####################################################################
# 0. Global parameters

co2_scaling_factor =  0.001 # internal co2 units per microbar
o2_scaling_factor = 1e-4 # internal o2 units per microbar

####################################################################
# I. Import the SBML model as a SloppyCell network, and identify
# various components

corn_net = si.fromSBMLFile('models/iEB2140x2.xml')
reactions = corn_net.reactions.copy()
reaction_ids = reactions.keys()
exchanges = [r for r in reaction_ids if 'sink' in r or 'tx' in r]

# At one time we relaxed the assumption of conservation of 
# cytosolic water and protons, but this is no longer necessary
nonconserved_ids = []

# For later reference, list parameters that need to be specified
all_parameters = ['bs_CO2_conductivity',
                  'rubisco_kc',
                  'rubisco_ko',
                  'rubisco_vomax_vcmax_ratio',
                  'pepc_kc',
                  'ms_CO2', 
                  'ms_oxygen', 
                  'bs_oxygen']

# External partial pressures may be treated either as variables or parameters,
# here, use parameters for simplicity in setting up elastic band models.       
basic_parameters = {'ms_oxygen': 200000. * o2_scaling_factor,
                    'ms_oxygen_chloroplast': 200000. * o2_scaling_factor,
                    'ms_CO2': 300. * co2_scaling_factor,
                    'ms_CO2_chloroplast': 300. * co2_scaling_factor
                    }

# Fix the kinetic parameters. Note these are currently treated as
# (usually fixed) variables, not hardwired parameters, because we
# might eventually allow them to take values within feasible ranges.

# In the following, the units are:
# bs_CO2_conductivity: flux units/ubar * (ubar/internal co2 unit) 
# bs_O2_conductivity: flux units/ubar * (ubar/internal O2 unit) 
# rubisco_kc: ubar * (internal co2 unit/ubar)
# rubisco_ko: ubar * (internal o2 unit/ubar)
# rubisco_vomax_vcmax_ratio: dimensionless
# pepc_kc: ubar * (internal co2 unit/ubar)

default_kinetic_parameters = {'bs_CO2_conductivity': 1.03e-3/co2_scaling_factor, 
                              'bs_O2_conductivity': 0.047*1.03e-3/o2_scaling_factor,
                              'rubisco_kc': 650. * co2_scaling_factor,
                              'rubisco_ko': 450000. * o2_scaling_factor,
                              'rubisco_vomax_vcmax_ratio': 0.2673, 
                              # (ko/(kc*S))       
                              'pepc_kc': 80. * co2_scaling_factor
                              }

default_max_light = 550. # uE m^{-2} s^{-1}

#########################################################################
# IV. Nonlinear model generator

def model(compile_model=True, free_biomass=True, 
          default_flux_bound=1000., max_light=default_max_light,
          extra_free_compartments=set(), biomass_reporters = True):

    """
    Add nonlinear kinetic constraints and return an optimization model.

    If free_biomass is True, use individual sink reactions for biomass 
    components; if False, suppress those and use the (otherwise inactivated)
    biomass reaction derived from iRS1563, consuming biomass components 
    in a fixed ratio.

    biomass_reporters - if true, variables representing total
        rates of production of various categories of biomass, as well
        as overall biomass production, in units of milligrams per square meter
        per second, are added. Ignored if free_biomass is false.

    """
    free_compartments = set(('external','intercellular_air_space','xylem',))
    free_compartments.update(extra_free_compartments)

    if free_biomass:
        free_compartments.add('biomass')

    model = nlcm.NonlinearNetworkModel('corn_nonlinear', 
                                       corn_net)
    max_kinetic_vmax = default_flux_bound
    model.set_bounds({r.id: (-1.0*default_flux_bound if r.reversible else 0.,
                             default_flux_bound) for r in model.reactions})

    # All species conserved by default; free some
    all_nonconserved = nonconserved_ids + [s.id for s in corn_net.species if
                                           s.compartment in free_compartments]
    model.do_not_conserve(*all_nonconserved)

    # Establish that certain quantities are parameters, not variables
    model.parameters.update(basic_parameters)

    #####################################################################
    # Objective function.
    #
    # The primary objective function in many cases is CO2 assimilation.
    # Note we invert the consumption rate so that minimizing the function
    # maximizes assimilation.
    model.set_objective('CO2_consumption','-1.0*ms_tx_CARBON_DIOXIDE')

    ######################################################################
    # Add particular constraints (kinetic laws, etc.)
    # Useful variable ids: (remember _chloroplast suffix where necessary)
    # CO2: ms_CO2, bs_CO2, 
    # O2:  ms_oxygen, bs_oxygen
    # (rubisco oxygenase) ms_RXN_961_chloroplast
    # (rubisco carboxlyase) ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast
    # plasmodesmata_ms_CO2_bs_CO2
    # plasmodesmata_ms_oxygen_bs_oxygen
    # bs_PEPCARBOX_RXN

    # Constraints on CO2 partial pressures
    # Plasmodesmata transport is currently defined with 
    # positive OUTWARD (bs -> me)
    model.add_constraint('CO2_leakiness_constraint',
                         'plasmodesmata_ms_CO2_bs_CO2'
                         ' - bs_CO2_conductivity * (bs_CO2 - ms_CO2)',
                         0.)
    model.add_constraint('bs_CO2_equality',
                                 'bs_CO2 - bs_CO2_chloroplast', 0.)
    # Constraints on O2 partial pressures
    model.add_constraint('O2_leakiness_constraint',
                         'plasmodesmata_ms_oxygen_bs_oxygen'
                         ' - bs_O2_conductivity * (bs_oxygen - ms_oxygen)',
                         0.)
    model.add_constraint('bs_O2_equality',
                                 'bs_oxygen - bs_oxygen_chloroplast', 0.)

    # Note that other partial pressures are specified as parameters
    # (otherwise we'd need additional equalities or diffusion relations)

    # To prevent infeasibilities, phrase rubisco kinetic laws in terms 
    # of a level of active rubisco, which may vary between zero and the 
    # true rubisco vcmax
    
    model.add_constraint('bs_rubisco_activity_constraint',
                         'bs_rubisco_vcmax - bs_active_rubisco',
                         (0., None))
    model.add_constraint('ms_rubisco_activity_constraint',
                         'ms_rubisco_vcmax - ms_active_rubisco',
                         (0., None))
    model.set_lower_bound('bs_active_rubisco',0.)
    model.set_bound('bs_rubisco_vcmax',(0., max_kinetic_vmax))
    model.set_lower_bound('ms_active_rubisco',0.)
    model.set_bound('ms_rubisco_vcmax',(0., max_kinetic_vmax))

    model.add_constraint('ms_rubisco_carboxylase_kinetic_constraint',
                         '(ms_active_rubisco * ms_CO2_chloroplast) / ' 
                         '(ms_CO2_chloroplast + rubisco_kc * '
                         '(1.0 + ms_oxygen_chloroplast / rubisco_ko)) - '
                         'ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast', 0.)
    model.add_constraint('bs_rubisco_carboxylase_kinetic_constraint',
                         '(bs_active_rubisco * bs_CO2_chloroplast) / ' 
                         '(bs_CO2_chloroplast + rubisco_kc * '
                         '(1.0 + bs_oxygen_chloroplast / rubisco_ko)) - '
                         'bs_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast', 0.)
    
    # As an alternative to setting the rubisco oxygenase rate through 
    # directly fixing the o/c ratio, use the same form of 
    # Michaelis-Menten constraint for both rubisco reactions.
    
    model.add_constraint('ms_rubisco_oxygenase_kinetic_constraint',
                         'rubisco_vomax_vcmax_ratio * ' + 
                         '(ms_active_rubisco * ms_oxygen_chloroplast) / ' + 
                         '(ms_oxygen_chloroplast + rubisco_ko * ' + 
                         '(1.0 + ms_CO2_chloroplast / rubisco_kc)) - ' + 
                         'ms_RXN_961_chloroplast', 0.)
    model.add_constraint('bs_rubisco_oxygenase_kinetic_constraint',
                         'rubisco_vomax_vcmax_ratio * ' + 
                         '(bs_active_rubisco * bs_oxygen_chloroplast) / ' + 
                         '(bs_oxygen_chloroplast + rubisco_ko * ' + 
                         '(1.0 + bs_CO2_chloroplast / rubisco_kc)) - ' + 
                         'bs_RXN_961_chloroplast', 0.)

    # Similarly, require PEPC flux in each compartment to be less than or 
    # equal to its MM expression in an analogous way. 
    # Here, (rate) <= (V_{pmax} C) / (C + K_c)
    # becomes 0 <= (V_{pmax} C) / (C + K_c) - rate
    # (Note this would allow arbitrary reverse rates if the reaction 
    # were set to reversible; it isn't, so we ignore this.)

    model.add_constraint('bs_pepc_activity_constraint',
                         'bs_pepc_vmax - bs_active_pepc',
                         (0., None))
    model.add_constraint('ms_pepc_activity_constraint',
                         'ms_pepc_vmax - ms_active_pepc',
                         (0., None))
    model.set_lower_bound('bs_active_pepc',0.)
    model.set_lower_bound('ms_active_pepc',0.)
    model.set_bound('ms_pepc_vmax',(0., max_kinetic_vmax))
    model.set_bound('bs_pepc_vmax',(0., max_kinetic_vmax))


    model.add_constraint('ms_pepc_kinetic_constraint',
                         '(ms_active_pepc * ms_CO2) / ' + 
                         '(pepc_kc + ms_CO2) - ms_PEPCARBOX_RXN',
                         0.)
    model.add_constraint('bs_pepc_kinetic_constraint',
                         '(bs_active_pepc * bs_CO2) / ' + 
                         '(pepc_kc + bs_CO2) - bs_PEPCARBOX_RXN',
                         0.)

    # Constraint on light absorption. Light is the only
    # species which can be taken up separately by both 
    # mesophyll and bundle sheath compartments; this allows
    # control of total light uptake, and is more convenient 
    # than rewriting the network to supply both compartments
    # from a single pool.
    
    model.add_constraint('max_light_constraint', 
                         'max_light_uptake - bs_tx__Light_ - ms_tx__Light_',
                         (0., None))
    model.parameters['max_light_uptake'] = max_light
    #
    ###################################################################

    model.set_bounds(default_kinetic_parameters)
    if free_biomass:
        model.set_bounds({'bs_CombinedBiomassReaction': 0.,
                          'ms_CombinedBiomassReaction': 0.,})
        if biomass_reporters:
            add_biomass_reporters(model)

    if compile_model:
        model.compile()

    return model

########################################
# Species representing total biomass production

masses = {'sink_1_3_beta_D_glucan_monomer_equivalent': -163.149, # UDP-GLUCOSE - UDP 
          'sink_1_4_alpha_D_Glucan_monomer_equivalent_chloroplast': -163.15, # ADP-D-GLUCOSE - ADP
          'sink_1_4_beta_D_xylan_monomer_equivalent':  -133.123, # UDP-D-XYLOSE - UDP
          'sink_ARG': -175.21,
          'sink_ASN': -132.119,
          'sink_ASCORBATE': -175.12,
          'sink_Arabinoxylan_monomer_equivalent': -266.246, # 1-4-beta-D-xylan
                                                            # +
                                                            # (UDP-L-ARABINOSE
                                                            # - UDP)
          'sink_CELLULOSE_monomer_equivalent': -163.149, # UDP-GLUCOSE - UDP
          'sink_CHLOROPHYLL_A': -893.503,
          'sink_CHLOROPHYLL_B': -907.486,
          'sink_CONIFERYL_ALCOHOL': -180.203,
          'sink_COUMARYL_ALCOHOL': -150.177,
          'sink_CPD_13612': -302.519,
          'sink_CPD_440': -313.35,
          'sink_CPD_649': -380.484,
          'sink_CYS': -121.154,
          'sink_DNA_base_equivalent': -307.9415, # average of DATP,
                                                 # DCTP, DGTP, TTP
                                                 # minus PPI
          'sink_GLN': -146.146,
          'sink_GLT': -146.122,
          'sink_GLY': -75.067,
          'sink_Glucomannan_monomer_equivalent': -163.15, # GDP-MANNOSE - GDP
          'sink_Glucuronoxylan_monomer_equivalent': -309.248, # 1-4-beta-D-xylan
                                                              # +
                                                              # (UDP-GLUCURONATE
                                                              # - UDP)
          'sink_HIS': -155.156,
          'sink_Homogalacturonan_monomer_equivalent': -176.125, # UDP-D-GALACTURONATE
                                                                # -
                                                                # UDP
          'sink_ILE': -131.174,
          'sink_LEU': -131.174,
          'sink_LINOLEIC_ACID': -279.442,
          'sink_LINOLENIC_ACID': -277.426,
          'sink_LYS': -147.197,
          # Comments on lipids. I believe the only way to produce
          # anything with a diacylglycerol in it is with oleate and
          # palmitate side groups (oleate, C18H33O2; palmitate
          # C16H31O2).  We can use this to work out the molecular
          # weights, for example by looking up the KEGG generic
          # lipids, for example, which provide chemical formulas with
          # subgroups as R; they include the carboxylic acids from
          # each acyl group explicitly, we add (C18H3302 + C16H3102 -
          # 2*COO(-) = C32H64) to recover the complete formula
          'sink_L_1_PHOSPHATIDYL_ETHANOLAMINE': -717.995, #C32H64 +
                                                          #C7H12NO8P =
                                                          #C39H76NPO8
          'sink_L_1_PHOSPHATIDYL_GLYCEROL': -749.0046, # C32H64 + C8H13O10P
          'sink_L_ALPHA_ALANINE': -89.094,
          'sink_L_ASPARTATE': -132.096,
          'sink_MET': -149.207,
          'sink_OCTADEC_9_ENE_118_DIOIC_ACID': -310.43,
          'sink_OLEATE_CPD': -281.457,
          'sink_PALMITATE': -255.42,
          'sink_PHE': -165.191,
          'sink_PHOSPHATIDYLCHOLINE': -760.0746, # C32H64 + C10H18NO8P
          'sink_PHYTOSPINGOSINE': -318.519,
          'sink_PRO': -115.132,
          'sink_RNA_base_equivalent': -320.4345, # average of the four
                                                 # triphosphates minus
                                                 # PPI
          'sink_SER': -105.093,
          'sink_SINAPYL_ALCOHOL': -210.229,
          'sink_STEARIC_ACID': -283.473,
          'sink_SUCROSE': -342.299,
          'sink_SULFOQUINOVOSYLDIACYLGLYCEROL': -821.145, # C32H64 + C11H16O12S
          'sink_THR': -119.12,
          'sink_TRP': -204.228,
          'sink_TYR': -181.191,
          'sink_VAL': -117.147,
          # Xylogalacturonan-monomer = Homogalacturonan-monomer +
          # (UDP-D-XYLOSE - UDP)
          'sink_Xylogalacturonan_monomer_equivalent': -309.248, 
          # Xyloglucan-monomer = Cellulose-monomer + (UDP-D-XYLOSE - UDP)
          'sink_Xyloglucan_monomer_equivalent': -296.272, 
          'sink__D_Galactosyl_12_diacyl_glycerols_': -757.087, # C32H64 + C11H16O10
          'sink__Galactosyl_galactosyl_diacyl_glycerols_': -919.226, # C32H64
                                                                     # +
                                                                     # C17H26O15
          'sink__L_1_phosphatidyl_inositols_': -837.0658, # C32H64 + C11H17O13P
          'sink_amylopectin_monomer_equivalent_chloroplast': -163.15, # same as 1-4-alpha-D-glucan
          'sink_polysaccharide_arabinose_unit': -133.123, # UDP-L-ARABINOSE - UDP
          'sink_polysaccharide_galactose_unit': -163.15, # GDP-L-GALACTOSE - GDP 
          'sink_polysaccharide_galacturonate_unit': -176.125, # UDP-D-GALACTURONATE
                                                              # - UDP
          'sink_polysaccharide_glucose_unit': -163.149, # UDP-GLUCOSE - UDP
          'sink_polysaccharide_glucuronate_unit': -176.125, # UDP-GLUCURONATE - UDP
          'sink_polysaccharide_mannose_unit': -163.15, # GDP-MANNOSE - GDP 
          'sink_polysaccharide_xylose_unit': -133.123, # UDP-D-XYLOSE - UDP
          'tx_CARBON_DIOXIDE': 44.01,
          'tx_MG_2': 24.305,
          'tx_NITRATE': 62.005, 
          'tx_OXYGEN_MOLECULE': 31.999,
          'tx_PROTON': 1.,
          'tx_SUCROSE': 342.299,
          'tx_SULFATE': 96.058,
          'tx_WATER': 18.015,
          'tx__Light_': 0.,
}          

biomass_types = {
    'starch': ['amylopectin_monomer_equivalent_chloroplast',
               '1_4_alpha_D_Glucan_monomer_equivalent_chloroplast',
               ],
    'chlorophyll': ['CHLOROPHYLL_A',
                    'CHLOROPHYLL_B',],
    'amino_acids': ['L_ALPHA_ALANINE',
                    'L_ASPARTATE',
                    'ARG',
                    'ASN',
                    'CYS',
                    'GLN',
                    'GLT',
                    'GLY',
                    'HIS',
                    'ILE',
                    'LEU',
                    'LYS',
                    'MET',
                    'PHE',
                    'PRO',
                    'SER',
                    'THR',
                    'TRP',
                    'TYR',
                    'VAL',],
    'monolignols': ['CONIFERYL_ALCOHOL',
                    'COUMARYL_ALCOHOL',
                    'SINAPYL_ALCOHOL'],
    'nucleic_acids': ['DNA_base_equivalent',
                      'RNA_base_equivalent',],
    'suberin_precursors': ['CPD_440',
                           'OCTADEC_9_ENE_118_DIOIC_ACID',],
    'lipids': ['L_1_PHOSPHATIDYL_ETHANOLAMINE',
               'L_1_PHOSPHATIDYL_GLYCEROL',
               'PHOSPHATIDYLCHOLINE',
               'PHYTOSPINGOSINE',
               'SULFOQUINOVOSYLDIACYLGLYCEROL',
               '_D_Galactosyl_12_diacyl_glycerols_',
               '_Galactosyl_galactosyl_diacyl_glycerols_',
               '_L_1_phosphatidyl_inositols_',
               'CPD_649',
               'CPD_13612',],
    'fatty_acids': ['STEARIC_ACID',
                    'OLEATE_CPD',
                    'PALMITATE',
                    'LINOLEIC_ACID',
                    'LINOLENIC_ACID',],
    'cellulose_hemicellulose': [
        'Arabinoxylan_monomer_equivalent',
        'CELLULOSE_monomer_equivalent',
        'Glucomannan_monomer_equivalent',
        'Glucuronoxylan_monomer_equivalent',
        'Homogalacturonan_monomer_equivalent',
        'Xylogalacturonan_monomer_equivalent',
        'Xyloglucan_monomer_equivalent',
        '1_3_beta_D_glucan_monomer_equivalent',
        '1_4_beta_D_xylan_monomer_equivalent',
    ],
   'other': ['ASCORBATE',
             'SUCROSE',]
}

def add_biomass_reporters(model,offset=0.1):
    total_coefficients = {'total_production_all_biomass': -1.}
    for category, instances in biomass_types.iteritems():
        coefficients = {}
        for instance in instances:
            sink = 'sink_%s' % instance
            mass = -0.001*masses[sink] # converting to mg/micromole
            for tag in ('ms','bs'):
                coefficients[tag + '_' + sink] = mass
        variable_name = 'total_production_' + category
        coefficients[variable_name] = -1.0
        total_coefficients[variable_name] = 1.0
        # Total production must always be positive; 
        # set a lower bound on it to help convergence, but
        # offset this bound from zero (also to help convergence)
        model.set_bound(variable_name,(-1.0*offset, None))
        constraint_name = variable_name + '_constraint'
        constraint = Linear(coefficients, name=constraint_name)
        model.constraints.set(constraint_name, constraint)
        model.set_bound(constraint_name, 0.)
    total_constraint = Linear(total_coefficients, 'total_biomass_constraint')
    model.constraints.set('total_biomass_constraint', total_constraint)
    model.set_bound('total_biomass_constraint', 0.)
                
