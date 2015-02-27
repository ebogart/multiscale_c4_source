"""
Reload the result of gradient_fit.py and then run FVA calculations.

The reload is accomplished somewhat awkwardly by setting up the entire
fitting machinery again and then filling in the attributes of the fit
object from the save

"""
import pickle, logging, time
from os import getpid
import numpy as np
import reduced_model as fm
import real_rna_data as rd
import real_enzyme_data as enz
import replica_fits as rf
import flexible as flex

########################################
# Set logging options.

#logger = logging.getLogger('fitting')
#logger.setLevel(logging.DEBUG)

########################################
# Set global IPOPT options.

import fluxtools.nlcm as nlcm
nlcm.default_ipopt_options = {
    'print_level': 0,
    'tol': 1e-5,
    'linear_solver': 'ma97',
    'ma97_print_level': -1,
    'max_iter': 500
}

########################################
# Set options for the fitting process

default_rescale = 1.
N = 15

########################################
# Set up the basic model.

_base_model = fm.model(default_flux_bound=1000.*default_rescale,
                       extra_free_compartments=('phloem',))
oxidases = ['bs_GenericNADOxidase_mod',
            'bs_GenericNADPOxidase_mod',
            'ms_GenericNADOxidase_mod',
            'ms_GenericNADPOxidase_mod']
oxidase_bounds = dict.fromkeys(oxidases,0.)
_base_model.set_bounds(oxidase_bounds)

phloem_transporters = ['bs_tx_CPD_397',
                       'bs_tx_LYS',
                       'bs_tx_ARG',
                       'bs_tx_TYR',
                       'bs_tx_GLT',
                       'bs_tx_GLY',
                       'bs_tx_ILE',
                       'bs_tx_L_ASPARTATE',
                       'bs_tx_THR',
                       'bs_tx_HIS',
                       'bs_tx_SUCROSE',
                       'bs_tx_GLUTATHIONE',
                       'bs_tx_PHE',
                       'bs_tx_L_ALPHA_ALANINE',
                       'bs_tx_ASN',
                       'bs_tx_VAL',
                       'bs_tx_LEU',
                       'bs_tx_MET',
                       'bs_tx_SER']
phloem_bounds = dict.fromkeys(phloem_transporters, 0.)
phloem_bounds.update({'bs_tx_SUCROSE': (-2.5, 2.5),
                      'bs_tx_GLY': (-2.5, 2.5),
                      'bs_tx_GLUTATHIONE': (-2.5, 2.5)})
_base_model.set_bounds(phloem_bounds)

# Simplify the model. Note, this does not change the feasible space
# and so does not affect the solution; rather, it removes redundancies
# in the way the problem is posed.
from simplification import simplify
base_model, details = simplify(_base_model)

########################################
# Prepare the data (with a function, for interactive use)

enzymes_to_reactions = enz.enzyme_to_variables

def prepare_data(fresh_weight_of_one_sq_m_in_g=150.,
                 rna_to_enzyme_scale=0.00490,
                 flux_rescale=default_rescale,
                 min_abs_err_fpkm=7.5, 
                 rna_min_relative_error=0.1,
                 start=0,
                 N=15,
                 M=1500, # i.e., all the reactions for which data can be had
                 offset=0):

    # Allow overall rescaling of fluxes away from micromoles/sq m/s
    rescale = flux_rescale
    bs_CO2_conductivity = rescale*fm.default_kinetic_parameters[
        'bs_CO2_conductivity'
    ]
    bs_O2_conductivity = rescale*fm.default_kinetic_parameters[
        'bs_O2_conductivity'
    ]

    # Load enzyme data, rescaling it and changing its units.  
    # Enzyme activity measurements are in nmol/min/g FW.
    assay_units_to_flux_units = ((1/60.) * 
                                 fresh_weight_of_one_sq_m_in_g * 
                                 0.001 *
                                 rescale)
    enzyme_data, enzyme_errors = enz.data_and_error(scale=assay_units_to_flux_units)
    enzyme_data_subset = {k:enzyme_data[k] for k in
                          enzymes_to_reactions.keys()} # necessary?

    # Load RNA data, controlling the overal scaling.

    # Least-squares fit of raw (mesophyll+bundle sheath) data for PEPC
    # to the enzyme data for PEPC in appropriate units with
    # weight_of_one_sq_m_in_g = 150 shows the best constant of
    # proportionality is 0.00490.
    scale = rescale * rna_to_enzyme_scale

    # RNA data below 'tiny_threshold' will be considered to be
    # effectively zero, in the sense that we don't need to know the
    # direction of the associated reaction-- a flux F and a flux -F
    # represent approximately equally bad deviations from the data
    # (ignoring, for a moment, the issue of scaling). There is no
    # technical reason this must equal the minimum experimental error
    # we apply, but conceptually if we don't believe (or, to
    # facilitate solving the problem, pretend not to believe) we can
    # measure expression levels with a resolution better than \delta,
    # data within \delta of zero might as well be zero.

    mae = min_abs_err_fpkm * scale
    tiny_threshold = mae
    all_data, all_errors = rd.data_and_error(
        scale=scale,
        min_relative_error=rna_min_relative_error,
        min_absolute_error=mae
    )
    
    # Make minor adjustments to the data so it matches the structure
    # of the model.

    # Ensure data is specified only for variables (excluding reactions
    # dropped in simplification, eg.)

    bad_data_keys = []
    for k in all_data.keys():
        if k not in base_model.variables:
            bad_data_keys.append(k)
            all_data.pop(k)

    # Drop RXN-961: both rubisco reactions will be taken into account
    # by fitting the active enzyme level

    for k in all_data.keys():
        if 'RXN_961_chloroplast' in k:
            all_data.pop(k)

    # Redirect the data for rubisco and pepc so that active enzyme
    # levels, not fluxes, are fit.

    for tag in ('ms_', 'bs_'):
        for reaction, vmax in (('PEPCARBOX_RXN', 'active_pepc'),
                               ('RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast',
                                'active_rubisco')):
            if tag == 'bs_' and reaction == 'PEPCARBOX_RXN':
                continue
            # Treating mesophyll and bundle sheath PEPC differently
            # helps numerical performance; theoretically, we speculate
            # that the relationship between PEPC transcripts and active
            # enzyme levels may vary between mesophyll and bundle sheath,
            # given the extent to which its function depends on localization.
            key = tag + reaction
            kinetic_key = tag + vmax
            datum = all_data.pop(key)
            erratum = all_errors.pop(key)
            all_data[kinetic_key] = datum
            all_errors[kinetic_key] = erratum

    # Optionally, select a subset of the data.

    # We consider N points along the leaf gradient, starting at index
    # 'start'.  We include data for the M reactions with the highest
    # overall expression levels (after skipping the top 'offset'
    # reactions).

    # Sort the RNA data and select the indicated portion.

    sorted_keys = sorted(all_data.keys(), key=lambda k:
                         np.sum(all_data[k]), reverse=True)
    data_keys = sorted_keys[offset:offset+M]
    data = {k: all_data[k][start:N+start] for k in data_keys}
    error = {k: all_errors[k][start:N+start] for k in data_keys}

    # Select the indicated points from the enzyme data (note that we
    # keep all enzymes)

    enzyme_data_subset = {k: v[start:N+start] for k,v in
                          enzyme_data_subset.iteritems()}

    data.update(enzyme_data_subset)
    
    return (data, error, bs_CO2_conductivity,
            bs_O2_conductivity, tiny_threshold, data_keys)

(data, error, bs_CO2_conductivity,
 bs_O2_conductivity, tiny_threshold, rna_keys) = prepare_data()

########################################
# Set up relationships among scale factors

# We force the scale factor for each reaction to be shared across
# mesophyll and bundle sheath compartments, so that different choices
# of scale factor can't wash out cell type specificity in the data.

# Note this doesn't apply to the scaling for data comparison
# to the rubisco vmax in each compartment; forcing those scale factors
# to be equal led to numerical problems, and it is clear from the 
# results that cell-type specificity of rubisco activity is maintained
# without this constraint.

relationships = {k:('scale_' + k[3:],) for k in data if k in 
                 rd.reduced_data}

########################################
# Set up the fitting model for an individual point on the gradient.

fva_file = 'data/corn_twocell_reduced_fva_cache.pickle' 
template_model = flex.FlexibleFittingModel(base_model, 
                                           load_fva_results=fva_file)
template_model.set_bound('bs_CO2_conductivity', bs_CO2_conductivity)
template_model.set_bound('bs_O2_conductivity', bs_O2_conductivity)
rxn_data = flex.PerReactionData(template_model, 'gradient_rna', 
                                rna_keys, 
                                scale_factor_relationships=relationships,
                                max_scale_factor=5.)
enzyme_ub_data = flex.EnzymeUpperBoundData(template_model, 'enzyme_bound',
                                           enzymes_to_reactions)
template_model.finalize(total_flux_decomposition_max=1e4*default_rescale)

########################################
# Set up the replica model, fitting all points of the gradient at once.

replica = rf.LeafModel(model=template_model,
                       name='reduced_flex',
                       N=N, n_processes=N,
                       net_phloem_bounds={'SUCROSE_phloem': (None, 0.)})

# Load the data. Note that appyling the enzyme data as upper bounds
# doesn't require the experimental uncertainties to be provided.

replica.data = data
replica.error = error
replica._load_data()

# The tuning of the scale factor prior relative to the 
# least squares costs is completely empirical at this point.

replica.fitting_model.parameters['scale_factor_tuning'] = 1.0
replica.template_model.tiny_threshold = tiny_threshold

# Set options.

# We will determine a starting point that should be approximately
# feasible by finding the minimum-flux best fit solution at each
# segment of the gradient and combining them; the following options
# help IPOPT take advantage of this relatively good guess, mostly by
# decreasing the extent to which the initialization routine (directly,
# or by choice of the starting dual variables and barrier parameter)
# pushes the initial guess away from the boundaries of the feasible
# regime.  Their values are empirically determined and not necessarily
# optimal.

opts = {'bound_push': 1e-6,
        'bound_frac': 1e-6,
        'bound_mult_init_val': 1e-1,
        'mu_init': 1e-3,
        'print_level': 5}
replica.fitting_model.ipopt_options.update(opts)

# Ensure that repeated_solve is called when optimizing fit for individual
# points; this seems to improve convergence
replica.template_model.repeated_solve_max_iter = 250
replica.template_model.repeated_solve_attempts = 3

########################################
# Reload results

with open('gradient_fit.pickle') as f:
    old_result = pickle.load(f)
replica.fitting_model.parameters.update(old_result['params'])
replica.fitting_model.soln = old_result['soln'].copy()
replica.fitting_model.soln_by_image = old_result['soln_by_image'].copy()
replica.fitting_model.cost_by_image = old_result['cost_by_image'].copy()
replica.fit_overall_value = old_result['cost']
replica.fitting_model.obj_value = old_result['cost']
replica.fit_result = old_result['traj']
replica.guesses = old_result['guesses']
replica.fitting_model.x = np.array([old_result['soln'][v] for v in 
                                    replica.fitting_model.variables])

########################################
# Do FVA calculations.
relative_delta = 0.001
filename_prefix = 'key_fva'
n_procs = 32
log_interval = 10
fva_model = replica.fitting_model

import time, pickle
import fluxtools.fva as fva

fva_model.repeated_solve_max_iter = 200
fva_model.repeated_solve_max_attempts = 10

key_targets = ['bs_tx_SUCROSE', 'ms_tx_CARBON_DIOXIDE',
               'total_production_all_biomass',
               'ms_PEPCARBOX_RXN',
               'bs_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast',
               'ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast',
               'bs_MALIC_NADP_RXN_chloroplast',
               'bs_PEPCARBOXYKIN_RXN',
               'plasmodesmata_ms_GAP_bs_GAP',
               'plasmodesmata_ms_G3P_bs_G3P',
               'ms_EC_1_2_1_13_chloroplast',
               'bs_CO2', 'bs_oxygen', 
               'bs_PhotosystemII_mod_chloroplast',
               'ms_PhotosystemII_mod_chloroplast',
               'bs_tx_GLY', 'bs_tx_GLUTATHIONE',
               'bs_PEPCARBOX_RXN']
key_targets_full = ['image%d_%s' % (i,v) for v in key_targets for i in xrange(15)]
variable_set = set(fva_model.variables)
key_targets_full = [t for t in key_targets_full if t in variable_set]

guess = fva_model.x.copy()
obj_bounds = (None, fva_model.obj_value * (1.0 + relative_delta))
fva_model.ipopt_options['print_level'] = 0
fva_model.ipopt_options['tol'] = 1e-4
key_extrema = fva.objective_constrained_fva(
    fva_model,
    objective_bounds=obj_bounds,
    variables=key_targets_full,
    n_procs=n_procs,
    guess=guess,
    log_filename=filename_prefix,
    log_interval=log_interval,
    check_failures=False
)

## Some failures may occur. We expect that image_4_bs_tx_GLY is the
# only one, and we can correct it by (first generating and then) using
# a different starting point; but we make this a general procedure.
failures = [k for k,result in key_extrema.iteritems() if 
            result == 'failure']
if failures:
    # Retrieve the full vector of variable values from one FVA optimization
    # that is expected to work.
    _ = fva.objective_constrained_fva(fva_model, objective_bounds=obj_bounds,
                                      variables=('image3_bs_tx_GLY',),
                                      n_procs=1,guess=guess)
    new_guess = fva_model.x
    retry_extrema = fva.objective_constrained_fva(
        fva_model, 
        objective_bounds=obj_bounds,
        variables=failures,
        n_procs=n_procs,
        guess=new_guess,
        check_failures=False # for debugging, return even if more errors
    )
    key_extrema.update(retry_extrema)

collate = {k: (np.array([key_extrema['image%d_%s' % (i,k)][0] for
                         i in xrange(15)]),
               np.array([key_extrema['image%d_%s' % (i,k)][1] for
                         i in xrange(15)])
           ) for k in key_targets}

with open('key_fva_broad.pickle', 'w') as f:
    pickle.dump(collate, f)

# Output CSV for inclusion as paper supplementary file
with open('key_fva.csv','w') as f:
    for variable, (lb, ub) in collate.iteritems():
        for tag, values in (('_lower_bound', lb),
                            ('_upper_bound', ub),):
            line = (variable + tag + '\t' + 
                    '\t'.join(['%.18e' % v for v in values]) 
                    + '\n')
            f.write(line)
