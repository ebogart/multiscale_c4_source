""" Generate FBA optimal solutions for comparison to the RNA fitting data. 

We seek the minimum-flux solution which achieves CO2 assimilation 
equal to that predicted at the tip in the gradient RNAseq data fit.

"""

import reduced_model as rm
import pickle
from fluxtools.utilities.total_flux import minimum_flux_solve, add_total_flux_objective

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
# Set up the basic model, as in gradient_fit.py,
# but tweaked to allow only source-tissue-type
# solutions (net export of C,N,S)

source_model = rm.model(default_flux_bound=1000.,
                       extra_free_compartments=('phloem',))
oxidases = ['bs_GenericNADOxidase_mod',
            'bs_GenericNADPOxidase_mod',
            'ms_GenericNADOxidase_mod',
            'ms_GenericNADPOxidase_mod']
oxidase_bounds = dict.fromkeys(oxidases,0.)
source_model.set_bounds(oxidase_bounds)

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
phloem_bounds.update({'bs_tx_SUCROSE': (-2.5, 0.),
                      'bs_tx_GLY': (-2.5, 0.),
                      'bs_tx_GLUTATHIONE': (-2.5, 0.)})
source_model.set_bounds(phloem_bounds)

# Look up what A we predict at the tip and constrain the 
# source model to match it.
with open('gradient_fit.pickle') as f:
    result = pickle.load(f)
    traj = result['traj']
    predicted_tip_A = traj['ms_tx_CARBON_DIOXIDE'][-1]
source_model.set_bound('ms_tx_CARBON_DIOXIDE', predicted_tip_A)

base_variables = source_model.variables
add_total_flux_objective(source_model)
x = minimum_flux_solve(source_model)

with open('source_fba_comparison.pickle','w') as f:
    pickle.dump({'soln': source_model.soln,
                 'variables': base_variables},
                f)
