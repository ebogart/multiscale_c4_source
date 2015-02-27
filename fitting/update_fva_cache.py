"""Find minimum/maximum fluxes for all reactions in the reduced model.

The result is important for determining which reactions are, in practice,
reversible. 

Nonlinear constraints are applied but other bounds are generally
relaxed. Sucrose import is allowed (in fact, all exchanges with the
phloem are allowed.) Both the individual biomass sinks and the
combined biomass reactions are allowed to operate.  The resulting
upper and lower bounds will thus be less tight than they will be in
practice in fitting calculations, where only one set of biomass sinks
should be used.

"""
import pickle
import numpy as np
import reduced_model as rm
from fluxtools.fva import do_fva
from fluxtools.nlcm import OptimizationFailure

# Load the model
base_model = rm.model(extra_free_compartments=('phloem',))
base_model.ipopt_options.update({'tol': 1e-5,
                                 'linear_solver': 'ma97',
                                 'ma97_print_level': -1,
                                 'print_level': 0})
base_model.set_bounds({'bs_CombinedBiomassReaction': (0.,None),
                       'ms_CombinedBiomassReaction': (0.,None)})

# Do FVA for all reactions, whether they are reversible or not
# Success of these calculations depends on the choice of starting point,
# and the viable choices of starting point differ for different variables;
# we rerun the variables that fail with one guess with another,
# checking to ensure at least one calculation completed successfully for 
# everything.
fva_keys = base_model.reactions.keys()
result_large = do_fva(base_model, variables=fva_keys, n_procs=50,
                      guess = 1e-1*np.ones(base_model.nvar),
                      check_failures=False)
failed = [r for r,result in result_large.iteritems() if result == 'failure']
if failed:
    backup_guesses = [base_model.solve(), 
                      1e-5*np.ones(base_model.nvar),
                      1e-3*np.ones(base_model.nvar),
                      np.zeros(base_model.nvar)]
    for guess in backup_guesses: 
        result_new = do_fva(base_model, 
                            variables=failed,
                            n_procs=min(50,len(failed)),
                            guess = guess,
                            check_failures=False)
        failed = [r for r,result in result_new.iteritems() if 
                  result == 'failure']
        result_large.update(result_new)
        if not failed:
            break
    if failed:
        raise OptimizationFailure('%d reactions failed after retrying' %
                                  len(second_pass_failures))
# If that succeeded, save.
with open('data/corn_twocell_reduced_fva_cache.pickle','w') as f:
    pickle.dump(result_large, f)
