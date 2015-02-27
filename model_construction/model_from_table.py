""" 
Generate FBA models from reaction tables.

"""

import re
import numpy as np
import fluxtools.stoichiometry_matrix as st
from table_utilities import read_reaction_table
        
def fba_model_generator(
        (stoichiometries, _, reversibilities),
        default_bound = 1000.,
        extras={}, 
        extra_bounds={},
        non_conserved=set(),
        do_conserve=[],
        free_compartments=('biomass',
                           'external',
                           'fixed_biomass',
                           'intercellular_air_space',
                           'xylem'),
        cost_keys=None):
    all_stoichiometries = stoichiometries.copy()
    all_stoichiometries.update(extras)
    bounds = dict.fromkeys(all_stoichiometries,
                           (0., default_bound))
    bounds.update(
        dict.fromkeys(
            (r for r in stoichiometries if
             reversibilities.get(r,False)),
            (-1.0*default_bound, default_bound)))
    bounds.update(extra_bounds)
    if cost_keys is None:
        cost_keys = all_stoichiometries.keys()
    objective = dict.fromkeys(
        cost_keys,
        (1., 1.))
    free_compartment_pattern = re.compile('_(' + 
                                          '|'.join(free_compartments) + 
                                          ')$')
    all_species = set()
    for stoichiometry in all_stoichiometries.values():
        all_species.update(stoichiometry)
    non_conserved = non_conserved.copy()
    non_conserved.update({s for s in all_species if
                          free_compartment_pattern.search(s)})
    for s in do_conserve:
        if s in non_conserved:
            non_conserved.remove(s)
    m = st.NonNegativeConstraintModel(all_stoichiometries)
    m.set_all_flux_bounds(bounds)
    m.set_objective_function(objective)
    m.do_not_conserve(*non_conserved)
    return m, bounds, objective, non_conserved

def fba_test_source(model, s, **kwargs):
    return fba_test_reaction(model, 'source', {s: 1.0}, **kwargs)
def fba_test_sink(model, s, **kwargs):
    return fba_test_reaction(model, 'sink', {s: -1.0}, **kwargs)
def fba_test_reaction(model, r_id, reaction=None, value=1.0, **kwargs):
    """ If reaction is None, assume the reaction is already
    in the model. """
    extras = kwargs.pop('extras',{})
    if reaction is not None:
        extras.update({r_id: reaction})
    extra_bounds = kwargs.pop('extra_bounds',{}).copy()
    extra_bounds.update({r_id: value})
#    print extras
#    print extra_bounds
#    print kwargs
    m, bounds, objective, n = fba_model_generator(
        model, 
        extras=extras,        
        extra_bounds=extra_bounds,
        **kwargs)
    result = m.solve()
#    print m.lp.status
#    print m.lp.obj.value
    return (m.lp.status=='opt'), {k:v for k,v in result.iteritems() if np.abs(v) > 1e-6}
