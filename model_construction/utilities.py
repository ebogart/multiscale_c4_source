""" Functions useful interactively that don't belong anywhere else yet. """

import pycyc
import numpy as np
from fluxtools import stoichiometry_matrix as st

# Testing pathways for missing reactions is complicated because
# entries in each pathway's reaction-list slot may be other pathways!
corn = pycyc.open('corn')
db = corn

def get_coefficient(reaction, compound, slot):
    """ Get the coefficient of a species in a rxn's left/right slot.

    If the species' COEFFICIENT annotation is NIL/None, return 1.

    The compound is not checked to ensure it is actually a value of the 
    slot.
    
    """
    coefficient = db.get_value_annot(reaction, slot, compound,
                                        'COEFFICIENT')
    # In some cases the coefficient is a non-frame LISP symbol, eg |n|, which
    # is parsed as a frame; try to catch those and return a string instead
    # (anticipating an exception later when this is cast to a float)
    if coefficient is None: 
        return 1
    elif isinstance(coefficient, int) or isinstance(coefficient, float):
        return coefficient
    else: 
        return str(coefficient)

def get_left_right_stoichiometries(r1):
    """ Get the stoichiometries of the left and right sides of a reaction. 

    Returns sorted lists of (compound_string, coefficient) tuples for
    left and right sides respectively.

    """
    r1_left = [(str(c), get_coefficient(r1,c,'left')) for c in 
               db[r1].slot_values('left')]
    r1_left.sort()
    r1_right = [(str(c), get_coefficient(r1,c,'right')) for c in 
                db[r1].slot_values('right')]
    r1_right.sort()
    
    return tuple(r1_left), tuple(r1_right)

def display(r1,model=None):
    """ Show stoichiometry, compartment info about a reaction. """
    r1 = str(r1)
    left, right = get_left_right_stoichiometries(r1)
    compartments = {}
    for slot, species_list in (('LEFT', left), ('RIGHT', right)):
        for s, coefficient in species_list:
            compartment = corn.get_value_annot(r1, slot, s, 'COMPARTMENT')
            if compartment:
                compartments.setdefault(str(compartment),[]).append(str(s))
    stoichiometry = {k: v for k,v in right}
    stoichiometry.update({k: -1.0*v for k,v in left})
    print '%s (%s):' % (r1, corn.get_name_string(r1,strip_html=True,
                                                 rxn_eqn_as_name=False))
    print corn.get_name_string(r1, strip_html=True)
    print 'Stoichiometry:'
    print '\t', stoichiometry
    print 'Direction:'
    print '\t', corn[r1].reaction_direction
    if compartments:
        print 'Compartments:'
        print '\t', compartments
    if model:
        parents = model[1]
        if r1 in parents.values():
            print 'Present as:'
            for k,v in parents.iteritems():
                if v == r1:
                    print '\t',k
                    print '\t\t',model[0][k]
    print '\n'


def all_reactions(pathway_id):
    """ Handle reactions with subpathways by recursively expanding those subpathways. """
    pathway = corn[pathway_id]
    pathway_reactions = map(str, pathway.slot_values('REACTION-LIST'))
    results = set()
    subpathways = []
    for c in pathway_reactions:
        if c in pathway_set:
            results.update(all_reactions(c))
        else:
            results.add(c)
    return results

# Broken... 
def fba_test_pathway(pathway_id, make_reversible=False):
    pathway = corn[pathway_id]
    pathway_reactions = all_reactions(pathway_id)
    t = corn.substrates_of_pathway(pathway)
    reactants, proper_reactants, products, proper_products = t
    if not proper_products:
        if not products:
            return (None, None, None, False)
        else:
            proper_products = products
    if not proper_reactants: 
        return (None, None, None, False)
    local_superreactions = {r for c in pathway_reactions for r in superreactions.get(c, [])}
    core = {r:s for r,s in fba_frame_stoichiometries.iteritems() if fba_parents[r]
             in pathway_reactions or fba_parents[r] in local_superreactions}
    
    sources = {'source_%s' % s: {s: 1.} for s in map(str, proper_reactants)}
    sinks = {'sink_%s' % s: {s: -1.} for s in map(str, proper_products)}
    local_cofactors = [s for s in cofactors if s not in proper_reactants and
                       s not in proper_products]
    cofactor_exchanges = {'exchange_%s' % s: {s: 1} for s in local_cofactors}
    sdict = core.copy()
    sdict.update(sinks)
    sdict.update(sources)
    sdict.update(cofactor_exchanges)
    bounds = dict.fromkeys(core, ((-1.0*b if make_reversible else 0.), b))
    for r in core:
        if r in fba_reversible_reactions:
            bounds[r] = (-1.0*b, b)
    bounds.update(dict.fromkeys(sources, (0., None)))
    bounds.update(dict.fromkeys(sinks, (1., None)))
    bounds.update(dict.fromkeys(cofactor_exchanges, (-1.0*b, b)))

    objective = dict.fromkeys(sdict, (1., 1.))
    m = st.NonNegativeConstraintModel(sdict)
    m.set_all_flux_bounds(bounds)
    m.set_objective_function(objective)
    result = m.solve()
    return (sdict, bounds, [r for r in core if np.abs(result[r])<1e-6],
            m.lp.status == 'opt')

