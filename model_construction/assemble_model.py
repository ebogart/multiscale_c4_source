""" Merge tables of reactions, biomass components, and exchange species. 

"""

from table_utilities import (read_reaction_table,
                             read_component_table,
                             write_reaction_table)
from fluxtools import stoichiometry_matrix as st

def make_biomass_sinks(species):
    """ Make ids, stoichiometries for conversions of species to biomass.

    For each species s in the argument, the returned dictionary 
    will contain an entry of the form:
        {'sink_s': {'s': -1., 's_biomass': 1.}}
    
    """
    return {'sink_%s' % s: {s: -1., '%s_biomass' %s: 1.} for s in species}

def make_external_transporters(species):
    """ Make ids, stoichiometries for import/export reactions of species.

    For each one-element list [s] in the argument, the returned dictionary 
    will contain an entry of the form:
        {'tx_s': {'s': 1., 's_external': -1.}}
    
    For each two element list [species_s, compartment_c] in the argument,
    the returned dictionary will contain:
        {'tx_species_s': {'species_s': 1., 'species_s_compartment_c': -1.}}
    or, if there is already a tx_species_s in the return dictionary,
        {'tx_species_s_compartment_c':
            {'species_s': 1., 'species_s_compartment_c': -1.}}
    
    """ 
    result = {}
    to_process = sorted(species, key=lambda l: len(l))
    for l in to_process:
        s = l[0]
        if len(l) == 1:
            compartment = 'external'
        else:
            compartment = l[1]
        reaction_name = 'tx_%s' % s
        if reaction_name in result:
            reaction_name = 'tx_%s_%s' % (s, compartment)
        stoichiometry = {s: 1., '%s_%s' % (s,compartment): -1.}
        result[reaction_name] = stoichiometry
            
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Assemble reactions/components from tables into an FBA model, writing a new table', 
        epilog='When a reaction appears in more than one input table, ' 
               'the data in the table appearing latest in the argument list '
               'will override earlier data for that reaction. ' 
               'Exchange reactions are reversible by default.')
    parser.add_argument('output_file', help='name of output table to write')
    parser.add_argument('tables',metavar='reaction_table_file',
                        nargs='+', help='a table of reaction data to read')
    parser.add_argument('--biomass',metavar='biomass_component_file',
                        help='a file with a list of species for which '
                             'biomass sink reactions should be added')
    parser.add_argument('--exchange',metavar='exchange_species_file',
                        help='a file with a list of species for which '
                             'external transport reactions should be added, '
                             'optionally with the compartment with which '
                             'the species should be exchanged ' 
                             "(default is 'external')")
    args = parser.parse_args()
    combined_parents = {}
    combined_reversibilities = {}
    combined_stoichiometries = {}
    for table_file in args.tables:
        stoichiometries, parents, reversibilities = read_reaction_table(table_file)
        combined_parents.update(parents)
        combined_reversibilities.update(reversibilities)
        combined_stoichiometries.update(stoichiometries)
    if args.biomass:
        biomass_precursors = read_component_table(args.biomass)
        combined_stoichiometries.update(make_biomass_sinks(biomass_precursors))
    if args.exchange:
        transported_species = read_component_table(args.exchange)
        transported_species = [t.split('\t') for t in transported_species]
        transporters = make_external_transporters(transported_species)
        combined_stoichiometries.update(transporters)
        combined_reversibilities.update(dict.fromkeys(transporters, True))

    write_reaction_table(args.output_file, combined_stoichiometries,
                        combined_parents, combined_reversibilities)
    
# def model_generator(extras={}, extra_bounds={}, free_exchange=fba_free_exchange):
#     test_sdict = fba_frame_stoichiometries.copy()
#     test_sdict.update(extras)
#     transporters = {'transport_%s' % s: 
#                     {s: 1.0} for s in free_exchange}
#     test_sdict.update(transporters)
#     bounds = dict.fromkeys(test_sdict, (0., b))
#     for r in fba_reversible_reactions:
#         if r in test_sdict:
#             bounds[r] = (-1.0*b, b)
#     for r in transporters:
#         bounds[r] = (-1.0*b, b)
#     for r in extras:
#         bounds[r] = extra_bounds.get(r,(-0.0*b, b))
#     objective = dict.fromkeys(test_sdict, (1., 1.))
#     objective.update(dict.fromkeys(special_stoichiometries, 0.))
#     m = st.NonNegativeConstraintModel(test_sdict)
#     m.set_all_flux_bounds(bounds)
#     m.set_objective_function(objective)
#     return m, bounds, objective
