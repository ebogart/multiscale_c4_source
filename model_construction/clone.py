"""
Local development branch of fluxtools.c4clone

Utilities for turning a one-cell SloppyCell network into a two-cell network.

"""
from fluxtools.reaction_networks import *
from fluxtools.gene_utilities import set_gene_association_from_list, genes_of_reaction

def strip_empty_compartments(network):
    """ Remove compartments from network if they contain no species.

    """ 
    occupied_compartment_ids = set([])
    for s in network.species:
        occupied_compartment_ids.add(s.compartment)
    
    print occupied_compartment_ids

    for compartment_id in network.compartments.keys():
        if compartment_id not in occupied_compartment_ids:
            network.compartments.removeByKey(compartment_id)

def clone(source, join_compartment_ids, new_network_id, new_name, a_label='ms',
          b_label='bs', reversibility_list = [], split_genes=False):
    """Make a two-cell network from a one-cell network.

    Returns a new network and a dictionary mapping ids in the old
    network to ids or pairs of ids in the new network as appropriate.

    If split_genes is True, genes associated with each split reaction will
    be processed in the same way as other ids. Genes associated with reactions
    not split across compartments will be unmodified.

    Currently notes fields of network components other than reactions
    or species are ignored; except as above, they are transferred
    unmodified to split versions of components.

    If reversibility_list is given, a list of the reactions in the new model
    corresponding to the entries of reversibility_list is returned as well.

    Note that the duplication of network components is not perfect:
    only compartments, species and reactions are considered, and attributes
    of those objects beyond what is relevant to FBA are currently ignored
    (initial values, kinetic laws, eg.) Note further that the component
    ids in the cloned network are not guaranteed to be valid SBML ids, though
    if a_label and b_label are valid beginnings of SBML ids and the old network
    ids are valid, they should be.

    """
    n = Network(new_network_id, new_name)
    
    a_tag = lambda s: a_label + '_' + s
    b_tag = lambda s: b_label + '_' + s
    
    replacement_table = {}

    # We will remove old reactions from the list of reversible reactions
    # one at a time, so we want to ensure each reaction appears at most once
    reversibility_list = list(set(reversibility_list))

    for c in source.compartments:
        if c.id in join_compartment_ids:
            n.add_compartment(c.id,name = c.name)
            replacement_table[c.id] = c.id
        else:
            n.add_compartment(a_tag(c.id),name = a_tag(c.name))
            n.add_compartment(b_tag(c.id),name = b_tag(c.name))
            replacement_table[c.id] = (a_tag(c.id), b_tag(c.id))

    for s in source.species:
        if s.compartment in join_compartment_ids:
            n.add_species(s.id,s.compartment,name = s.name)
            n.species.get(s.id).notes = s.notes.copy()
            replacement_table[s.id] = s.id
        else:
            n.add_species(a_tag(s.id),a_tag(s.compartment),
                                      name = a_tag(s.name))
            n.species.get(a_tag(s.id)).notes = s.notes.copy()
            n.add_species(b_tag(s.id),b_tag(s.compartment),
                                      name = b_tag(s.name)) 
            n.species.get(b_tag(s.id)).notes = s.notes.copy()
            replacement_table[s.id] = (a_tag(s.id), b_tag(s.id)) 
            
    for r in source.reactions:
        species = [source.species.getByKey(s) for s in r.stoichiometry.keys()]
        if all([s.compartment in join_compartment_ids for s in species]):
            n.addReaction(Reactions.Reaction, r.id, r.stoichiometry, 
                                             name=r.name, kineticLaw = None)
            replacement_table[r.id] = r.id
            n.reactions.get(r.id).notes = r.notes.copy()
        else:
            if r.id in reversibility_list:
                reversibility_list.remove(r.id)
                reversibility_list += [a_tag(r.id), b_tag(r.id)]
            a_stoichiometry = {}
            b_stoichiometry = {}
            for s in species:
                if s.compartment in join_compartment_ids:
                    a_stoichiometry[s.id] = r.stoichiometry[s.id]
                    b_stoichiometry[s.id] = r.stoichiometry[s.id]
                else:
                    a_stoichiometry[a_tag(s.id)] = r.stoichiometry[s.id]
                    b_stoichiometry[b_tag(s.id)] = r.stoichiometry[s.id]

            n.addReaction(Reactions.Reaction,a_tag(r.id), a_stoichiometry, 
                                             name=a_tag(r.name),
                                             kineticLaw = None)
            n.addReaction(Reactions.Reaction,b_tag(r.id), b_stoichiometry, 
                                             name=b_tag(r.name),
                                             kineticLaw = None)

            for tag in (a_tag, b_tag):
                n.reactions.get(tag(r.id)).notes = r.notes.copy()
                if hasattr(r, 'reversible'):
                    n.reactions.get(tag(r.id)).reversible = r.reversible

            if split_genes:
                for tag in (a_tag, b_tag):
                    new_reaction = n.reactions.get(tag(r.id))
                    old_genes = genes_of_reaction(new_reaction)
                    set_gene_association_from_list(new_reaction,
                                                   map(tag, old_genes))

            replacement_table[r.id] = (a_tag(r.id), b_tag(r.id))

    if reversibility_list:
        return n, replacement_table, reversibility_list
    else: 
        return n, replacement_table
    
def give_exchange_reaction(network, species_pair, prefix='plasmodesmata_',
                           name = None):
    """ Add a reaction to the network that interconverts two species. """
    r_id = prefix + species_pair[0] + '_' + species_pair[1]
    if name is None:
        name = r_id
    network.addReaction(Reactions.Reaction, r_id,
                        dict(zip(species_pair, (1.0, -1.0))),
                        name = name,
                        kineticLaw = None)
    return r_id

if __name__ == '__main__':
    import argparse
    from fluxtools import sbml_interface as si
    parser = argparse.ArgumentParser(
        description='Make a two-comparmtent model from a one-cell model'
    )
    parser.add_argument('input_file', help='name of SBML file to clone')
    parser.add_argument('--join_compartments',nargs='*',
                        help='compartments which are shared by both '
                        'cell types (eg external or biomass)')
    parser.add_argument('--ms_only_compartments',nargs='*',
                        help='compartments which may interact only '
                        'with the mesophyll (eg intercellular air space)')
    parser.add_argument('--bs_only_compartments',nargs='*',
                        help='compartments which may interact only '
                        'with the bundle sheath (eg vasculature)')
    parser.add_argument('--split_genes',action='store_const',
                        const=True, default=False, help='split gene '
                        'assocations of child reactions across '
                        'compartments (default is to leave them '
                        'unmodified)')
    parser.add_argument('--transport_whitelist', 
                        help='file with list of species IDs in the original model '
                        'for which inter-cell-type exchange reactions should be '
                        'added (default is to use all species in the "cytoplasm" '
                        'compartment.)')
    args = parser.parse_args()
    join_compartments = args.join_compartments or []
    network = si.fromSBMLFile(args.input_file)
    new_network, _ = clone(network, join_compartments, 'two_cell_' + network.id,
                           network.name + ' (two cell version)', 
                           split_genes=args.split_genes)
    # Add plasmodesmata for all species in the cytoplasm 
    # (note typically in the past I have restricted this further, at least
    # to prevent bulk ATP transfer etc)
    if args.transport_whitelist is None:
        transported_ids = [s.id for s in network.species if s.compartment == 
                           'cytoplasm']
    else: 
        with open(args.transport_whitelist) as f:
            transported_ids = f.read().split()
    for species in transported_ids:
            give_exchange_reaction(new_network, ('ms_' + species, 'bs_' + species))

    # Apply the ms_only_compartments and bs_only_compartments
    reaction_to_compartments = {}
    reactions_removed_for_exclusivity = []
    for r in new_network.reactions:
        compartments = set()
        for s in r.stoichiometry:
            compartments.add(new_network.species.get(s).compartment)
        reaction_to_compartments[r.id] = compartments
    for target_list, bad_tag in [[args.ms_only_compartments, 'bs'],
                                 [args.bs_only_compartments, 'ms']]:
        for target_compartment in target_list:
            bad = []
            for reaction, compartments in reaction_to_compartments.iteritems():
                if target_compartment in compartments:
                    if [c for c in compartments if c.startswith(bad_tag)]:
                        bad.append(reaction)
            for bad_reaction in bad:
                new_network.reactions.remove_by_key(bad_reaction)
                reaction_to_compartments.pop(bad_reaction)
                reactions_removed_for_exclusivity.append(bad_reaction)

    print si.toSBMLString(new_network)

