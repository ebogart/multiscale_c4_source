""" 
Remove structurally blocked reactions/species from an SBML file, writing new file.

"""
from fluxtools.deblock import deblock
from fluxtools import sbml_interface as si
from fluxtools.stoichiometry_matrix import sdict_from_sloppy_net as net_to_dict

def remove_blocked(network, protect_compartments=[],
                   blocked_reaction_filename=None):
    """ Remove blocked reactions/species from network.

    Blocked species will be identified as those in which singleton
    metabolites participate, and removed; a new set of singleton metabolites
    will be generated, and the process repeated, until no blocked reactions
    remain. 

    Species in the compartments listed in protect_compartments will
    never be considered singletons. 

    Changes are made to the network's 'reactions' and 'species' attributes
    in place. Nothing is returned. 

    If blocked_reaction_filename is given, a list (one per line)
    of blocked reaction IDs will be written there.

    Empty compartments may result.

    """ 
    protect_species = [s.id for s in network.species if s.compartment in protect_compartments]
    stoichiometries = net_to_dict(network)
    unblocked_stoichiometries, removed_reactions, _ = deblock(stoichiometries, protect_species=protect_species)
    unblocked_species = {species for stoichiometry in unblocked_stoichiometries.values() for
                         species in stoichiometry}
    reactions_to_remove = [r.id for r in network.reactions if
                           r.id not in unblocked_stoichiometries]
    species_to_remove = [s.id for s in network.species if
                         s.id not in unblocked_species]
    for r in reactions_to_remove:
        network.reactions.removeByKey(r)
    for s in species_to_remove:
        network.species.removeByKey(s)

    if blocked_reaction_filename:
        with open(blocked_reaction_filename, 'w') as f:
            f.write('\n'.join(removed_reactions))
            f.write('\n')
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Remove blocked reactions from SBML model')
    parser.add_argument('input_file', help='name of SBML file to deblock')
    parser.add_argument('--protect_compartments',nargs='*',
                        help='compartments in which species should never be considered blocked')
    parser.add_argument('--blocked_reaction_file', help='optional file to write list of removed reactions to')
    args = parser.parse_args()
    protect_compartments = args.protect_compartments or []
    network = si.fromSBMLFile(args.input_file)
    remove_blocked(network, protect_compartments, args.blocked_reaction_file)
    print si.toSBMLString(network)
