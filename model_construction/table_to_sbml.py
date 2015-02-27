""" 
Generate SBML models from reaction tables.

Some code from SloppyCell.IO.

"""

import re
import libsbml
from fluxtools.sbml_interface import rxn_add_stoich

sid_pattern = re.compile(r'^[A-Za-z_]\w*$')
not_alphanumeric = re.compile(r'[^\w]')

def clean_id(string):
    """ Convert string to a valid SBML SId. 

    """
    if sid_pattern.match(string):
        return 0, string
    else:
        clean, n_replacements = not_alphanumeric.subn('_',string)
        if clean[0].isdigit():
            clean = '_' + clean
            n_replacements += 1
        return n_replacements, clean

def write_sbml_note_table(dictionary):
    """Convert key-value dictionary to COBRA-style SBML notes.

    Returns an HTML string suitable for use as the optional
    'notes' element on an SBML component. This may be verified with
    libsbml.SyntaxChecker_hasExpectedXHTMLSyntax(). 

    Per the supplementary information to doi:10.1038/nprot.2011.308

    """
    
    records = []
    for key_value_pair in dictionary.iteritems():
        records.append('\n    <p>%s: %s</p>' % key_value_pair)

    template = "<body xmlns='http://www.w3.org/1999/xhtml'>%s\n</body>"
    return template % ''.join(records)

def table_to_SBML(
        (stoichiometries, reaction_parents, reversibilities),
        network_id, 
        network_name,
        name_hints = {},
        default_compartment = 'cytoplasm',
        compartments=(),
        sbml_version = 4,
        sbml_level = 2
    ):
    """Translate table of reactions to SBML file. 

    This is currently somewhat brittle. Names of reactions and species
    are munged to conform with the SBML sID type by replacing all
    unacceptable characters with underscores and prepending an
    underscore if the name starts with a number; if this process would
    lead to a clash, an exception is thrown.

    All compartment strings must be valid sIDs themselves. (Omit leading
    underscores, which will be inferred.)

    The name_hints argument specifies values to be used for
    the SBML name attribute of the components (given without
    compartment tags, ie, {'|Pi|': 'phosphate'}, not
    {'|Pi|_mitochondrion': 'phosphate_mitochondrion'}. Where 
    these names translate more cleanly to sIDs than the original
    component IDs, they will be used for that purpose as well.

    Returns an SBML document, as a string.

    """
    compartment_pattern = re.compile('^(.*?)_(' + 
                                     '|'.join(compartments) + 
                                     ')$')
    class_pattern = re.compile('^\|(.*)\|$')
    # Track the SBML component IDs we generate to ensure there are no
    # duplicates.
    unique_ids = set()
    # Determine the best valid sID for a component of the model,
    # using either its current ID or the relevant entry in 
    # name_hints, if available, depending on which one
    # requires the most characters to be replaced by 
    # underscores to become sID-compliant.
    def best_id(old_id):
        if compartment_pattern.match(old_id):
            root, compartment = compartment_pattern.match(old_id).groups()
        else:
            root = old_id
            compartment = None
        id_bases = [root]
        hint = name_hints.get(root, None)
        if hint:
            id_bases.append(hint)
        if class_pattern.match(root):
            id_bases.append(class_pattern.match(root).groups()[0])
        possible_ids = [clean_id(base) for base in id_bases]
        if compartment:
            possible_ids = [(n, id_ + '_' + compartment) for n, id_ in
                            possible_ids]
        possible_ids.sort(key=lambda t: t[0])
        for n, id_ in possible_ids:
            if id_ in unique_ids:
                continue
            else:
                unique_ids.add(id_)
                return id_
        raise ValueError('No good ID for %s (tried %s)' % (old_id,
                                                           ''.join(map(str,
                                                                       possible_ids))))
    def get_name(component):  
        if compartment_pattern.match(component):
            root, compartment = compartment_pattern.match(component).groups()
        else:
            root = component
            compartment = None
        hint = (name_hints.get(root, None) or
                name_hints.get(reaction_parents.get(component, None), 
                               None))
        if hint and (hint.upper() != root.upper()):
            name = '%s (%s)' % (hint, root)
        else:
            name = root
        if compartment:
            name += ' [%s]' % compartment
        return name

    species_to_id = {}
    new_stoichiometries = {}
    new_reversibilities = {}
    compartments = {}
    component_names = {} 
    # Data to be written to each component's notes element
    data = {} 
    
    for reaction, stoichiometry in stoichiometries.iteritems():
        new_stoichiometry = {}
        for species, coefficient in stoichiometry.iteritems():
            if species not in species_to_id:
                new_species = best_id(species)
                species_to_id[species] = new_species
            new_stoichiometry[species_to_id[species]] = coefficient
        new_reaction = best_id(reaction)
        new_reversibilities[new_reaction] = reversibilities[reaction]
        new_stoichiometries[new_reaction] = new_stoichiometry
        component_names[new_reaction] = get_name(reaction)
        data[new_reaction] = {'PARENT_FRAME': reaction_parents.get(reaction, None)}

    for old_species, new_species in species_to_id.iteritems():
        compartment_match = compartment_pattern.match(old_species)
        compartments[new_species] = (compartment_match.groups()[1] if
                                     compartment_match else
                                     default_compartment)
        component_names[new_species] = get_name(old_species)

    m = libsbml.Model(sbml_level, sbml_version)
    m.setId(network_id)
    m.setName(network_name)
    
    for id_ in set(compartments.values()):
        sc = libsbml.Compartment(sbml_level, sbml_version)
        sc.setId(id_)
        # Don't look for these in component_names as we have 
        # not populated that for compartments, but do check
        # to see if there are expected expansions (eg '_m' ->
        # '_mitochondrion') 
        sc.setName(name_hints.get(id_,id_))
        sc.setConstant(True)
        sc.setSize(1.0)
        m.addCompartment(sc)
    
    for id_, compartment in compartments.iteritems():
        ss = libsbml.Species(sbml_level, sbml_version)
        ss.setId(id_)
        ss.setName(component_names[id_])
        ss.setCompartment(compartment)
        ss.setBoundaryCondition(False)
        if id_ in data:
            ss.setNotes(write_sbml_note_table(data[id_]))
        m.addSpecies(ss)
    
    for id_, stoichiometry in new_stoichiometries.iteritems():
        srxn = libsbml.Reaction(sbml_level, sbml_version)
        srxn.setId(id_)
        srxn.setName(component_names[id_])
        for rid, stoich in stoichiometry.iteritems():
            rxn_add_stoich(srxn, rid, stoich)
        kl = libsbml.KineticLaw(sbml_level, sbml_version)
        kl.setFormula('0')
        srxn.setKineticLaw(kl)
        # Reversibility attribute is optional in SBML, but
        # we should have it for all reactions.
        srxn.setReversible(new_reversibilities[id_])
        if id_ in data:
            srxn.setNotes(write_sbml_note_table(data[id_]))
        m.addReaction(srxn)
    
    d = libsbml.SBMLDocument(sbml_level, sbml_version)
    d.setModel(m)
    sbmlStr = libsbml.writeSBMLToString(d)
    return sbmlStr

if __name__ == '__main__':
    import argparse
    from table_utilities import read_reaction_table
    parser = argparse.ArgumentParser(description='Convert table to SBML')
    parser.add_argument('input_file', help='name of reaction table to load')
    parser.add_argument('model_id', help='id of model in SBML file')
    parser.add_argument('--model_name', help='name of model in SBML file')
    parser.add_argument('--name_hints', help='table of component name hints')
    parser.add_argument('--compartments',nargs='*',
                        help='compartment tags to recognize in the input table')
    args = parser.parse_args()
    name_hints = []
    if args.name_hints:
        with open(args.name_hints) as f:
            for line in f:
                name_hints.append(line.strip().split('\t',1))
    name_hints = dict(name_hints)
    model = read_reaction_table(args.input_file)
    print table_to_SBML(
        model,
        network_id=args.model_id, 
        network_name=args.model_name or args.model_id,
        name_hints = name_hints,
        default_compartment = 'cytoplasm',
        compartments=args.compartments or [],
    )
    
