"""Assign reactions from table to compartments, write new table. 

Given a table of reactions in standard format, and a table of the form

reaction1    compartment1 [compartment2 compartment3 ...]
reaction2 ...

write a new reaction table where reaction1 is present in compartment1,
compartment2, and compartment3, etc.; that is, the new table contains
a reaction 'reaction1_compartment1' with stoichiometry
{'reactant1_compartment1': ...}, and so on.

If a compartment assignment is given for RXN-101, it will be applied
not just to the reaction with name 'RXN-101' but all instantiated
versions, etc., with parent 'RXN-101', unless a separate assignment is
given for such a reaction by its particular name.

Blank lines and comment lines starting with '#' are okay in the
compartment table.

Note that, if a reaction is specified multiple times in the table, it
will be present (in only one copy) in all the compartments listed in
any of its lines. For readability it is probably best to avoid this,
however.

By default, the original, uncompartmentalized version of each reaction
will not be present in the new table. By specifying eg
'--default_compartment cytoplasm' reactions may be listed as

reaction1 cytoplasm mitochondrion

and the 'cytoplasm' indication will be interpreted as an instruction
to leave the original, unmarked copy of the reaction in place.

With the '--respect_compartments' option the script can be told to
leave reactants already given compartment tags alone when moving the
reactant in which they participate; for example, if the reaction
'reaction2: {'A': -1, 'B_thylakoid_lumen': 1}' should become
'reaction2_chloroplast: {'A_chloroplast': -1, 'B_thylakoid_lumen':
1}'.

When a compartment indication is given for a reaction not present in
the input table, an exception is raised.

A blank compartment will also raise an exception (these usually
result from typos.)

"""

import re
from table_utilities import read_reaction_table, read_component_table, write_reaction_table

def compartmentalize(reaction, stoichiometry, compartment, 
                     respect_test=None):
    tag = '_' + compartment
    new_reaction = reaction + tag
    new_stoichiometry = {}
    for species, coefficient in stoichiometry.iteritems():
        if respect_test is None or not respect_test(species):
            new_stoichiometry[species + tag] = coefficient
        else:
            new_stoichiometry[species] = coefficient
    return new_reaction, new_stoichiometry

if __name__ == '__main__':
    import argparse, logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Move reactions from input table to compartments and write new table', 
        epilog='If "reaction_1 RXN-100 {"A": -1, "B": 1} False" is in the input table and ' 
               '"reaction_1 chloroplast" is in the file of compartment information, '
               'the output file will contain "reaction_1_chloroplast RXN-100 ' 
               '{"A_chloroplast": -1, "B_chloroplast": 1} False". Multiple compartments '
               'may be specified per line. All input files should be tab-delimited.')
    parser.add_argument('input_file', help='input table to read')
    parser.add_argument('compartment_file', help='file of compartment indications')
    parser.add_argument('output_file', help='output table to write')
    parser.add_argument('--respect_compartments',nargs='*',
                        help='compartment tags already present in the input table '
                             'which should be preserved where they have been applied '
                             'to substrates of reactions being moved.')
    parser.add_argument('--default_compartment',
                        help='Interpret this compartment as an instruction to leave '
                             'an unmodified version of the reaction in the new file (by '
                             'default, the original version is replaced with the new'
                             'compartmentalized version(s).') 
    args = parser.parse_args()
    
    respect_pattern = re.compile('(' + 
                                 '|'.join(args.respect_compartments) +
                                 ')$')
    respect_test = lambda s: respect_pattern.search(s)

    stoichiometries, parents, reversibilities = read_reaction_table(args.input_file)
    
    compartment_assignments = {}
    all_parents = set(parents.values())
    with open(args.compartment_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if not fields[0]:
                continue
            if (fields[0] not in stoichiometries and
                fields[0] not in all_parents):
                raise ValueError('Compartment specified for '
                                 'invalid reaction %s' % fields[0])
            # We allow for the possibility that a reaction is mentioned
            # more than once in the compartment file.
            compartment_assignments.setdefault(fields[0],set()).update(fields[1:])

    new_stoichiometries = {}
    for reaction, stoichiometry in stoichiometries.iteritems():
        if reaction in compartment_assignments:
            compartments = compartment_assignments[reaction]
        elif parents[reaction] in compartment_assignments:
            compartments = compartment_assignments[parents[reaction]]
        else:
            new_stoichiometries[reaction] = stoichiometry
            continue
        
        for compartment in compartments:
            if compartment.isspace():
                raise ValueError('Blank compartment specified for '
                                 'reaction %s' % reaction)
            if compartment == args.default_compartment:
                new_stoichiometries[reaction] = stoichiometry
                logging.info('Leaving %s in %s' %
                             (reaction, compartment))
                continue
            new_reaction, new_stoichiometry = compartmentalize(
                reaction, stoichiometry, compartment, respect_test
            )
            new_stoichiometries[new_reaction] = new_stoichiometry
            parents[new_reaction] = parents[reaction]
            reversibilities[new_reaction] = reversibilities[reaction]
            logging.info('Moving %s to %s' % (reaction, compartment))
            logging.info('with stoichiometry %s' % new_stoichiometry)

    write_reaction_table(args.output_file,
                         new_stoichiometries,
                         parents,
                         reversibilities)
    
