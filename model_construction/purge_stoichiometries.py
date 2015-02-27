""" Remove reactions with a given stoichiometry from a table, writing a new table. """

import argparse
from ast import literal_eval
from table_utilities import read_reaction_table, read_component_table, write_reaction_table

parser = argparse.ArgumentParser(
    description='Remove reactions with a given stoichiometry from a table, writing a new table.'
    )
parser.add_argument('stoichiometry_file', help='file of stoichiometries to exclude')
parser.add_argument('input_file',  help='input reaction table')
parser.add_argument('output_file', help='clean reaction table file to write')
args = parser.parse_args()
stoichiometries, parents, reversibilities = read_reaction_table(args.input_file)
bad_stoichiometries = map(literal_eval,read_component_table(args.stoichiometry_file))
to_remove = []
for reaction, stoichiometry in stoichiometries.iteritems():
    if stoichiometry in bad_stoichiometries:
        to_remove.append(reaction)
        print 'Removing %s:\t%s' % (reaction, stoichiometry)
for reaction in to_remove:
    stoichiometries.pop(reaction)
write_reaction_table(args.output_file, stoichiometries, parents, reversibilities)
