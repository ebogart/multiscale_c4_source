""" Remove reactions from a table, writing a new table. """

import argparse
from table_utilities import read_reaction_table, read_component_table, write_reaction_table

parser = argparse.ArgumentParser(
    description='Remove reactions from a table, writing a new table.'
    )
parser.add_argument('reaction_file', help='file of reaction names (or parents) to exclude')
parser.add_argument('input_file',  help='input reaction table')
parser.add_argument('output_file', help='clean reaction table file to write')
args = parser.parse_args()
stoichiometries, parents, reversibilities = read_reaction_table(args.input_file)
bad_reactions = set(read_component_table(args.reaction_file))
to_remove = []
for reaction, stoichiometry in stoichiometries.iteritems():
    if (reaction in bad_reactions) or (parents[reaction] in bad_reactions):
        to_remove.append(reaction)
        print 'Removing %s:\t%s' % (reaction, stoichiometry)
for reaction in to_remove:
    stoichiometries.pop(reaction)
write_reaction_table(args.output_file, stoichiometries, parents, reversibilities)
