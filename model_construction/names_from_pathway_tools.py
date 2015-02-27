""" Extract reasonable names for the frames in a Pathway Tools db. 

Calls get_name_string for every reaction and species in the database
specified as first argument, and prints the results as a tab-delimited
table.

These can then be used for element names and (in some cases, where they
are closer to complying with the sID format than the frame names themselves)
IDs in an SBML model.

"""

import sys
import pycyc

db_id = sys.argv[1]
db = pycyc.open(db_id)

for frame in (db.reactions+db.compounds):
    print '%s\t%s' % (frame, 
                        db.get_name_string(frame, rxn_eqn_as_name=False,
                                           strip_html=True))
