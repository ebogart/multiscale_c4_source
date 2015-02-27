""" Tools for working with metabolic network models as text tables. """

from ast import literal_eval

def read_reaction_table(filename):
    """Load a table of reaction ids and stoichiometries.

    Arguments:
    filename - name of tab-delimited text file to load. Each row should
        contain the following fields in order:
            - an ID string for the reaction,
            - the frame in the parent database with which the reaction
              is associated, or None,
            - a stoichiometry, represented as a string which will be passed
              to eval() to obtain a dict; 
            - the reversibility of the reaction (True or False)
        Blank lines and lines beginning with '#' will be ignored.

    'None', 'True', and 'False' are case-insensitive.

    Returns: 
        stoichiometries: a dictionary of stoichiometry dictionaries,
        parents: dictionary mapping reaction IDs to their specified
            parent frames in the DB, where those were not None;
        reversibilities: dictionary of booleans giving the reversibility
            of each reaction

    """
    stoichiometries = {}
    parents = {}
    reversibilities = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = [f.strip() for f in line.strip().split('\t')]
            r = fields[0]
            if fields[1].upper() == 'NONE':
                parents[r] = None
            else:
                parents[r] = fields[1]
            stoichiometries[r] = literal_eval(fields[2])
            reversibilities[r] = (fields[3].upper() == 'TRUE')
            
    return (stoichiometries, parents, reversibilities)

def write_reaction_table(filename, stoichiometries,
                        parents={}, reversibilities={}):
    """Write a table of reaction ids and stoichiometries.

    Writes a tab-delimited text file with one row per reaction,
    each containing the following fields in order:

    - an ID string for the reaction,
    - the frame in the parent database with which the reaction
        is associated, or None,
    - a stoichiometry, represented as a string which will be passed
        to eval() to obtain a dict; 
    - the reversibility of the reaction (True or False)

    Arguments:
    filename: file to write
    stoichiometries: a dictionary of stoichiometry dictionaries
    parents: dictionary mapping some reaction IDs to their 
        parent frames in the DB (will use None where unspecified)
    reversibilities: dictionary of booleans giving the reversibility
        of each reaction (defaults to False where unspecified)

    Returns: 
    None. 
    
    """

    with open(filename,'w') as f:
        for tag in sorted(stoichiometries.keys()):
            line = '\t'.join(map(str,[tag, parents.get(tag,None),
                                      stoichiometries[tag], 
                                      reversibilities.get(tag,False)]))
            f.write(line)
            f.write('\n')

def read_component_table(filename):
    """ Load a table of items ignoring lines beginning with '#' or blank. 

    Returns a list. 

    """
    items = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            items.append(line.strip())
    return items

