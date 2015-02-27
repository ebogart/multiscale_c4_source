"""Add GRA information from a table to the Notes element of SBML reactions.

First argument: filename of table of tab-separated gene/reaction
    pairs, one per line
Second argument: SBML model filename
Third argument: a field of the Notes element of the SBML reaction
    objects which will be used to map them to the reactions given in
    the table
Fourth argument: a file of special cases. Each line should consist of
    a gene followed by zero or more reactions, tab-delimited. These
    reactions will replace the reactions with which that gene would
    ordinarily be associated based on the previous arguments, if there
    were any. Note that here the reactions _must_ be specified by
    their SBML ID (the mapping by a given field of the Notes attribute
    is not used.) This is really awkward but necessary for cases where
    what we are tweaking is the assignment of particular genes to
    particular compartments, e.g.

"""
from fluxtools.gene_utilities import set_gene_association_from_list

def load_GRA_table(filename):
    gra = {}
    with open(filename) as f:
        for line in f:
            gene, reaction = line.strip().split('\t')
            gra.setdefault(gene,set()).add(reaction)
    return gra

def load_override_file(filename):
    gra = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            items = line.strip().split('\t')
            if not items:
                continue
            gene = items[0]
            reactions = items[1:]
            gra[gene] = set(reactions)
    return gra

if __name__ == '__main__':
    import sys
    from fluxtools.sbml_interface import toSBMLString, fromSBMLFile

    # Eventually I should convert this to use argparse.
    gra_file = sys.argv[1]
    model_file = sys.argv[2]
    key_field = sys.argv[3]
    override_file = sys.argv[4]
    
    net = fromSBMLFile(model_file)
    genes_to_keys = load_GRA_table(gra_file)
    keys_to_reactions = {}
    for reaction in net.reactions:
        key = reaction.notes.get(key_field, None)
        keys_to_reactions.setdefault(key,set()).add(reaction.id)
    genes_to_reactions = {g: {r for k in keys for r in 
                              keys_to_reactions.get(k,[])} for g, keys in
                          genes_to_keys.iteritems()}
    replacement_genes_to_reactions = load_override_file(override_file)
    genes_to_reactions.update(replacement_genes_to_reactions)
    
    reactions_to_genes = {}
    for gene, reactions in genes_to_reactions.iteritems():
        for r in reactions:
            reactions_to_genes.setdefault(r,set()).add(gene)
    
    for reaction in net.reactions:
        genes = reactions_to_genes.get(reaction.id)
        if genes:
            set_gene_association_from_list(reaction,
                                           genes)
    
    print toSBMLString(net)
