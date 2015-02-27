############################################
# Convert per-gene data to per-reaction data
import numpy as np

def get_gra(net):
    """ Extract GRA from 'GENE_ASSOCIATION' notes.

    Currently this is very fragile and can handle only ' or ' 
    relationships without parentheses, etc.

    """
    gra = {}
    for reaction in net.reactions:
        genes = reaction.notes.get('GENE_ASSOCIATION',None)
        if genes:
            for g in genes.split(' or '):
                gra.setdefault(reaction.id,set()).add(g)
    return gra

def get_reverse_gra(net):
    """ Extract list of reactions associated with each gene.

    As with get_gra, this can handle only simple ' or ' relationships.

    """
    associations = {}
    gra = get_gra(net)
    for reaction, gene_set in gra.iteritems():
        for g in gene_set:
            associations.setdefault(g, set()).add(reaction)
    return associations

def gene_data_to_reaction_data(gene_data, reactions_to_genes):
    """ Naively combine data by adding data for all genes associated with reaction.

    This is theoretically unjustifiable but we can at least
    handwave that the different genes probably have a high sequence similarity
    and thus the RNA-seq bias is probably comparable for them at least
    to some level. 

    """
    gene_associations = reactions_to_genes

    reaction_data = {}
    for reaction, genes in gene_associations.iteritems():
        data = []
        for g in genes:
            if g in gene_data:
                data.append(gene_data[g])
        if data:
            reaction_data[reaction]= sum(data)

    return reaction_data

def split_gene_data_to_reaction_data(gene_data, genes_to_reactions):
    """ Split data for genes across their associated reactions and add.

    This is theoretically problematic in several ways! Unlike
    gene_data_to_reaction_data above, it won't over-count the expression
    data for genes with multiple associated reactions. (In practice, this
    will likely overallocate data to some reactions and underallocate data to
    others, of course.)

    Data should be vectors.

    """
    N_points = len(gene_data.values()[0])
    reaction_data = {}
    for gene, data in gene_data.iteritems():
        reactions_of_gene = genes_to_reactions.get(gene,[])
        N_reactions = len(reactions_of_gene)
        for reaction in reactions_of_gene:
            if reaction not in reaction_data:
                reaction_data[reaction] = np.zeros(N_points)
            reaction_data[reaction] += data/N_reactions
    return reaction_data

def split_gene_error_to_reaction_error(gene_error, genes_to_reactions):
    """Combine uncertainties for genes, obtain uncertainties for reactions.

    This is the natural companion to split_gene_data_to_reaction_data,
    above, and is likewise theoretically problematic.

    The errors should be vectors.

    """
    N_points = len(gene_error.values()[0])
    reaction_variances = {}
    for gene, std in gene_error.iteritems():
        reactions_of_gene = genes_to_reactions.get(gene,[])
        if reactions_of_gene:
            N_reactions = len(reactions_of_gene)
            contribution = (std/N_reactions)**2
        for reaction in reactions_of_gene:
            if reaction not in reaction_variances:
                reaction_variances[reaction] = np.zeros(N_points)
            reaction_variances[reaction] += contribution

    reaction_error = {k: np.sqrt(v) for k,v in
                      reaction_variances.iteritems()}
    return reaction_error
