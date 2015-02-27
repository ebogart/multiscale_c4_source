""" Load experimental data sets for use with the (reduced) two-cell model.

Exports:

data_and_error(...): returns data, error as dicts whose keys are reactions in the
    model and values are arrays of data or uncertainties

"""
import pickle
import numpy as np
import reduced_model

########################################
# LOAD DATA
data_filename = 'data/twocell_data.csv' #tab-delimited,
                                        #compartment-specific,
                                        #per-gene
gene_data_array = np.genfromtxt(data_filename,delimiter='\t',
                                usecols=range(1,16))
gene_ids = np.genfromtxt(data_filename,delimiter='\t',
                         dtype=None,
                         usecols=[0])
gene_data = dict(zip(gene_ids, gene_data_array))

std_filename = 'data/twocell_std.csv'
gene_std_array =  np.genfromtxt(std_filename,delimiter='\t',
                                usecols=range(1,16))
std_gene_ids = np.genfromtxt(std_filename,delimiter='\t',
                         dtype=None,
                         usecols=[0])
gene_std = dict(zip(std_gene_ids, gene_std_array))

# Convert it to reaction data
from utilities import get_reverse_gra, split_gene_data_to_reaction_data
from utilities import split_gene_error_to_reaction_error
reduced_gra = get_reverse_gra(reduced_model.corn_net)
reduced_data = split_gene_data_to_reaction_data(gene_data, reduced_gra)
reduced_error = split_gene_error_to_reaction_error(gene_std, reduced_gra)


########################################
# PROCESS AND RETURN IT AS NECESSARY
def data_and_error(n_points=15, n_reactions=None,
                   min_data_sum=0., 
                   min_absolute_error=0., min_relative_error=0., scale=1.):
    """ Return expression data and error after filtering and rescaling.

    Note that the minimum data sum and minimum absolute error are
    enforced after rescaling.

    """

    _data = reduced_data
    _error = reduced_error

    return_data = {k: scale*v for k,v in _data.iteritems()}
    data_sums = {k: np.sum(v) for k,v in return_data.iteritems()}
        
    if n_reactions:
        sorted_keys = sorted(data_sums.keys(), key=lambda k: data_sums[k],
                         reverse=True)[:n_reactions]
        return_data = {k: v for k,v in return_data.iteritems() 
                       if k in sorted_keys}

    return_error = {}
    for k in return_data:
        v = _error[k]
        # Careful: compare the scaled v to the
        # minumum relative error times the scaled 
        # return_data
        v = np.fmax(scale*v, min_relative_error*return_data[k])
        v = np.fmax(min_absolute_error, v)
        return_error[k] = v

    return return_data, return_error

