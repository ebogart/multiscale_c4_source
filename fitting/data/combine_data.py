""" Combine LCM and whole-leaf RNAseq data. 

This version assumes a mesophyll/total ratio of 0.5 for genes
where the LCM expression measurements are too low to calculate the 
true ratio.

"""
import numpy as np

# load the whole-leaf fpkm averages and standard deviations
gradient_array = np.genfromtxt('wl_fpkm_avg.txt',
                               skiprows=0,
                               usecols=range(1,16))
gradient_ids = np.genfromtxt('wl_fpkm_avg.txt',
                             dtype=None,skiprows=0,usecols=[0])
std_array = np.genfromtxt('wl_fpkm_std.txt',
                               skiprows=0,
                               usecols=range(1,16))
std_ids = np.genfromtxt('wl_fpkm_std.txt',
                             dtype=None,skiprows=0,usecols=[0])
gradient_data = dict(zip(gradient_ids,gradient_array))
gradient_std = dict(zip(std_ids, std_array))
gradient_totals = dict(zip(gradient_ids, np.sum(gradient_array,
                                                axis=1)))

# load the lcm data 
data_array = np.genfromtxt('lcm_data.csv',
                            delimiter=',',skiprows=1,
                           usecols=range(1,13))
lcm_gene_ids = np.genfromtxt('lcm_data.csv',delimiter=',',
                          dtype=None,skiprows=1,usecols=[0])
bs_avg = data_array[:,(0,2,4)]
me_avg = data_array[:,(1,3,5)]
bs_std = data_array[:,(6,8,10)]
me_std = data_array[:,(7,9,11)]

me_data = dict(zip(lcm_gene_ids, me_avg))
bs_data = dict(zip(lcm_gene_ids, bs_avg))
me_errors = dict(zip(lcm_gene_ids, me_std))
bs_errors = dict(zip(lcm_gene_ids, bs_std))

me_totals = {g: np.sum(v) for g,v in me_data.iteritems()}
bs_totals = {g: np.sum(v) for g,v in bs_data.iteritems()}

# calculate the ratios. Where the LCM total RPKM is less than a threshold,
# or is entirely absent, use a 1:1 ratio arbitrarily.
me_ratio = me_avg/(me_avg+bs_avg)
all_ratios = {}
for g, ratio in zip(lcm_gene_ids, me_ratio):
    above_threshold = (me_data[g] + bs_data[g]) > 0.1
    filtered_ratio = np.array([(r if check else 0.5) for r,check in 
                               zip(ratio, above_threshold)])
    all_ratios[g] = filtered_ratio
for g in gradient_data:
    if g not in lcm_gene_ids:
        all_ratios[g] = 0.5*np.ones(3)

# interpolate the ratios to all 15 points

# It is possible that we should instead interpolate the LCM data and
# then take ratios, but I have chosen this route for convenience, for
# the moment.

# Here we assume that the ratio exactly at the base is 0.5. (Other
# options include assuming that the ratio at -1 cm extends all the way
# to the base, or somehow allowing this to be optimizable.)
extended_ratios = {}
for gene, ratios in all_ratios.iteritems():
    v = np.zeros(4)
    v[0] = 0.5
    v[1:] = ratios
    extended_ratios[gene] = v

weights = ((1.00, 0.00, 0, 0),
           (0.67, 0.33, 0, 0),
           (0.33, 0.67, 0, 0),
           (0.00, 1.00, 0, 0),
           (0.0, 0.8, 0.2, 0),
           (0.0, 0.6, 0.4, 0),
           (0.0, 0.4, 0.6, 0),
           (0.0, 0.2, 0.8, 0),
           (0.0, 0.0, 1.0, 0),
           (0.0, 0.0, 0.8, 0.2),
           (0.0, 0.0, 0.6, 0.4),
           (0.0, 0.0, 0.4, 0.6),
           (0.0, 0.0, 0.2, 0.8),
           (0.0, 0.0, 0.0, 1.0),
           (0.0, 0.0, 0.0, 1.0))
weights = np.array(weights)
interpolated_ratios = {g: np.dot(weights,extended_ratios[g]) for g
                       in all_ratios}

# apply the interpolated ratios to produce new data
split_data = {}
split_std = {}
skipped = set()
for g, ratios in interpolated_ratios.iteritems():
    if g not in gradient_data:
        skipped.add(g)
        continue
    # KEY POINT: 'ms_', not 'me_', expected by FBA model
    split_data['ms_' + g] = ratios * gradient_data[g]
    split_data['bs_' + g] = (1.0-ratios) * gradient_data[g]
    split_std['ms_' + g] = ratios * gradient_std[g]
    split_std['bs_' + g] = (1.0-ratios) * gradient_std[g]

with open('twocell_data.csv','w') as f:
    for gene_id, data in split_data.iteritems():
        f.write('%s\t' % gene_id)
        f.write('\t'.join(map(str, data)))
        f.write('\n')

with open('twocell_std.csv','w') as f:
    for gene_id, std in split_std.iteritems():
        f.write('%s\t' % gene_id)
        f.write('\t'.join(map(str, std)))
        f.write('\n')
