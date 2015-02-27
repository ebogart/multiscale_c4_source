""" 
Reformat the whole-leaf FPKM data file, obtaining averages of replicates.

Writes wl_fpkm_avg.txt, wl_fpkm_std.txt, wl_fpkm_total.txt.

25 June 2014

"""
import numpy as np

# The master data file, 'WL_maizev2FGS_fpkm.xls', contains
# 5 replicates at each of 15 segments for each gene, numbered 
# nonconsecutively and given in an unhelpful order. The sections and 
# replicates are labeled as follows:
sections = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
replicates = [1, 2, 3, 4, 6]
# Each column is labeled 'rep%dsec%d' % (replicate, section). 
# We will load this and rearrange it into an array of shape 
# (N_genes, 15, 5), then take the average and standard deviation
# over the replicates, and write new files.
# First, load the gene IDs:
genes = np.genfromtxt('WL_maizev2FGS_fpkm.xls', usecols=(0,),
                      skiprows=1, dtype=None)
N_genes = len(genes)
# Then the data as a structured array:
data = np.genfromtxt('WL_maizev2FGS_fpkm.xls', names=True,
                     usecols=range(1,76))
# Set up a new array of the desired shape. (There is 
# probably some clever way to achieve this by reshaping
# the existing array.)
by_replicate = np.zeros((N_genes, len(sections), len(replicates)))
for i, section_i in enumerate(sections):
    for j, replicate_j in enumerate(replicates):
        label = 'rep%dsec%d' % (replicate_j, section_i)
        by_replicate[:,i,j] = data[label]
mean_fpkm = np.mean(by_replicate, axis=-1)
std_fpkm = np.std(by_replicate, axis=-1)

with open('wl_fpkm_avg.txt','w') as mean_file:
    for gene, values in zip(genes, mean_fpkm):
        mean_file.write('%s\t' % gene)
        mean_file.write('\t'.join(map(str, values)))
        mean_file.write('\n')

with open('wl_fpkm_std.txt','w') as std_file:
    for gene, values in zip(genes, std_fpkm):
        std_file.write('%s\t' % gene)
        std_file.write('\t'.join(map(str, values)))
        std_file.write('\n')

with open('wl_fpkm_total.txt','w') as total_file:
    for gene, values in zip(genes, mean_fpkm):
        total_file.write('%s\t' % gene)
        total_file.write(str(np.sum(values)))
        total_file.write('\n')

