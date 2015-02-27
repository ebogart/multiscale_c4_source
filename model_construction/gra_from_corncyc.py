"""
Extract a mapping of maize genes to CornCyc reactions from CornCyc.

Also writes some auxiliary tables. 

INPUTS: 

CornCyc Pathway Tools database (will be read through PyCyc interface)
    
OUTPUTS: 

corncyc_nonstandard_id_mappings.txt: 
    Lists cases where a standard gene accession for the gene frame could
    not be determined. First column the gene frame ID, second column a
    (space-delimited) list of the names in the frame's 'names' slot.
    Associations between reactions and these genes will not be written
    to the files below.

gene_to_corncyc_reaction_[direct|indirect].txt:
    tab-delimited table, one row per (gene, reaction) pair, first
    column a standard Gramene gene id (GRMZM2G... etc), second column
    the name of a reaction frame in CornCyc with which that gene 
    is associated; one file lists genes directly associated with
    the indicated reaction frames, the other genes associated with 
    generic forms of the reaction frames.


"""
#############################
## CHOOSE DB TO WORK WITH
db_id = 'corn'
db_name = '%scyc' % db_id
#############################

import re
import pycyc
import sys

db = pycyc.open(db_id)

gene = re.compile(r'(GRMZM\dG\d\d\d\d\d\d|[AE][CFY]\d\d\d\d\d\d.\d_FG\d\d\d)$')
model = re.compile(r'(GRMZM\dG\d\d\d\d\d\d_[TtPp]\d\d|[AE][CFY]\d\d\d\d\d\d.\d_FG[P|T]\d\d\d)$')

def write_table(filename, list_of_pairs):
    with open(filename,'w') as f:
        for pair in list_of_pairs:
            f.write('%s\t%s\n' % pair)

def read_table(filename):
    with open(filename) as f:
        pairs = []
        for line in f:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

###############################
# ESTABLISH FRAME TO GENE ID MAP
# The relationship of gene frames to gene IDs is different for 
# CornCyc (one frame per transcript/protein model) and MaizeCyc
# (one frame per gene in the usual sense). 

def corn_frame_to_accession():
    frame_to_accession = {}
    skipped_frames = {}
    nonmodel_matches = 0 
    multiple_name_matches = 0
    for g in db.genes:
        accession = g.accession_1 or ''
        gene_id = ''
        if gene.match(accession):
            nonmodel_matches += 1
            frame_to_accession[str(g)] = accession
            continue
        if model.match(accession):
            gene_id = accession
        else:
            name_matches = 0 
            for name in g.names:
                if gene.match(name):
                    name_matches += 1
                    frame_to_accession[str(g)] = name 
                if model.match(name):
                    gene_id = name
                    name_matches += 1 
            if name_matches > 1: 
                multiple_name_matches += 1
        if frame_to_accession.get(str(g), None): # matched a gene name already
            pass
        elif gene_id:
            if gene_id.startswith('G'):
                frame_to_accession[str(g)] = gene_id[:-4]
            if gene_id.startswith('A') or gene_id.startswith('E'):
                gene_id = gene_id[:-4] + gene_id[-3:] # drop 'P' or 'T'
                frame_to_accession[str(g)] = gene_id
        else:
            skipped_frames[str(g)] = g.names
    print 'Skipped %d frames without recognizable accession.' % len(skipped_frames)
    for frame, g in frame_to_accession.iteritems():
        if not gene.match(g):
            print 'Bad accession value %s for frame %s' % (g,frame)
    return frame_to_accession, skipped_frames

print 'Determining accessions for gene frames in %s...' % db_name
if db_id == 'corn':
    frame_to_accession, skipped_frames = corn_frame_to_accession()
else:
    raise

accession_filename = '%s_gene_id_mappings.txt' % db_name
write_table(accession_filename, sorted(frame_to_accession.items()))
bad_accession_filename = '%s_nonstandard_id_mappings.txt' % db_name
with open(bad_accession_filename, 'w') as f:
    for frame_id, name_list in sorted(skipped_frames.items()):
        if not isinstance(name_list, list):
            name_list = [name_list]
        f.write('%s\t%s\n' % (frame_id, ' '.join(name_list)))
print 'Done.'

###############################
# MAP GENE ACCESSIONS TO REACTION FRAMES

all_reaction_frame_names = {str(r) for r in db.reactions}
def gene_to_reaction():
    annotations = {}
    for r in all_reaction_frame_names:
        for g in db.genes_of_reaction(r) or ():
            a = frame_to_accession.get(str(g),None)
            if a:
                annotations.setdefault(r, set()).add(a)
    return annotations

print 'Extracting gene-reaction associations...'
direct_annotations = gene_to_reaction()

indirect_annotations = {}
for r in all_reaction_frame_names:
    indirect_genes = set()
    generic_forms = map(str, db.nonspecific_forms_of_rxn(r) or [])
    for generic_reaction_frame in generic_forms:
        indirect_genes.update(direct_annotations.get(generic_reaction_frame, set()))
    indirect_annotations[r] = indirect_genes
    
output_filename = 'gene_to_%s_reaction_direct.txt' % db_name
write_table(output_filename, [(gene, reaction) for reaction, genes in
                              direct_annotations.iteritems() for gene
                              in genes])
output_filename = 'gene_to_%s_reaction_indirect.txt' % db_name
write_table(output_filename, [(gene, reaction) for reaction, genes in
                              indirect_annotations.iteritems() for gene
                              in genes])

print 'Done.'

