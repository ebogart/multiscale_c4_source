"""Check PPDB annotations for genes of reactions present in multiple compartments.

Reads relevant files from data/ and the comparmentalized, pre-GRA one-cell SBML
model, corn_fba.xml and gra_overrides.txt and corncyc_global_gra.txt.

'data/maize321_annotation_ppdb.csv' is an export of data from the
Plant Proteome Database, ppdb.tc.cornell.edu, provided by Qi Sun on 2014-06-09. 

Some relevant genes aren't represented there, so I manually downloaded addition
records into 'data/ppdb_export.txt' (most recently on 2014-09-04). That file
should be a tab-delimited table with one header row, first column the protein
accession, second column the standard annotation, third column the TargetP prediction,
and fourth column not important (PPDB exports tabs after every field...)

Writes compartment_check_result.txt, listing, for every reaction frame in
CornCyc which has instances in more than one compartment in the model,
each associated gene's
- absence from gra_overrides.txt? (!)
- accession
- total expression level
- curated location
- TargetP prediction
- other reaction assocations
- PPDB 'lab annot'
- PPDB 'std annot'

"""

import csv, time
annots = {}
curated_locations = {}
mapman_bins = {}
mapman_names = {}

def accession_to_gene(accession):
    """ Strip extra information from PPDB accessions.

    Accessions are given complete with transcript/protein model, 
    eg "GRMZM2G101042_P02"; our data is by gene, so we strip the final 
    tag-- except in the cases of accessions beginning in AC or EF, eg 
    AC155624.2_FGP011, where the final piece differentiates completely
    genes. 

    """
    accession = accession.strip()
    if accession.startswith('GRMZM'):
        gene = accession.split('_')[0]
    elif accession.startswith('AC') or accession.startswith('EF'):
        gene = accession.replace('_FGP','_FG')
    else:
        raise ValueError(accession)
    return gene

with open('data/maize321_annotation_ppdb.csv') as f:
    reader = csv.reader(f,delimiter='\t')
    # Skip header row
    reader.next()
    for accession, annot, location, bin1, name1, bin2, name2 in reader:
        gene = accession_to_gene(accession)
        annots.setdefault(gene,set()).add(annot)
        curated_locations.setdefault(gene,set()).add(location)
        if bin1 != 'NULL':
            mapman_bins.setdefault(gene,set()).add(bin1)
            mapman_names.setdefault(gene,set()).add(name1)
        if bin2 != 'NULL':
            mapman_bins.setdefault(gene,set()).add(bin2)
            mapman_names.setdefault(gene,set()).add(name2)
nonnull_locations = {k:v for k,v in curated_locations.iteritems() if v != {'NULL'}}

with open('gra_overrides.txt') as f:
    override_genes = set()
    for line in f:
        if not (line.startswith('#') or line.isspace()):
            override_genes.add(line.strip().split()[0])

std_annots = {}
targetp_locations = {}
with open('data/ppdb_export.txt') as f:
    reader = csv.reader(f,delimiter='\t')
    # Skip header row
    reader.next()
    for accession, annot, targetp, _ in reader:
        gene = accession_to_gene(accession)
        std_annots.setdefault(gene,set()).add(annot)
        targetp_locations.setdefault(gene,set()).add(targetp)

totals = {}
with open('data/wl_fpkm_total.txt') as f:
    reader = csv.reader(f,delimiter='\t')
    for accession, value in reader:
        totals[accession] = float(value)

from fluxtools.sbml_interface import fromSBMLFile
net = fromSBMLFile('corn_fba.xml')
compartments = ['chloroplast','mitochondrion','peroxisome']
frame_to_reaction = {}
frame_to_compartments = {}
parents = set()
for r in net.reactions:
    parent = (r.notes.get('PARENT_FRAME') or 'None')
    if parent == 'None':
        continue
    parents.add(parent)
    frame_to_reaction.setdefault(parent, []).append(r.id)
    found = False
    for c in compartments:
        if r.id.endswith(c):
            frame_to_compartments.setdefault(parent,set()).add(c)
            found = True
            break
    if not found:
        frame_to_compartments.setdefault(parent,set()).add('cytosol')

from add_gra import load_GRA_table
gra = load_GRA_table('corncyc_global_gra.txt')
frame_to_genes = {}
for gene, frames in gra.iteritems():
    for frame in frames:
        frame_to_genes.setdefault(frame,set()).add(gene)
tagged_gra = {g: [r if r in parents else '(x) ' + r for r in 
                  reactions] for g,reactions in gra.iteritems()}
multi_compartment_frames = [f for f,compartments in
                            frame_to_compartments.iteritems() if
                            len(compartments) > 1]

report = ''
relevant_unknowns = set()
for f in sorted(multi_compartment_frames):
    report += f + ': ' + ','.join(frame_to_reaction[f]) + '\n'
    report += '-----\n'
    
    gene_info = []
    for g in frame_to_genes.get(f,[]):
        if g not in nonnull_locations and g not in targetp_locations:
            relevant_unknowns.add(g)
        gene_info.append([('! ' if g not in override_genes else '  ') + g,
                          round(totals.get(g,-1),1),
                          list(curated_locations.get(g,[None])),
                          list(targetp_locations.get(g,['?'])),
                          [other_frame for other_frame in tagged_gra[g] if
                           f != other_frame],
                          list(annots.get(g,[None])), list(std_annots.get(g,[None]))])
        gene_info.sort(key=lambda t: (str(t[2]), t[1]),reverse=True)

    report += '? ' + '\t'.join(('gene','expression','curated loc','targetp','other reactions','curated annot',
                         'std annot')) + '\n'
    

    for line in gene_info:
        report += '\t'.join(map(str,line)) + '\n'

    report += '\n\n'
    
with open('compartment_check_result.txt','w') as f:
    f.write(report)
    f.write('There were %d relevant genes missing from both data files.\n' % len(relevant_unknowns))
    f.write(time.ctime())

# Genes to request from PPDB to regenerate ppdb_export.txt while also searching for
# the relevant unknowns:

query = sorted([g for g in relevant_unknowns.union(std_annots)])
# Now we have to reformat some accessions the other way to be able to find them in PPDB...
query = [g.replace('_FG','_FGP') for g in query]
# Also, if you try to search PPDB for accessions starting with 'AC' 
# and accessions starting with 'GRMZM' at the same time, only the results
# for the 'GRMZM' accessions are returned, so it's best to submit two queries
# and concatentate the results.

# Note that, however I munge the name, I can't find 'AC147602.5_FG004' in PPDB today (2014-08-19).
# Not sure why; it remains in MaizeGDB (and KEGG), annotated as a sedoheptulose bisphosphatase.
