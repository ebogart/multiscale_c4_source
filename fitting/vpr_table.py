""" 
Summarize reactions of the C4 system and count their associated genes.

"""
import fluxtools.sbml_interface as si
from utilities import get_gra
twocell = si.fromSBMLFile('models/iEB2140x2.xml')
gra = get_gra(twocell)

frames = [('adenylate kinase', 'ADENYL-KIN-RXN'),
          ('malate dehydrogenase (NADP)', 'MALATE-DEHYDROGENASE-NADP+-RXN'),
          ('alanine aminotransferase', 'ALANINE-AMINOTRANSFERASE-RXN'),
          ('NADP-malic enzyme', 'MALIC-NADP-RXN'),
          ('PPDK', 'PYRUVATEORTHOPHOSPHATE-DIKINASE-RXN'),
          ('PEPCK', 'PEPCARBOXYKIN-RXN'),
          ('NAD-malic enzyme', '1.1.1.39-RXN'),
          ('aspartate aminotransferase', 'ASPAMINOTRANS-RXN'),
          ('pyrophosphatase', 'INORGPYROPHOSPHAT-RXN')]
frame_set = {t[1] for t in frames}
frame_to_name = {r:n for n,r in frames}

additional = ['plasmodesmata pyruvate exchange',
              'plasmodesmata PEP exchange',
              'plasmodesmata alanine exchange',
              'plasmodesmata aspartate exchange',
              'plasmodesmata malate exchange',
              'chloroplast pyruvate, PEP, malate and OAA exchange'] 
genes = {r.id: len(gra[r.id]) for r in twocell.reactions if
         r.notes.get('PARENT_FRAME',None) in frame_set}
lines = []
for r,n in genes.iteritems():
    frame = frame_to_name[twocell.reactions.get(r).notes['PARENT_FRAME']]
    lines.append('%s & %s & %s \\' % (frame, r, n))
lines.sort()
for l in lines:
    print l
print 'Additional important processes:'
for p in additional:
    print p
