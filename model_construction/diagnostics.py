"""Corn model quick diagnostic tool.

Tests the production of everything in the biomass compartment,
production of sucrose from CO2 without exchange of triose phosphate
other than G3P with the chloroplast, and the ability of every reaction
in the special cases file, intracellular transport file, etc., to
carry flux under standard conditions (trying forward first, then
reverse.)

Note that this is somewhat sloppy as a test of the species
of the fixed-composition biomass reactions adapted from iRS1563, 
because the 'fixed_biomass' compartment is freed from conservation
rules by default in model_from_table. It should suffice as a test
of the production of first-tier components of that biomass 
composition (that is, those which are produced by reactions which pull
substrates directly from the cytoplasmic compartment, where 
conservation is enforced.)

"""
import sys
import numpy as np
from table_utilities import read_reaction_table
from model_from_table import fba_test_reaction, fba_test_sink

zero_threshold = 1e-6

extras = {}
#extras = {'fake_ferredoxin_reductase': {'|Oxidized-ferredoxins|': -1., '|Reduced-ferredoxins|': 1.}}

if __name__ == '__main__': 
    model = read_reaction_table(sys.argv[1])

    print 'Checking production of biomass species...'
    species = {compound for (reaction, stoichiometry) in
               model[0].iteritems() for compound in stoichiometry}
    targets = [s for s in species if s.endswith('_biomass')]
    biomass_failures = []
    for s in targets:
        if not fba_test_sink(model, s, extras=extras, do_conserve=(s,))[0]:
            biomass_failures.append(s)
            print 'Cannot synthesize %s' % s

    print 'Checking special cases...'
    reaction_targets = []
    special = read_reaction_table('special_cases.txt')
    for s in special[0]:
        candidates = [r for r in model[0] if r.startswith(s)]
        if len(candidates) < 1:
            print 'Misplaced special case reaction %s' % s
        else:
            reaction_targets += candidates
    transport = read_reaction_table('corn_intracellular_transport.txt')
    reaction_targets += transport[0].keys()
    reaction_failures = []
    default_value = 0.1
    for r in reaction_targets:
        if not fba_test_reaction(model, r, value=default_value,
                                 extras=extras)[0]:
            if not fba_test_reaction(model, r, extras=extras,
                                     value=-1.*default_value)[0]:
                print 'Reaction %s can carry no flux' % r
                reaction_failures.append(r)

    print 'Checking photosynthetic sucrose production...'
    sucrose = fba_test_sink(
        model, 'SUCROSE',
        extras=extras,
        extra_bounds={'tx_SUCROSE': 0.,
                      'chloroplast_TPT_DHAP_exchange': 0.,
                      'chloroplast_TPT_GAP_exchange': 0.})[0]
    print('OK' if sucrose else 'failed')
        
    print 'Checking rubisco oxygenase...'
    oxygenase = fba_test_reaction(
        model, 'RXN-961_chloroplast', extras=extras)[0]
    print('OK' if oxygenase else 'failed')

    # Error checking of stoichiometries
    bad_stoichiometries = []
    for r, s in model[0].iteritems():
        if s == {} or [v for v in s.values() if np.abs(v) < zero_threshold]:
            print 'Suspicious stoichiomietry for %s' % r
            bad_stoichiometries.append(r)
            
    t = (len(biomass_failures), len(reaction_failures))
    print '%d biomass components unreachable. %d special case reactions inoperative' % t
    if bad_stoichiometries:
        print 'There were bad stoichiometries.'

    if not (sucrose and oxygenase and biomass_failures == []):
        print 'There were critical failures.'
    
    
