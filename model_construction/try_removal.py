"""Remove lower-confidence reactions not necessary for basic functionality.

In particular, we find essential reactions (in corn_full.txt) which
are (1) associated with a parent frame with no associated genes or (2)
associated with a parent frame with no pathway assignment, then
discard all other reactions falling into those classes, unless they
are listed in 'try_removal_exclusions.txt'. Then we additionally
discared blocked reactions 

Requires a connection to Pathway Tools.

"""

import numpy as np
import pycyc
from table_utilities import read_reaction_table, write_reaction_table
from table_utilities import read_component_table
from model_from_table import fba_test_reaction, fba_test_sink

full_model_file = 'corn_full.txt'
gra_file = 'corncyc_global_gra.txt'
# Remove these reactions only if they can't carry flux anyway:
exclusions_file = 'try_removal_exclusions.txt' 
zero_threshold = 150

########################################
# Load GRA, corncyc reactions; identify reactions in CornCyc 
# without a pathway assignment
corn = pycyc.open('corn')
corncyc = set(map(str, corn.reactions))
corncyc_nonpathway = set(map(str, [r for r in corn.reactions if
                                   not r.in_pathway]))
gra = {}
with open(gra_file) as f:
    for line in f:
        gene, reaction = line.strip().split()
        gra.setdefault(reaction,set()).add(gene)

########################################
# Load model, find reactions in model that are questionable 
base_model = read_reaction_table(full_model_file)
stoichiometries, parents, reversibilities = base_model
nongenetic_reactions = [r for r in stoichiometries if 
                        parents[r] not in gra and
                        parents[r] is not None]
nonpathway_reactions = [r for r in stoichiometries if 
                        parents[r] in corncyc_nonpathway and
                        parents[r] is not None]
neither_reactions = [r for r in stoichiometries if
                     parents[r] in corncyc_nonpathway and
                     parents[r] is not None]

species = {compound for (reaction, stoichiometry) in
           base_model[0].iteritems() for compound in stoichiometry}
biomass_targets = [s for s in species if s.endswith('_biomass')]

##################################################
# Set up the function we will use to see which of the questionable
# reactions must be kept
def try_removing_set(model, bad_reactions, check_special=True):
    """ Report reactions in a test set necessary for basic functionality. """
    # Additional reactions can be added to the model while tests are
    # run, for debugging
    extras = {}
    # Collect the bad reactions associated with each essential 
    # species or reaction
    dependencies = {}

    # 1. Test biomass species.
    for s in biomass_targets:
        success, fluxes = fba_test_sink(
            model, s, extras=extras, do_conserve=(s,),
            cost_keys=bad_reactions, 
            non_conserved=set(('SUCROSE_phloem',)),
            default_bound=10000.
        )
        # Note we assume it is possible to achieve all essential
        # functions if the use of the 'bad' reactions is allowed; if
        # this assumption is violated, stop.
        if not success:
            raise Exception()
        overlap = {r for r in fluxes if r in bad_reactions}
        if overlap:
            dependencies[s] = overlap

    # 2. Test photosynthetic sucrose production, specifying some
    # chloroplast transport bound to avoid a particular unrealistic
    # case.
    success, fluxes = fba_test_sink(
        model, 'SUCROSE',
        extras=extras,
        extra_bounds={'tx_SUCROSE': 0.,
                      'chloroplast_TPT_DHAP_exchange': 0.,
                      'chloroplast_TPT_GAP_exchange': 0.},
        cost_keys=bad_reactions)
    if not success:
        raise Exception()
    overlap = {r for r in fluxes if r in bad_reactions}
    if overlap:
        dependencies['photosynthetic sucrose'] = overlap        

    # 3. Test rubisco oxygenase.
    success, fluxes = fba_test_reaction(
        model, 'RXN-961_chloroplast', extras=extras,
        cost_keys=bad_reactions)
    overlap = {r for r in fluxes if r in bad_reactions}
    if overlap:
        dependencies['rubisco oxygenase'] = overlap        
        
    # 4. Test various special cases: reactions in compartments,
    # transporters connected to the phloem compartment, intracellular
    # transport reactions-- on the theory that we have taken
    # particular care to put these reactions into the model and should
    # not break them in the simplification process. Note that in some
    # cases they do _not_ work in the full model (even if all 'bad'
    # reactions are allowed); in that case, warn and move on.
    reaction_targets = {r for r in stoichiometries if 
                        r.endswith('mitochondrion') or
                        r.endswith('chloroplast') or 
                        r.endswith('peroxisome')}
    special = read_reaction_table('special_cases.txt')
    # Some of these may have been compartmentalized; warn if they have
    # disappeared completely from the model.
    for s in special[0]:
        candidates = [r for r in stoichiometries if r.startswith(s)]
        if len(candidates) < 1:
            print 'Misplaced special case reaction %s' % s
        else:
            reaction_targets.update(candidates)
    transport = read_reaction_table('corn_intracellular_transport.txt')
    reaction_targets.update(transport[0].keys())
    # Phloem transporters are handled separately as we are interested
    # in one direction only (exporting).
    phloem_tx_targets = {r for r in stoichiometries if 'tx' in r and
                         [c for c in stoichiometries[r] if 'phloem' in c]}
    phloem_tx_bounds = dict.fromkeys(phloem_tx_targets,(-100.,0.))

    # List the reactions which do not work even in the full model.
    skipped_specials = []
    if check_special:
        for r in reaction_targets:
            # Test forward direction.
            success, fluxes = fba_test_reaction(
                model, r, extras=extras, 
                cost_keys = bad_reactions, value=0.1,
                non_conserved=set(('SUCROSE_phloem',)),
                default_bound=10000.
            )
            # If that failed, test reverse direction if 
            # applicable.
            if (not success) and (model[2][r]):
                success, fluxes = fba_test_reaction(
                    model, r, extras=extras, 
                    cost_keys = bad_reactions,
                    value=-0.1,
                    non_conserved=set(('SUCROSE_phloem',)),
                    default_bound=10000.
                )
            if success:
                overlap = {r for r in fluxes if r in bad_reactions}
                if overlap:
                    dependencies[r] = overlap 
            else:
                # Neither direction carries flux even if all 
                # 'bad' reactions are allowed.
                skipped_specials.append(r)

        for r in phloem_tx_targets:
            success, fluxes = fba_test_reaction(
                model, r, extras=extras, 
                cost_keys = bad_reactions, value=-0.1,
                free_compartments=(
                    'biomass',
                    'external',
                    'fixed_biomass',
                    'intercellular_air_space',
                    'xylem',
                    'phloem'),
                extra_bounds=phloem_tx_bounds,
                default_bound=10000.)
            if success:
                overlap = {r for r in fluxes if r in bad_reactions}
                if overlap:
                    dependencies[r] = overlap 
            else:
                skipped_specials.append(r)
                
    if skipped_specials:
        print 'Could not complete at any cost:'
        print skipped_specials
    return dependencies, {r for v in dependencies.values() for r in v}

############################################
# Protect some reactions from removal. Unless they turn out to be
# blocked in the final model, we will not remove of the special
# case/compartmentalized/transport reactions themselves...
protected_reactions = {r for r in stoichiometries if
                       r.endswith('mitochondrion') or
                       r.endswith('chloroplast') or
                       r.endswith('peroxisome') or
                       'tx' in r}
special = read_reaction_table('special_cases.txt')
protected_reactions.update(special[0])
transport = read_reaction_table('corn_intracellular_transport.txt')
protected_reactions.update(transport[0])
# ... or the reactions exempted in the exclusion file.
exclusions = read_component_table(exclusions_file)
protected_reactions.update(exclusions)

############################################
# First simplification step.
# Remove non-essential reactions without pathway assignments.
# Note as of this writing no _essential_ reactions without
# pathway assignments (that aren't otherwise exempted.)

nonprotected_nonpathway = [r for r in nonpathway_reactions 
                           if r not in protected_reactions]
nonpathway_details, nonpathway_essential = try_removing_set(
    base_model, nonprotected_nonpathway
)

reduced_stoichiometries = stoichiometries.copy()
removed_in_step1 = []
for r in nonprotected_nonpathway: 
    if r not in nonpathway_essential:
        reduced_stoichiometries.pop(r)
        removed_in_step1.append(r)
reduced_model = [reduced_stoichiometries, parents, reversibilities]

############################################
# Second simplification step.
# Remove non-essential reactions without genes.

nonprotected_nongenetic = [r for r in reduced_stoichiometries if
                           r in nongenetic_reactions and 
                           r not in protected_reactions]
nong_details, nongenetic_essential = try_removing_set(
    reduced_model, nonprotected_nongenetic
)

removed_in_step2 = []
for r in nonprotected_nongenetic:
    if r not in nongenetic_essential:
        reduced_stoichiometries.pop(r)
        removed_in_step2.append(r)

############################################
# Final simplification:
# Remove all reactions which are now blocked (except transporters.)

# Next find all the blocked reactions:
blocked = [r for r in reduced_stoichiometries if 
           ((not fba_test_reaction(reduced_model,
                                   r, default_bound=10000.,
                                   value=0.1)[0]) 
            and ((not reversibilities[r]) or 
                 (not fba_test_reaction(reduced_model, 
                                        r, default_bound=10000.,
                                        value=-0.1)[0]))
           )]
for r in blocked:
    # Keep all the transporters, but warn
    if 'tx' in r:
        print 'Blocked transporter %s kept' % r
        continue
    if r in protected_reactions:
        print 'Removing blocked special reaction %s' % r
    reduced_stoichiometries.pop(r)


############################################
# Finally write out a new one-cell file, corn_reduced.txt
write_reaction_table('corn_reduced.txt', 
                     reduced_stoichiometries, 
                     parents,
                     reversibilities)

### OLD, TO DELETE
# But we can go farther!

# # Many reactions involving NAD(P)H exist in both instantiated and raw forms:
# generics = [r for r in reduced_stoichiometries if 'NADH-P-OR-NOP' in r and 
#             r not in protected_reactions]
# generic_details, generic_essential = try_removing_set(reduced_model, bad_reactions=generics)
# for r in generics:
#     if r not in generic_essential:
#         reduced_stoichiometries.pop(r)


# # Now find all species which are connected to the rest of the network
# # only by instances of the same frame. Typically these are secondary
# # metabolites produced by an NADH/NADPH pair. (Ignore biomass species.) 
# species_parents = {}
# species_reactions = {}
# for r, stoichiometry in reduced_stoichiometries.iteritems():
#     for s in stoichiometry:
#         if s.endswith('biomass'):
#             continue
#         species_reactions.setdefault(s,set()).add(r)
#         species_parents.setdefault(s,set()).add(parents[r])
# pseudo_singletons = [s for s,p in species_parents.iteritems() if
#                      len(p) == 1 and None not in p]
# for ps in pseudo_singletons:
#     for r in species_parents[ps]:
#         print 'removing %s (%s)' % (r,ps)
#         reduced_stoichiometries.pop(r)

