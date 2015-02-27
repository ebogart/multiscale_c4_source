"""Assess conservation problems, activity of reactions, etc.

In particular, we want to know:

1. For each reaction, may it carry flux under standard boundary conditions?

(We may then reduce the model to remove reactions which can carry no flux
before continuing to the following:)

2. For each reaction, what is the minimum size of cycle it may participate in
   if the system is closed (s.t. it carries a unit of flux?)

3. For each species, can we demonstrate that it may be created or
   destroyed as part of a conservation-of-mass-violating process, and
   if so, what is the minimum total flux required for such a process?

4. For each reaction, how many of the minimum-flux nonphysical processes in step 3 
   does it participate in?

The numbers of reactions and species are large enough that it makes
sense to parallelize these. In Step 1, we could make the process more
efficient by updating the results for all reactions after each
simulation, but rather than deal with interprocess communication on
that scale we simply accept some unnecessary duplication of effort.

"""
import pickle
import numpy as np

import fluxtools.stoichiometry_matrix as st
import multiprocessing as mp

from model_from_table import fba_model_generator
from table_utilities import read_reaction_table
from deblock import deblock

zero_threshold = 1e-6
default_bound = 1e5
n_processes = 4

def assess(model, filename_stem='diagnostic_'):
    """Perform many tests on model. 

    Model should be supplied as a tuple from read_reaction_table.

    """
    # First, deblocking.
    # Add a reaction so that external and biomass species 
    # will not be treated as singletons. The easiest way to 
    # identify them (for the lazy): call model_from_table
    # and look at the nonconserved species:
    _, _, _, biomass_and_external = fba_model_generator(model)
    # The details of the reaction do not matter so long
    # as all these species have nonzero coefficient
    boundary_reaction = dict.fromkeys(biomass_and_external, 1.)
    stoichiometries = model[0].copy()
    stoichiometries['_boundary_reaction'] = boundary_reaction
    (unblocked_stoichiometries, blocked_by_singletons,
     true_singletons) = deblock(stoichiometries,
                                filename_stem + 'log.txt',
                                filename_stem + 'blocked_by_singletons.txt',
                                filename_stem + 'deblock_removed_species.txt')
    unblocked_stoichiometries.pop('_boundary_reaction')
    revised_model = (unblocked_stoichiometries, 
                     model[1],
                     model[2])

    errors = []
    # Second, further flux checking
    blocked_other = []
    cycle_processes = {}
    min_cycle_fluxes = []
    flux_results = parallel_controller(revised_model,
                                       reaction_test_worker,
                                       unblocked_stoichiometries.keys()[:])
    for r, (any_flux, min_cycle, cycle_fluxes) in flux_results.iteritems():
        if 'error' in [any_flux, min_cycle]:
            errors.append(r)
            continue
        if not any_flux:
            blocked_other.append(r)
            unblocked_stoichiometries.pop(r)
        elif min_cycle is None:
            continue
        else:
            min_cycle_fluxes.append((min_cycle, r))
            cycle_processes[r] = cycle_fluxes
    min_cycle_fluxes.sort()

    minimal_stoichiometries = {r:s for r,s in unblocked_stoichiometries.iteritems()
                               if r not in blocked_other}
    
    # Third, conservation and reachability checking
    # Check even species which are not present in the reduced, unblocked model
    reaching_processes = {}
    rescues = {}
    creation_processes = {}
    destruction_processes = {}
    min_nonconservation_fluxes = []
    bad_processes_by_reaction = {}

    all_species = {s for stoichiometry in
                   model[0].values() for s in stoichiometry}
    species_results = parallel_controller(model,
                                          species_test_worker,
#                                          ['CPD-7384'])
                                          list(all_species)[:])
    for (species,
         (reachable, reaching_process,
         created, min_creation_flux, creation_process,
          destroyed, min_destruction_flux, destruction_process)) in \
        species_results.iteritems():
        min_flux = None
        if 'error' in [reachable, created, destroyed]:
            errors.append(species)
            continue
        if reachable: 
            reaching_processes[species] = reaching_process
            local_rescues = []
            for r in reaching_process:
                if (r in blocked_by_singletons or 
                    r in blocked_other):
                    local_rescues.append(r)
            if local_rescues:
                rescues[species] = local_rescues

        if created:
            min_flux = min_creation_flux
            creation_processes[species] = creation_process
            for r in creation_process:
                bad_processes_by_reaction[r] = bad_processes_by_reaction.get(r,0) + 1
        if destroyed:
            if min_flux is None or min_flux > min_destruction_flux:
                min_flux = min_destruction_flux
            destruction_processes[species] = destruction_process
            for r in destruction_process:
                bad_processes_by_reaction[r] = bad_processes_by_reaction.get(r,0) + 1
        if created or destroyed:
            min_nonconservation_fluxes.append((min_flux, species))

    min_nonconservation_fluxes.sort()
    bad_reaction_counts = [(count, r) for r,count in
                           bad_processes_by_reaction.iteritems()]
    bad_reaction_counts.sort()
    bad_reaction_counts.reverse()

    # Fifth, save results

    def save_list(l,filename):
        with open(filename, 'w') as f:
            for k in l:
                f.write(str(k) + '\n')
    save_list(blocked_other, filename_stem + 'blocked_other.txt')
    save_list(min_cycle_fluxes, filename_stem + 
              'reaction_cycles_by_flux.txt')
    save_list(min_nonconservation_fluxes, filename_stem + 
              'nonconserved_species_by_flux.txt')
    save_list(bad_reaction_counts, filename_stem + 'reactions_by_badness.txt')
    with open(filename_stem + 'bad_processes.pickle', 'w') as f:
        pickle.dump((creation_processes, destruction_processes), f)
    with open(filename_stem + 'reachability_data.pickle', 'w') as f:
        pickle.dump((rescues, reaching_processes), f)

#    (unblocked_stoichiometries, blocked_by_singletons,
#     true_singletons) = deblock(stoichiometries,
    if creation_processes or destruction_processes:
        print 'There were conservation violations.' 
    if errors:
        print 'There were optimization failures.'
    return (minimal_stoichiometries, bad_reaction_counts, creation_processes,
            destruction_processes, cycle_processes, reaching_processes, 
            rescues, errors, flux_results, species_results)
    
def parallel_controller(model, test, targets):
    target_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = [mp.Process(target=test,
                            args=(model, target_queue, result_queue))
                 for i in xrange(n_processes)]

    for r in targets:
        target_queue.put(r)

    for p in processes:
        p.start()
        # Be sure to signal workers to terminate
        target_queue.put(None)

    results = {}
    for i in xrange(len(targets)):
        results.update(result_queue.get())
    
    for i,p in enumerate(processes):
        p.join()

    return results


def reaction_test_worker(model, job_queue, result_queue):
    """ Test reactions in model to see if they can carry flux. 

    Entries in job queue: reaction strings

    Entries in result queue: 
    {reaction_string: (True if etc. else False,
                       minimum total flux of closed cycle involving reaction or None,
                       flux distribution for closed cycle)}

    In case of optimization failure the string 'error' will replace a result.

    """
    # Set up model here
    t = fba_model_generator(model, 
                            default_bound=default_bound)
    m, standard_bounds, objective, non_conserved = t

    # Detecting boundary reactions (to close system) may become tricky
    # with more complex biomass sinks...
    boundary_reactions = [r for r in m.reactions if r.startswith('tx_') or
                          r.startswith('sink_')]
    closed_bounds = standard_bounds.copy()
    closed_bounds.update({r: 0. for r in boundary_reactions})
    
    while True:
        reaction = job_queue.get()
        if reaction is None:
            return

        # Only try to achieve flux in direction(s)
        # allowed by the bounds
        fluxes_to_try = []
        lb, ub = standard_bounds.get(reaction, (-1000, 1000))
        if ub > 0:
            fluxes_to_try.append(1.)
        if lb < 0:
            fluxes_to_try.append(-1.)
        
        # Reset model to clean state
        m.set_all_flux_bounds(standard_bounds)

        carries_flux = False
        local_bounds = standard_bounds.copy()
        for v in fluxes_to_try:
            local_bounds[reaction] = v
            m.set_all_flux_bounds(local_bounds)
            m.solve()
            if m.lp.status == 'opt':
                carries_flux = True
                break
            elif m.lp.status != 'nofeas':
                carries_flux = 'error'

        min_cycle_flux = None
        cycle = {}
        # The closed conditions are strictly
        # more restrictive than standard, so skip
        # this step if no flux under standard conditions
        if carries_flux:
            local_bounds = closed_bounds.copy()
            for v in fluxes_to_try:
                local_bounds[reaction] = v
                m.set_all_flux_bounds(local_bounds)
                fluxes = m.solve()
                if m.lp.status == 'opt':
                    if min_cycle_flux is None or m.lp.obj.value < min_cycle_flux:
                        min_cycle_flux = m.lp.obj.value
                        cycle = {r: f for r,f in fluxes.iteritems()
                                 if np.abs(f) > zero_threshold} 
                elif m.lp.status != 'nofeas':
                    min_cycle_flux = 'error'

        result_queue.put({reaction: (carries_flux,
                                     min_cycle_flux,
                                     cycle)})
        
def species_test_worker(model, job_queue, result_queue):
    """ Test species in model to see if they can be created or destroyed.

    Also test if they may be produced under standard conditions
    (not necessarily as a conservation violation.)

    Entries in job queue: species strings

    Entries in result queue: 
    {species_string: (True if can be produced under normal conditions else False,
                       minimal flux distribution reaching species under normal 
                       conditions,
                      True if can be destroyed else False,
                       minimum total flux of closed cycle destroying species,
                       flux distribution destroying species,
                      True if can be created else False,
                       minimum total flux etc,
                       flux distribution creating etc.
                       )}

    In case of optimization failure the string 'error' will replace a result.

    """
    # Set up model here
    t = fba_model_generator(model, 
                            default_bound=default_bound)
    m, standard_bounds, objective, non_conserved = t

    # Creating or destroying is interesting only under closed conditions
    non_conserved_set = set(non_conserved)
    boundary_reactions = [r for r,stoichiometry in model[0].iteritems() if
                          non_conserved_set.intersection(stoichiometry)]
    closed_bounds = standard_bounds.copy()
    closed_bounds.update({r: 0. for r in boundary_reactions})

    standard_rhs = m.right_hand_side[:]
    creation_rhs = [(0., None)] * m.num_compounds
    destruction_rhs = [(None, 0.)] * m.num_compounds
    
    while True:
        species = job_queue.get()
        if species is None:
            return
        # Test reachability under normal conditions
        reachable = False
        reaching_fluxes = {}
        m.set_all_flux_bounds(standard_bounds)
        m.right_hand_side = standard_rhs[:]
        m.right_hand_side[m.compounds.index(species)] = 1.
        fluxes = m.solve()
        if m.lp.status == 'opt':
            reachable = True
            reaching_fluxes = {r: f for r,f in fluxes.iteritems()
                               if np.abs(f) > zero_threshold}
        elif m.lp.status != 'nofeas':
            reachable = 'error'
        
        m.set_all_flux_bounds(closed_bounds)
        # Try creation
        creation = False
        creation_min_flux = 0.
        creation_fluxes = {}
        m.right_hand_side = creation_rhs[:]
        m.right_hand_side[m.compounds.index(species)] = 1.
        fluxes = m.solve()
        if m.lp.status == 'opt':
            creation = True
            creation_min_flux = m.lp.obj.value
            creation_fluxes = {r: f for r,f in fluxes.iteritems()
                               if np.abs(f) > zero_threshold}
        elif m.lp.status != 'nofeas':
            creation = 'error'
        m.right_hand_side[m.compounds.index(species)] = (0., None)

        # Try destruction
        destruction = False
        destruction_min_flux = 0.
        destruction_fluxes = {}
        m.right_hand_side = destruction_rhs[:]
        m.right_hand_side[m.compounds.index(species)] = -1.
        fluxes = m.solve()
        if m.lp.status == 'opt':
            destruction = True
            destruction_min_flux = m.lp.obj.value
            destruction_fluxes = {r: f for r,f in fluxes.iteritems()
                                  if np.abs(f) > zero_threshold}
        elif m.lp.status != 'nofeas':
            destruction = 'error'
        m.right_hand_side[m.compounds.index(species)] = (None, 0.) 

        result_queue.put({species: (reachable, reaching_fluxes,
                                     creation, creation_min_flux, creation_fluxes,
                                     destruction, destruction_min_flux, 
                                     destruction_fluxes,)})

