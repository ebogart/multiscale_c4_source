"""Extract reaction stoichiometries from a .sol file.

Reads the .sol output file from a Pathway Tools FBA calculation
(specified as first argument) and writes an easier-to-work-with table
describing the FBA model (in a file specified as second argument,)
interpreting the .sol file in the context of the PGDB whose id is
given as the third argument.

All reactions will be read, probably including suggested try-reactions
from a file produced in model development mode; I have not tested
this.

The first part of this process is straightforward; after that, though,
the script tries to partially reverse the Pathway Tools FBA model
generation process to translate the reactions in the export file,
presented in terms of their components' common names, to a
presentation in terms of their components frame labels, that is,

    {'H2O': -1.0, 'D-ribose 5-phosphate': 1.0, 'trans-zeatin': 1.0, 
    'trans-zeatin riboside monophosphate': -1.0}

vs

    {'CPD-4206': -1.0, 'CPD-4210': 1.0, 'WATER': -1.0, 'RIBOSE-5P': 1.0}

All sorts of special cases are encountered in this process and the
code is something of a mess as I did not originally anticipate them
and have added additional complexity in an ad hoc way rather than
comprehensively rewriting the script, which was never intended to be
used more than once or twice, as is so often the case.

Where corresponding frames could not be determined for all 
susbstrates of a reaction, it is not written to the output file;
a message is printed.

All reactions resulting from the polymerization reaction
instantiation process (all those whose name in the .sol
file contains 'POLYMER') are also excluded, with a message.

Reactions with null effective stoichiometry (e.g. 'A -> A')
will be excluded with a message.

The output file includes a line for each reaction in the .sol file,
giving as tab-delimited fields 
    - the reaction's name in the .sol file,
    - the reaction frame in the original PGDB with which it appears to be
      associated,
    - a string representation of a dictionary giving the reaction's 
      stoichiometry in terms of species frames, 
    - a Boolean flag indicating whether the reaction
      is presented as reversible in the .sol file

Note that the directionality of the reactions as presented
may differ from their usual representation.

"""

import sys, re, logging
import numpy as np
import pycyc
import table_utilities

input_file = sys.argv[1]
output_file = sys.argv[2]
target_db = sys.argv[3]

logging.basicConfig(
    filename='%s_parse_log.txt' % output_file,
    level=logging.DEBUG)

class ParsingException(Exception):
    pass

db = pycyc.open(target_db)
reaction_frames = map(str, db.reactions)

# Each reaction is described on a line of the form 
# Flux:   0.063125        (PEPDEPHOS-RXN) phosphoenolpyruvate + ADP + H+  ->  pyruvate + ATP
# or 
# (RXN-11811 *spontaneous*)       ammonium  ->  ammonia + H+

line_pattern = re.compile(r'^(Flux:\s+[\d.]+\s+)?\(([^\)]+)\)\s+(.*)\s->\s(.*)$')
records = []
with open(input_file) as f:
    for line in f:
        match = line_pattern.match(line)
        if match:
            records.append((match.group(2), match.group(3), match.group(4)))

# For each reaction, determine:
# - its parent frame
# - its instantiation status
# - its spontaneity status
# - its stoichiometry
# - its reversibility 
# Not all of this information is currently used.

parents = {}
instantiated_reactions = []
spontaneous_reactions = []
stoichiometries = {}
reversible_reactions = []

reactant_pattern = re.compile(r'(-?[\d\.]+\s)?\s*(.*[^-\d\.]+.*)')

def extract_stoichiometry(one_side):
    # Stoichiometries are initially extracted as ordered lists,
    # not dictionaries, to allow us later to spot cases where
    # different frames may appear on left and right sides 
    # under the same name. 
    s = []
    reactants = one_side.split(' + ')
    reactants = [r.strip() for r in reactants]
    for r in reactants:
        try:
            coefficient, species = reactant_pattern.match(r).groups()
        except TypeError:
            raise ParsingException('Malformed reactant %s' % r)
        if coefficient:
            coefficient = float(coefficient)
        else:
            coefficient = 1.
        s.append((species,coefficient))
    return s

def reverse(tuple_list):
    return [(k, -1.0*v) for k,v in tuple_list]

for reaction_block, left, right in records:
    l = reaction_block.split()
    tag = l[0]
    if '*spontaneous*' in l:
        spontaneous_reactions.append(tag)
    if '*instantiated*' in l:
        instantiated_reactions.append(tag)
        # Instantiated reactions look like:
        # RXN0-6369-NITRATE/CPD-9956//NITRITE/UBIQUINONE-8/WATER.45
        # If we have eg RXN-1234-CPD-5678/... it is not obvious where
        # to split the string to obtain the base reaction frame id,
        # so assume it is the first '-'-delimited substring 
        # which is a reaction frame.
        segments = tag.split('/')[0].split('-')
        possible_reactions = ['-'.join(segments[:i]) for i in
                                       range(1,len(segments))]
        for possible_reaction in possible_reactions:
            if possible_reaction in reaction_frames:
                parents[tag] = possible_reaction
        if tag not in parents:
            raise Exception('Failed to identify base reaction of %s' % tag)
    else:
        parents[tag] = tag
    
    stoichiometry = extract_stoichiometry(right)
    for k,v in reverse(extract_stoichiometry(left)):
        stoichiometry.append((k,v))
    
    if tag in stoichiometries:
        if sorted(stoichiometries[tag]) != sorted(reverse(stoichiometry)):
            logging.debug('Inconsistent stoichiometries for %s:' % tag)
            logging.debug(stoichiometries[tag])
            logging.debug(reverse(stoichiometry))
            raise Exception('Inconsistent stoichiometries for %s' % tag)
        else:
            reversible_reactions.append(tag)
    else:
        stoichiometries[tag] = stoichiometry

############################
# For comparison with AraMeta and other purposes, we want everything
# translated back to frame ids. We can look up the frame associated
# with each name in the .sol file stoichiometry, but determining which
# frame in the database has a particular name is time-consuming, so
# purely to speed up the table generation process, in straightforward
# cases we can look up a reaction's stoichiometry directly from
# Pathway Tools:

def get_coefficient(reaction, compound, slot):
    """ Get the coefficient of a species in a rxn's left/right slot.

    If the species' COEFFICIENT annotation is NIL/None, return 1.

    The compound is not checked to ensure it is actually a value of the 
    slot.
    
    """
    coefficient = db.get_value_annot(reaction, slot, compound,
                                        'COEFFICIENT')
    # In some cases the coefficient is a non-frame LISP symbol, eg |n|, which
    # is parsed as a frame; try to catch those and return a string instead
    # (anticipating an exception later when this is cast to a float)
    if coefficient is None: 
        return 1
    elif isinstance(coefficient, int) or isinstance(coefficient, float):
        return coefficient
    else: 
        return str(coefficient)

def get_left_right_stoichiometries(r1):
    """Get the stoichiometries of the left and right sides of a reaction. 

    Returns sorted lists of (compound_string, coefficient) tuples for
    left and right sides respectively.

    """
    r1_left = [(str(c), get_coefficient(r1,c,'left')) for c in 
               db[r1].slot_values('left')]
    r1_left.sort()
    r1_right = [(str(c), get_coefficient(r1,c,'right')) for c in 
                db[r1].slot_values('right')]
    r1_right.sort()
    
    return tuple(r1_left), tuple(r1_right)

def direct_stoichiometry(r, direction_reference = None):
    """Determine a reaction frame's stoichiometry directly from ptools.

    In the context of interpreting an FBA output file this is
    complicated by the fact that we want the direction of the
    stoichiometry (ie sign of the coefficients) to match that shown in
    the .sol file, if only one direction is. Probably we could just
    consult the REACTION-DIRECTION slot of the reaction frame to
    determine whether it will be reversible in the FBA model or not,
    but to be cautious, we can instead compare the default direction
    (LEFT slot to RIGHT slot) to the presentation in the .sol file
    (the argument direction_reference) by generating a new
    stoichiometry in terms of the common names of the reactant frames.

    """

    left, right = get_left_right_stoichiometries(r)
    direct = sorted(reverse(left) + list(right))
    if direction_reference:
        # Apparently the Pathway Tools FBA module sometimes 
        # uses the 'N-NAME' slot.
        l1 = [(db.get_name_string(s,strip_html=True), coefficient) for
             s, coefficient in direct]
        l2 = [(db.get_name_string(s,strip_html=True,
                                  name_slot='N-NAME'), coefficient)
              for s, coefficient in direct]
        for l in [l1, l2]:
            l.sort()
            if sorted(direction_reference) == l:
                return direct
            elif sorted(direction_reference) == reverse(l):
                return reverse(direct)
        # If none of the above works...
        raise ParsingException('Unable to determine direction of %s' % 
                               r) 
    else:
        return direct

# Because looking up frames by name string is time-consuming,
# we cache the results, with None indicating an error:
_frame_cache = {}

def species_to_frame(s, reaction_context=None, instantiated=False):
    """Given the name of a species, determine its frame.

    In some cases, the name may be associated with more than one frame
    in the db. The function will try to resolve this ambiguity by
    requiring the frame to be a reactant in a particular reaction, if
    reaction_context is given, or a subclass or instance of such a
    reactant, if (in addition) instantiated is True.

    If the matching frame may not be determined with confidence
    (there is none, or more than one), return None.

    """
    try:
        try:
            l = _frame_cache[s]
        except KeyError:
            l = map(str, db.get_frame_labeled(s))
            _frame_cache[s] = l
    except pycyc.PathwayToolsError:
        _frame_cache[s] = None
        l = None
    if l is None:
        return None
    if len(l) > 1:
        logging.debug('Matching %s' % s)
        logging.debug('Found %s' % (' '.join(l)))
        if reaction_context:
            if instantiated:
                expected = set()
                for substrate in db[reaction_context].substrates:
                    expected.add(str(substrate))
                    if db.is_class(substrate):
                        instances = (db.get_class_all_instances(substrate) +
                                     db.get_class_all_instances(substrate))
                        expected.update(map(str,instances))                                    
            else:
                expected = map(str, db[reaction_context].substrates)
            logging.debug('Candidates: ')
            logging.debug(expected)
            matches = [s for s in l if s in expected]
            if len(matches) == 1:
                logging.debug('Chosen: ')
                logging.debug(matches[0])
                return matches[0]
            else: 
                logging.warning('Unresolvable ambiguity: %s, %s' % (s, reaction_context))
                return None
        else:
            return None
    return l[0]

# In a few cases reactions in the PGDB have specified compartments
# for one or more of their reactants, which are specified
# in brackets after the name in the .sol file; we need to detect these,
# strip them before trying to find the underlying name, add 
# a compartment tag back to the frame ID, and make a note 
# of this for export. 
# We don't want to assume that _any_ species name ending with 
# a substring enclosed in square brackets is necessarily
# compartmentalized, though in practice this may be true.
compartment_pattern = re.compile(r'(.*\S+)\[(\S+)\]$')
# Names which cannot be matched to a frame (more general 
# than 'ambiguities', above)
frame_matching_failures = [] 
def compartmentalized_species_to_frame(s, **kwargs):
    """ Find frame matching a species name with compartment tag. 

    Returns the frame (as string), with a compartment
    label appended if appropriate, or None if none found, and 
    a boolean which is true if the argument has successfully
    been interpreted as a species name plus compartment label.

    """
    frame = species_to_frame(s, **kwargs)
    compartment_specific = False
    if frame is None:
        # In some cases we have compound names with compartment labels
        # appended, e.g. 'nitrite[out]'; we have to preserve this
        # distinction, so we map this string to a frame name
        # with a compartment label attached, if possible.
        match = compartment_pattern.search(s)
        if match:
            frame = species_to_frame(match.group(1), **kwargs)
            if frame is None:
                frame_matching_failures.append(s)
            else:    
                compartment_specific = True
                frame += '_%s' % match.group(2)
        else:
            frame_matching_failures.append(s)
    return frame, compartment_specific

# Finally, loop over all the stoichiometries extracted from the 
# .sol file and translate each one in turn, noting which reactions
# have reactants which cannot be translated to frames, and which
# have reactants with compartment labels.
frame_stoichiometries = {}
reactions_with_invalid_frames = set()
compartment_specific_reactions = set()
total = len(stoichiometries)
for i, (reaction, stoichiometry) in enumerate(stoichiometries.iteritems()):
    logging.debug('Processing %d/%d (%s)' % (i,total,reaction))
    compartmentalized = any([compartment_pattern.search(s[0]) for 
                             s in stoichiometry])
    instantiated = reaction in instantiated_reactions
    new_stoichiometry = []
    if not compartmentalized and not instantiated:
        new_stoichiometry = direct_stoichiometry(reaction,stoichiometry)
    if not new_stoichiometry:
        for species, coefficient in stoichiometry:
            kwargs = {'s': species,
                      'reaction_context': parents[reaction],
                      'instantiated': reaction in instantiated_reactions}
            t =  compartmentalized_species_to_frame(**kwargs)
            frame, compartment_specific = t
            if frame is None:
                reactions_with_invalid_frames.add(reaction)
                frame = species
            if compartment_specific:
                compartment_specific_reactions.add(reaction)
            new_stoichiometry.append((frame, coefficient))
    frame_stoichiometries[reaction] = new_stoichiometry

#
#################
# Finally, export the table, excluding polymerization, reactions
# with invalid frames.

def tuples_to_dict(tuple_list):
    d = {}
    for k,v in tuple_list:
        d[k] = d.get(k,0.) + v
    d = {k:v for k,v in d.iteritems() if
                     np.abs(v) > 1e-10}
    return d

export_stoichiometries = {}
for tag, s in frame_stoichiometries.iteritems():
    if 'POLYMER' in tag: 
        print 'Excluding polymerization reaction %s' % tag
    elif tag in reactions_with_invalid_frames:
        print 'Excluding reaction %s as not all reactants could be mapped to frames.'
    else:
        effective_stoichiometry = tuples_to_dict(s)
        if not effective_stoichiometry:
            print 'Excluding reaction %s (null stoichiometry)'
        else:
            export_stoichiometries[tag] = effective_stoichiometry

export_reversibilities = {tag: tag in reversible_reactions for tag
                          in frame_stoichiometries}
table_utilities.write_reaction_table(
    output_file,
    export_stoichiometries, 
    parents,
    export_reversibilities)
