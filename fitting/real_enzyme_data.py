""" Load experimental enzyme data sets for use with the (reduced) two-cell model.

Provides:

data_and_error(...): returns data, error as dicts whose keys are enzyme names
    (not reactions in the model) and values are arrays of data or uncertainties

enzymes_to_variables: dictionary mapping keys in the data to tuples of variables
    in the model. For speed, this is hard-coded rather than regenerated every
    time the module is reloaded.

find_reactions(): method which generates the mapping from keys
    in the data to variables in the model, for use in updating 
    the enzymes_to_variables attribute after a change in the model
    structure, for example.

"""
import pickle
import numpy as np
import reduced_model

########################################
# LOAD DATA
data_filename = 'data/enzyme_means.csv' #tab-delimited,
                                        #not cell-type-specific
                                        #labels based on enzyme assay labels
enzyme_data_array = np.genfromtxt(data_filename,delimiter='\t',
                                usecols=range(1,16))
enzyme_ids = np.genfromtxt(data_filename,delimiter='\t',
                         dtype=None,
                         usecols=[0])
enzyme_data = dict(zip(enzyme_ids, enzyme_data_array))

std_filename = 'data/enzyme_stds.csv'
enzyme_std_array =  np.genfromtxt(std_filename,delimiter='\t',
                                usecols=range(1,16))
std_enzyme_ids = np.genfromtxt(std_filename,delimiter='\t',
                         dtype=None,
                         usecols=[0])
enzyme_std = dict(zip(std_enzyme_ids, enzyme_std_array))

########################################
# PROCESS AND RETURN IT AS NECESSARY
def data_and_error(n_points=15, min_absolute_error=0.,
                   min_relative_error=0., scale=1.):
    """Return enyzme data and error after rescaling.

    If scale is 1.0, the data have units of nmol/min/g FW.

    Note that the minimum absolute error is enforced after rescaling.

    """

    return_data = {k: scale*v[:n_points] for k,v in enzyme_data.iteritems()}

    return_error = {}
    for k in return_data:
        v = enzyme_std[k][:n_points]
        # Careful: compare the scaled v to the
        # minumum relative error times the scaled 
        # return_data
        v = np.fmax(scale*v, min_relative_error*return_data[k])
        v = np.fmax(min_absolute_error, v)
        return_error[k] = v

    return return_data, return_error

########################################
# ESTABLISH CORRESPONDENCE BETWEEN ENZYMES AND REACTIONS
#
# We provide a dictionary mapping enzyme labels to reactions (or, in
# some cases, kinetic law variables), a dictionary mapping enzymes to
# CornCyc frames, a dictionary of special cases, and the method which
# may be used to regenerate the former from the latter two. (Users are
# on their own as far as looking up all the relevant frame names in
# CornCyc goes.)

def parent(r):
    return r.notes.get('PARENT_FRAME','')

def children(frame, model):
    return {r.id for r in model.reactions if parent(r) == frame}

enzymes_to_frames = {
    'gapdh_nadp': ('1.2.1.13-RXN',), 
    'gapdh_nad': ('GAPOXNPHOSPHN-RXN',),
    'udp_glucose_pyrophosphorylase': ('GLUC1PURIDYLTRANS-RXN',),
    'fbp_aldolase': ('F16ALDOLASE-RXN',),
    'alanine_aminotransferase': ('ALANINE-AMINOTRANSFERASE-RXN', 'RXN-13698',),
    'glutamate_dh_nad': ('GLUTAMATE-DEHYDROGENASE-RXN',),
    'transketolase': ('1TRANSKETO-RXN', '2TRANSKETO-RXN'),
    'phosphoglucomutase': ('PHOSPHOGLUCMUT-RXN',),
    'aspartate_aminotransferase': ('RXN-13697', 'ASPAMINOTRANS-RXN'),
    'phosphoglycerokinase': ('PHOSGLYPHOS-RXN',),
    'malate_dh_nad': ('MALATE-DEH-RXN',),
    'triose_phosphate_isomerase': ('TRIOSEPISOMERIZATION-RXN',),
    'phosphofructokinase_atp': ('6PFRUCTPHOS-RXN',),
    'phosphoglucose_isomerase_total': ('PGLUCISOM-RXN', 'RXN-13720', 'RXN-6182'),    
    'glycerokinase': ('GLYCEROL-KIN-RXN',),
    'malate_dh_nadp_total': ('MALATE-DEHYDROGENASE-NADP+-RXN',),
}

special_cases = {
    'pep_carboxylase': ('ms_active_pepc', 'bs_active_pepc'),
    'rubisco_maximal': ('ms_active_rubisco', 'bs_active_rubisco'),
    'phosphoglucose_isomerase_cytosolic': ('bs_PGLUCISOM_RXN',
                                           'ms_PGLUCISOM_RXN',
                                           'ms_RXN_6182',
                                           'bs_RXN_6182')
}
# Note that a more rigorous treatment of phosphoglucose isomerase
# would probably subtract the cytosolic data from the total data 
# to estimate a bound on the chloroplastic rates.

# Currently, we've left out the following channels from the
# enzyme assay data:
# rubisco_initial (not sure how to incorporate it)
# malate_dh_nadp_initial (not sure how to incorporate it)
# (Also, though an entry is provided for it in the dictionary,
# glycerokinase; GLYCEROL-KIN-RXN is dropped from the (full) model in
# the phase of removing blocked reactions.)

def find_reactions():
    import reduced_model as rm
    base_model = rm.model()
    enzyme_to_variables = {}
    enzyme_to_variables.update(special_cases)
    for enzyme, frames in enzymes_to_frames.iteritems():
        variables = []
        for frame in frames:
            variables += children(frame, base_model)
        enzyme_to_variables[enzyme]=tuple(variables)
    return enzyme_to_variables

enzyme_to_variables = {
    'alanine_aminotransferase': ('bs_ALANINE_AMINOTRANSFERASE_RXN',
                                 'ms_ALANINE_AMINOTRANSFERASE_RXN'),
    'aspartate_aminotransferase': ('ms_ASPAMINOTRANS_RXN',
                                   'bs_ASPAMINOTRANS_RXN'),
    'fbp_aldolase': ('ms_F16ALDOLASE_RXN',
                     'bs_F16ALDOLASE_RXN',
                     'bs_F16ALDOLASE_RXN_chloroplast',
                     'ms_F16ALDOLASE_RXN_chloroplast'),
    'gapdh_nad': ('bs_GAPOXNPHOSPHN_RXN', 'ms_GAPOXNPHOSPHN_RXN'),
    'gapdh_nadp': ('bs_EC_1_2_1_13_chloroplast', 'ms_EC_1_2_1_13_chloroplast'),
    'glutamate_dh_nad': ('bs_GLUTAMATE_DEHYDROGENASE_RXN',
                         'ms_GLUTAMATE_DEHYDROGENASE_RXN'),
    'glycerokinase': (),
    'malate_dh_nad': ('ms_MALATE_DEH_RXN',
                      'bs_MALATE_DEH_RXN_peroxisome',
                      'bs_MALATE_DEH_RXN',
                      'ms_MALATE_DEH_RXN_mitochondrion',
                      'bs_MALATE_DEH_RXN_mitochondrion',
                      'ms_MALATE_DEH_RXN_peroxisome'),
    'malate_dh_nadp_total': ('bs_MALATE_DEHYDROGENASE_NADP__RXN_chloroplast',
                             'ms_MALATE_DEHYDROGENASE_NADP__RXN_chloroplast'),
    'pep_carboxylase': ('ms_active_pepc', 'bs_active_pepc'),
    'phosphofructokinase_atp': ('bs__6PFRUCTPHOS_RXN', 'ms__6PFRUCTPHOS_RXN'),
    'phosphoglucomutase': ('ms_PHOSPHOGLUCMUT_RXN_chloroplast',
                           'bs_PHOSPHOGLUCMUT_RXN_chloroplast',
                           'bs_PHOSPHOGLUCMUT_RXN',
                           'ms_PHOSPHOGLUCMUT_RXN'),
    'phosphoglucose_isomerase_total': ('bs_PGLUCISOM_RXN',
                                       'bs_PGLUCISOM_RXN_chloroplast',
                                       'ms_PGLUCISOM_RXN_chloroplast',
                                       'ms_PGLUCISOM_RXN',
                                       'ms_RXN_6182',
                                       'bs_RXN_6182'),
    'phosphoglycerokinase': ('ms_PHOSGLYPHOS_RXN_chloroplast',
                             'bs_PHOSGLYPHOS_RXN_chloroplast',
                             'ms_PHOSGLYPHOS_RXN',
                             'bs_PHOSGLYPHOS_RXN'),
    'rubisco_maximal': ('ms_active_rubisco', 'bs_active_rubisco'),
    'transketolase': ('bs__1TRANSKETO_RXN_chloroplast',
                      'ms__1TRANSKETO_RXN',
                      'bs__1TRANSKETO_RXN',
                      'ms__1TRANSKETO_RXN_chloroplast',
                      'bs__2TRANSKETO_RXN',
                      'bs__2TRANSKETO_RXN_chloroplast',
                      'ms__2TRANSKETO_RXN',
                      'ms__2TRANSKETO_RXN_chloroplast'),
    'triose_phosphate_isomerase': ('ms_TRIOSEPISOMERIZATION_RXN_chloroplast',
                                   'bs_TRIOSEPISOMERIZATION_RXN_chloroplast',
                                   'ms_TRIOSEPISOMERIZATION_RXN',
                                   'bs_TRIOSEPISOMERIZATION_RXN'),
    'udp_glucose_pyrophosphorylase': ('ms_GLUC1PURIDYLTRANS_RXN',
                                      'bs_GLUC1PURIDYLTRANS_RXN')}

