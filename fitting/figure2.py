"""
Generate classical model/IPOPT result comparison for figure 2.

"""
import pickle
import numpy as np

# Classical results
from classical_c4_model import C4Model
# Case 1: A vs mesophyll CO2 level for varying Vpmax values
vp_range = [110., 90., 70., 50., 30.]
classical_crange = np.linspace(0,450,1000)
# We set rm and rd to zero rather than enforcing a minimal
# level of non-photorespiratory respiration in the FBA model.
model1 = C4Model(vpr=120.,rm=0.,rd=0.)
case1_classic = []
for vp in vp_range:
    model1.parameters['vpmax'] = vp
    case1_classic.append(model1.A(cm=classical_crange, force_enzyme_limited=True))

# Case 2: A vs maximum rubisco regeneration for varying Vcmax values
vc_range = [70., 60., 50., 40., 30.]
classical_vprrange = np.linspace(0,100,1000)
model2 = C4Model(vpr=120.,rm=0.,rd=0.)
model2.parameters['cm'] = 300.
case2_classic = [] 
for vc in vc_range:
    model2.parameters['vcmax'] = vc
    case2_classic.append(model2.A(vpr=classical_vprrange, force_enzyme_limited=True))

# IPOPT results
import reduced_model as rd
import fluxtools.nlcm as nlcm
nlcm.default_ipopt_options = {
    'print_level': 5,
    'tol': 1e-5,
    'linear_solver': 'ma97',
    'ma97_print_level': -1,
    'max_iter': 500
}

# In both cases, the model's configuration and parameters differ from
# those used for fitting; set up both models and adjust them appropriately:
ipopt_model1 = rd.model(free_biomass=False,max_light=2000.)
ipopt_model2 = rd.model(free_biomass=False,max_light=2000.)
# Use the generic C4 conductivities from von Caemmerer 2000, not the experimental
# maize conductivities we use for data-fitting
conductivities = {'bs_CO2_conductivity': 3e-3/rd.co2_scaling_factor, 
                  'bs_O2_conductivity': 0.047*3e-3/rd.o2_scaling_factor}
for m in (ipopt_model1, ipopt_model2):
    m.set_bound('ms_rubisco_vcmax',0.)
    m.set_bound('bs_rubisco_vcmax',60.)
    m.set_bound('ms_pepc_vmax',120.)
    m.set_bounds(conductivities)
    # Allow assimilated carbon to be partitioned into biomass or sucrose
    m.do_not_conserve('SUCROSE_phloem')
    m.set_bound('bs_tx_SUCROSE',(None, 0.)) # export only
    # Note that the classical model does not anticipate any O2
    # consumption in the bundle sheath; to ensure we are modeling the
    # same problem, we need to keep the BS O2 concentration higher
    # than the mesophyll, otherwise (eg) flux through the Mehler
    # reaction can perturb the result a little.
    m.set_bound('bs_oxygen',(20., None))

# Case 1: A vs mesophyll CO2 level, varying Vpmax
ipopt_crange = rd.co2_scaling_factor * np.arange(15.,465,15)
case1_ipopt = []
for vp in vp_range:
    ipopt_model1.set_bound('ms_pepc_vmax',vp)
    a = []
    for cm in ipopt_crange:
        ipopt_model1.parameters['ms_CO2'] = cm
        ipopt_model1.parameters['ms_CO2_chloroplast'] = cm
        ipopt_model1.solve()
        a.append(-1.0*ipopt_model1.obj_value)
    case1_ipopt.append(np.array(a))

# Case 2: A vs maximum rubisco regeneration for varying Vcmax values
#
# Here we control the rubisco regeneration rate by an upper bound 
# on the total decarboxylation rate, but the same effect could be achieved
# in various ways.
# First we ensure that only chloroplastic NADP-ME is active by inactivating
# all the other potential decarboxylases (urease is included because it carries
# flux in some of these poorly constrained FVA problem solutions, but it's not
# clear if it can be part of a functional carbon concentrating mechanism in 
# the model.) 
decarbs = ['bs_MALIC_NADP_RXN', 'bs_PEPCARBOXYKIN_RXN', 'bs_EC_1_1_1_39',
           'bs_MALIC_NADP_RXN_chloroplast', 'bs_UREASE_RXN']
ipopt_model2.set_bounds(dict.fromkeys(decarbs, 0.))
# Also, the classical model doesn't account for CO2 consumption/release
# by biomass production; this isn't relevant above, but leads to modest
# discrepancies here, so we partition all carbon into sucrose instead
for tag in ('ms_','bs_'):
    ipopt_model2.set_bound(tag + 'CombinedBiomassReaction', 0.)

ipopt_vprrange = np.arange(5,105.,5.)
case2_ipopt = []
for vc in vc_range:
    ipopt_model2.set_bound('bs_rubisco_vcmax',vc)
    a = []
    for vpr in ipopt_vprrange:
        ipopt_model2.set_bound('bs_MALIC_NADP_RXN_chloroplast',(0., vpr))
        ipopt_model2.solve()
        a.append(-1.0*ipopt_model2.obj_value)
    case2_ipopt.append(np.array(a))

with open('figure2_data.pickle','w') as f:
    pickle.dump({'case1_classic': case1_classic,
                 'case2_classic': case2_classic,
                 'case1_ipopt': case1_ipopt,
                 'case2_ipopt': case2_ipopt,
                 'vp_range': vp_range,
                 'vc_range': vc_range,
                 'classical_vprrange': classical_vprrange,
                 'ipopt_vprrange': ipopt_vprrange,
                 'classical_crange': classical_crange,
                 'ipopt_crange': ipopt_crange}, 
                f)
                
