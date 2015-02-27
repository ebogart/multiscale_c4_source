""" 
Load data and draw figures. 

"""
import pickle
import numpy as np
import matplotlib as mpl
mpl.rc_file('figures_rc')
mpl.use('GTK3Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

with open('gradient_fit.pickle') as f:
    result = pickle.load(f)
traj = result['traj']

with open('key_fva_broad.pickle') as f:
    fva = pickle.load(f)

with open('proteomics/mRNA_proteomics.pickle') as f:
    mrnap = pickle.load(f)

standard_xticks = np.arange(1,16)
standard_xticklabels = ['1','','3','','5','','7','','9','','11','','13','','15']

leg_small_fontsize = 10

# Fig. 1 is an illustration without any actual data.
# Generate a placeholder.
def fig1():
    f = plt.figure()
    a = f.gca()
    a.text(0.5,0.5,'placeholder',horizontalalignment='right',
           verticalalignment='center', transform=a.transAxes,
           fontsize=14)
    return f

def fig2():
    """ Show comparison between optimization and classical results. """
    with open('figure2_data.pickle') as f:
        data = pickle.load(f)
    f, ax = plt.subplots(1,2,sharey=True)
    # Case 1: A vs ci at varying vpmax
    ax1 = ax[0]
    for a in data['case1_classic']:
        ax1.plot(data['classical_crange'],
                 a, 'b', label='_nolegend_')
    # Subsample the IPOPT results above cm=100ppm because otherwise
    # the graph is too crowded
    subset = range(8) + range(9,30,2)
    ipopt_crange = 1e3*data['ipopt_crange']
    for a, symbol, vp in zip(data['case1_ipopt'],
                             ('+','o','<','p','>'),
                             data['vp_range']):
        
        ax1.plot(ipopt_crange[subset], a[subset], 'k'+symbol,
                 fillstyle='none',
                 label='%.1f' % vp)
#    ax1.legend(loc='best')
    ax1.set_ylabel('$\mathrm{CO_2}$ assimilation ' + 
                   '($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$)')
    ax1.set_xlabel('mesophyll $\mathrm{CO_2}$ level ($\mu$mol/mol)')
    ax1.set_xticks((0.,100.,200.,300.,400.))
    ax1.set_xticks((50.,150.,250.,350.,450.),minor=True)
    ax1.set_ylim((0.,70.))        
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    # Case 2: A vs vpr_max at varying vcmax    
    ax2 = ax[1]
    for a in data['case2_classic']:
        ax2.plot(data['classical_vprrange'],
                 a, 'b', label='_nolegend_')
    # All the curves run together at the low end, so truncate some of the
    # datasets near that end to avoid crowding
    starting_indices = [0,6,5,4,3]
    for a, symbol, vc, start in zip(data['case2_ipopt'],
                                    ('x','s','v','*','^'),
                                    data['vc_range'],
                                    starting_indices):
        ax2.plot(data['ipopt_vprrange'][start:], a[start:], 'k'+symbol, 
                 fillstyle='none',
                 label='%.1f' % vc)
#    ax2.legend(loc='best')
    ax2.set_xlabel('maximum decarboxylation rate\n' + 
                 '($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$)')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')


    ax1.text(1.0,1.0,'a',horizontalalignment='right',
           verticalalignment='top', transform=ax1.transAxes,
           fontsize=14)
    ax2.text(1.0,1.0,'b',horizontalalignment='right',
           verticalalignment='top', transform=ax2.transAxes,
           fontsize=14)


    f.tight_layout()
    return f

def reset_fontsizes(value):
    for param in ('font.size', 'axes.labelsize', 
                  'xtick.labelsize', 'ytick.labelsize'):
        mpl.rcParams[param] = value

def fig3():
    """ 
    Illustrate the source-sink transition. 
    
    The second subfigure is left blank so I can paste in the 
    the 14C labeling results.

    """
    # This figure is different in size; adjust the font accordingly
    old_fontsize = mpl.rcParams['font.size']
    reset_fontsizes(9)
    # To match the width of the axes the image needs to be ~2.1in high.
    f, ax = plt.subplots(3,sharex=True,figsize=(5.19,1.7*4.05))
    a,b,c = ax
    
    x = np.arange(0.5,15.5)
    a.set_ylabel('carbon uptake\n' + 
                 '$\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.spines['bottom'].set_position(('data',0)) 
    a.spines['bottom'].set_zorder(-10)  
    # set_position calls reset_ticks on the x-axis, which may be why 
    # we need to set labelbottom=False _below_.
    a.xaxis.set_ticks_position('bottom')
    a.tick_params(axis='x',direction='inout', labelbottom=False, )
    a.yaxis.set_ticks_position('left')
    a.plot(x,12*traj['bs_tx_SUCROSE'],'r-o',label='sucrose')
    a.plot(x,12*fva['bs_tx_SUCROSE'][0],'r:',label='_nolegend_')
    a.plot(x,12*fva['bs_tx_SUCROSE'][1],'r:',label='_nolegend_')
    a.plot(x,traj['ms_tx_CARBON_DIOXIDE'],'b-o',
           label='$\mathrm{CO_2}$')
    a.plot(x,fva['ms_tx_CARBON_DIOXIDE'][0],'b:',label='_nolegend_')
    a.plot(x,fva['ms_tx_CARBON_DIOXIDE'][1],'b:',label='_nolegend_')
    a.set_xlim(0,15)
    a.set_xticks(range(1,16))
    a_legend = a.legend(bbox_to_anchor=(0.433,0.25),ncol=2)
    a.text(1.0,1.0,'a',horizontalalignment='right',
           verticalalignment='top', transform=a.transAxes,
           fontsize=14)
# issues: 
# legend text vertical alignment is off
# can we load the image with mpl and still save a good PDF?

#    b.plot(x,np.zeros(15))
    b.set_yticks([])
    b.xaxis.set_ticks_position('none')
    for spine in b.spines.values():
        spine.set_color('none')
    b.text(1.0,1.0,'b',horizontalalignment='right',
           verticalalignment='top', transform=b.transAxes,
           fontsize=14)

    c.plot(x,traj['total_production_all_biomass'],'b-o')
    c.plot(x,fva['total_production_all_biomass'][0],'b:')
    c.plot(x,fva['total_production_all_biomass'][1],'b:')
    c.set_ylabel('biomass production\n$\mathrm{mg}$ $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    c.set_xlabel('cm from leaf base')
    c.spines['right'].set_color('none')
    c.spines['top'].set_color('none')
    c.xaxis.set_ticks_position('bottom')
    c.yaxis.set_ticks_position('left')
    c.set_ylim(0,0.25)
    c.set_yticks(np.linspace(0,0.25,6))
    c.text(1.0,1.0,'c',horizontalalignment='right',
           verticalalignment='top', transform=c.transAxes,
           fontsize=14)

    f.tight_layout()
    reset_fontsizes(old_fontsize)
    return f

def add_fva(axes,x,key,colorletter,rescale=1., symbol=':'):
    axes.plot(x,rescale*fva[key][0],colorletter+symbol,label='_nolegend_')
    axes.plot(x,rescale*fva[key][1],colorletter+symbol,label='_nolegend_')

def fig4():
    """ 
    Illustrate the operation of the C4 cycle. 

    """
    f = plt.figure()
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 0, 'borderaxespad': 0.6}
    a = f.add_subplot(221)
    b = f.add_subplot(222, sharex=a, sharey=a)
    c = f.add_subplot(223, sharex=a)
    d = f.add_subplot(224, sharex=a)
    for axis in (a,b,c):
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')        
    x = np.arange(0.5,15.5)

    a.plot(x,traj['ms_PEPCARBOX_RXN'],'b-o',label='PEPC')
    add_fva(a,x,'ms_PEPCARBOX_RXN','b')
    a.plot(x,traj['bs_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast'],
           'g-s',label='rubisco (b)')
    add_fva(a,x,'bs_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast','g')
    a.plot(x,traj['ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast'],
           'r-^',label='rubisco (m)')
    add_fva(a,x,'ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast','r')
    a.text(0.98,0.98,'a',horizontalalignment='right',
           verticalalignment='top', transform=a.transAxes,
           fontsize=14)
    a.legend(loc='upper left', **leg_args)
    a.set_ylabel('$\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    a.set_xlabel('leaf segment (cm)')


    add_fva(b,x,'bs_MALIC_NADP_RXN_chloroplast','b')
    add_fva(b,x,'bs_PEPCARBOXYKIN_RXN','g')
    b.plot(x,traj['bs_MALIC_NADP_RXN_chloroplast'],'b-o',label='NADP-ME')
    b.plot(x,traj['bs_PEPCARBOXYKIN_RXN'],'g-s',label='PEPCK')
    b.text(0.98,0.98,'b',horizontalalignment='right',
           verticalalignment='top', transform=b.transAxes,
           fontsize=14)
    b.legend(loc='upper left', **leg_args)
    b.set_xlabel('leaf segment (cm)')


    c.text(0.98,0.98,'c',horizontalalignment='right',
           verticalalignment='top', transform=c.transAxes,
           fontsize=14)
    c.set_ylabel('$\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$',
                 labelpad=-3)
    #c.legend(loc='upper left')
    c.plot(x,traj['ms_EC_1_2_1_13_chloroplast'],'b-o',label='GAPDH (m)')
    add_fva(c,x,'ms_EC_1_2_1_13_chloroplast','b')
    c.plot(x,traj['plasmodesmata_ms_GAP_bs_GAP'],'g-s',label='GAP ex.')
    add_fva(c,x,'plasmodesmata_ms_GAP_bs_GAP','g')
    c.plot(x,traj['plasmodesmata_ms_G3P_bs_G3P'],'r-^',label='3PGA ex.')
    add_fva(c,x,'plasmodesmata_ms_G3P_bs_G3P','r')

    #c.plot(traj['bs_PHOSGLYPHOS_RXN_chloroplast'],'b'
    c.spines['bottom'].set_position(('data',0)) 
    c.spines['bottom'].set_zorder(-10) 
    c.spines['left'].set_zorder(-10) 
    # set_position calls reset_ticks on the x-axis, which may be why 
    # we need to set labelbottom=False _below_.
    c.xaxis.set_ticks_position('bottom')
    c.tick_params(axis='y',zorder=-10)
    c.tick_params(axis='x',direction='inout', labelbottom=False,zorder=-10)
    c.legend(loc='upper left', **leg_args)

    d.text(0.98,0.98,'d',horizontalalignment='right',
           verticalalignment='top', transform=d.transAxes,
           fontsize=14)
    dprime = d.twinx()
    o2_plot = dprime.plot(x,1e1*traj['bs_oxygen'],'r-s',label='$\mathrm{O_2}$')
    add_fva(dprime,x,'bs_oxygen','r',rescale=10.)
    dprime.set_ylim(0.,400.)
    dprime.set_yticks(np.linspace(0.,400.,9))
    dprime.set_ylabel('$\mathrm{O_2}$ (mbar)')

    d.plot(0.3*np.ones(16),'b-',label='_nolegend_')
    dprime.plot(200*np.ones(16),'r-',label='_nolegend_')
    co2_plot = d.plot(x,traj['bs_CO2'],'b-o',label='$\mathrm{CO_2}$')
    d.set_ylim(0., 10.)
    d.set_ylabel('$\mathrm{CO_2}$ (mbar)',labelpad=-2)
    d.set_yticks(np.linspace(0.,10.,6))
    add_fva(d,x,'bs_CO2','b')

    d_lines, d_labels = d.get_legend_handles_labels()
    dprime_lines, dprime_labels = dprime.get_legend_handles_labels()
    d.legend(d_lines + dprime_lines, d_labels + dprime_labels, 
             loc = 'upper left', **leg_args)

    a.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)
    f.tight_layout()
    return f

def fig5():
    """ 
    Illustrate the rescaling and fitting process on two examples.

    """
    f, ax = plt.subplots(2,2,sharex=True)
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    a,b,c,d = ax.flat
    for axis, label in zip(ax.flat,'abcd'):
        print axis
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')
        if label in 'bd':
            axis.text(0.98,0.98,label,horizontalalignment='right',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
        else:
            axis.text(0.03,0.98,label,horizontalalignment='left',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
        axis.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)
    x = np.arange(0.5,15.5)
    c.set_xlabel('leaf segment (cm)')
    d.set_xlabel('leaf segment (cm)')
    exp_label = 'rescaled expression data'
    flux_label = '$\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$'
    a.set_ylabel(exp_label)
    c.set_ylabel(exp_label)
    b.set_ylabel(flux_label)
    d.set_ylabel(flux_label)

    # a,b: the chlorophyllide a biosynthesis pathway
    # Note our model ignores the branch to heme production
    # (cytochromes, etc) after protoporphyrin IX.
    pathway = [('ms_UROGENDECARBOX_RXN','uroporphyrinogen decarboxylase'),
               ('ms_RXN0_1461','coproporphyrinogen oxidase'),
               ('ms_PROTOPORGENOXI_RXN', 'protoporphyrinogen oxidase'),
               ('ms_RXN1F_20','magnesium chelatase'),
               ('ms_RXN_MG_PROTOPORPHYRIN_METHYLESTER_SYN',
                'magnesium protoporphyrin IX methyltransferase'),
               ('ms_RXN_13191',
                'magnesium protoporphyrin IX monomethyl ester cyclase'),
               ('ms_RXN1F_72','divinyl chlorophyllide a 8-vinyl-reductase'),
               ('ms_RXN1F_10','protochlorophyllide reductase')]
    N = len(pathway)
    colors = [cm.jet(s) for s in np.linspace(0.,0.99,N)]
    a.set_color_cycle(colors)
    b.set_color_cycle(colors)
    for r,name in pathway:
#        print result['data'][r]
#        print result['error'][r]
#        if r in result['data']:
        a.errorbar(x,result['data'][r],yerr=result['error'][r],label=name)
    a.legend(map(str,range(1,N+1)), ncol=2, **leg_args)
    a.set_ylim(0,12)

    for (r,name),offset in zip(pathway,np.linspace(-0.1,0.1,N)):
        scale = np.exp(-1.0*result['soln']['scale_%s'%(r[3:])])
        b.errorbar(x+offset,
                   scale*result['data'][r],
                   label=name,
                   yerr=scale*result['error'][r])
    for r,name in pathway[0:1]:
        if r in result['data']:
            b.plot(x,np.abs(result['traj'][r]),'ko',linewidth=2)
    b.set_ylim(0,0.06)

    # c,d: branch point at arogenate between phenylalanine and tyrosine
    # production 

    branch =[('bs_PREPHENATE_TRANSAMINE_RXN','prephenate transaminase'),
             ('bs_RXN_5682','arogenate dehydrogenase'),
             ('bs_CARBOXYCYCLOHEXADIENYL_DEHYDRATASE_RXN','arogenate dehydratase')]
    N = len(branch)
    colors = 'kbg'
    c.set_color_cycle(colors)
    d.set_color_cycle(colors)

    for r,name in branch:
        c.errorbar(x,result['data'][r],yerr=result['error'][r],label=name)
    c.legend([t[1] for t in branch],loc='best',**leg_args)
    c.set_ylim(0,0.6)

    ebs = []
    for (r,name),offset in zip(branch,np.linspace(-0.1,0.1,N)):
        scale = np.exp(-1.0*result['soln']['scale_%s'%(r[3:])])
        eb = d.errorbar(x+offset,
                        scale*result['data'][r],
                        label=name,
                        yerr=scale*result['error'][r],
                        linestyle='-')
        ebs.append(eb)
    ebs[0][-1][0].set_alpha(0.3)
    ebs[0][-2][0].set_alpha(0.3)
    ebs[0][-2][1].set_alpha(0.3)
    for r,name in branch:
        d.plot(x,np.abs(result['traj'][r]),':o',linewidth=1)
    plt.ylim(0,0.08)

    f.tight_layout()
    return f

def error_by_rxn(result,r):
    s = 'image%d_gradient_rna_error_%s'
    return np.sum([result['soln'][s % (i,r)]**2/result['error'][r][i]**2 for 
                   i in xrange(15)])
           
def fig6():
    """ 
    Illustrate aggregate information about fit quality. 

    a - cost by image
    b - distribution of correlation coefficients (total, mesophyll,
    bundle sheath)
    (skipped) - scatter: reaction cost vs rescaled mean RNAseq data
    d/e: comparison: flux vs expression level before and after rescaling,
    at segment 14. 

    (skipped): cost (or r) vs mean flux (as platform for comparing
    to case w/o scale factors) 

    """
    f = plt.figure()
    a = f.add_subplot(221)
    x = np.arange(0.5,15.5)
    a.plot(x,1e-3*result['cost_by_image'],'-o')
    a.set_ylabel('cost/1000')
    a.set_xlabel('leaf segment (cm)')
    a.set_xlim(0,15)
    a.set_ylim(0,7.2)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)

    data_subset = {k:v for k,v in result['data'].iteritems() if k
                   in result['traj']}
    data_6 = np.array([np.sum(v) for k,v in data_subset.iteritems()])
    values_6 = np.array([np.mean(np.abs(result['traj'][k])) for
                         k in data_subset])
    error_6 = np.array([error_by_rxn(result, r) for r in data_subset])
    r_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                    k,v in data_subset.iteritems()])
    br_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('bs')])
    mr_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('ms')])

    b = f.add_subplot(222)
    b.hist(r_6,50,normed=True,cumulative=True)
    # bundle sheath and mesophyll distributions pretty much indistinguishable
    # from bulk distribution
#    b.hist(mr_6,50,normed=True,cumulative=True,alpha=0.3)
#    b.hist(br_6,50,normed=True,cumulative=True,alpha=0.3)
    b.set_xlabel('correlation coefficient $r$')
    
    # several scales have nonstandard names; ignore them for the moment
    has_scales = {k for k in data_subset if ('scale_' + k[3:]) in
                  result['soln']}
    tip_values = np.array([np.abs(result['traj'][k][14]) for k in 
                           has_scales])
    tip_data = np.array([data_subset[k][14] for k in 
                           has_scales])
    tip_scales = np.array([result['soln']['scale_' + k[3:]] for k in 
                           has_scales])

    c = f.add_subplot(223)
    c.loglog(tip_data, tip_values, 'o', alpha=0.4)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    c.set_ylabel(flux_label)
    c.set_xlabel('expression data')
    d = f.add_subplot(224, sharey=c)
    d.loglog(tip_data, np.exp(tip_scales)*tip_values, 'o',alpha=0.4)
    d.set_ylabel('flux $\cdot$ scale factor')
    # Note some low-flux outliers are excluded.
    c.set_ylim(1e-5,1e3)
    d.set_xlabel('expression data')

    for axis, label in zip((a,b,c,d),'abcd'):
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')
        if label == 'a':
            axis.text(0.98,0.98,label,horizontalalignment='right',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
        else:
            axis.text(0.05,0.98,label,horizontalalignment='left',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
 
    f.tight_layout()
    return f

def fig6_alt1():
    """ 
    Illustrate aggregate information about fit quality. 

    Here, mRNA/proteomics correlation for seg. 14 is shown overlaid
    on the rescaled flux/expression data correlation. For better 
    comparison, the RPKM data from the proteomics paper is rescaled 
    by the same overall factor (0.0049) as the expression data used
    for fitting. 

    a - cost by image
    b - distribution of correlation coefficients (total, mesophyll,
    bundle sheath)
    (skipped) - scatter: reaction cost vs rescaled mean RNAseq data
    d/e: comparison: flux vs expression level before and after rescaling,
    at segment 14. 

    (skipped): cost (or r) vs mean flux (as platform for comparing
    to case w/o scale factors) 

    """
    f = plt.figure()
    a = f.add_subplot(221)
    x = np.arange(0.5,15.5)
    a.plot(x,1e-3*result['cost_by_image'],'-o')
    a.set_ylabel('cost/1000')
    a.set_xlabel('leaf segment (cm)')
    a.set_xlim(0,15)
    a.set_ylim(0,7.2)
    a.set_xticks((1,3,5,7,9,11,13,15))

    data_subset = {k:v for k,v in result['data'].iteritems() if k
                   in result['traj']}
    data_6 = np.array([np.sum(v) for k,v in data_subset.iteritems()])
    values_6 = np.array([np.mean(np.abs(result['traj'][k])) for
                         k in data_subset])
    error_6 = np.array([error_by_rxn(result, r) for r in data_subset])
    r_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                    k,v in data_subset.iteritems()])
    br_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('bs')])
    mr_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('ms')])

    b = f.add_subplot(222)
    b.hist(r_6,50,normed=True,cumulative=True)
    # bundle sheath and mesophyll distributions pretty much indistinguishable
    # from bulk distribution
#    b.hist(mr_6,50,normed=True,cumulative=True,alpha=0.3)
#    b.hist(br_6,50,normed=True,cumulative=True,alpha=0.3)
    b.set_xlabel('correlation coefficient $r$')
    
    # several scales have nonstandard names; ignore them for the moment
    has_scales = {k for k in data_subset if ('scale_' + k[3:]) in
                  result['soln']}
    tip_values = np.array([np.abs(result['traj'][k][14]) for k in 
                           has_scales])
    tip_data = np.array([data_subset[k][14] for k in 
                           has_scales])
    tip_scales = np.array([result['soln']['scale_' + k[3:]] for k in 
                           has_scales])

    c = f.add_subplot(223)
    c.loglog(tip_data, tip_values, 'o', alpha=0.4)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    c.set_ylabel(flux_label)
    c.set_xlabel('expression data')

    d = f.add_subplot(224, sharey=c)
    d.loglog(tip_data, np.exp(tip_scales)*tip_values, 'o',alpha=0.4,
             label='flux $\cdot$ scale factor')
#    d.set_ylabel('flux $\cdot$ scale factor')
    # Note some low-flux outliers are excluded.
    c.set_ylim(1e-5,1e3)
    d.set_xlabel('expression data')

    proteomics_expression = 0.0049 * mrnap['RPKM'][:,-1]
    nsaf = mrnap['NSAF'][:,-1]
    d.loglog(proteomics_expression, nsaf, 'ro', alpha=0.75, label='NSAF')
    d.legend(bbox_to_anchor=(0.72,0.95),numpoints=1, handlelength=0, borderaxespad=0.6)
    

    for axis, label in zip((a,b,c,d),'abcd'):
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')
        if label == 'a':
            axis.text(0.98,0.98,label,horizontalalignment='right',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
        else:
            axis.text(0.05,0.98,label,horizontalalignment='left',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
 
    f.tight_layout()
    return f

def fig6_alt2():
    """Illustrate aggregate information about fit quality. 

    Here, raw and rescaled fluxes are overlaid in the comparison to
    data in panel C, and the mRNA/proteomics scatter for seg. 14 is
    shown on its own in panel d. The RPKM data
    from the proteomics paper is rescaled by the same overall factor
    (0.0049) as the expression data used for fitting.

    a - cost by image
    b - distribution of correlation coefficients (total, mesophyll,
    bundle sheath)
    (skipped) - scatter: reaction cost vs rescaled mean RNAseq data
    d/e: comparison: flux vs expression level before and after rescaling,
    at segment 14. 

    (skipped): cost (or r) vs mean flux (as platform for comparing
    to case w/o scale factors)

    """
    f = plt.figure()
    a = f.add_subplot(221)
    x = np.arange(0.5,15.5)
    a.plot(x,1e-3*result['cost_by_image'],'-o')
    a.set_ylabel('cost/1000')
    a.set_xlabel('leaf segment (cm)')
    a.set_xlim(0,15)
    a.set_ylim(0,7.2)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)

    data_subset = {k:v for k,v in result['data'].iteritems() if k
                   in result['traj']}
    data_6 = np.array([np.sum(v) for k,v in data_subset.iteritems()])
    values_6 = np.array([np.mean(np.abs(result['traj'][k])) for
                         k in data_subset])
    error_6 = np.array([error_by_rxn(result, r) for r in data_subset])
    r_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                    k,v in data_subset.iteritems()])
    br_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('bs')])
    mr_6 = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                     k,v in data_subset.iteritems() if k.startswith('ms')])

    b = f.add_subplot(222)
    b.hist(r_6,50,normed=True,cumulative=True)
    # bundle sheath and mesophyll distributions pretty much indistinguishable
    # from bulk distribution
#    b.hist(mr_6,50,normed=True,cumulative=True,alpha=0.3)
#    b.hist(br_6,50,normed=True,cumulative=True,alpha=0.3)
    b.set_xlabel('correlation coefficient $r$')
    
    # several scales have nonstandard names; ignore them for the moment
    has_scales = {k for k in data_subset if ('scale_' + k[3:]) in
                  result['soln']}
    tip_values = np.array([np.abs(result['traj'][k][14]) for k in 
                           has_scales])
    tip_data = np.array([data_subset[k][14] for k in 
                           has_scales])
    tip_scales = np.array([result['soln']['scale_' + k[3:]] for k in 
                           has_scales])

    c = f.add_subplot(223)
    c.loglog(tip_data, tip_values, 'o', alpha=0.4)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    c.set_ylabel(flux_label)
    c.set_xlabel('expression data')
    c.loglog(tip_data, np.exp(tip_scales)*tip_values, 'ro',alpha=0.75,
             label='flux $\cdot$ scale factor')
    # No legend, remember to explain this in caption
    c.set_ylim(1e-5,1e3)

    d = f.add_subplot(224)

#    d.set_ylabel('flux $\cdot$ scale factor')
    # Note some low-flux outliers are excluded.

    d.set_xlabel('expression data')
    d.set_ylabel('NSAF')
    proteomics_expression = 0.0049 * mrnap['RPKM'][:,-1]
    nsaf = mrnap['NSAF'][:,-1]
    d.loglog(proteomics_expression, nsaf, 'co', alpha=0.75, label='NSAF')
#    d.set_xlim(1e-4,1e3)
#    d.set_ylim(1e-6,1e-1)

    for axis, label in zip((a,b,c,d),'abcd'):
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')
        if label == 'a':
            axis.text(0.98,0.98,label,horizontalalignment='right',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
        else:
            axis.text(0.05,0.98,label,horizontalalignment='left',
                      verticalalignment='top', transform=axis.transAxes,
                      fontsize=14)
 
    f.tight_layout()
    return f



def fig_biomass():
    """ 
    Show variation in biomass composition

    """
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    f, ax = plt.subplots(3,sharex=True) # ,figsize=(5.19,4.05))
    a,b,c = ax
    x = np.arange(0.5,15.5)
    a.plot(x,traj['total_production_cellulose_hemicellulose'],'k-o',
           label='cellulose and hemicellulose')
    a.plot(x,traj['total_production_amino_acids'],'b-^',
           label='amino acids')
    a.plot(x,traj['total_production_nucleic_acids'],'c-v',
           label='nucleic acids')
    a.plot(x,traj['total_production_chlorophyll'],'g-p',
           label='chlorophyll')
    a.plot(x,(traj['total_production_lipids'] + 
              traj['total_production_fatty_acids']),'r-s',
           label='lipids and fatty acids')
#    a.set_ylabel('biomass production\n$\mathrm{mg}$ $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    a.plot(x,traj['total_production_other'],'m->',
           label='ascorbate')
    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.set_ylim(0,0.075)
    a.set_yticks(np.linspace(0,0.075,6))
    a.text(1.0,1.0,'a',horizontalalignment='right',
           verticalalignment='top', transform=a.transAxes,
           fontsize=14)
    a.set_xlim(0,15)
    a.legend(loc='best',ncol=2, **leg_args)

    b.plot(x,traj['total_production_suberin_precursors'],'b-*',
           label='suberin precursors')
    b.plot(x,traj['total_production_monolignols'],'g-x',
           label='monolignols')
    b.plot(x,traj['total_production_starch'],'c-+',
           label='starch')
    b.legend(loc='upper right',**leg_args)
    b.set_ylabel('biomass production\n$\mathrm{mg}$ $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    b.spines['right'].set_color('none')
    b.spines['top'].set_color('none')
    b.xaxis.set_ticks_position('bottom')
    b.yaxis.set_ticks_position('left')
    b.set_ylim(0,0.01)
    b.set_yticks(np.linspace(0,0.01,6))
    b.text(1.01,1.0,'b',horizontalalignment='right',
           verticalalignment='top', transform=b.transAxes,
           fontsize=14)

    # mesophyll vs bundle sheath starch production
    starch_sinks = ['sink_amylopectin_monomer_equivalent_chloroplast',
                    'sink_1_4_alpha_D_Glucan_monomer_equivalent_chloroplast']
    ms_starch = sum([0.16315*traj['ms_' + k] for k in starch_sinks])
    bs_starch = sum([0.16315*traj['bs_' + k] for k in starch_sinks])
    c.spines['right'].set_color('none')
    c.spines['top'].set_color('none')
    c.xaxis.set_ticks_position('bottom')
    c.yaxis.set_ticks_position('left')
    c.set_ylim(0,0.003)
    c.set_yticks(np.linspace(0,0.003,7))
    c.text(1.0,1.0,'c',horizontalalignment='right',
           verticalalignment='top', transform=c.transAxes,
           fontsize=14)
    c.plot(x,ms_starch,'b-+',label='starch (mesophyll)')
    c.plot(x,bs_starch,'g-+',label='starch (bundle sheath)')
    c.legend(loc='best',**leg_args)
    c.set_xlabel('leaf postition (cm)')

    f.tight_layout()
    return f

def fig_ps2():
    """ 
    Illustrate photosystem II activity.

    """
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    f, ax = plt.subplots(1) # ,figsize=(5.19,4.05))
    a = ax
    x = np.arange(0.5,15.5)
    a.plot(x,traj['ms_PhotosystemII_mod_chloroplast'],'b-<',
           label='photosystem II (mesophyll)')
    a.plot(x,traj['bs_PhotosystemII_mod_chloroplast'],'g->',
           label='photosystem II (bundle sheath)')
    add_fva(a,x,'ms_PhotosystemII_mod_chloroplast','b')
    add_fva(a,x,'bs_PhotosystemII_mod_chloroplast','g', symbol='-.')
    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.set_ylim(0,20)
    a.set_yticks(np.linspace(0,20,5))
    # a.text(1.0,1.0,'a',horizontalalignment='right',
    #        verticalalignment='top', transform=a.transAxes,
    #        fontsize=14)
    a.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)
    flux_label = 'flux ($\mathrm{\mu}$mol oxygen $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    a.set_ylabel(flux_label)
    a.legend(loc='best')
    a.set_xlabel('leaf postition (cm)')

    f.tight_layout()
    return f

def fig_summary(result_file, description, fixed_biomass=False):
    """ 
    Summarize properties of a fitting result other than the default.

    Note, correlation values between predicted fluxes which are zero
    everywhere and their associated data are taken to be zero.

    """ 
    with open(result_file) as f:
        other = pickle.load(f)
    other_traj = other['traj']
    figure = plt.figure()
    # This figure needs a slightly smaller size
    old_fontsize = mpl.rcParams['font.size']
    reset_fontsizes(9)

    a = figure.add_subplot(321)
    b = figure.add_subplot(322, sharex=a)
    c = figure.add_subplot(323, sharex=a)
    d = figure.add_subplot(324, sharex=a)
    e = figure.add_subplot(325, sharex=a)
    f = figure.add_subplot(326)

    leg_args = {'fontsize': 8, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}

    x = np.arange(0.5,15.5)
    a.set_xlim(0,15)
    a.set_xticks(range(1,16))
    e.set_xlabel('leaf postition (cm)')
    d.set_xlabel('leaf postition (cm)')
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'

    for ax, letter in zip((a,b,c,d,e,f),'abcdef'):
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(1.0,1.125,letter,horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize=14)

    # a. Source-sink
    a.set_ylabel('carbon uptake\n' + 
                 '$\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    a.spines['bottom'].set_position(('data',0)) 
    a.spines['bottom'].set_zorder(-10)  
    a.tick_params(axis='x',direction='inout', labelbottom=False, )
    a.plot(x,12*other_traj['bs_tx_SUCROSE'],'r-o',label='sucrose')
    a.plot(x,other_traj['ms_tx_CARBON_DIOXIDE'],'b-o',
           label='$\mathrm{CO_2}$')
    a.legend(loc='best',**leg_args)

    # b. Rubisco and pepc
    b.set_ylabel(flux_label)
    b.plot(x,other_traj['ms_PEPCARBOX_RXN'],'b-o',label='PEPC')
    b.plot(x,other_traj['bs_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast'],
           'g-s',label='rubisco (b)')
    b.plot(x,other_traj['ms_RIBULOSE_BISPHOSPHATE_CARBOXYLASE_RXN_chloroplast'],
           'r-^',label='rubisco (m)')
    b.legend(loc='best', **leg_args)
    b.tick_params(axis='x',direction='in', labelbottom=False, )

    # c. Linear pathway example
    # Note that, in the fixed biomass case, this pathway
    # is completely blocked. 
    c.set_ylabel(flux_label)
    if fixed_biomass:
        c.plot(np.zeros(15),'k-o',label='chlorophyllide A synthesis (mesophyll)')
        c.set_ylim(-1,1)
        c.set_yticks((-1,0,1))
    else:
        c.plot(other_traj['ms_RXN_13191'],'k-o',
               label='chlorophyllide A synthesis (m)')
        # Ensure we don't exaggerate varations in results which are
        # well within the solver tolerance and effectively noise
        ymin, ymax = c.get_ylim()
        if (ymax-ymin) < 1e-5:
            c.set_ylim(ymin, ymin+1e-5)
    c.legend(loc='best', **leg_args)
    c.tick_params(axis='x',direction='in', labelbottom=False, )
    c.get_yaxis().get_major_formatter().set_powerlimits((3,3))

    # d. Branch point example

    branch =[('bs_PREPHENATE_TRANSAMINE_RXN','prephenate transaminase'),
             ('bs_RXN_5682','arogenate dehydrogenase'),
             ('bs_CARBOXYCYCLOHEXADIENYL_DEHYDRATASE_RXN','arogenate dehydratase')]
    colors = 'kbg'
    symbols = 'os^'
    for (reaction, label), color, symbol in zip(branch,colors, symbols):
        d.plot(other_traj[reaction],color + '-' + symbol,label=label)
    d.legend(loc='best', **leg_args)
    d.set_ylabel(flux_label)
    d.get_yaxis().get_major_formatter().set_powerlimits((3,3))

    # e. CO2 and O2 levels
    eprime = e.twinx()
    o2_plot = eprime.plot(x,1e1*other_traj['bs_oxygen'],'r-s',label='$\mathrm{O_2}$')
#    eprime.set_ylim(0.,400.)
#    eprime.set_yticks(np.linspace(0.,400.,9))
    eprime.set_ylabel('$\mathrm{O_2}$ (mbar)')

    e.plot(0.3*np.ones(16),'b-',label='_nolegend_')
    eprime.plot(200*np.ones(16),'r-',label='_nolegend_')
    co2_plot = e.plot(x,other_traj['bs_CO2'],'b-o',label='$\mathrm{CO_2}$')
#    e.set_ylim(0., 10.)
#    e.set_yticks(np.linspace(0.,10.,6))
    e.set_ylabel('$\mathrm{CO_2}$ (mbar)')

    e_lines, e_labels = e.get_legend_handles_labels()
    eprime_lines, eprime_labels = eprime.get_legend_handles_labels()
    e.legend(e_lines + eprime_lines, e_labels + eprime_labels, 
             loc = 'upper left', **leg_args)

    # f. Flux-data correlations

    standard_subset = {k:v for k,v in result['data'].iteritems() if k
                       in result['traj']}
    standard_r = np.array([pearsonr(v,np.abs(result['traj'][k]))[0] for
                           k,v in standard_subset.iteritems()])

    other_subset = {k:v for k,v in other['data'].iteritems() if k
                   in other['traj']}
    other_r = np.array([pearsonr(v,np.abs(other['traj'][k]))[0] for
                    k,v in other_subset.iteritems()])
    other_r[np.isnan(other_r)] = 0.
    fprime = f.twinx()
    h_other = f.hist(other_r,50,normed=False,cumulative=False,
                     label=description)
    f.set_xlabel('correlation coefficient $r$')
    f.set_ylabel('reactions\n(%s)' % description)
    fprime.set_ylabel('reactions (standard method)')
    h_standard = fprime.hist(standard_r,50,normed=False,cumulative=False,
                             color='r',alpha=0.9,label='standard method')
    f_lines, f_labels = f.get_legend_handles_labels()
    fprime_lines, fprime_labels = fprime.get_legend_handles_labels()
    f.legend(f_lines + fprime_lines, f_labels + fprime_labels,
             loc='upper left')

    figure.tight_layout()
    reset_fontsizes(old_fontsize)
    return figure

def fig_phloem():
    """ 
    Illustrate phloem nitrogen and sulfur transport.

    """
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    f, ax = plt.subplots(2) # ,figsize=(5.19,4.05),sharex=True)
    a,b = ax
    x = np.arange(0.5,15.5)
    a.plot(x,traj['bs_tx_GLY'],'b-o',
           label='glycine import')
    add_fva(a,x,'bs_tx_GLY','b')
    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.spines['bottom'].set_position(('data',0)) 
    a.spines['bottom'].set_zorder(-10)  
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.set_ylim(-0.3,0.7)
    a.set_yticks(np.linspace(-0.3,0.7,11))
    # a.text(1.0,1.0,'a',horizontalalignment='right',
    #        verticalalignment='top', transform=a.transAxes,
    #        fontsize=14)
    a.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    a.set_ylabel(flux_label)
    a.legend(loc='best')

    b.plot(x,traj['bs_tx_GLUTATHIONE'],'g-o',
           label='glutathione import')
    add_fva(b,x,'bs_tx_GLUTATHIONE','g')
    b.spines['right'].set_color('none')
    b.spines['top'].set_color('none')
    b.spines['bottom'].set_position(('data',0)) 
    b.spines['bottom'].set_zorder(-10)  
    b.xaxis.set_ticks_position('bottom')
    b.yaxis.set_ticks_position('left')
    b.set_ylim(-0.05,0.11)
    b.set_yticks(np.linspace(-0.04,0.1,8))
    # b.text(1.0,1.0,'b',horizontalalignment='right',
    #        verticalalignment='top', transform=b.transAxes,
    #        fontsize=14)
    b.set_xlim(0,15)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    b.set_ylabel(flux_label)
    b.legend(loc='best')
    b.set_xlabel('leaf postition (cm)',labelpad=20)

    f.tight_layout()
    return f


def fig_fixed_biomass():
    """ Illustrate biomass production rate in the fixed-biomass case. """

    with open('fit_fixed_biomass.pickle') as f:
        fixed = pickle.load(f)

    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    f, ax = plt.subplots(1) # ,figsize=(5.19,4.05))
    a = ax
    x = np.arange(0.5,15.5)
    a.plot(x,fixed['traj']['ms_CombinedBiomassReaction'],'b-o',
           label='mesophyll')
    a.plot(x,fixed['traj']['bs_CombinedBiomassReaction'],'g-s',
           label='bundle sheath')

    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.set_ylim(0,0.2)
    a.set_yticks(np.linspace(0,0.2,11))
    # a.text(1.0,1.0,'a',horizontalalignment='right',
    #        verticalalignment='top', transform=a.transAxes,
    #        fontsize=14)
    a.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)
    a.set_ylabel('biomass production\n$\mathrm{mg}$ $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$')
    a.set_xlabel('cm from leaf base')
    a.legend(loc='best')


    f.tight_layout()
    return f

def fig_objective_scatter():
    """
    Compare best-fit to FBA to E-Flux results at the tip.

    """
    with open('source_fba_comparison.pickle') as f:
        fba = pickle.load(f)
    with open('fit_eflux.pickle') as f:
        eflux = pickle.load(f)
    # Compare values of variables in the base model,
    # excluding a few whose values are unconstrained in 
    # these calculations
    keys = [v for v in fba['variables'] if 'max' not in v] 
    v_fba = np.array([fba['soln'][k] for k in keys])
    v_eflux = np.array([eflux['traj'].get(k,[0.])[-1] for k in keys])
    v_standard = np.array([result['traj'].get(k,[0.])[-1] for k in keys])
    f,ax = plt.subplots(1,2,sharey=True)
    a,b = ax

    a.plot(v_standard, v_fba, 'o')
    a.set_ylabel('FBA predictions',labelpad=40)
    a.set_xlabel('data-fitting predictions',labelpad=95)
    
    b.plot(v_eflux, v_fba, 'o')
    b.set_xlabel('E-Flux predictions', labelpad=95)

    # c.plot(v_standard, v_eflux, 'o')
    # c.set_xlabel('data-fitting predictions')
    # c.set_ylabel('E-Flux predictions')

    for subfig in (a,b,):
        subfig.spines['right'].set_color('none')
        subfig.spines['top'].set_color('none')
        subfig.spines['bottom'].set_position(('data',0)) 
        subfig.spines['left'].set_position(('data',0)) 
        subfig.xaxis.set_ticks_position('bottom')
        subfig.yaxis.set_ticks_position('left')
    a.set_xlim(-50,150)
    a.set_ylim(-20,60)
    b.set_xlim(-10,15)

    f.tight_layout()
    return f

def fig_bs_pepc():
    """ 
    Illustrate bundle sheath PEPC activity.

    """
    leg_args = {'fontsize': leg_small_fontsize, 'numpoints': 1,
                'handlelength': 1,}
                                 # 'borderaxespad': 0.6}
    f, ax = plt.subplots(1) # ,figsize=(5.19,4.05))
    a = ax
    x = np.arange(0.5,15.5)
    a.plot(x,traj['bs_PEPCARBOX_RXN'],'b-o',
           label='PEPC (bundle sheath)')
    add_fva(a,x,'bs_PEPCARBOX_RXN','b')
    # a.plot(x,traj['ms_PEPCARBOX_RXN'],'g-s',
    #        label='PEPC (mesophyll)')
    a.spines['right'].set_color('none')
    a.spines['top'].set_color('none')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.set_ylim(0,10)
    a.set_yticks(np.linspace(0,10,6))
    # a.text(1.0,1.0,'a',horizontalalignment='right',
    #        verticalalignment='top', transform=a.transAxes,
    #        fontsize=14)
    a.set_xlim(0,15)
    a.set_xticks(standard_xticks)
    a.set_xticklabels(standard_xticklabels)
    flux_label = 'flux ($\mathrm{\mu}$mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1})$'
    a.set_ylabel(flux_label)
    a.legend(loc='best')
    a.set_xlabel('leaf postition (cm)')

    f.tight_layout()
    return f


if __name__ == '__main__':
    figs = [fig2(), fig3(), fig4(), fig5(), fig6_alt2()]
    figs[0].savefig('Fig2.pdf')    
    figs[1].savefig('Fig3_template.pdf')    
    figs[2].savefig('Fig4.pdf')    
    figs[3].savefig('Fig5.pdf')    
    figs[4].savefig('Fig6.pdf')    

    fig_ps2().savefig('S2_Figure.pdf')
    fig_phloem().savefig('S1_Figure.pdf')
    fig_bs_pepc().savefig('S3_Figure.pdf')
    fig_fixed_biomass().savefig('S8_Figure.pdf')
    
    ub_flex = fig_summary('fit_eflux.pickle','E-Flux')
    ub_fixed = fig_summary('fit_eflux_fixed.pickle','E-Flux (fixed biomass)',
                           fixed_biomass=True)
    no_scales = fig_summary('fit_zero_scales.pickle','no scale factors')
    fixed_biomass = fig_summary('fit_fixed_biomass.pickle',
                                'fixed biomass composition',
                                fixed_biomass=True)
    
    ub_flex.savefig('S5_Figure.pdf')
    ub_fixed.savefig('S6_Figure.pdf')
    no_scales.savefig('S4_Figure.pdf')
    fixed_biomass.savefig('S7_Figure.pdf')
    
    objective_scatter = fig_objective_scatter()
    objective_scatter.savefig('S18_Figure.pdf')
    # Then manually convert pdf to tiff as necessary.
