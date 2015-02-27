"""
Implement the C4 photosynthesis model presented in von Caemmerer, _Biochemical
Models of Leaf Photosynthesis_, 2000. 

"""

import numpy as np

class C4Model:
    def __init__(self,**parameters):
        """
        Create a model, overriding default parameters or adding new ones.

        Setting parameters through keyword arguments sets this instance's
        default parameters (which may then be overridden when calling
        C4Model.A() to calculate an assimilation rate.) See the code for
        a list of parameters which may be set, their default values
        if not specified, and their units. 

        CAUTION: Mesophyll oxygen and carbon dioxide partial pressures
        must be specified in microbar and g_s, the bundle sheath 
        carbon dioxide conductivity, must be specified in micromole
        per (square meter * second * microbar). This is different from
        the text, where g_s is consistently given as millimole per 
        (square meter * second), with an apparent intent that this be 
        read as millimole per (square meter * second * bar). 
        """
        # Supply default parameter values from table 4.1
        defaults = {'vcmax': 60., # umol/(m^2 s)
                    'kc': 650., # ubar
                    'ko': 450000., # ubar
                    'gamma': 0.5/2590., # dimensionless
                    'vpmax': 120., # umol/(m^2 s)
                    'vpr': 80., # umol/(m^2 s),
                    'kp': 80., # ubar,
                    'gs': 0.003, # umol/(m^2 s ubar) -- my interpretation
                    'alpha': 0., # dimensionless
                    'x': 0.4, # dimensionless
                    'jmax': 400., # umol electrons / (m^2 s)
                    # The next three parameters are given as typical 
                    # in section 2.3.4.2
                    'theta': 0.7, # irradiance/electron transport curvature
                    'f': 0.15, # light spectral quality correction
                    'absorptance': 0.85, # dimensionless
                    # No default values for mesophyll O2 or irradiance
                    # are given in the text, but I add them here for 
                    # convenience, using the typical value for om and 
                    # a high-irrandiance value.
                    'om': 200000. ,# ubar
                    'i': 1e4 # micromole quanta / (m^2 s); 
                    }

        # Override the defaults with user-supplied parameters.
        defaults.update(parameters)

        # Unless they have been supplied by the user, set parameters
        # which by default depend on other parameters to their default values.
        defaults.setdefault('rd', 0.01*defaults['vcmax'])
        defaults.setdefault('rm', 0.5*defaults['rd'])
        # g_o is actually never used in the model as written;
        # the following relationship is just assumed to be true.
        #defaults.setdefault('go', 0.047*defaults['g_s'])

        self.parameters = defaults

    def A(self, cm = None, force_enzyme_limited = False, 
          **additional_parameters):
        """
        Evaluate assimilation rate as a function of mesophyll carbon dioxide.

        Arguments:
        cm -- mesophyll carbon dioxide partial pressure (ubar). If omitted,
        the calculation will look for a cm parameter in self.parameters.

        force_enzyme_limited -- if True, don't even calculate the light-limited
        rate. (Otherwise, the result is the minimum of the light-limited and
        enzyme-limited rates.)

        Additional parameters, if present, will override parameter values
        set on initialization of the object. 

        One or more arguments/parameters may be numpy arrays (of the same 
        shape), in which case the output will be a corresponding array of 
        A values.
        """

        variables = {}
        variables.update(self.parameters)
        variables.update(additional_parameters)
        if cm is not None:
            variables['cm'] = cm

        # Evaluate the enzyme-limited assimilation rate
        variables.setdefault('vp',eval('np.minimum(vpr, (cm*vpmax)/(cm+kp))',
                                       globals(),variables))

        
        a = '1.0 - (alpha * kc)/(0.047 * ko)'
        b = '-1.0 * ((vp - rm + gs*cm) + \
                    (vcmax - rd) + \
                    gs * (kc * (1. + om/ko)) + \
                    alpha * (gamma*vcmax + rd*kc/ko) / 0.047)'
        c = '((vcmax - rd) * (vp - rm + gs*cm) - \
             (vcmax * gs * gamma * om + rd * gs * kc * (1. + om/ko)) \
             )'

        a = eval(a,variables)
        b = eval(b,variables)
        c = eval(c,variables)

        A_enzyme = (-1.0*b - np.sqrt(b**2 - 4*a*c)) / (2*a)

        if force_enzyme_limited:
            return A_enzyme 

        # Don't use setdefault for the following as if jt has been 
        # set explicitly we don't want to even assume the related 
        # parameters have values.
        if 'jt' not in variables:
            if 'i2' not in variables:
                variables['i2'] = eval('i * absorptance * (1.0 - f) / 2',
                                       variables)
            pass
            j = '(i2 + jmax - np.sqrt((i2 + jmax)**2 - 4*theta*i2*jmax)) / \
                 (2 * theta)'
            variables['jt'] = eval(j, globals(), variables)
        
        d = '1.0 - 7 * gamma * alpha / (3.0 * 0.047)'
        e = '-1.0*((x*jt*0.5 - rm + gs * cm) + \
                  ((1.0 - x) * jt / 3.0 - rd) + \
                  gs * (7.0 * gamma * om / 3.0) + \
                  (alpha * gamma / (3.0*0.047)) * ((1.0-x) * jt + \
                                                   7.0 * rd) \
                  )'
        f = '((x*jt*0.5 - rm + gs * cm)*((1.0-x)*jt/3.0 - rd) - \
             gs * gamma * om * ((1.0-x) * jt + 7.0*rd)/3.0 \
             )'

        d = eval(d,variables)
        e = eval(e,variables)
        f = eval(f,variables)

        A_electron = (-1.0*e - np.sqrt(e**2 - 4*d*f)) / (2*d)

        return np.minimum(A_enzyme, A_electron)

     
if __name__ == '__main__':
    """ Reproduce some figures from the source paper. """

    import matplotlib.pyplot as plt
#    plt.ion()
    # We can reproduce results for the enzyme-limited case fairly
    # easily, eg. fig. 4.7(a) (remembering that the text's units for 
    # gs are a smaller than our units for gs, and noting that 
    # where the figure has Vcmax/Vpmax = 1.8 it appears to mean
    # Vpmax/Vcmax = 1.8, and setting the max PEP regeneration rate high
    # to ensure it does not limit A):

    case1 = C4Model(cm = 100., rd = 0., vpmax = 90., gs = 0.003, vpr=1000.)
    vcmax = np.arange(0.,90.,1.)
    A1 = case1.A(vcmax = vcmax, force_enzyme_limited = True)
    
    case2 = C4Model(cm = 100., rd = 0., vpr=1000.)
    vcmax = np.arange(0.,90.,1.)
    vpmax = vcmax*1.8
    gs = 5e-5 * vcmax # 50 Vcmax / bar = 5e-5 Vcmax / microbar
    A2 = case2.A(vcmax = vcmax, vpmax = vpmax, gs = gs, 
                 force_enzyme_limited = True)
    
    case3 = C4Model(cm = 100., rd = 0., gs = 0.003, vpr=1000.)
    vcmax = np.arange(0.,90.,1.)
    vpmax = vcmax*1.8
    A3 = case3.A(vcmax = vcmax, vpmax = vpmax, force_enzyme_limited = True)

    plt.figure()
    plt.plot(vcmax, A1, 'k', label='constant PEPC and gs')
    plt.plot(vcmax, A2, 'k--', label='scaled PEPC and gs')
    plt.plot(vcmax, A3, 'k:', label='scaled PEPC only')
    plt.xlabel('Vcmax (micromole / (square meter * second))')
    plt.ylabel('CO2 assimilation rate (micromole / (square meter * second))')
    plt.legend(loc = 'upper left')
    plt.title('Figure 4.7(a): effects of scaling PEPC and conductivity with Rubisco')

    plt.savefig('reproduced_47a.png')
    
    # Note that if we did not relax the vpr=80. limit, which table 4.1
    # claims is used by default, we would see effects for cases 2 and 3 
    # in the high-Vcmax regime. Indeed, that value appears to have been
    # tacitly ignored for many of the figures in the chapter and enforced
    # only for figure 4.13.
    
    # We can't quite reproduce the light response figures: something limits 
    # A. For example, the inset in figure 4.22:

    plt.figure()

    m = C4Model(cm = 100., rd = 0., vpr = 1000.,i=2000.)
    x = np.arange(0.,1.01,0.005)
    A_vs_x = m.A(x=x)
    plt.plot(x,A_vs_x,'k',label='standard')
    plt.xlabel('fraction x of electron transport allocated to C4 cycle')
    plt.ylabel('CO2 assimilation rate (micromole / (square meter * second))')
    plt.title('Figure 4.22, inset: effects of electron transport partitioning')
    plt.axis([0.,1.,0.,62.5])
    plt.text(0.35,45,'should not be flat!')

    # With the parameters in table 4.1,
    # at Cm = 100 ubar, Vp = 120 * 100 / (100 + 80) ~ 66.7. To sustain
    # A = 59, as we see for the optimal value of x in fig. 4.22, L is then
    # at most 7.7, corresponding to a CO2 enrichment in the BS vs the 
    # mesophyll of L/g = 7.7/0.003 = 2567 ubar. Then C_bs = 2667 ubar
    # and Vc = 60 * 2667 / (2667 + 650 (1 + Os/Ks)) <= 
    # 60 * 2667 / (2667 + 650) = 48.2, inconsistent with A = 59. 

    # (The oxygenation rate would be (1/S)(o/c) * Vc, on the order of 
    # (1/2590.)*(200000/2667) ~ 0.03; even if it is 0.6, bringing L to
    # 8.0 would increase C_bs by an insignificant 100 ubar.)

    # So possibly this figure is drawn assuming the system is 
    # electron-transport-limited, ignoring enzymatic restrictions.

    # A curve with unrealistically high enzyme levels 
    # appears to agree with the text:
    A_unlimited_enzymes = m.A(x=x,vpmax = 300.,vcmax = 100.)
    plt.plot(x,A_unlimited_enzymes,'k--',label='no enzyme limitation')
    plt.legend()
    plt.savefig('reproduced_422inset.png')

    # In fact it appears that most of the figures in the discussions 
    # of enzyme-limited or light-limited photosynthesis were made without
    # considering the other limitations at all; note for example that
    # the high A values in fig. 4.7 would require an increased maximum
    # electron transport rate, as can be confirmed by rerunning the 
    # calculations without the 'force_enzyme_limited' option set!
    # This is entirely reasonable but not made explicit in the text.
    plt.show()

