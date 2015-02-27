import os
import pickle
import logging
import multiprocessing as mp
from Queue import Empty
import numpy as np
import fluxtools.nlcm as nlcm
from fluxtools.nlcm import NonlinearModel
from fluxtools.functions import Function, Linear
from fluxtools.fva import do_fva

# The literature on mixing the basic logging module and 
# multiprocessing suggests that all sorts of unfortunate things may happen 
# if this logger is accessed from multiple processes at once under the 
# wrong circumstances (e.g., it is trying to write to a file;) currently I ignore
# this.
logger = logging.getLogger('fitting.leastsquares')

def make_sum_of_squares(data_ids, uncertainty_ids, scale_factor_ids, name=None):
    """ Write a least-squares style objective function and derivatives.

    Returns the function (as fluxtools.functions.Function).

    """ 
    terms = []
    derivs = {}
    second_derivs = {}
    for var, datum_id in data_ids.iteritems():
        uncertainty = uncertainty_ids[var]
        scale_factor = scale_factor_ids[var]
        terms.append('((%s*%s-%s)**2)*%s' % (var, scale_factor,
                                             datum_id,
                                             uncertainty))
        derivs[var] = '2*(%s*%s-%s)*%s*%s' % (var, scale_factor,
                                              datum_id,
                                              scale_factor,
                                              uncertainty)
        second_derivs[(var,var)] = '2*%s*(%s)**2' % (uncertainty,
                                                     scale_factor)
    math = nlcm.tree_sum(terms)
    return Function(math, first_derivatives=derivs,
                    second_derivatives=second_derivs, name=name)

def setup_exponentially_scaled_sum_of_squares(model, data_ids, inv_uncertainty_ids,
                                              scale_factor_ids, sign_ids, name=None):
    """Give model a least-squares objective function with auxiliary variables.
    
    For each variable v in data_ids, a new variable error_v is
    introduced and constrained to equal (datum_v - v * sign_v *
    exp(scale_factor_v)); then the objective is set to
        \sum_i inv_uncertainty_v**2 * error_v_i**2 

    A dictionary {v: error_v, ...} is returned. 

    """
    # Set up a template auxiliary constraint
    template_math = 'error_v - (datum_v - v * sign_v * exp(scale_factor_v))'
    template = Function(template_math, name='error_template')
    template.all_first_derivatives()
    template.all_second_derivatives()
    
    # Collect terms and derivatives of the objective, and auxiliary variable names
    terms = []
    derivs = {}
    second_derivs = {}
    error_variables = {}
    for var, datum_id in data_ids.iteritems():
        # Set up the error constraint
        inv_uncertainty = inv_uncertainty_ids[var]
        scale_factor = scale_factor_ids[var]
        sign = sign_ids[var]
        error_variable = 'error_%s' % var
        error_variables[var] = error_variable
        error_constraint = template.substitute({'v': var,
                                                'sign_v': sign,
                                                'scale_factor_v': scale_factor,
                                                'error_v': error_variable,
                                                'datum_v': datum_id})
        error_constraint.name = '_error_constraint_%s' % var
        model.constraints.set(error_constraint.name, error_constraint)
        model.set_bound(error_constraint.name, 0.)
        
        # Add a term to the objective function 
        terms.append('%s**2 * %s**2' % (inv_uncertainty, error_variable))
        derivs[error_variable] = '2*%s*(%s**2)' % (error_variable, inv_uncertainty)
        second_derivs[(error_variable, 
                       error_variable)] = '2*(%s**2)' % inv_uncertainty 

    # Write the objective function and apply it
    math = nlcm.tree_sum(terms)
    if name is None:
        name = 'automatically generated least squares'
    model.objective_function = Function(math, first_derivatives=derivs,
                                        second_derivatives=second_derivs, name=name)
    return error_variables

##########
# Fitting objects

class LeastSquaresModel(NonlinearModel):

    """ Find feasible points of a model closest to given data. """

    def __init__(self, base_model, data_variables):
        """ Create a least-squares fitting model from an existing model.

        Arguments:
        base_model - existing model to copy (should be compiled, or have
            an up-to-date list of variables)
        data_variables - variables which may be fit to data. Here it is 
            assumed all these have an appropriate sign (ie, reversible
            reactions are not handled correctly.)

        See self.setup_objective for details on the objective function.

        """

        NonlinearModel.__init__(self)
        self._copy_model(base_model)
        # Allow for the possibility of lazily initializing the model
        # from a data set
        self.data_variables = list(data_variables)
        self.setup_objective()
        self.compile()

    def _copy_model(self, base_model):
        """Copy the structure of base_model to this problem.
        
        In the future maybe this could be replaced using __getstate__ and
        __setstate__ .


        """
        self.constraints = base_model.constraints.copy()
        self.parameters = base_model.parameters.copy()
        # Allow for the possibility that the base model may have a nontrivial
        # get_lower[upper]_bound method. This is where we assume the base model
        # has been compiled, or at least has an up-to-date variables attribute.
        bounds = {v: (base_model.get_lower_bound(v),
                      base_model.get_upper_bound(v)) for v in
                  (list(base_model.variables) +
                   base_model.constraints.keys())}
        self.set_bounds(bounds)

    def setup_objective(self):
        """Create a least-squares objective function for this model. 

        Gives the model an objective function which amounts to minimizing
        the sum of squares of discrepancies between the variables in
        self.data_variables and some data (as yet unspecified.) 

        self.setup_basic_parameters is called to add data, scale
        factors, and squared uncertainties for the data_variables, and
        an objective function of the form 

            \sum_i inv_sq_uncertainty_i * (scale_factor_i * variable_i
                   - data_i)**2 

        is applied.

        Derivative caching is used to pre-calculate the objective
        derivatives and second derivatives efficiently. 

        Using an inverse uncertainty allows terms to be dropped from the
        objective function (temporarily) by setting the inverse
        uncertainty to 0. (Those terms will still contribute to the cost
        of evaluating the objective function and its derivatives, and will
        add nonzero terms to the problem's Hessian, increasing the cost of
        solving linear systems inside IPOPT, so it is generally best to
        exclude irrelevant terms outright; however, if doing repeated
        calculations with one model and different combinations of fit
        variables, it may be faster or at least more convenient to avoid
        compiling the objective function anew each time.) 

        This method does not recompile the model.

        Nothing is returned.

        """
        self.setup_basic_parameters()
        self.objective_function = make_sum_of_squares(self.data_parameters,
                                             self.inverse_squared_uncertainties,
                                             self.scale_factors)

    def setup_basic_parameters(self):
        """Add data, uncertainty, and scale factor parameters for variables.

        Data, scale factors, and squared uncertainties are stored as
        parameters; the names of the datum, scale factor, and inverse
        squared uncertainty parameters for variable v may be found in
        self.data_parameters[v], self.scale_factors[v],
        self.inverse_squared_uncertainties[v]. These are not
        variables, and no derivatives wrt them are taken. Inverse
        uncertainties and data are given default values of 0 in this
        step; scale factors are given the default value 1.0.

        """
        data_ids = {}
        is_uncertainty_ids = {}
        inv_uncertainty_ids = {}
        scale_factor_ids = {}
        for var in self.data_variables:
            datum_id = 'datum_%s' % var
            data_ids[var] = datum_id
            self.parameters[datum_id] = 0.

            is_uncertainty_id = 'inv_sq_uncertainty_%s' % var
            is_uncertainty_ids[var] = is_uncertainty_id
            self.parameters[is_uncertainty_id] = 0.

            inv_uncertainty_id = 'inv_uncertainty_%s' % var
            inv_uncertainty_ids[var] = inv_uncertainty_id
            self.parameters[inv_uncertainty_id] = 0.

            scale_factor_id = 'scale_factor_%s' % var
            scale_factor_ids[var] = scale_factor_id
            self.parameters[scale_factor_id] = 1.

        self.data_parameters = data_ids
        self.inverse_uncertainties = inv_uncertainty_ids
        self.inverse_squared_uncertainties = is_uncertainty_ids
        self.scale_factors = scale_factor_ids

    def set_data(self, data):
        """Set data for fitting variables from a dict.

        Data settings for variables not in the dict are unchanged.

        """
        for variable, datum in data.iteritems():
            self.parameters[self.data_parameters[variable]] = datum

    def set_uncertainties(self, uncertainties):
        """Set uncertainties in data from a dict. 

        Uncertainty parameters for variables not in the dict are
        unchanged.

        """
        for variable, uncertainty in uncertainties.iteritems():
            inv_sq_delta = uncertainty ** -2.
            inv_sq_parameter = self.inverse_squared_uncertainties[variable]
            self.parameters[inv_sq_parameter] = inv_sq_delta
            inv_parameter = self.inverse_uncertainties[variable]
            self.parameters[inv_parameter] = 1./uncertainty

    def set_data_assuming_uncertainties(self, data,
                                        relative_uncertainty_factor=0.1,
                                        min_absolute_uncertainty=1.0):
        """ Set data, and assign uncertainties dependent on that data.
        
        If variable v has data d, the uncertainty in d will be
            delta_d = max(min_absolute_uncertainty,relative_uncertainty_factor.)

        Note that data and uncertainties for variables not in the 'data' dict
        will not be changed.

        """
        assumed_uncertainties = {k: max(min_absolute_uncertainty,
                                        relative_uncertainty_factor*v) for
                                 k, v in data.iteritems()}
        self.set_data(data)
        self.set_uncertainties(assumed_uncertainties)

    def set_scale_factors(self, scale_factors):
        """ Set scale factors for fitting variables from a dict.

        Scale factors for variables not in the dict are unchanged. 

        """
        for variable, scale_factor in scale_factors.iteritems():
            self.parameters[self.scale_factors[variable]] = scale_factor

class ImprovedFittingModel(LeastSquaresModel):
    """ Data-fitting class with useful features for large-scale RNA-seq fitting.

    This model uses exponential scale factors which may have 
    complex interrelationships, specifies independent variables for the
    purposes of imposing a prior cost on scale factors, and handles sign
    sign differences between data and variables.

    """

    def __init__(self, base_model, data_variables, scale_relationships={},
                 zero_threshold=1e-4, load_fva_results=None, save_fva_results=None,
                 n_procs=1, max_scale_factor=10.):
        """Set up a fitting problem allowing reversible reactions with positive data. 

        Arguments:
        base_model - existing model to copy (should be compiled, or have
            an up-to-date list of variables)
        data_variables - variables which may be fit to data.
        scale_relationships - dict of variables in data_variables, 
            specifying a set of quantities which must add together to give
            the variable's scale factor: that is, an entry
                {'bs_RXN_FOO_instance1': ('scale_RXN_FOO', 'scale_bs')}
            indicates that scale_bs_RXN_FOO = scale_RXN_FOO + scale_bs.
            If ignored, all scale factors will be independent. The attribute
            self.independent_scale_factors lists the independent scale factors;
            each key in scale_relationships is removed from the set, and each 
            variable referenced in a value is added. 
        load_fva_results - either a dict or the name of a file containing
            a pickled dict of fva results {'variable': (lower_bound, upper_bound)...}
            which should be assumed rather than recalculated. Use cautiously
            as nonsense may result if an incorrect set of results is assumed.
        save_fva_results - filename to which all the FVA results determined
            in the initialization process should be saved after initialization. 
        zero_threshold - float; quantities less than this in absolute values may
            be considered zero at various points in the code. Note that this should
            not be smaller than the IPOPT tolerance!
        max_scale_factor: all scale factors (direct and/or independent) 
            will be constrained to lie in (-1.*max_scale_factor, max_scale_factor).

        See self.setup_objective for details on the objective function.

        Note that this class distinguishes between bounds set manually
        or explicitly using set_bound()/set_bounds(), and bounds which may
        be imposed automatically when a variable's sign parameter is changed.
        
        Getting and setting bounds works normally, but setting has the
        added effect of updating a cache of bounds
        (self.manual_lower_bounds or self.manual_upper_bounds) which
        persists when set_sign() changes bounds to ensure positive
        variables do not become large and negative, and vice versa.

        """
        self._base_model_cache = base_model.copy()
        self.threshold = zero_threshold
        self.n_procs = n_procs 
        self.manual_lower_bounds = {} # These will be populated automatically
        self.manual_upper_bounds = {} # when _copy_model() calls set_bounds()
        self.scale_relationships = scale_relationships 
        self.max_scale_factor = max_scale_factor
        LeastSquaresModel.__init__(self, base_model, data_variables)
        if load_fva_results:
            self._load_fva_cache(load_fva_results)
        self.check_signs()
        if save_fva_results:
            with open(save_fva_results,'w') as f:
                pickle.dump(getattr(self, '_fva_cache', {}), f)


    def set_lower_bound(self, v, bound, cache=True):
        LeastSquaresModel.set_lower_bound(self, v, bound)
        # Some unexpected behavior could occur here if resolve_name
        # is being used nontrivially
        if cache:
            self.manual_lower_bounds[v] = bound

    def set_upper_bound(self, v, bound, cache=True):
        LeastSquaresModel.set_upper_bound(self, v, bound)
        # Some unexpected behavior could occur here if resolve_name
        # is being used nontrivially
        if cache:
            self.manual_upper_bounds[v] = bound

    def _load_fva_cache(self, cache):
        """ Set FVA result cache from dict or filename of pickle file.

        This will cause nonsense results if a bad cache is given!

        """
        if not isinstance(cache, dict):
            with open(cache) as f:
                cache = pickle.load(f)
        self._fva_cache = cache

    def setup_objective(self):
        """Create a least-squares objective function with sign parameters. 

        Gives the model an objective function which amounts to minimizing
        the sum of squares of discrepancies between the variables in
        self.data_variables and some data (as yet unspecified), with the 
        variables and data in some cases having different signs.

        The objective function is mathematically equivalent to

            \sum_i inv_sq_uncertainty_i * (exp(scale_factor_i) * sign_i * variable_i
                   - data_i)**2 
        
        though it is expressed differently (TODO: expand this.) 
        
        """
        self.setup_basic_parameters()
        errors = setup_exponentially_scaled_sum_of_squares(
            self, self.data_parameters,
            self.inverse_uncertainties,
            self.scale_factors,
            self.sign_parameters
        )
        self.error_variables = errors
        self.setup_scale_relationships(self.scale_relationships)

    def setup_scale_relationships(self, relationships):
        """Constrain direct scale factors to be sums of underlying scale factors.

        Adds constraints (but does not compile), sets bounds on direct
        scale factors which become variables, adds underlying scale
        factors as parameters with value 0., and sets up the
        self.independent_scale_factors attribute.

        Assumes self.scale_factors is already set up.

        """
        self.independent_scale_factors = set(self.scale_factors.values())

        for data_variable, indirect_factors in relationships.iteritems():
            direct_factor = self.scale_factors[data_variable]
            self.independent_scale_factors.remove(direct_factor)
            self.parameters.pop(direct_factor)
            for f in indirect_factors:
                self.independent_scale_factors.add(f)

            self.set_bound(direct_factor, (-1.*self.max_scale_factor,
                                           self.max_scale_factor))

            constraint_id = 'decomposition_%s' % direct_factor
            coefficients = dict.fromkeys(indirect_factors, 1.)
            coefficients[direct_factor] = -1.
            g = Linear(coefficients, constraint_id)
            self.constraints.set(constraint_id, g)
            self.set_equality(constraint_id, 0.)
        
        for f in self.independent_scale_factors:
            self.parameters[f] = 0.

    def setup_basic_parameters(self):
        """Add data, uncertainty, scale factor, and sign parameters for variables. 

        Scale factors start at 0, and signs at 1, by default.
        """
        LeastSquaresModel.setup_basic_parameters(self)
        sign_parameters = {}
        for variable in self.data_variables:
            sign_id = 'sign_%s' % variable
            sign_parameters[variable] = sign_id
            self.parameters[sign_id] = 1.
        self.sign_parameters = sign_parameters
        for p in self.scale_factors.values():
            self.parameters[p] = 0.


    def check_signs(self):
        """Determine which variables with data may be positive/negative. 

        All variables in self.data_variables are checked (by looking up
        upper and lower bounds and, if that is not determinative, performing
        FVA calculations). 

        Two attributes are then populated:
            - self.unsigned, those which truly may take values of either sign 
            - self.always_zero, those which may not differ from zero by more
              than self.threshold

        Next, for each variable with a fixed sign, the corresponding sign
        parameter is set:
            self.parameters[self.sign_parameters[v]] = sign(v)

        For variables in self.unsigned and self.always_zero, sign parameters
        are set to 1 arbitrarily. (Note that signs must be +/- 1--
        this is assumed in calculating the derivative of the objective
        function.)
        
        Note the results, and thus the behavior of the fitting process,
        may change if variable upper and lower bounds are changed.

        """
        # Populate three collections of variables depending on 
        # what we determine about their signs:
        fixed_signs = {}
        always_zero = []
        either_sign = [] 
        
        # First, examine upper and lower bounds, listing 
        # variables which need further attention
        to_check = [] 

        for v in self.data_variables:
            lb, ub = self.get_bounds(v)
            # Ignore the possibility that some bounds
            # are within self.threshold of 0...
            if lb is not None and lb >= 0.: # always positive
                fixed_signs[v] = 1.
                logger.info('check_signs setting %s forward based on its bounds.' % v)
            elif ub is not None and ub <= 0.: # always negative
                fixed_signs[v] = -1.
                logger.info('check_signs setting %s reverse based on its bounds.' % v)
            elif ub == lb == 0.:
                always_zero.append(v)
                logger.info('check_signs setting %s zero based on its bounds.' % v)
            else:
                to_check.append(v)

        # Do an FVA step to determine which of the variables which
        # appear to be allowed to take both signs in fact can.
        if to_check:
            # The FVA model could also be self.copy()...?
            # We do not do fva on self directly because it would
            # destroy the objective function (note also we would need
            # to guarantee the model had been compiled by this point; 
            # I think this is true now but in the future it might not be.) 
            fva_model = self._base_model_cache.copy()
            fva_cache = getattr(self, '_fva_cache', {})
            fva_results = do_fva(fva_model, to_check, n_procs=self.n_procs,
                                 cache=fva_cache)
            self._fva_cache = fva_results
            for variable, (lb, ub) in fva_results.iteritems():
                can_be_negative = lb < -1.0*self.threshold
                can_be_positive = ub > self.threshold
                if can_be_negative and can_be_positive:
                    either_sign.append(variable)
                    logger.info('check_signs found %s can take either sign based on FVA.' % variable)
                elif can_be_negative:
                    fixed_signs[variable] = -1.
                    logger.info('check_signs setting %s reverse based on FVA.' % variable)
                elif can_be_positive:
                    fixed_signs[variable] = 1.
                    logger.info('check_signs setting %s forward based on FVA.' % variable)
                else:
                    logger.info('check_signs setting %s zero based on FVA.' % variable)
                    always_zero.append(variable)
            
        # Record results and update the sign parameters
        self.always_zero = always_zero
        self.unsigned = either_sign
        for v, sign in fixed_signs.iteritems():
            self.parameters[self.sign_parameters[v]] = sign
        # Sign parameters for indeterminate or always-zero variables
        # are arbitary; set them to 1. 
        for v in always_zero + either_sign:
            self.parameters[self.sign_parameters[v]] = 1.

    def _set_objective_coefficient(self, variable, coefficient):
        """ Set a data variable's coefficient in the objective function.

        This is equivalent to setting the inverse uncertainty 
        of that variable.

        The structure of the objective function is not changed.

        The previous value of the corresponding parameter is returned.

        """
#        parameter = self.inverse_squared_uncertainties[variable]
#        old_value = self.parameters[parameter]
#        self.parameters[parameter] = coefficient
        parameter = self.inverse_uncertainties[variable]
        old_value = self.parameters[parameter]
        self.parameters[parameter] = coefficient
        return old_value

    def set_sign(self, v, sign, tolerance=1.):
        """Set sign of a variable with sign parameter. 
        
        The bounds on v are reset to those in self.manual_lower_bounds
        and self.manual_upper_bounds (relaxing bounds imposed by 
        previous calls to set_sign, if any.) 

        Then, if the sign is postive (resp. negative), the variable itself
        is constrained to be greater than -1*tolerance (resp. less
        than 1*tolerance), if this is a tighter lower (resp. upper)
        bound than the one already in place.

        """ 
        self.parameters[self.sign_parameters[v]] = sign
        lb = self.manual_lower_bounds[v]
        ub = self.manual_upper_bounds[v]
        self.set_bound(v, (lb, ub))
        if sign < 0:
            if ub is None or ub > tolerance:
                self.set_upper_bound(v, tolerance, cache=False)
        if sign > 0:
            if lb is None or lb < -1.*tolerance:
                self.set_lower_bound(v, -1.*tolerance, cache=False)

    def get_datum(self, v):
        return self.parameters[self.data_parameters[v]]

    def optimize_signs(self, to_assign, guess, skip_tiny=True):
        raise NotImplementedError()

    def fit(self, guess=None, skip_signs=False):
        """Fit to data, assigning signs to variables as needed.

        Variables in self.always_zero are excluded from the fit.  
        
        """
        if guess is None:
            guess = np.ones(self.nvar)
        # Ignore all zero variables
        for v in self.always_zero:
            self._set_objective_coefficient(v, 0.)
        
        # Try to determine the optimal sign choice for the reversible reactions
        if not skip_signs:
            return_values = self.optimize_signs(self.unsigned, guess=guess)
        # If we are skipping sign choice, return None (TODO: remove dependency on 
        # pointless return value)
        else:
            return_values = None

        ########################################
        # Do the final optimization -- this is not _entirely_
        # redundant; skip_signs may be set, or the last optimization
        # in the sign-fixing process may have been associated with a
        # suboptimal choice of sign
        self.solve(guess)
        return return_values

from fluxtools.utilities.total_flux import minimum_flux_solve, add_total_flux_objective
class SignChoiceTest(ImprovedFittingModel):
    def __init__(self, base_model, data_variables, **kwargs):
        ImprovedFittingModel.__init__(self, base_model, data_variables, **kwargs)
        add_total_flux_objective(self, reactions=data_variables)

    def optimize_signs(self, to_assign, guess, skip_tiny=True,
                       improvement_threshold=None, tolerate_failures=True):
        """Find the best signs for some variables, heuristically.

        First the fitting problem is solved with the current set of
        signs and the contribution of each variable-with-data to the
        objective function is determined (the reference solution.)

        Signs of variables with very low data (datum < self.threshold)
        are then given sign 1 arbitrarily, if skip_tiny is True.


        The remaining unsigned reversible reactions with data are then
        sorted in descending order of their corresponding data, and
        each in turn is added to the objective function first with
        sign -1 and then with sign +1. Minimum-flux optimal solutions
        are determined for each sign, and the change \delta in
        objective function value relative to the reference, restricted
        to active reactions, computed for each. 

        If the difference between the deltas is smaller than
        improvement_threshold, the original sign of the reaction is
        kept; otherwise, the sign with the best (most negative) delta
        is chosen.  (Here we could potentially accelerate the process
        by setting the signs of all the other reactions involved in
        the best solution.)

        If improvement_threshold is not specified, it defaults to
        self.threshold.

        If tolerate_failures is true, when an optimization failure is
        encountered in trying to determine the best sign of a
        particular reaction (not in the first fit to the irreversible
        reactions,) and trying any readily available alternative guess
        doesn't resolve it, the reaction will be given sign 1
        arbitrarily and sign choice will continue.  A warning will be
        printed to the logger and to standard output, and at the
        conclusion of sign choice a count of such reactions will be
        logged and printed, and they are returned in the first return value,
        which otherwise is empty.

        TODO: return values

        """
        # It is not clear how best to determine the 
        # offset to use in finding minimum-flux optimal states. 
        # The units are different from the implicit flux units of 
        # self.threshold (TODO: check that I have consistently used
        # that only for fluxes...). Here I have set an empirically
        # helpful value.
        minimum_flux_offset = 1e-5

        if improvement_threshold is None:
            improvement_threshold = self.threshold
        to_assign = list(to_assign)
        to_assign.sort(key=lambda v: self.get_datum(v),
                       reverse=True)

        already_assigned = [k for k in self.data_variables if 
                            k not in to_assign]

        self.solve(guess)
        reference_solution = self.soln.copy()

        original_signs = {r: self.parameters[self.sign_parameters[r]]
                          for r in to_assign}
        
        def delta(variables, solution):
            """ Objective change relative to reference, restricted to variables.
            
            Negative is improvement. 

            """
            # print 'Reference:'
            # print {v: (reference_solution[v], reference_solution[self.error_variables[v]]) for
            #        v in variables}
            # print 'New:' 
            # print {v: (solution[v], solution[self.error_variables[v]]) for
            #        v in variables}
            reference_value = np.sum([reference_solution[self.error_variables[v]]**2
                                      for v in variables])
            new_value = np.sum([solution[self.error_variables[v]]**2
                                for v in variables])
            return new_value - reference_value

        # Temporarily remove all reversible reactions from the
        # objective, saving the values of their coefficients for later
        unsigned_coefficients = {v: self._set_objective_coefficient(v, 0)
                                 for v in to_assign}

        if skip_tiny:
            tiny_threshold = getattr(self, 'tiny_threshold', self.threshold)
            for r in to_assign[:]:
                datum_id = self.data_parameters[r]
                datum = self.parameters[datum_id]
                if datum < tiny_threshold:
                    self.set_sign(r, 1)
                    logger.info(('Reaction %s given sign +1 without optimization ' +
                                 'because its associated datum is too small.') % r)
                    to_assign.remove(r)
                    already_assigned.append(r)
                    self._set_objective_coefficient(r, unsigned_coefficients[r])

        # Relax any bounds on the reversible reactions which may have 
        # been set in accordance with their prior sign choices, 
        # by setting the sign to zero for each:
        remaining_unsigned = set(to_assign)
        for v in to_assign:
            self.set_sign(v, 0)

        ########################################
        # Preprocesssing step: Fit to the irreversible reactions.
        # 
        # If the solution is an improvement, taking into account 
        # the reversible reactions which must be nonzero to achieve it,
        # set their signs too.
        n_fixed_signs = len(self.data_variables) - len(to_assign) - len(self.always_zero)
        if n_fixed_signs:
            logger.info('Fitting data for %d irreversible/signed reactions '
                        'before choosing signs for %d other reactions.' % 
                        (n_fixed_signs, len(to_assign)))

            minimum_flux_solve(self, guess, offset=minimum_flux_offset)
            improved_guess = self.x.copy()
            logger.info('Initial best fit value %.3g' % self.soln['_objective'])
            
        else:
            logger.info('Skipping first stage of direction assignment '
                        'because there are too few irreversible reactions.')
            improved_guess = guess.copy()
        
        ########################################
        # Iteratively add reversible reactions to the objective,
        # determining the optimal sign for each.
        # 
        problem_reactions = [] 
        n_errors_tolerated = 0
        logger.debug('to_assign')
        logger.debug(str(to_assign))

        this_pid = os.getpid()        
        for current_reaction in to_assign:
            print ('process %d: current_reaction is %s (%d remain)' %
                   (this_pid, current_reaction, len(remaining_unsigned)))
            remaining_unsigned.remove(current_reaction)
            self._set_objective_coefficient(current_reaction, 
                                            unsigned_coefficients[current_reaction])
            try: 
                comparison = self._compare_signs(current_reaction, 
                                                 remaining_unsigned,
                                                 unsigned_coefficients,
                                                 already_assigned,
                                                 delta, improved_guess,
                                                 guess, minimum_flux_offset)
                fluxes, values, deltas, n_fixed, results = comparison
            except nlcm.OptimizationFailure:
                message = 'Optimization failure choosing sign for %s' % current_reaction
                print message
                logger.warn(message)
                problem_reactions.append(current_reaction)
                n_errors_tolerated += 1
                sign = 1
            else:
                # Set the sign depending on the results.
                forward_vs_backward = deltas[1]-deltas[-1]
                logger.debug('%s margin: %.4g (negative implies forward direction is better)' %
                             (current_reaction, forward_vs_backward)) 
                if np.abs(forward_vs_backward) < improvement_threshold:
                    sign = original_signs[current_reaction]
                    logger.info('Insignificant difference between signs; keeping sign %d for %s' % 
                                (sign, current_reaction))
                elif forward_vs_backward < 0:
                    # Positive sign results in a significantly better
                    # change relative to the reference
                    sign = 1
                    logger.info('Forward is significantly better; %s set forward' %
                                current_reaction)
                else:
                    # Negative sign results in a significantly better
                    # change relative to reference
                    sign = -1
                    logger.info('Reverse is significantly better; %s set to reverse' % 
                                current_reaction)
                # Use the minimum-flux, best-fit-for-all-nonzero-reactions
                # case for whichever sign we ultimately decided on
                # as the guess for the next calculation
                improved_guess = results[sign]
                              
            self.set_sign(current_reaction, sign)
            already_assigned.append(current_reaction)
            print 'Sign choice for %s concludes' % current_reaction

        final_parameters = self.parameters.copy()
        print 'Sign choice concludes.'
        if problem_reactions:
            print 'There were %d signs chosen arbitrarily after failures' % len(problem_reactions)
            print problem_reactions
        return problem_reactions, final_parameters

    def _compare_signs(self, current_reaction, remaining_unsigned, unsigned_coefficients, 
                       already_assigned, delta, improved_guess, guess, minimum_flux_offset):
        values = {}
        fluxes = {}
        results = {}
        deltas = {}
        n_fixed = {}

        for sign in (-1, 1):
            self.set_sign(current_reaction, sign)
            try:
                minimum_flux_solve(self, improved_guess,
                                   offset=minimum_flux_offset)
            except nlcm.OptimizationFailure:
                logger.debug('Optimization failure with improved guess testing %s' % r)
                minimum_flux_solve(self, guess, offset=minimum_flux_offset)
            # For debugging purposes, track the value _before_ 
            # adding in other implicated reactions, because
            # this should be nondecreasing....
            values[sign] = self.soln['_objective']
            now_set = {r for r,f in self.soln.iteritems() if r in 
                       remaining_unsigned and
                       np.abs(f) > self.threshold}
            report_string = '; '.join(['%s (%.3g)' % (r, (self.soln[r]))
                                      for r in now_set])
            logger.debug('Setting %s to sign %d fixes %s' % (current_reaction,
                                                             sign,
                                                             report_string))
            n_fixed[sign] = len(now_set)
            if n_fixed:
                for other_reaction in now_set:
                    self._set_objective_coefficient(
                        other_reaction, 
                        unsigned_coefficients[other_reaction]
                    )
                    self.set_sign(
                        other_reaction, 
                        np.sign(self.soln[other_reaction])
                    )
                try:
                    # Note that at this point we don't _need_
                    # a minimum-flux solution-- we only care about
                    # the value, immediately-- but I have 
                    # used one here so that the guess provided
                    # to the next iteration will have a nice,
                    # low-flux structure. Whether this makes a
                    # practical difference, I don't know. 
                    minimum_flux_solve(self, improved_guess,
                                       offset=minimum_flux_offset)
                except nlcm.OptimizationFailure:
                    logger.debug('Optimization failure trying to min-flux-solve '
                                 'with improved guess; retrying w/ default guess')
                    minimum_flux_solve(self, guess,
                                       offset=minimum_flux_offset)
            fluxes[sign] = self.soln[current_reaction]
            results[sign] = self.x.copy()
            delta_variables = {current_reaction}
            delta_variables.update(already_assigned)
            delta_variables.update(now_set)
            deltas[sign] = delta(delta_variables, self.soln)
            for other_reaction in now_set:
                self._set_objective_coefficient(other_reaction, 0.)
                self.set_sign(other_reaction, 0.)

        message = ('%s, %s fixes %d: flux %.3g, bare value %.3g, ' +
                   'delta %.3g')
        for tag, i in (('Reversed', -1), ('Forward', 1)):
            logger.info(message % (tag, current_reaction,
                                   n_fixed[i], fluxes[i],
                                   values[i], deltas[i]))
        return fluxes, values, deltas, n_fixed, results


class BatchSignChoice(ImprovedFittingModel):
    def __init__(self, base_model, data_variables, **kwargs):
        ImprovedFittingModel.__init__(self, base_model, data_variables, **kwargs)
        add_total_flux_objective(self, reactions=data_variables)

    def optimize_signs(self, to_assign, guess, skip_tiny=True,
                       improvement_threshold=None):
        """Find the best signs for some variables, heuristically.

        First the fitting problem is solved with the current set of
        signs and the contribution of each variable-with-data to the
        objective function is determined (the reference solution.)

        Signs of variables with very low data (datum < self.threshold)
        are then given sign 1 arbitrarily, if skip_tiny is True.

        Then all remaining data of unknown sign are temporarily
        ignored and the fitting problem for irreversible reactions is
        solved in such a way as to minimize the total flux through all
        reactions with data.

        The reversible reactions-with-data which take nonzero values
        in the solution to that problem are then reintroduced to the
        objective function, with whatever sign they took in in the
        solution, and the problem is resolved.  

        # UNTRUE
        If the resulting solution
        is better, considering only contributions from the reactions involved,
        than the reference solution, those signs are adopted; otherwise, 
        the algorithm continues without fixing any signs. 
        ###

        The remaining unsigned reversible reactions with data are then
        sorted in descending order of their corresponding data, and
        each in turn is added to the objective function first with
        sign -1 and then with sign +1. Minimum-flux optimal solutions
        are determined for each sign, and the change \delta in
        objective function value relative to the reference, restricted
        to active reactions, computed for each. 
        
        If the difference between the deltas is smaller than
        improvement_threshold, the original sign of the reaction is
        kept; otherwise, the sign with the best (most negative) delta
        is chosen.  The signs of all the other reactions involved in
        the best solution are also set.

        If improvement_threshold is not specified, it defaults to
        self.threshold.

        TODO: return values

        """
        # It is not clear how best to determine the 
        # offset to use in finding minimum-flux optimal states. 
        # The units are different from the implicit flux units of 
        # self.threshold (TODO: check that I have consistently used
        # that only for fluxes...). Here I have set an empirically
        # helpful value.
        minimum_flux_offset = 1e-5

        if improvement_threshold is None:
            improvement_threshold = self.threshold
        to_assign = list(to_assign)
        to_assign.sort(key=lambda v: self.get_datum(v),
                       reverse=True)

        already_assigned = [k for k in self.data_variables if 
                            k not in to_assign]

        self.solve(guess)
        reference_solution = self.soln.copy()

        original_signs = {r: self.parameters[self.sign_parameters[r]]
                          for r in to_assign}
        
        def delta(variables, solution):
            """ Objective change relative to reference, restricted to variables.
            
            Negative is improvement. 

            """
            # print 'Reference:'
            # print {v: (reference_solution[v], reference_solution[self.error_variables[v]]) for
            #        v in variables}
            # print 'New:' 
            # print {v: (solution[v], solution[self.error_variables[v]]) for
            #        v in variables}
            reference_value = np.sum([reference_solution[self.error_variables[v]]**2
                                      for v in variables])
            new_value = np.sum([solution[self.error_variables[v]]**2
                                for v in variables])
            return new_value - reference_value

        # Temporarily remove all reversible reactions from the
        # objective, saving the values of their coefficients for later
        unsigned_coefficients = {v: self._set_objective_coefficient(v, 0)
                                 for v in to_assign}

        if skip_tiny:
            tiny_threshold = getattr(self, 'tiny_threshold', self.threshold)
            for r in to_assign[:]:
                datum_id = self.data_parameters[r]
                datum = self.parameters[datum_id]
                if datum < tiny_threshold:
                    self.set_sign(r, 1)
                    logger.info(('Reaction %s given sign +1 without optimization ' +
                                 'because its associated datum is too small.') % r)
                    to_assign.remove(r)
                    already_assigned.append(r)
                    self._set_objective_coefficient(r, unsigned_coefficients[r])

        # Relax any bounds on the reversible reactions which may have 
        # been set in accordance with their prior sign choices, 
        # by setting the sign to zero for each:
        for v in to_assign:
            self.set_sign(v, 0)

        ########################################
        # Preprocesssing step: Fit to the irreversible reactions.
        # 
        # If the solution is an improvement, taking into account 
        # the reversible reactions which must be nonzero to achieve it,
        # set their signs too.
        n_fixed_signs = len(self.data_variables) - len(to_assign) - len(self.always_zero)
        if n_fixed_signs:
            logger.info('Fitting data for %d irreversible/signed reactions '
                        'before choosing signs for %d other reactions.' % 
                        (n_fixed_signs, len(to_assign)))

            minimum_flux_solve(self, guess, offset=minimum_flux_offset)
            improved_guess = self.x.copy()
            logger.info('Initial best fit value %.3g' % self.soln['_objective'])
            now_set = {r for r,f in self.soln.iteritems() if r in 
                       to_assign and
                       np.abs(f) > self.threshold}
            logger.info('Initial best fit sets %d' % len(now_set))
            report_string = '; '.join(['%s (%.3g)' % (r, (self.soln[r]))
                                       for r in now_set])
            logger.debug('Those set are %s' % report_string)
            for other_reaction in now_set:
                self._set_objective_coefficient(
                    other_reaction, 
                    unsigned_coefficients[other_reaction]
                )
                self.set_sign(
                    other_reaction, 
                    np.sign(self.soln[other_reaction])
                )
                to_assign.remove(other_reaction)

            
        else:
            logger.info('Skipping first stage of direction assignment '
                        'because there are too few irreversible reactions.')
            improved_guess = guess.copy()
        
        ########################################
        # Iteratively add reversible reactions to the objective,
        # determining the optimal sign for each.
        # 
        problem_reactions = [] # This is just here to provide a useless return value
        logger.debug('to_assign')
        logger.debug(str(to_assign))
        
        this_pid = os.getpid()
        while to_assign:
            current_reaction = to_assign[0]
            print ('process %d: current_reaction is %s (%d remain)' %
                   (this_pid, current_reaction, len(to_assign)))
            to_assign.remove(current_reaction)
            self._set_objective_coefficient(current_reaction, 
                                            unsigned_coefficients[current_reaction])
            values = {}
            fluxes = {}
            results = {}
            deltas = {}
            n_fixed = {}
            others_set = {}

            for sign in (-1, 1):
                self.set_sign(current_reaction, sign)
                try:
                    minimum_flux_solve(self, improved_guess,
                                       offset=minimum_flux_offset)
                except nlcm.OptimizationFailure:
                    logger.debug('Optimization failure with improved guess testing %s' % r)
                    minimum_flux_solve(self, guess, offset=minimum_flux_offset)
                # For debugging purposes, track the value _before_ 
                # adding in other implicated reactions, because
                # this should be nondecreasing....
                values[sign] = self.soln['_objective']
                now_set = {r:np.sign(f) for r,f in self.soln.iteritems() if r in 
                           to_assign and
                           np.abs(f) > self.threshold}
                others_set[sign] = now_set.copy()
                report_string = '; '.join(['%s (%.3g)' % (r, (self.soln[r]))
                                          for r in now_set])
                logger.debug('Setting %s to sign %d fixes %s' % (current_reaction,
                                                                 sign,
                                                                 report_string))
                n_fixed[sign] = len(now_set)
                if n_fixed:
                    for other_reaction, other_reaction_sign in now_set.iteritems():
                        self._set_objective_coefficient(
                            other_reaction, 
                            unsigned_coefficients[other_reaction]
                        )
                        self.set_sign(
                            other_reaction, 
                            other_reaction_sign
                        )
                    try:
                        # Note that at this point we don't _need_
                        # a minimum-flux solution-- we only care about
                        # the value, immediately-- but I have 
                        # used one here so that the guess provided
                        # to the next iteration will have a nice,
                        # low-flux structure. Whether this makes a
                        # practical difference, I don't know. 
                        minimum_flux_solve(self, improved_guess,
                                           offset=minimum_flux_offset)
                    except nlcm.OptimizationFailure:
                        logger.debug('Optimization failure trying to min-flux-solve '
                                     'with improved guess; retrying w/ default guess')
                        minimum_flux_solve(self, guess,
                                           offset=minimum_flux_offset)
                fluxes[sign] = self.soln[current_reaction]
                results[sign] = self.x.copy()
                delta_variables = {current_reaction}
                delta_variables.update(already_assigned)
                delta_variables.update(now_set)
                deltas[sign] = delta(delta_variables, self.soln)
                for other_reaction in now_set:
                    self._set_objective_coefficient(other_reaction, 0.)
                    self.set_sign(other_reaction, 0.)

            message = ('%s, %s fixes %d: flux %.3g, bare value %.3g, ' +
                       'delta %.3g')
            for tag, i in (('Reversed', -1), ('Forward', 1)):
                logger.info(message % (tag, current_reaction,
                                       n_fixed[i], fluxes[i],
                                       values[i], deltas[i]))

            # Set the sign depending on the results.
            forward_vs_backward = deltas[1]-deltas[-1]
            logger.debug('%s margin: %.4g (negative implies forward direction is better)' %
                         (current_reaction, forward_vs_backward)) 
            if np.abs(forward_vs_backward) < improvement_threshold:
                sign = original_signs[current_reaction]
                logger.info('Insignificant difference between signs; keeping sign %d for %s' % 
                            (sign, current_reaction))
            elif forward_vs_backward < 0:
                # Positive sign results in a significantly better
                # change relative to the reference
                sign = 1
                logger.info('Forward is significantly better; %s set forward' %
                            current_reaction)
            else:
                # Negative sign results in a significantly better
                # change relative to reference
                sign = -1
                logger.info('Reverse is significantly better; %s set to reverse' % 
                            current_reaction)

            logger.info('... and setting %d others' % len(others_set[sign]))
            for other_reaction, other_sign in others_set[sign].iteritems():
                self._set_objective_coefficient(
                    other_reaction, 
                    unsigned_coefficients[other_reaction]
                )
                self.set_sign(
                    other_reaction, 
                    other_reaction_sign
                )
                to_assign.remove(other_reaction)
                already_assigned.append(other_reaction)
                        
            self.set_sign(current_reaction, sign)
            already_assigned.append(current_reaction)
            # Use the minimum-flux, best-fit-for-all-nonzero-reactions
            # case for whichever sign we ultimately decided on
            # as the guess for the next calculation
            improved_guess = results[sign]
            print 'Sign choice for %s concludes' % current_reaction
            #print 'New guess:'
            #print dict(zip(self.variables, improved_guess))

        final_parameters = self.parameters.copy()
        print 'Sign choice concludes.'
        return problem_reactions, final_parameters


