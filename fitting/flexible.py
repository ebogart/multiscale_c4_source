import os
import pickle
import logging
import numpy as np
import fluxtools.nlcm as nlcm
from fluxtools.nlcm import NonlinearModel, OptimizationFailure
from fluxtools.functions import Function, Linear
from fluxtools.fva import do_fva
from fluxtools.utilities.total_flux import minimum_flux_solve, add_total_flux_objective
from utilities import get_gra, get_reverse_gra

# The literature on mixing the basic logging module and 
# multiprocessing suggests that all sorts of unfortunate things may happen 
# if this logger is accessed from multiple processes at once under the 
# wrong circumstances (e.g., it is trying to write to a file;) currently I ignore
# this.
logger = logging.getLogger('fitting.flexible')

class FlexibleFittingModel(NonlinearModel):
    """ Data-fitting class with many complex data-variable relationships.

    """

    def solve(self, x0=None, options={}):
        """Solve the nonlinear optimization problem.

        Here I have awkwardly allowed attributes of the model to
        control whether the usual solve() method or repeated_solve()
        should be called (because trying to specify this in all the
        various methods which do calculations with this model would be
        a bit cumbersome at this point.)

        If self.repeated_solve_attempts is greater than 1,
        repeated_solve() will be called rather than solve, using that
        value for max_attempts and self._repeated_solve_max_iter
        for max_iter.

        """
        repeated_solve_attempts = getattr(self,'repeated_solve_attempts', 1)
        if repeated_solve_attempts > 1:
            repeated_solve_max_iter = getattr(self,
                                              'repeated_solve_max_iter', 1)
            return self.repeated_solve(x0,max_iter=repeated_solve_max_iter,
                                       max_attempts=repeated_solve_attempts,
                                       options=options)
        else:
            return NonlinearModel.solve(self, x0=x0,options=options)

    def repeated_solve(self, x0, max_iter, max_attempts, options):
        """Solve the nonlinear problem, restarting as required.

        This is essentially the same as
        NonlinearModel.repeated_solve() but must be modified here
        because if NonlinearModel.repeated_solve() is called from
        solve(), it will call solve() again in turn.

        """
        all_options = {'max_iter': max_iter}
        all_options.update(options)
        attempt = 0
        x = x0
        while attempt < max_attempts:
            try:
                x = NonlinearModel.solve(self, x0=x, options=all_options)
            except OptimizationFailure as e:
                if self.status not in {-1, -2}:
                    raise e
                else:
                    x = self.x.copy()
            else:
                return x
            attempt += 1
        # We have made the maximum number of restart attempts
        # but still not converged, or still ended up with a restoration 
        # failure.
        raise OptimizationFailure('No convergence after %d attempts (status %d)' %
                                  (max_attempts, self.status))

    def __init__(self, base_model,
                 load_fva_results=None,
                 save_fva_results=None,
                 n_procs=1,
                 zero_threshold=1e-4,
    ):
        """ Create a fitting model from an existing model.

        After initialization, the model may be set up to fit 
        data in various ways by creating appropriate Dataset 
        objects attached to the model.

        After all data sets have been added, the finalize() method
        should be called to set up the objective function and compile 
        the model.

        Arguments:
        base_model - existing model to copy (should be compiled, or have
            an up-to-date list of variables)
        load_fva_results - either a dict or the name of a file containing
            a pickled dict of fva results {'variable': (lower_bound, upper_bound)...}
            which should be assumed rather than recalculated. Use cautiously
            as nonsense may result if an incorrect set of results is assumed.
        save_fva_results - filename to which all the FVA results determined
            in the initialization process should be saved after initialization. 
        zero_threshold - float; quantities less than this in absolute values may
            be considered zero at various points in the code. Note that this should
            not be smaller than the IPOPT tolerance!
        n_procs: max number of processes to use

        """
        self.manual_lower_bounds = {} # These will be populated automatically
        self.manual_upper_bounds = {} # when _copy_model() calls set_bounds()
        NonlinearModel.__init__(self)
        self._copy_model(base_model)
        self._base_model_cache = base_model.copy()
        self.data_parameters = {}
        self.inverse_uncertainty_parameters = {}
        self.sign_parameters = {} # sign_parameters.keys() doubles as
                                  # a list of the signed variables.
                                  # Note that I assume that setting the 
                                  # sign to zero ensures that the derivatives
                                  # of all objective function terms with 
                                  # respect to the variable become zero
                                  # everywhere.
        self.critical_signs = set() # These signs must be chosen carefully and 
                                    # respected regardless of the magnitude of the
                                    # data associated with them
        self.max_sign_violation = 1.0 # An ostensibly negative (positive) variable
                                      # will be constrained to be less than this 
                                      # (greater than -1.0*this)-- should be small
                                      # but nonzero; best choice will depend on the 
                                      # details of the datasets
        self.datasets = {}  # {id_string: dataset_object} 
        self.threshold = zero_threshold
        self.n_procs = n_procs 
        if load_fva_results:
            self._load_fva_cache(load_fva_results)
        self.save_fva_results = save_fva_results
        
        # Allow the exclusion of some variables from 
        # sign choice, when otherwise they would be included
        self.suppress_signs = set()

    def finalize(self, total_flux_decomposition_max=1e4):
        # Identify variables of unknown sign
        self.check_signs()
        if self.save_fva_results:
            with open(self.save_fva_results,'w') as f:
                pickle.dump(getattr(self, '_fva_cache', {}), f)
        
        # Set up the objective function, taking into account all data 
        # sets. Note we could probably speed this up by treating this 
        # as a Linear object but I can't remember exactly when it's possible
        # to get away with using parameters, rather than literal numbers,
        # as coefficients for the Linear class. 
        #
        # Also track whether the objective is guaranteed to be positive (for
        # sensible choices of the tuning parameters, at least.)
        self.nonnegative_objective = True
        objective_terms = []
        self.dataset_weights = {}
        for dataset_id, dataset in self.datasets.iteritems():
            if not dataset.nonnegative_objective_term:
                self.nonnegative_objective = False
            if dataset.cost_variable is None:
                continue
            tuning_parameter = 'tuning_%s' % dataset.id
            self.parameters[tuning_parameter] = 1.
            self.dataset_weights[dataset_id] = tuning_parameter
            objective_terms.append((tuning_parameter,
                                    dataset.cost_variable))
        objective_math = ' + '.join(['%s*%s' % (variable, coefficient) for
                                     variable, coefficient in 
                                     objective_terms])
        print 'math %s' % objective_math
        print self.dataset_weights
        print self.datasets
        objective = Function(objective_math, name='combined_objective')
        self.objective_function = objective

        # Add minimum flux objective.
        # In this model flux minimization is used for two distinct purposes:
        # 
        # 1. In the process of sign assignment, we want to identify
        # the minimum number of reactions with as-yet-undetermined
        # sign which must be nonzero to achieve the best objective
        # value in concert with the reaction whose sign we are
        # currently testing (see compare_signs()).
        # 
        # 2. In the final fitting step, we (may possibly) want to
        # choose a single optimal flux distribution as the best
        # fitting solution.
        # 
        # In principle, for case 1, it would be best to minimize the
        # flux through only those reactions whose sign has not yet
        # been set; while in case 2, it is standard to minimize the
        # flux through all reactions.
        # 
        # Tweaking the set of reactions whose flux is minimized for
        # each new reaction tested in case 1 is tedious and so far we
        # have obtained acceptable results without doing this, so I
        # have somewhat lazily used the same flux-minimizing objective
        # function, considering all reactions, for both cases. If this
        # becomes a problem it should not be too hard to revisit this
        # decision and set up some way to dynamically change the
        # coefficients in the total-flux function at each step in case
        # 1.

        # (Not sure the list of variables will be updated yet, but 
        # we expect all the reactions in the base model to become variables
        # in this model.)
        min_flux_reactions = [r.id for r in self._base_model_cache.reactions]
        add_total_flux_objective(self, reactions=min_flux_reactions,
                                 flux_upper_bound=total_flux_decomposition_max)

        # add_total_flux_objective compiles the model, so we don't need to
        # do so here explicitly.

    def get_sign_weights(self):
        """ Calculate the approximate importance of each signed reaction.

        This will take into account contributions from all the datasets,
        and will be used to prioritize reactions for sign choice.

        Return value: a dictionary {reaction1: float_value1, ... }
        giving a value (default zero) for each reaction 
        in self.sign_parameters. 

        """
        weights = dict.fromkeys(self.sign_parameters, 0.)
        for dataset_id, dataset_weight_param in self.dataset_weights.iteritems():
            dataset_weight = self.parameters[dataset_weight_param]
            dataset = self.datasets[dataset_id]
            if dataset.sign_weight_coefficients is None:
                continue
            for reaction, coefficients in dataset.sign_weight_coefficients.iteritems():
                if reaction not in weights:
                    continue
                total_weight = 0.
                for parameter, coefficient in coefficients.iteritems():
                    total_weight += coefficient*self.parameters[parameter]
                weights[reaction] += total_weight * dataset_weight
        return weights

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

    def optimize_signs(self, to_assign, guess, skip_tiny=True,
                        skip_threshold=None,
                        improvement_threshold=None, tolerate_failures=True):
        """Find the best signs for some variables, heuristically.

        (Note signed variables are described as 'reversible reactions'
        below but they may be of other types as well.)

        First the fitting problem is solved with the current set of
        signs and the contribution of each variable-with-data to the
        objective function is determined (the reference solution.)
        
        The variables are then ordered by self.get_sign_weights(). If
        the weight value is less than skip_threshold (defaults to
        self.tiny_threshold or, if that is absent, self.threshold) and
        skip_tiny is true, the variable is given sign 1 arbitrarily.

        The remaining unsigned reversible reactions with data are then
        sorted in descending order of their corresponding data. For
        each reaction, the problem is solved with the corresponding
        sign set to first -1 and then +1. Minimum-flux optimal
        solutions are determined for each sign, and for each the
        objective function value is computed _taking into account the
        fluxes carried by other as-yet-unsigned reactions_. 

        If the difference between the values is smaller than
        improvement_threshold, the original sign of the reaction is
        kept; otherwise, the sign with the best (lowest) value is
        chosen.  (Here we could potentially accelerate the process by
        setting the signs of all the other reactions involved in the
        best solution, but this is somewhat dangerous.)

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
        logged and printed, and they are returned in the first return
        value, which otherwise is empty.

        Optimization failures which are not otherwise handled are
        detected, and the original signs are restored before reraising
        the optimization failure exception and exiting.

        Returns a list of signs chosen arbitrarily after failures, and
        the final set of parameters after all signs have been chosen.

        """
        print 'Entered optimize_signs (%s)' % to_assign
        # It is not clear how best to determine the 
        # offset to use in finding minimum-flux optimal states. 
        # The units are different from the implicit flux units of 
        # self.threshold (TODO: check that I have consistently used
        # that only for fluxes...). Here I have set an empirically
        # helpful value.
        minimum_flux_offset = 1e-5

        # Skip those reactions which this object has been set to 
        # ignore for whatever reason (e.g., we are confident they have 
        # been assigned correctly in previous steps of an iterative process)
        to_assign = [r for r in list(to_assign) if r not in self.suppress_signs]

        if improvement_threshold is None:
            improvement_threshold = self.threshold
        priorities = self.get_sign_weights()
        to_assign.sort(key=lambda v: priorities[v],
                       reverse=True)
        already_assigned = [k for k in self.sign_parameters if 
                            k not in to_assign]
        original_signs = {r: self.parameters[self.sign_parameters[r]]
                          for r in to_assign}

        if skip_threshold is None:
            skip_threshold = getattr(self, 'tiny_threshold', self.threshold)
        if skip_tiny:
            n_tiny = 0
            for r in to_assign[:]:
                priority = priorities[r]
                if priority < skip_threshold:
                    if r in self.critical_signs:
                        # We cannot skip this one (for whatever reason; mostly because
                        # we will really need this variable's absolute value, which 
                        # we get by multiplying it by its sign.)
                        continue
                    n_tiny += 1
                    self.set_sign(r, 1, leave_bounds=True)
                    logger.info(('Reaction %s given sign +1 without optimization ' +
                                 'because its associated datum is too small.') % r)
                    to_assign.remove(r)
                    already_assigned.append(r)
            logger.info('Total %d reactions with "tiny" data processed.' % n_tiny)

        # Relax any bounds on the reversible reactions which may have 
        # been set in accordance with their prior sign choices, 
        # by setting the sign to zero for each. Note this will
        # also effectively remove these reactions from the objective function.
        remaining_unsigned = set(to_assign)
        for v in to_assign:
            self.set_sign(v, 0)

        ########################################
        # Preprocesssing step: Fit to the irreversible reactions.
        # 
        logger.info('Fitting data for irreversible/signed reactions '
                    'before choosing signs for %d other reactions.' % 
                    len(to_assign))

        try:
            minimum_flux_solve(self, guess, offset=minimum_flux_offset)
        except nlcm.OptimizationFailure as e:
            for variable, sign in original_signs.iteritems():
                self.set_sign(variable, sign)
            raise e
        improved_guess = self.x.copy()
        logger.info('Initial best fit value %.3g' % self.soln['_objective'])

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
            try: 
                comparison = self._compare_signs(current_reaction, 
                                                 remaining_unsigned,
                                                 improved_guess,
                                                 guess, minimum_flux_offset)
                fluxes, values, n_fixed, results = comparison
                # Note values is a dict with keys +/- 1, not a tuple/list--
                # as indeed all these are
            except nlcm.OptimizationFailure:
                message = ('Optimization failure choosing sign for %s' %
                           current_reaction)
                print message
                logger.warn(message)
                problem_reactions.append(current_reaction)
                n_errors_tolerated += 1
                sign = 1
            else:
                # Set the sign depending on the results.
                forward_vs_backward = values[1]-values[-1]
                logger.debug('%s margin: %.4g (negative implies '
                             'forward direction is better)' %
                             (current_reaction, forward_vs_backward)) 
                if np.abs(forward_vs_backward) < improvement_threshold:
                    sign = original_signs[current_reaction]
                    logger.info('Insignificant difference between signs; '
                                'keeping sign %d for %s' % 
                                (sign, current_reaction))
                elif forward_vs_backward < 0:
                    sign = 1
                    logger.info('Forward is significantly better; %s set forward' %
                                current_reaction)
                else:
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
            print ('There were %d signs chosen arbitrarily after failures' %
                   len(problem_reactions))
            print problem_reactions
        return problem_reactions, final_parameters

    def _compare_signs(self, current_reaction, remaining_unsigned,
                       improved_guess, guess, minimum_flux_offset):
        values = {}
        fluxes = {}
        results = {}
        n_fixed = {}

        for sign in (-1, 1):
            self.set_sign(current_reaction, sign)
            try:
                minimum_flux_solve(self, improved_guess,
                                   offset=minimum_flux_offset)
            except nlcm.OptimizationFailure:
                logger.debug('Optimization failure with improved guess testing %s' %
                             current_reaction)
                minimum_flux_solve(self, guess, offset=minimum_flux_offset)

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
            values[sign] = self.soln['_objective']
            for other_reaction in now_set:
                self.set_sign(other_reaction, 0.)

        message = ('%s, %s fixes %d: flux %.3g, value %.3g')
        for tag, i in (('Reversed', -1), ('Forward', 1)):
            logger.info(message % (tag, current_reaction,
                                   n_fixed[i], fluxes[i],
                                   values[i]))
        return fluxes, values, n_fixed, results

    def set_lower_bound(self, v, bound, cache=True):
        NonlinearModel.set_lower_bound(self, v, bound)
        # Some unexpected behavior could occur here if resolve_name
        # is being used nontrivially
        if cache:
            self.manual_lower_bounds[v] = bound

    def set_upper_bound(self, v, bound, cache=True):
        NonlinearModel.set_upper_bound(self, v, bound)
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

    def set_sign(self, v, sign, tolerance=None, leave_bounds=False):
        """Set sign of a variable with sign parameter. 
        
        The bounds on v are reset to those in self.manual_lower_bounds
        and self.manual_upper_bounds (relaxing bounds imposed by
        previous calls to set_sign, if any.)

        Unless leave_bounds is True, if the sign is postive
        (resp. negative), the variable itself is constrained to be
        greater than -1*tolerance (resp. less than 1*tolerance), if
        this is a tighter lower (resp. upper) bound than the one
        already in place. The tolerance defaults to self.max_sign_violation.

        """ 
        if tolerance is None:
            tolerance = self.max_sign_violation
        self.parameters[self.sign_parameters[v]] = sign
        lb = self.manual_lower_bounds[v]
        ub = self.manual_upper_bounds[v]
        self.set_bound(v, (lb, ub))
        if not leave_bounds:
            if sign < 0:
                if ub is None or ub > tolerance:
                    self.set_upper_bound(v, tolerance, cache=False)
            if sign > 0:
                if lb is None or lb < -1.*tolerance:
                    self.set_lower_bound(v, -1.*tolerance, cache=False)

    def set_data(self, data):
        """Set data for fitting variables from a dict.

        Data settings for variables not in the dict are unchanged.

        """
        for variable, datum in data.iteritems():
            for parameter in self.data_parameters[variable]:
                self.parameters[parameter] = datum

    def set_uncertainties(self, uncertainties):
        """Set uncertainties in data from a dict. 

        Uncertainty parameters for variables not in the dict are
        unchanged.

        """
        for variable, uncertainty in uncertainties.iteritems():
            inv_parameters = self.inverse_uncertainty_parameters[variable]
            for parameter in inv_parameters:
                self.parameters[parameter] = 1./uncertainty

    def fit(self, guess=None, skip_signs=False):
        """Fit to data, assigning signs to variables as needed.

        """
        if guess is None:
            guess = np.ones(self.nvar)
        
        # Try to determine the optimal sign choice for the reversible reactions
        if not skip_signs:
            return_values = self.optimize_signs(self.unknown_signs, guess=guess)
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

    def check_signs(self):
        """Determine which signed variables may be positive/negative. 

        All variables in self.sign_parameters are checked (by looking
        up upper and lower bounds and, if that is not determinative,
        performing FVA calculations).

        Two attributes are then populated:
            - self.unknown_signs, those which truly may take values of either sign 
            - self.always_zero, those which may not differ from zero by more
              than self.threshold.

        Next, for each variable with a fixed sign, the corresponding
        sign parameter is set (through self.set_sign.)

        For variables in self.unsigned and self.always_zero, sign
        parameters are set to 1 arbitrarily. 
        
        Note the results, and thus the behavior of the fitting
        process, may change if variable upper and lower bounds are
        changed.

        """
        # Populate three collections of variables depending on 
        # what we determine about their signs:
        fixed_signs = {}
        always_zero = []
        either_sign = [] 
        
        # First, examine upper and lower bounds, listing 
        # variables which need further attention
        to_check = [] 

        for v in self.sign_parameters:
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
        self.unknown_signs = either_sign
        self.unsigned = self.unknown_signs # This is a misleading
                                           # name, and using two names
                                           # for this list like this
                                           # is asking for trouble,
                                           # but we need it for
                                           # compatibility with the
                                           # replica fitting code.
        for v, sign in fixed_signs.iteritems():
            # Note that, without the leave_bounds flag, this call to
            # set_sign will establish upper/lower bounds on variables
            # which, for whatever reason, we know already cannot
            # violate those bounds. This will probably be harmless in
            # most cases.
            self.set_sign(v, sign)
        # Sign parameters for indeterminate or always-zero variables
        # are arbitary; set them to 1. 
        for v in always_zero + either_sign:
            self.parameters[self.sign_parameters[v]] = 1.

class Dataset():
    def __init__(self, model, id_, max_scale_factor):
        self.model = model
        self.model.datasets[id_] = self
        self.id = id_ 
        self.max_scale_factor = max_scale_factor
        # List the contsraints, variables, scale factors, etc., in the model
        # which are owned by this dataset. Note that signs are shared among
        # datasets.
        self.constraints = {} # {id_string: function_object}
        self.constraint_bounds = {} # {id_string: float or tuple or None} 
        self.data_parameters = {} # {'var1': 'datum_var1', ...}
        self.inverse_uncertainty_parameters = {} # {'var1':
                                                 # 'inv_uncertainty_var1'...}
        # A note about scale factors. I have split them up into various subsets
        # at different times. Here, the local scale factors-- and the global
        # scale factors when they are treated as optimizable variables in the 
        # replica problem-- should be bounded above and below by self.
        # max_scale_factor and should contribute to a scale factor prior
        # cost term (if there is one.) 
        # The direct scale factors are the quantities which directly
        # multiply other variables to rescale them. These may be 
        # local or global scale factors as well (but not both); if they
        # are neither, they are assumed to be specified by linear combinations
        # of other variables, (but are still given specific upper and lower 
        # bounds), and do not contribute to scale factor prior cost terms.
        # The values of the self.direct_scale_factors dictionary 
        # are the variables scaled. 
        self.global_scale_factors = set() 
        self.local_scale_factors = set() 
        self.direct_scale_factors = {}

        self.cost_variable = None
        self.sign_weight_coefficients = None

        # By default, indicate that the model will never
        # contribute a negative term to the overall objective function
        self.nonnegative_objective_term = True

        self.default_datum = 0.
        self.default_scale = 0.
        self.default_inverse_uncertainty = 1.

        # Some variables, e.g. sums of squares, must be
        # nonnegative when constraints are satisfied, and
        # setting a lower bound on them can improve convergence 
        # behavior, but I have found it best to use a lower
        # bound slightly below zero.
        self.nonnegative_lower_bound = -1.
        self.setup()
        self.finalize()

    def ensure_signed(self, variable, critical=False):
        if critical: 
            self.model.critical_signs.add(variable)
        return self.model.sign_parameters.setdefault(variable,
                                                     'sign_%s' % variable)
    
    def finalize(self):
        for constraint_id, function_object in self.constraints.iteritems():
            self.model.constraints.set(constraint_id, function_object)
            bounds = self.constraint_bounds[constraint_id]
            self.model.set_bound(constraint_id, bounds)
        for key, parameter in self.data_parameters.iteritems():
            self.model.data_parameters.setdefault(key,[]).append(parameter)
        for variable, data_parameter in self.data_parameters.iteritems():
            self.model.parameters[data_parameter] = self.default_datum
        for key, parameter in self.inverse_uncertainty_parameters.iteritems():
            self.model.inverse_uncertainty_parameters.setdefault(key, []).append(parameter)
        for variable, parameter in self.inverse_uncertainty_parameters.iteritems():
            self.model.parameters[parameter] = self.default_inverse_uncertainty
        for scale_factor in self.global_scale_factors:
            self.model.parameters[scale_factor] = self.default_scale
        for scale_factor in self.local_scale_factors:
            self.model.set_bounds(scale_factor, (-1.0*self.max_scale_factor,
                                                 self.max_scale_factor))

    def setup_errors(self, targets, data_keys=None, use_signs=True):
        """ Add variables representing deviation of targets from data. 

        Returns a list of the error variables, in order.

        If data_keys is given, it should be a dictionary 
            {'target_variable_1': 'name_v1', ...}
        Then the parameter associated with target_variable_1 will
        be listed in self.data_parameters as eg:
            {'name_v1': 'datum_target_variable_1'}
        rather than the default
            {'target_variable_1': 'datum_target_variable_1'}

        This does not affect the internal operation of the model, but changes
        the keys of the dictionaries supplied to the set_data() and set_error()
        parameters of the associated FlexibleModel instance.

        """
        if data_keys is None:
            data_keys = {k:k for k in targets}

        error_variables = []
        # Future work should introduce a new variable to represent the
        # rescaled variable, to further isolate the nonlinearity 
        # (at a guess)
        if use_signs:
            error_math = 'error_v - (datum_v - v * sign_v * exp(scale_factor_v))'
        else:
            error_math = 'error_v - (datum_v - v * exp(scale_factor_v))'
        error_template = Function(error_math, name='error_template')
        error_template.all_first_derivatives()
        error_template.all_second_derivatives()
        for r in targets:
            datum_id = '%s_datum_%s' % (self.id, r)
            self.data_parameters[data_keys[r]] = datum_id
            error_variable = '%s_error_%s' % (self.id, r)
            error_variables.append(error_variable)
            scale_id = '%s_scale_%s' % (self.id, r)
            self.direct_scale_factors[r] = scale_id

            # substitute into error_template and 
            # impose the resulting constraint
            substitutions = {'v': r,
                             'scale_factor_v': scale_id,
                             'error_v': error_variable,
                             'datum_v': datum_id}
            if use_signs:
                sign = self.ensure_signed(r)
                substitutions['sign_v'] = sign
            error_constraint_id = '%s_constraint' % error_variable
            error_constraint = error_template.substitute(
                substitutions
            )
            error_constraint.name = error_constraint_id
            self.constraints[error_constraint_id] = error_constraint
            self.constraint_bounds[error_constraint_id] = 0.
            
        return error_variables

    def setup_leastsquares_cost(self, variables, error_variables, data_keys=None):
        """ 
        Constrain a variable to equal the sum of squares of the error variables.

        Returns the id of the variable equal to the cost. 

        data_keys functions here to set the keys of the inverse uncertainty
        parameters, as it does in setup_errors to control the keys of the
        data parameters.

        """
        if data_keys is None:
            data_keys = {k:k for k in variables}

        ###
        # costs = []
        # single_cost_math = 'cost_id - (iu_id**2 * error_id**2)'
        # single_cost_template = Function(single_cost_math,
        #                                 name='single_cost_template')
        # single_cost_template.all_first_derivatives()
        # single_cost_template.all_second_derivatives()

        # for variable, error_variable in zip(variables, error_variables):
        #     cost_id = '%s_cost_term_%s' % (self.id, variable)
        #     costs.append(cost_id)
        #     iu_id = '%s_inverse_uncertainty_%s' % (self.id, variable)
        #     self.inverse_uncertainty_parameters[data_keys[variable]] = iu_id
        #     # Constrain cost_id to equal iu_id**2 * error_variable**2
        #     substitutions = {'cost_id': cost_id,
        #                      'iu_id': iu_id,
        #                      'error_id': error_variable}
        #     single_cost_constraint_id = '%s_constraint' % cost_id
        #     single_cost_constraint = single_cost_template.substitute(
        #         substitutions
        #     )
        #     single_cost_constraint.name = single_cost_constraint_id
        #     self.constraints[single_cost_constraint_id] = single_cost_constraint
        #     self.constraint_bounds[single_cost_constraint_id] = 0.
        #     self.model.set_bound(cost_id, (self.nonnegative_lower_bound, None))
        
        # name = '%s_least_squares' % self.id
        # cost_id = '%s_overall_cost' % name
        # terms = dict.fromkeys(costs, 1.)
        # terms['cost_id'] = -1.
        # cost_function = Linear(terms, name)
        # self.constraints[name] = cost_function
        # self.constraint_bounds[name] = 0.
        # self.model.set_bound(cost_id, (self.nonnegative_lower_bound, None))
        # return cost_id

        ###
        # Collect terms and derivatives of the cost, and auxiliary variable names
        terms = []
        derivs = {}
        second_derivs = {}
        for variable, error_variable in zip(variables, error_variables):
            iu_id = '%s_inverse_uncertainty_%s' % (self.id, variable)
            self.inverse_uncertainty_parameters[data_keys[variable]] = iu_id
            # Add a term to the objective function 
            terms.append('%s**2 * %s**2' % (iu_id, error_variable))
            derivs[error_variable] = '2*%s*(%s**2)' % (error_variable, iu_id)
            second_derivs[(error_variable, 
                           error_variable)] = '2*(%s**2)' % iu_id
        
        # Future (probable) improvement: add one variable per 
        # squared cost, set up the sum as a linear function
        name = '%s_least_squares' % self.id
        cost_id = '%s_cost' % name
        self.model.set_bound(cost_id, (self.nonnegative_lower_bound, None))
        terms.append('-1.0*%s' % cost_id)
        derivs[cost_id] = '-1.0'
        math = nlcm.tree_sum(terms)
        cost_function = Function(math, first_derivatives=derivs,
                                 second_derivatives=second_derivs, 
                                 name=name)
        self.constraints[name] = cost_function
        self.constraint_bounds[name] = 0.
        return cost_id
    
class PerReactionData(Dataset):
    """ Fit flux through reactions to specified data. """
    def __init__(self, model, id_, reactions, max_scale_factor=5.,
                 scale_factor_relationships={}, local_scales=set()):
        self.reactions = reactions
        self.scale_factor_relationships = scale_factor_relationships
        self.local_scale_factors = local_scales
        Dataset.__init__(self, model, id_, max_scale_factor)

    def setup(self):
        # Set up the error terms
        error_variables = self.setup_errors(self.reactions)
        
        # The least-squares cost
        least_squares_cost_id = self.setup_leastsquares_cost(self.reactions,
                                                             error_variables)
        
        # Then the scale factor relationships
        self.setup_scale_relationships(self.scale_factor_relationships)
        
        # The scale factor prior cost, for the local scale factors, 
        # if there are any 
        if self.local_scale_factors:
            terms = []
            derivs = []
            second_derivs = []
            self.local_scale_priors = {}
            for factor in self.local_scale_factors:
                prior = '%s_prior' % factor # need to tag this with self.id?
                self.local_scale_priors[factor] = prior
                self.model.parameters[prior] = 0.
                terms.append('(%s-%s)**2' % (factor, prior))
                derivs[factor] = '2.*(%s-%s)' % (factor, prior)
                second_derivs[(factor, 
                               factor)] = '2.'
            math = nlcm.tree_sum(terms)
            name = '%s_local_scales' % self.id
            local_scale_cost_id = '%s_cost' % name
            terms.append('-1.0*%s' % local_scale_cost_id)
            derivs[local_scale_cost_id] = '-1.0'
            cost_function = Function(math, first_derivatives=derivs,
                                     second_derivatives=second_derivs, 
                                     name=name)
            self.constraints[name] = cost_function
            self.constraint_bounds[name] = 0.
        else:
            local_scale_cost = None
            
        # The overall cost
        total_cost_id = '%s_total_cost' % self.id
        total_cost_terms = [('-1.0',total_cost_id)]
        least_squares_coefficient = '%s_least_squares_coefficient' % self.id
        self.model.parameters[least_squares_coefficient] = 1.
        total_cost_terms.append((least_squares_coefficient, 
                                 least_squares_cost_id))
        if local_scale_cost:
            local_scale_coefficient = '%s_least_squares_coefficient' % self.id
            self.model.parameters[local_scale_coefficient] = 1.
            total_cost_terms.append((local_scale_coefficient, 
                                     local_scale_cost))
        total_cost_math = ' + '.join(['%s*%s' % (variable, coefficient) for
                                      variable, coefficient in 
                                      total_cost_terms])
        total_cost_constraint_name = '%s_total_cost_constraint' % self.id
        total_cost_function = Function(total_cost_math, name=total_cost_constraint_name)
        self.constraints[total_cost_constraint_name] = total_cost_function
        self.constraint_bounds[total_cost_constraint_name] = 0.

        self.cost_variable = total_cost_id
        
        # The sign choice priority hints
        # These are straightforward here:
        self.sign_weight_coefficients = {r: {self.data_parameters[r]:
                                             1.} for r in self.reactions}

    def setup_scale_relationships(self, relationships):
        """Constrain direct scale factors to be sums of underlying scale factors.

        Adds constraints (but does not compile), sets bounds on direct
        scale factors which become variables, adds underlying
        scale factors to self.global_scale_factors unless they 
        are already in self.local_scale_factors; checks to 
        ensure all entries in self.local_scale_factors
        are sensible.

        Assumes self.direct_scale_factors is already populated.

        Arguments:
        relationships - a dictonary {'ms_v1': ['overall_scale',
                                                            'ms_scale',
                                                            'v1_scale',...] ...}
                        giving for each direct scale factor the set of 
                        indirect scale factors which are added together to 
                        obtain it.
        
        """
        independent_scale_factors = set()

        for var, direct_factor in self.direct_scale_factors.iteritems():
            if var in relationships:
                indirect_factors = relationships[var]
                for f in indirect_factors:
                    independent_scale_factors.add(f)
                n_factors = len(indirect_factors)
                self.model.set_bound(direct_factor, 
                                     (-1.*n_factors*self.max_scale_factor,
                                      n_factors*self.max_scale_factor))
                constraint_id = '%s_decomposition_%s' % (self.id, direct_factor)
                coefficients = dict.fromkeys(indirect_factors, 1.)
                coefficients[direct_factor] = -1.
                g = Linear(coefficients, constraint_id)
                self.constraints[constraint_id] = g
                self.constraint_bounds[constraint_id] = 0.
            else:
                independent_scale_factors.add(direct_factor)
        
        for f in self.local_scale_factors:
            if f not in independent_scale_factors:
                raise ValueError('Invalid local scale factor specified.')
            else:
                independent_scale_factors.remove(f)
        
        self.global_scale_factors.update(independent_scale_factors)

class EnzymeUpperBoundData(Dataset):
    """ Use enzyme data to set upper bounds on multiple reactions. """
    def __init__(self, model, id_, enzymes_to_reactions):
        """
        Arugments:
        id_ - string
        enzymes_to_reactions - dictionary of the form 
        {'enzyme_name': ('reaction_1', 'reaction_2', ....)}
        
        This will create a variable 
            id_total_flux_enzyme_name 
        constrained to equal
            sign_reaction_1 * reaction_1 + ...
        and constrained to be in the range 
            [-1.0*id_datum_enzyme_name, id_datum_enzyme_name]
        with 
            self.data_parameters['enzyme_name'] = id_datum_enzyme_name

        There is no associated cost and no contribution to the 
        reaction direction choice priorities.
        
        """
        self.enzymes = list(enzymes_to_reactions)
        # Some of the reactions listed in the structure argument may
        # not be valid variables; remove those (we could force the
        # user/calling code to fix this, but this makes the interface
        # more consistent with the metabolite dataset, though the
        # invalid-variables problem is somewhat different there)
        self.valid_variables = set(model._base_model_cache.variables)
        self.structure = {k: tuple([r for r in v if r in self.valid_variables]) for
                          k,v in enzymes_to_reactions.iteritems()}
        Dataset.__init__(self, model, id_, max_scale_factor=None)

    def setup(self): 
        # Impose constraints
        for enzyme, reactions in self.structure.iteritems():
            # First, constrain the sum of the signed fluxes
            # to equal the enzyme's total flux:
            total_flux_variable = '%s_total_flux_%s' % (self.id,
                                                        enzyme)
            # This quantity will always be positive:
            self.model.set_lower_bound(total_flux_variable,
                                       self.nonnegative_lower_bound)

            terms = ['-1.0*%s' % total_flux_variable]
            for reaction in reactions:
                sign_parameter = self.ensure_signed(reaction, critical=True)
                terms.append('%s*%s' % (sign_parameter, reaction))
            
            flux_math = ' + '.join(terms)
            flux_constraint_id = '%s_constraint' % total_flux_variable
            flux_function = Function(flux_math, name=flux_constraint_id)
            self.constraints[flux_constraint_id] = flux_function
            self.constraint_bounds[flux_constraint_id] = 0.

            # Next, require the absolute value of the total flux
            # to be less than the datum for this enzyme:
            datum_id = '%s_datum_%s' % (self.id, enzyme)
            self.data_parameters[enzyme] = datum_id

            # add constraints requiring total_flux to be in correct range
            # The lower bound constraint is redundant, as the total 
            # flux is guaranteed to be positive. TODO: remove this constraint.
            lb_id = '%s_lb' % total_flux_variable
            lb_constraint = Linear({datum_id: 1.,
                                    total_flux_variable: 1.},
                                   lb_id)
            self.constraints[lb_id] = lb_constraint
            self.constraint_bounds[lb_id] = (0., None)

            ub_id = '%s_ub' % total_flux_variable
            ub_constraint = Linear({datum_id: 1.,
                                    total_flux_variable: -1.},
                                   ub_id)
            self.constraints[ub_id] = ub_constraint
            self.constraint_bounds[ub_id] = (0., None)

        # We must have a cost variable even if it is None:
        self.cost_variable = None

        # These are straightforward here:
        self.sign_weight_coefficients = None


class ReactionUpperBoundData(Dataset):
    """ Use data to set upper bounds on individual reactions. """
    def __init__(self, model, id_, reactions):
        """Arugments:
        id_ - string
        reactions - list of reactions
        
        This will create for each variable in reactions two constraint
        functions, representing the lower and upper sides of a constraint
        requiring the variable to be in the range
            [-1.0*id_datum_reactionid, id_datum_reactionid]
        with 
            self.data_parameters['reactionid'] = id_datum_reactionid

        This may seem like a needlessly indirect way to proceed when
        we could instead set upper and lower bounds on the variable
        itself, but there is no automatic way to update lower and
        upper bounds on variables when a parameter is changed. This
        also ensures that the data-derived upper and lower bounds are
        never relaxed by accident.

        There is no associated cost and no contribution to the 
        reaction direction choice priorities.

        """
        self.reactions = reactions
        # Some of the reactions listed in the structure argument may
        # not be valid variables; if so, raise an exception        
        self.valid_variables = set(model._base_model_cache.variables)
        if [v for v in reactions if v not in self.valid_variables]:
            raise ValueError('Invalid reactions.')
        Dataset.__init__(self, model, id_, max_scale_factor=None)

    def setup(self): 
        # Impose constraints
        for reaction in self.reactions:
            datum_id = '%s_datum_%s' % (self.id, reaction)
            self.data_parameters[reaction] = datum_id

            # add constraints requiring total_flux to be in correct range
            lb_id = '%s_%s_lb' % (self.id, reaction)
            lb_constraint = Linear({datum_id: 1.,
                                    reaction: 1.},
                                   lb_id)
            self.constraints[lb_id] = lb_constraint
            self.constraint_bounds[lb_id] = (0., None)
 
            ub_id = '%s_%s_ub' % (self.id, reaction)  
            ub_constraint = Linear({datum_id: 1.,
                                    reaction: -1.},
                                   ub_id)
            self.constraints[ub_id] = ub_constraint
            self.constraint_bounds[ub_id] = (0., None)

        # We must have a cost variable even if it is None:
        self.cost_variable = None

        # These are straightforward here:
        self.sign_weight_coefficients = None


class ObjectiveDataset(Dataset):
    """ Dataset class which adds a contribution to the objective term. """
    def __init__(self, model, id_, math, nonnegative=False):
        """ Add a generic term to the objective function.

        Sometimes (for example, using only datasets which provide upper
        and lower bounds, but no cost term) it is necessary to supply
        a contribution to the objective function which is not directly
        associated with data. Having a class for this is slightly silly
        but maintains the pattern of setting up the various components
        of the model, then calling finalize() (rather than setting up 
        the data components, calling finalize(), and resetting the 
        objective function.) This also provides an easy way to
        set up multiple terms in the objective function and adjust
        their relative weights.

        Internally this will become a constraint 
            cost_id - (math) = 0

        Arugments:
        id_ - string
        math - expression giving objective function 
        nonnegative - boolean specifying whether the expression will
            always be nonnegative (if all terms in the objective are,
            it may be bounded below by zero to improve solver performance.)
 
        There is no contribution to the reaction direction choice
        priorities.

        """
        self.math = math
        Dataset.__init__(self, model, id_, max_scale_factor=None)
        self.nonnegative_objective_term = nonnegative


    def setup(self): 
        # Constrain a variable to be equal to this term 
        # in the objective function
        self.cost_variable = 'cost_%s' % self.id
        constraint_id = '%s_constraint' % self.id
        math = '%s - (%s)' % (self.cost_variable, self.math)
        constraint = Function(math, name = constraint_id)
        self.constraints[constraint_id] = constraint
        self.constraint_bounds[constraint_id] = 0.

        # These are straightforward here:
        self.sign_weight_coefficients = None

class EnzymeData(Dataset):
    """ Fit to enzyme activity data, allowing one enzyme <-> multiple reactions.

    """
    def __init__(self, model, id_, enzymes_to_reactions, max_scale_factor=5.):
        """
        Arugments:
        id_ - string
        enzymes_to_reactions - dictionary of the form 
        {'enzyme_name': ('reaction_1', 'reaction_2', ....)}
        
        This will create a variable 
            id_total_flux_enzyme_name 
        constrained to equal
            sign_reaction_1 * reaction_1 + ...
        with 
            self.data_parameters['enzyme_name'] = id_datum_enzyme_name
            self.inverse_uncertainty_parameters['enzyme_name'] = (etc...)

        A cost is then associated with the discrepancy between the
        total flux through each enzyme and the associated data, 
        in the usual way, with scale factors. 

        Data for an enzyme is divided up into equal contributions to 
        the direction choice priorities of its associated reactions.
        
        """
        self.enzymes = list(enzymes_to_reactions)
        # Some of the reactions listed in the structure argument may
        # not be valid variables; remove those (we could force the
        # user/calling code to fix this, but this makes the interface
        # more consistent with the metabolite dataset, though the
        # invalid-variables problem is somewhat different there)
        self.valid_variables = set(model._base_model_cache.variables)
        self.structure = {k: tuple([r for r in v if r in self.valid_variables]) for
                          k,v in enzymes_to_reactions.iteritems()}
        Dataset.__init__(self, model, id_, max_scale_factor)

    def setup(self): 
        # For each enzyme, set up the total flux variable
        total_flux_variables = [] 
        variables_to_enzymes = {}
        for enzyme in self.enzymes:
            # First, constrain the sum of the signed fluxes
            # to equal the enzyme's total flux:
            total_flux_variable = '%s_total_flux_%s' % (self.id,
                                                        enzyme)
            variables_to_enzymes[total_flux_variable] = enzyme
            total_flux_variables.append(total_flux_variable)

            # This quantity will always be positive:
            self.model.set_lower_bound(total_flux_variable,
                                       self.nonnegative_lower_bound)

            terms = ['-1.0*%s' % total_flux_variable]
            for reaction in self.structure[enzyme]:
                sign_parameter = self.ensure_signed(reaction, critical=True)
                terms.append('%s*%s' % (sign_parameter, reaction))
            
            flux_math = ' + '.join(terms)
            flux_constraint_id = '%s_constraint' % total_flux_variable
            flux_function = Function(flux_math, name=flux_constraint_id)
            self.constraints[flux_constraint_id] = flux_function
            self.constraint_bounds[flux_constraint_id] = 0.

        # Set up the error variables
        errors=self.setup_errors(total_flux_variables,
                                 data_keys=variables_to_enzymes,
                                 use_signs=False)

        # Specify that all scale factors should be considered global
        self.global_scale_factors.update(self.direct_scale_factors.values())

        # Set up the overall cost
        self.cost_variable = self.setup_leastsquares_cost(total_flux_variables, 
                                                          errors,
                                                          data_keys=variables_to_enzymes)
        
        # Set up the sign choice weights:
        self.sign_weight_coefficients = {}
        for enzyme, reactions in self.structure.iteritems():
            n_reactions = len(reactions)
            datum_id = self.data_parameters[enzyme]
            for reaction in reactions:
                self.sign_weight_coefficients.setdefault(reaction,{})[datum_id] = 1./n_reactions

class MetaboliteReporter(Dataset):
    """ 
    Track the flux through metabolite pools.

    """
    def __init__(self, model, id_, metabolites_to_species):
        """
        Arugments:
        id_ - string
        metabolites_to_species - dictionary of the form 
            {'metabolite_name': ('species_1', 'species_2', ....)}
            where species_1, species_2, etc., are keys of the 
            model's _base_model_cache.species list.

        This will create a variable 
            id_metabolite_name_pool_flux 

        The pool flux variable will be constrained to equal
            sign_reaction_1 * abs(stoichiometry_reaction_1) * reaction_1 + ...

        where stoichiometry_reaction_1 is the sum of the coefficients
        in reaction 1 of the metabolites in question (so that, e.g., a
        transport reaction moving the metabolite between compartments
        does not contribute to the total.) The sum runs over all
        reactions in model._base_model_cache.reactions.

        """
        self.metabolites = list(metabolites_to_species)
        self.structure = metabolites_to_species
        Dataset.__init__(self, model, id_, max_scale_factor=None)

    def setup(self, zero_threshold=1e-6): 
        # For each pool, set up the total flux variable
        total_flux_variables = [] 
        variables_to_metabolites = {}
        reactions = self.model._base_model_cache.reactions
        valid_variables = set(self.model._base_model_cache.variables)
        cache_stoichiometries = {}
        for metabolite in self.metabolites:
            relevant_species = self.structure[metabolite]
            # First, find the net stoichiometry coefficient of the pool
            # in each reaction
            stoichiometries = {}
            # (Alternatively, we could identify the relevant species,
            # then iterate through the reactions only once.)
            for r in reactions:
                # In some cases-- if the base model has been obtained
                # by calling the simplification code on a more
                # complicated model, e.g.-- the list of reactions may
                # include some whose fluxes are not to be taken as
                # meaningful variables in the base model. We check for 
                # this and ignore them if found.
                if r.id not in valid_variables:
                    continue
                net_stoichiometry = 0
                for species, coefficient in r.stoichiometry.iteritems():
                    if species in relevant_species:
                        net_stoichiometry += coefficient
                # Stoichiometry coefficients can be floats, so check 
                # over-paranoidly...
                if np.abs(net_stoichiometry) > zero_threshold:
                    stoichiometries[r.id] = np.abs(net_stoichiometry)

            if not stoichiometries:
                msg = 'Ignoring metabolite %s; it participates in no reactions.'
                logger.warn(msg % metabolite)
                continue

            cache_stoichiometries[metabolite] = stoichiometries

            # Then, constrain a variable to equal the total flux 
            # through the pool:
            total_flux_variable = '%s_%s_pool_flux' % (self.id,
                                                       metabolite)
            variables_to_metabolites[total_flux_variable] = metabolite
            total_flux_variables.append(total_flux_variable)

            # This quantity will always be positive:
            self.model.set_lower_bound(total_flux_variable,
                                       self.nonnegative_lower_bound)

            terms = ['-1.0*%s' % total_flux_variable]
            for reaction, coefficient in stoichiometries.iteritems():
                sign_parameter = self.ensure_signed(reaction, critical=True)
                ### TODO: serious problem if the coefficient 
                # needs more than a few decimal places of precision
                terms.append('%f*%s*%s' % (coefficient,sign_parameter, reaction))
            
            # TODO: tree_sum
            flux_math = ' + '.join(terms)
            flux_constraint_id = '%s_constraint' % total_flux_variable
            flux_function = Function(flux_math, name=flux_constraint_id)
            self.constraints[flux_constraint_id] = flux_function
            self.constraint_bounds[flux_constraint_id] = 0.

        self.cost_variable = None
        self.sign_weight_coefficients = None
        self.stoichiometries = cache_stoichiometries

        self.total_flux_variables = total_flux_variables
        self.variables_to_metabolites = variables_to_metabolites


class MetaboliteData(MetaboliteReporter):
    """ 
    Fit the total flux through metabolite pools to data on their concentration.

    This is theoretically dubious, but may be a useful heuristic.

    """

    def __init__(self, model, id_, metabolites_to_species, max_scale_factor=5.):
        """
        Arugments:
        id_ - string
        metabolites_to_species - dictionary of the form 
        {'metabolite_name': ('species_1', 'species_2', ....)}
        where species_1, species_2, etc., are keys of the 
        model's _base_model_cache.species list.

        This will create a variable 
            id_metabolite_name_pool_flux 
        with 
            self.data_parameters['enzyme_name'] = id_datum_enzyme_name
            self.inverse_uncertainty_parameters['enzyme_name'] = (etc...)

        The pool flux variable will be constrained to equal
            sign_reaction_1 * abs(stoichiometry_reaction_1) * reaction_1 + ...
        where stoichiometry_reaction_1 is the sum of the coefficients in reaction 1 
        of the metabolites in question (so that, e.g., a transport reaction
        moving the metabolite between compartments does not contribute to the
        total.) The sum runs over all reactions in
        model._base_model_cache.reactions. 

        A cost is then associated with the discrepancy between the
        total flux through each metabolite pool and the associated data, 
        in the usual way, with scale factors. 

        Data for a metabolite is divided up into contributions to 
        the direction choice priorities of its associated reactions
        according to weights TBD.
        
        """
        self.metabolites = list(metabolites_to_species)
        self.structure = metabolites_to_species
        Dataset.__init__(self, model, id_, max_scale_factor)

    def setup(self, zero_threshold=1e-6): 
        MetaboliteReporter.setup(self, zero_threshold)
        total_flux_variables = self.total_flux_variables
        variables_to_metabolites = self.variables_to_metabolites

        # Set up the error variables
        errors=self.setup_errors(total_flux_variables,
                                 data_keys=variables_to_metabolites,
                                 use_signs=False)

        # Specify that all scale factors should be considered global
        self.global_scale_factors.update(self.direct_scale_factors.values())

        data_keys = variables_to_metabolites
        self.cost_variable = self.setup_leastsquares_cost(total_flux_variables,
                                                          errors,
                                                          data_keys=data_keys)
        
        # Set up the sign choice weights:
        self.sign_weight_coefficients = {}
        for metabolite, reaction_weights in self.stoichiometries.iteritems():
            total_weight = np.sum(reaction_weights.values())
            datum_id = self.data_parameters[metabolite]
            for reaction, weight in reaction_weights.iteritems():
                c = float(weight)/total_weight
                self.sign_weight_coefficients.setdefault(reaction,{})[datum_id] = c
            
