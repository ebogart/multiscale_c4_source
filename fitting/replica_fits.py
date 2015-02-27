""" 
Fit fluxes to data at a series of points using replicas with shared scale factors.

"""
import time
import logging
logger = logging.getLogger('fitting.replica_fits')

from Queue import Empty
import multiprocessing as mp
import numpy as np
import fluxtools.fva as fva
import fluxtools.replica as replica
import leastsquares as ls
from fluxtools.functions import Linear, Function
from fluxtools.nlcm import OptimizationFailure
from fit_interface import FitInterface, FittingFailure

def _presolve_worker(model, use_soln, skip_signs, job_queue, result_queue):

    """Worker function for handling individual points in parallel.

    For use as a target of multiprocessing.Process. Each entry in
    job_queue should be an integer index .  The
    corresponding entry of result_queue will be 
        {index: result of model.solve_one_segment(index, use_soln, skip_signs)}
    unless an optimization failure occurs, in which case it will be 
        {index: 'failure'}.

    """

    done = 0

    while True:
        try:
            key = job_queue.get(timeout=30.0)
        except Empty:
            print 'Presolve worker finishing after completing %d tasks' % done
            return 
        try:
            result = model.solve_one_segment(key, use_soln, skip_signs)
            result_queue.put({key: result})
        except OptimizationFailure:
            result_queue.put({key: 'failure %d' % model.template_model.status})
        done += 1

class ReplicaFit(FitInterface):
    def _setup_fit_infrastructure(self, nonnegative_objective=True, **kwargs):
        # expects kwarg N
        self.N = kwargs['N']
        self._setup_template_model(**kwargs)
        split_parameters = (self.base_data_parameters + 
                            self.base_uncertainty_parameters + 
                            self.base_sign_parameters)
        logger.debug('Initializing replica model.')
        self.fitting_model = replica.ReplicaModel(self.template_model,
                                                  self.N, 
                                                  split_parameters=split_parameters)
        logger.debug('Freeing scale parameters')
        self._free_scales()
        # Constrain the variable 'combined_objective' 
        # to equal the sum of the images' objective functions
        logger.debug('Setting up objective function.')
        coefficients = {'combined_objective': -1.}
        coefficients.update({self.fitting_model.metaname('objective_value', i):
                             1. for i in xrange(self.N)})
        objective_sum = Linear(coefficients, name='objective_sum')
        self.fitting_model.constraints.set('objective_sum', objective_sum)
        self.fitting_model.set_bound('objective_sum', 0.)
        if nonnegative_objective:
            # In this case, we assume the objectives cannot meaningfully
            # be negative; enforce this explicitly to try to prevent IPOPT
            # from getting trapped in infeasible regions.
            for i in xrange(self.N):
                objective_i = self.fitting_model.metaname('objective_value', i)
                self.fitting_model.set_bound(objective_i, (-1., None))

        self._setup_scale_prior() # constraints 'scale_factor_prior'
                                  # and bounds scale factors as nec.
        overall_objective = Function('scale_factor_prior*scale_factor_tuning + combined_objective',
                                     name='overall_objective')
        self.fitting_model.objective_function = overall_objective
        self.fitting_model.parameters['scale_factor_tuning'] = 1.
        self._fixed_signs = {i:set() for i in xrange(self.N)}
        logger.debug('Compiling fitting model.')
        self.fitting_model.compile()
        logger.debug('Initialization complete.')
        
    def _free_scales(self):
        for s in self.base_scale_parameters:
            self.fitting_model.parameters.pop(s)
            self.fva_keys.add(s)

    def _setup_template_model(self):
        raise NotImplementedError
        # sets up:
        # self.template_model
        # self.base_data_parameters (a list)
        # self.base_uncertainty_parameters (a list)
        # self.base_sign_parameters (a list)
        # self.base_scale_parameters (a list)
        # self.starting_scale

    def _setup_scale_prior(self):
        raise NotImplementedError
        # constrains 'scale_factor_prior'

    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        """
        Find the best-fit solution to a single point of the replica. 

        i - index of image to fit
        use_soln - if true, start from the values of this image's variables
            in the most recent solution to the collective fit
        skip_signs - if true, do not try to optimize signs when fitting 
            this image

        """
        print 'Entered solve_one_segment (%d, %s, %s)' % (i, use_soln, skip_signs)
        logger.debug('Fitting segment %d separately' % i)
        logger.debug('... using previous solution' if use_soln else
                     '... not using previous solution')
        if skip_signs:
            logger.debug('... skipping sign check.')

        original_signs = {}
        for p in self.base_sign_parameters:
            meta_parameter = self.fitting_model.metaname(p, i)
            value = self.fitting_model.parameters[meta_parameter]
            self.template_model.parameters[p] = value
            original_signs[p] = value

        self.template_model.set_data({k:v[i] for k,v in 
                                      self.data.iteritems()})
        self.template_model.set_uncertainties({k:v[i] for k,v in 
                                               self.error.iteritems()})

        # Skip signs we are confident are correctly set for this image
        self.template_model.suppress_signs = self._fixed_signs[i]

        if use_soln:
            # Sometimes the solution is not a good starting guess!
            try:
                guess = self.fitting_model.x_by_image[:,i]
                self.template_model.fit(guess, skip_signs=skip_signs)
            except OptimizationFailure:
                logger.debug('In solve_one_segment use_soln, optimization failure!')
                guess = self.guess(self.template_model.nvar)
                self.template_model.fit(guess, skip_signs=skip_signs)
            finally:
                self.template_model.suppress_signs = set()
        else:
            guess = self.guess(self.template_model.nvar)
            try:
                self.template_model.fit(guess, skip_signs=skip_signs)
            finally:
                self.template_model.suppress_signs = set()


        all_signs = {p: self.template_model.parameters[p] for p, value
                     in original_signs.iteritems()}
        all_bounds = {v: self.template_model.get_bounds(v) for v, sign in 
                      self.template_model.sign_parameters.iteritems() if
                      sign in all_signs}
        changed_signs = {p: self.template_model.parameters[p] for p, value
                         in original_signs.iteritems() if value !=
                         self.template_model.parameters[p]}

        return (self.template_model.soln.copy(), self.template_model.obj_value, 
                all_signs, all_bounds, changed_signs)

    def presolve(self, use_soln=False):
        """Fit points separately, optimizing signs, to obtain a guess for the full problem.

        Returns:
        - a vector x0 of length self.fitting_model.nvar, which
          should be close to the optimal solution restricted to the case
          where each scale factor is fixed to its current value in the
          full problem (or self.starting_scale, if use_soln is False),
        - a summary of any signs which have been changed in the
          process, 
        - a dictionary of the _old_ values of parameters in self.fitting_model
          which have been changed,
        - a dictionary of the _old_ bounds on variables in self.fitting_model
          where those have been changed.

        The latter two should allow resetting the fitting model to its
        initial state if the result of presolve() is to be rejected.

        If self.skip_signs is True, signs are fixed in the subproblems
        to whatever value is set in the full problem.

        If use_soln is True, the values of the scale factors in
        self.fitting_model.soln are used in the subproblems, and
        self.fitting_model.x_by_image is mined for initial guesses for
        the optimization steps in the subproblems (otherwise,
        self.guess is called.)

        """
        print 'Entered presolve (%s)' % use_soln
        skip_signs = getattr(self, 'skip_signs', False)
        if skip_signs:
            logger.warn('Skipping sign check for debugging purposes')

        if use_soln:
            x0 = self.fitting_model.x.copy()
            # In this case, entries in x0 corresponding to the scale
            # parameters will never be changed
            for p in self.base_scale_parameters:
                if p in self.fitting_model.soln:
                    self.template_model.parameters[p] = self.fitting_model.soln[p]
        else:
            if self.random_guess:
                x0 = self.guess(self.fitting_model.nvar)
            else:
                x0 = np.zeros(self.fitting_model.nvar)
            # Here we need to explicitly set the entries corresponding to the
            # scale parameters to self.starting_scale (though in fact this
            # will typically be zero and we are not entirely prepared for it 
            # not to be, as that would affect scale factor priors, which we'd
            # also need to provide reasonable guesses for)...
            for p in self.base_scale_parameters:
                # Note that in some cases we will have turned the scales
                # _back_ into parameters, for testing or comparison
                # purposes
                if p in self.fitting_model.variables:
                    self.template_model.parameters[p] = self.starting_scale
                    x0[self.fitting_model.var_index[p]] = self.starting_scale

        objectives = []
        signs_changed = []

        per_segment_results = {}
        if self.n_processes == 1:
            for i in xrange(self.N):
                per_segment_results[i] = self.solve_one_segment(i, 
                                                                use_soln,
                                                                skip_signs)
        else:
            # Split the segments across processors.
            # We set up a process pool:
            argument_queue = mp.Queue()
            result_queue = mp.Queue()
            worker_args = (self, use_soln, skip_signs,
                           argument_queue, result_queue)
            n_workers = min(self.n_processes, self.N)
            processes = [mp.Process(target=_presolve_worker, args=worker_args) for 
                         i in xrange(n_workers)]
            # Populate the queue of segments to solve:
            for i in xrange(self.N):
                argument_queue.put(i)
    
            for p in processes:
                p.start()
                
            # Harvest the results in arbitrary order:
            for j in xrange(self.N):
                per_segment_results.update(result_queue.get())
            # Close the workers
            n_processes_joined = 0
            for p in processes:
                n_processes_joined += 1
                print 'Joining %d' % n_processes_joined
                p.join()
                print 'Joined %d' % n_processes_joined

            # Check for optimization failures and raise
            failed_segments = [(i,result) for i, result in 
                               per_segment_results.iteritems() if 
                               (isinstance(result, str) and 
                                result.startswith('failure'))]
            if failed_segments:
                failure_info = (len(failed_segments), str(failed_segments))
                message = ('Presolve encountered %d optimization failures (%s)' % 
                           failure_info)
                raise OptimizationFailure(message)

        print 'Processing results (presolve)...'
        reset_params = {}
        reset_bounds = {}
        for i in xrange(self.N):
            soln, value, signs, bounds, local_signs_changed = per_segment_results[i]
            for k,v in soln.iteritems():
                meta_k = self.fitting_model.metaname(k, i)
                x0[self.fitting_model.var_index[meta_k]] = v
            objectives.append(value)
            meta_obj = 'image%d_objective_value' % i
            x0[self.fitting_model.var_index[meta_obj]] = value

            for param, value in signs.iteritems():
                meta_parameter = self.fitting_model.metaname(param, i)
                old = self.fitting_model.parameters[meta_parameter]
                reset_params[meta_parameter] = old
                self.fitting_model.parameters[meta_parameter] = value
                if param in local_signs_changed:
                    signs_changed.append((i, param, old, value))

            for variable, bounds in bounds.iteritems():
                meta_variable = self.fitting_model.metaname(variable, i)
                reset_bounds[meta_variable] = self.fitting_model.get_bounds(
                    meta_variable
                )
                self.fitting_model.set_bound(meta_variable, bounds)
        print 'Done with result-processing loop (presolve)'

        # Set the remaining variable value...
        combined_index = self.fitting_model.var_index['combined_objective']
        x0[combined_index] = np.sum(objectives)

        if signs_changed:
            logger.info('%d signs changed: ' % len(signs_changed))
            logger.debug(str(signs_changed))

        return x0, signs_changed, reset_params, reset_bounds
        
    def _load_data(self):
        """Set data-related parameters in self.fitting_model.

        The approach here-- loading the data and error for each point
        i into the template model, then copying the relevant
        parameters from the template model, renaming them
        appropriately, and applying them to the full model-- may seem
        needlessly circuitous. It was designed to allow the replica
        model code to remain agnostic as to how the data are treated
        internally-- for example, the template model might handle
        uncertainties by storing them directly as parameters, or by
        storing the inverse of each uncertainty, or the inverse
        square...

        """
        for i in xrange(self.N):
            local_dataset = {k: v[i] for k,v in self.data.iteritems()}
            local_error = {k: v[i] for k,v in self.error.iteritems()}
            self.template_model.set_data(local_dataset)
            self.template_model.set_uncertainties(local_error)
            for base_p in (self.base_data_parameters + 
                      self.base_uncertainty_parameters):
                new_p = self.fitting_model.metaname(base_p, i)
                value = self.template_model.parameters[base_p]
                self.fitting_model.parameters[new_p] = value

    def _fit(self, step_limit=2, use_soln=False, rs_max_iter=200,
             rs_max_attempts=5, keep_signs=False):
        self.fit_log = [] # debugging
        self.guesses = []
        self.parameter_sets = []
        self.signs_log = {}
        self.scales_log = {}
        for v in self.template_model.unsigned:
            self.signs_log[v] = []
        for p in self.base_scale_parameters:
            self.scales_log[p] = [self.starting_scale]
        self._load_data()

        # If we are not reusing an old solution, clear the 
        # record of signs we believe to have been chosen correctly, 
        # unless otherwise instructed
        if not (use_soln or keep_signs):
            self._fixed_signs = {i:set() for i in xrange(self.N)}

        # Solve each point separately to find an approximately 
        # feasible initial guess
        logger.debug('Finding initial guess before step 0.')
        self.fit_log.append('Presolve called: %s' % time.ctime())
        guess, _, _, _ = self.presolve(use_soln=use_soln)
        self.fit_log.append('Presolve returned: %s' % time.ctime())
        for v in self.template_model.unsigned:
            p = self.template_model.sign_parameters[v]
            signs = [self.fitting_model.parameters[self.fitting_model.metaname(p,i)]
                     for i in xrange(self.N)]
            self.signs_log[v].append(signs)
        self.guesses.append(guess)
        step = 0 
        signs_changed = True
        while signs_changed and step < step_limit:
            logger.debug('Fitting step %d' % step)

            guess_value = self.fitting_model.eval_f(guess)
            self.fit_log.append('value before %d: %.5g' % (step, guess_value))
            if step > 0 and guess_value > self.fitting_model.obj_value:
                # Trying to find improved choices of sign with the
                # current values of the scale factors has led to a
                # worse overall result.  We really would like to
                # imagine this won't happen, but it has occasionally
                # in testing; the sign choice algorithm is far from
                # perfect.  We could continue and see if solving the
                # full model with this sign choice leads to a net
                # improvement, but then we have to preserve the old
                # solution, and it may be expensive; we can revisit
                # this choice if this comes up a lot.
                self.fitting_model.parameters.update(reset_params)
                self.fitting_model.set_bounds(reset_bounds)
                self.fit_log.append('Rejecting proposal and terminating.') 
                logger.warning('New choice of signs has increased objective function. Halting.')
                break

            self.parameter_sets.append(self.fitting_model.parameters.copy())
            self.fit_log.append('Solve called: %s' % time.ctime())
            self.fitting_model.repeated_solve(guess,max_iter=rs_max_iter,
                                              max_attempts=rs_max_attempts)
            self.fit_log.append('Solve returned: %s' % time.ctime())
            self.fit_log.append('value after %d: %.5g' % 
                                (step, 
                                 self.fitting_model.eval_f(self.fitting_model.x)))
            logger.info('Fitting step %d: value %.3g' % (step, 
                                                         self.fitting_model.obj_value))
            for p in self.base_scale_parameters:
                self.scales_log[p].append(self.fitting_model.soln.get(p,None))
                    
            logger.debug('Checking for sign changes after step %d' % step)
            
            self.fit_log.append('Presolve (from solution) called: %s' % time.ctime())
            pre_result = self.presolve(use_soln=True)
            guess, signs_changed, reset_params, reset_bounds = pre_result
            self.fit_log.append('Presolve (from solution) returned: %s' % time.ctime())
            self.guesses.append(guess)
            for v in self.template_model.unsigned:
                p = self.template_model.sign_parameters[v]
                signs = []
                old_signs = self.signs_log[v][-1]
                for i in xrange(self.N):
                    meta_parameter = self.fitting_model.metaname(p,i)
                    meta_value = self.fitting_model.parameters[meta_parameter]
                    signs.append(meta_value)
                    # As a heuristic, assume a sign which has persisted
                    # through two rounds of sign choice is correct.
                    if meta_value == old_signs[i]:
                        pass
                        # if v not in self._fixed_signs[i]:
                        #     logger.info(
                        #         'Sign of %s (image %d) fixed to %d' %
                        #         (v,i,meta_value)
                        #     )
                        # self._fixed_signs[i].add(v)
                        
                self.signs_log[v].append(signs)

            self.fit_log.append(signs_changed)
            step += 1

        if not signs_changed:
            logger.info('Fitting concludes after step %d: value %.3g'
                        % (step, self.fitting_model.obj_value))
        if signs_changed and step >= step_limit: 
            #raise FittingFailure()
            logger.warn('Terminating; overran step limit while trying to fix signs.')
        self._fit_postprocess()

    def _fit_postprocess(self):
        # establishes self.fit_x_by_image
        # and self.fit_result, fit_other, fit_parameters,
        # fit_overall_value, fit_value_by_image
        self.fit_result = {v: np.array([self.fitting_model.soln_by_image[i][v]
                                        for i in xrange(self.N)]) for
                           v in self.base_model.variables}
        # fit_other and fit_parameters could be improved by automatic detection
        # of names that appear to have been broadcast across images,
        # and rolling up of their values into arrays
        self.fit_other = {v: self.fitting_model.soln[v] for v in 
                          self.fitting_model.variables if v not in
                          self.base_model.variables}
        self.fit_parameters = self.fitting_model.parameters.copy()
        for p in (self.base_uncertainty_parameters + self.base_data_parameters + 
                  self.base_sign_parameters):
            meta_parameters = [self.fitting_model.metaname(p,i) for i in xrange(self.N)]
            v = np.array([self.fitting_model.parameters[p] for p in meta_parameters])
            self.fit_parameters[p] = v
        self.fit_overall_value = self.fitting_model.obj_value
        objectives_by_image = [self.fitting_model.metaname('objective_value', i) for
                               i in xrange(self.N)]
        self.fit_value_by_image = np.array([self.fitting_model.soln[v] for 
                                            v in objectives_by_image])
        self.fit_x_by_image = self.fitting_model.x_by_image.copy()

    def _objective_constrained_fva(self, objective_offset):
        # not tested yet
        objective_bounds = (None, self.fit_overall_value + objective_offset)
        keys = []
        for v in self.fva_keys:
            if v in self.base_model.variables:
                for i in xrange(self.N):
                    keys.append(self.fitting_model.metaname(v,i))
            else:
                keys.append(v)

        bad = [k for k in keys if k not in self.fitting_model.variables]

        if bad:
            print bad
            raise KeyError()

        extrema = fva.objective_constrained_fva(self.fitting_model,
                                                objective_bounds=objective_bounds,
                                                variables = keys,
                                                n_procs = self.n_processes,
                                                guess = self.fitting_model.x.copy(),
                                                check_failures=False)
        result = {}
        failures = 0
        for v in self.fva_keys:
            if v in self.base_model.variables:
                rows = []
                for i in xrange(self.N):
                    ex = extrema[self.fitting_model.metaname(v, i)]
                    if ex == 'failure':
                        failures += 1
                        rows.append((np.nan, np.nan))
                    else:
                        rows.append(ex)
                result[v] = np.array(rows)
            else:
                ex = extrema[v]
                if ex == 'failure':
                    failures += 1
                    result[v] = np.array((np.nan, np.nan))
                else:
                    result[v] = np.array(ex)
        self.extrema = result
        return failures

class ExampleSubclass(ReplicaFit):
    def _setup_template_model(self):
        # sets up:
        # self.template_model
        # self.base_data_parameters
        # self.base_uncertainty_parameters
        # self.base_sign_parameters
        # self.base_scale_parameters
        # self.starting_scale
        pass

    def _setup_scale_prior(self):
        # constrains 'scale_factor_prior' to be the sum
        # of the prior costs for the scale factors and 
        # bounds the scale_factor variables as needed
        pass

    def _set_type(self):
        pass
        # update self.metadata['type']

class ImprovedReplica(ReplicaFit):
    def _setup_template_model(self, **kwargs):
        relationships = kwargs.get('scale_relationships', {})
        self.template_model = ls.ImprovedFittingModel(
            self.base_model,
            self.data_keys,
            load_fva_results=self.fva_cache,
            n_procs=self.n_processes,
            scale_relationships=relationships
        )
        tm = self.template_model
        self.base_data_parameters = tm.data_parameters.values()
        self.base_uncertainty_parameters = tm.inverse_uncertainties.values()
        self.base_sign_parameters = tm.sign_parameters.values()
        self.base_scale_parameters = list(tm.independent_scale_factors)
        self.starting_scale = 0.

    def _set_type(self):
        self.metadata['type'] = 'replica_fits.ImprovedReplica'

    def _setup_scale_prior(self, **kwargs):
        # constrains 'scale_factor_prior' to be the sum
        # of the prior costs for the scale factors
        # Here, scale_factor_prior = scale_factor_prior_v0 + 
        # scale_factor_prior_v1 + ...
        # and scale_factor_prior_vi = scale_factor_vi**2
        # Also constrains scale factor priors to be nonnegative
        # (they will be automatically if the square constraint is obeyed,
        # but their explicit bounds will be respected even while the solver
        # is trying to find the feasible manifold; I hope this
        # will help prevent cases where IPOPT gets stuck trying to 
        # restore feasibility for a long time.
        template = Function('scale_factor_i**2 - cost_i')
        template.all_first_derivatives()
        template.all_second_derivatives()
        cost_variables = []
        for sf in self.template_model.independent_scale_factors:
            #self.fitting_model.set_bound(sf, (-10., 10.)) # do this in ls
            cost_variable = 'prior_%s' % sf
            name_map = {'scale_factor_i': sf, 
                        'cost_i': cost_variable}
            cost_variables.append(cost_variable)
            new_id = 'prior_%s_constraint' % sf
            g = template.substitute(name_map, new_id)
            self.fitting_model.constraints.set(new_id, g)
            self.fitting_model.set_bound(new_id, 0.)
            # Bounding the cost variable below by zero explicitly
            # seems to cause problems when, e.g., fixing the 
            # scale factors to 0
            self.fitting_model.set_bound(cost_variable, (-1., 120.))
        coefficients = dict.fromkeys(cost_variables, 1.)
        coefficients['scale_factor_prior'] = -1.
        overall_id = 'scale_factor_prior_constraint'
        g = Linear(coefficients, overall_id)
        self.fitting_model.constraints.set(overall_id, g)
        self.fitting_model.set_bound(overall_id, 0.)

    def _free_scales(self):
        for s in self.base_scale_parameters:
            self.fitting_model.parameters.pop(s)
            self.fva_keys.add(s)
            self.fitting_model.set_bound(s, (-10., 10.))

class SignChoiceReplica(ImprovedReplica):
    def _setup_template_model(self, **kwargs):
        relationships = kwargs.get('scale_relationships', {})
        self.template_model = ls.SignChoiceTest(self.base_model,
                                      self.data_keys,
                                      load_fva_results=self.fva_cache,
                                                n_procs=self.n_processes,
                                                scale_relationships=relationships)
        tm = self.template_model
        self.base_data_parameters = tm.data_parameters.values()
        self.base_uncertainty_parameters = tm.inverse_uncertainties.values()
        self.base_sign_parameters = tm.sign_parameters.values()
        self.base_scale_parameters = list(tm.independent_scale_factors)
        self.starting_scale = 0.

    def _set_type(self):
        self.metadata['type'] = 'replica_fits.SignChoiceReplica'

class BatchSignChoiceReplica(ImprovedReplica):
    def _setup_template_model(self, **kwargs):
        # TODO: handle scale relationships
        raise NotImplementedError()
        self.template_model = ls.BatchSignChoice(self.base_model,
                                                 self.data_keys,
                                                 load_fva_results=self.fva_cache,
                                                 n_procs=self.n_processes)
        tm = self.template_model
        self.base_data_parameters = tm.data_parameters.values()
        self.base_uncertainty_parameters = tm.inverse_uncertainties.values()
        self.base_sign_parameters = tm.sign_parameters.values()
        self.base_scale_parameters = tm.scale_factors.values()
        self.starting_scale = 0.

    def _set_type(self):
        self.metadata['type'] = 'replica_fits.BatchSignChoiceReplica'


class SucrosePoolModel(ImprovedReplica):
    
    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        old_bound = self.template_model.get_bounds('bs_tx_SUCROSE')
        default_bound = getattr(self,'per_point_sucrose_bounds', (None, 50.))
        if use_soln:
            local_sucrose = self.fitting_model.soln_by_image[i]['bs_tx_SUCROSE']
            self.template_model.set_bound('bs_tx_SUCROSE', local_sucrose)
        else:
            self.template_model.set_bound('bs_tx_SUCROSE', default_bound)
        result = ImprovedReplica.solve_one_segment(self, i, use_soln=use_soln,
                                                   skip_signs=skip_signs)
        self.template_model.set_bound('bs_tx_SUCROSE', old_bound)
        return result

    def _setup_fit_infrastructure(self, **kwargs):
        ImprovedReplica._setup_fit_infrastructure(self, **kwargs)
        logger.info('Adding net sucrose consumption constraint begins')
        coefficients = {('image%d_bs_tx_SUCROSE' % i): 1.0 for i in xrange(self.N)}
        coefficients['net_sucrose'] = -1.
        net_sucrose = Linear(coefficients, name='net_sucrose_constraint')
        self.fitting_model.constraints.set('net_sucrose_constraint', net_sucrose)
        self.fitting_model.set_bound('net_sucrose_constraint', 0.)
        bounds = kwargs.get('net_sucrose_bounds', (None, 5.))
        logger.info('Setting net sucrose bounds to %s' % str(bounds))
        self.fitting_model.set_bound('net_sucrose', bounds)
        self.fitting_model.compile()
        logger.info('Adding net sucrose consumption constraint finishes')


class SignedSucrosePoolModel(SignChoiceReplica):
    
    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        old_bound = self.template_model.get_bounds('bs_tx_SUCROSE')
        default_bound = getattr(self,'per_point_sucrose_bounds', (None, 50.))
        if use_soln:
            local_sucrose = self.fitting_model.soln_by_image[i]['bs_tx_SUCROSE']
            self.template_model.set_bound('bs_tx_SUCROSE', local_sucrose)
        else:
            self.template_model.set_bound('bs_tx_SUCROSE', default_bound)
        result = SignChoiceReplica.solve_one_segment(self, i, use_soln=use_soln,
                                                   skip_signs=skip_signs)
        self.template_model.set_bound('bs_tx_SUCROSE', old_bound)
        return result

    def _setup_fit_infrastructure(self, **kwargs):
        SignChoiceReplica._setup_fit_infrastructure(self, **kwargs)

        logger.info('Adding net sucrose consumption constraint begins')
        coefficients = {('image%d_bs_tx_SUCROSE' % i): 1.0 for i in xrange(self.N)}
        coefficients['net_sucrose'] = -1.
        net_sucrose = Linear(coefficients, name='net_sucrose_constraint')
        self.fitting_model.constraints.set('net_sucrose_constraint', net_sucrose)
        self.fitting_model.set_bound('net_sucrose_constraint', 0.)
        bounds = kwargs.get('net_sucrose_bounds', (None, 5.))
        logger.info('Setting net sucrose bounds to %s' % str(bounds))
        self.fitting_model.set_bound('net_sucrose', bounds)
        self.fitting_model.compile()
        logger.info('Adding net sucrose consumption constraint finishes')

class FlexibleReplica(ReplicaFit): ### FINISH
    def __init__(self, model, n_processes=1, random_seed=None, 
                 fva_keys=None, name=None, **kwargs):
        # This method differs from fit_model.__init__ because
        # here we build the replica from a model with datasets and 
        # costs already implmeneted, which knows its own fva cache, etc.,
        # and specifies its own data keys.
        # self.base_model is used only for its list of variables,
        # typically to clarify what per-image variables should be collapsed
        # into vector valued global variables when reporting results
        self.base_model = model 
        self.template_model = model
        self.data_keys = model.data_parameters.keys()
        self.n_processes = n_processes
        if random_seed:
            self.random_guess = True
            self.random_seed = random_seed
            self.prng = np.random.RandomState(random_seed) 
        else:
            self.random_seed = None
            self.random_guess = False
        if fva_keys:
            self.fva_keys = set(fva_keys)
        else:
            self.fva_keys = {v for v in model._base_model_cache.variables}
        self._setup_attributes()
        self._setup_fit_infrastructure(
            nonnegative_objective=model.nonnegative_objective,
            **kwargs
        )
        self._set_type()
        self.name = name


    def _setup_template_model(self, **kwargs): 
        tm = self.template_model
        self.base_data_parameters = [p for parameters in tm.data_parameters.values()
                                     for p in parameters]
        self.base_uncertainty_parameters = [p for parameters in
                                            tm.inverse_uncertainty_parameters.values()
                                            for p in parameters]
        self.base_sign_parameters = tm.sign_parameters.values()
        self.base_scale_parameters = []
        self.base_scale_bounds = {}
        for d in tm.datasets.values():
            max_scale = d.max_scale_factor
            if max_scale is not None:
                scale_bounds = (-1.0*max_scale, max_scale)
                for scale in d.global_scale_factors:
                    self.base_scale_parameters.append(scale)
                    self.base_scale_bounds[scale] = scale_bounds
        self.starting_scale = 0.

    def guess(self, N):
        """Return random/deterministic guess for N variables.

        This tries to ensure that values that will be used as
        exponents are not so large as to cause overflow (by drawing
        them from a normal distribution centered at zero, not the
        lognormal distribution of the other variables, and
        guaranteeing they are in the range (-3,3).)

        I don't think there's a convenient list of all scale factors
        (note that self.base_scale_parameters includes only the global
        ones); possibly I could build this from the datasets but here
        I just assume I am interested in anything with the substring
        'scale'.

        """
        if not self.random_guess:
            return None
            
        guess = self.prng.lognormal(1.5,1.5,(N,))
        if N == len(self.template_model.variables):
            variables = self.template_model.variables
        elif N == self.fitting_model.nvar:
            variables = self.fitting_model.variables
        else:
            variables = []

        for i, (variable, value) in enumerate(zip(variables,
                                                  guess)):
            if 'scale' in variable:
                value = np.random.randn()
                if value > 3.:
                    value = 3.
                elif value < -3.:
                    value = -3.
                guess[i] = value
        return guess
            

    def _free_scales(self):
        for s in self.base_scale_parameters:
            self.fitting_model.parameters.pop(s)
            self.fva_keys.add(s)
            self.fitting_model.set_bound(s, self.base_scale_bounds[s])

    def _setup_scale_prior(self, **kwargs): ### PROOFREAD
        # constrains 'scale_factor_prior' to be the sum
        # of the prior costs for the global scale factors
        # Here, scale_factor_prior = scale_factor_prior_v0 + 
        # scale_factor_prior_v1 + ...
        # and scale_factor_prior_vi = scale_factor_vi**2
        # Also constrains scale factor priors to be nonnegative
        # (they will be automatically if the square constraint is obeyed,
        # but their explicit bounds will be respected even while the solver
        # is trying to find the feasible manifold; I hope this
        # will help prevent cases where IPOPT gets stuck trying to 
        # restore feasibility for a long time.
        template = Function('scale_factor_i**2 - cost_i')
        template.all_first_derivatives()
        template.all_second_derivatives()
        cost_variables = []
        for sf in self.base_scale_parameters:
            cost_variable = 'prior_%s' % sf
            name_map = {'scale_factor_i': sf, 
                        'cost_i': cost_variable}
            cost_variables.append(cost_variable)
            new_id = 'prior_%s_constraint' % sf
            g = template.substitute(name_map, new_id)
            self.fitting_model.constraints.set(new_id, g)
            self.fitting_model.set_bound(new_id, 0.)
            # Bounding the cost variable below by zero explicitly
            # seems to cause problems when, e.g., fixing the 
            # scale factors to 0
            self.fitting_model.set_bound(cost_variable, (-1., 120.))
        coefficients = dict.fromkeys(cost_variables, 1.)
        coefficients['scale_factor_prior'] = -1.
        overall_id = 'scale_factor_prior_constraint'
        g = Linear(coefficients, overall_id)
        self.fitting_model.constraints.set(overall_id, g)
        self.fitting_model.set_bound(overall_id, 0.)

    def _set_type(self):
        self.metadata['type'] = 'replica_fits.FlexibleReplica'

class LeafModel(FlexibleReplica):
    """ Model individual segments along the leaf interacting via phloem transport. """

    def _setup_fit_infrastructure(self, **kwargs):
        # kwargs may include bounds on the net export/import of 
        # compounds through the phloem, as 
        # {'net_phloem_bounds': {'bs_tx_SUCROSE': (None, 0.)...}, ...}
        # Bounds on individual reactions, which will be applied at each segment 
        # (during presolving and the collective fit) may be set at the template
        # model level: 
        # template_model.set_bound('bs_tx_SUCROSE': (None, 20.))
        FlexibleReplica._setup_fit_infrastructure(self, **kwargs)
        logger.info('Adding phloem interaction constraints')
        # Identify the phloem species
        self.phloem_species = [s.id for s in self.template_model._base_model_cache.species
                          if s.compartment == 'phloem']
        # Note we assume these species are not conserved at the base-model level.
        net_bounds = kwargs.get('net_phloem_bounds', {})
        phloem_set = set(self.phloem_species)
        # Identify the transporters carefully as those reactions which 
        # interact with the phloem compartment and which are valid variables
        # in the template model (if they have been removed through a simplfication
        # process, e.g. if they are set to zero, they may not be variables, but
        # will still be in _base_model_cache.reactions)
        template_variable_set = set(self.template_model.variables)
        self.phloem_transporters = [r.id for r in
                                    self.template_model._base_model_cache.reactions if
                                    phloem_set.intersection(r.stoichiometry) 
                                    and r.id in template_variable_set]
        for transporter in self.phloem_transporters: 
            coefficients = {('image%d_%s' % (i,transporter)): 1.0 for i in xrange(self.N)}
            net_variable = 'net_%s' % transporter
            coefficients[net_variable] = -1.
            constraint_name = 'net_%s_constraint' % transporter
            constraint = Linear(coefficients, name=constraint_name)
            self.fitting_model.constraints.set(constraint_name,constraint)
            self.fitting_model.set_bound(constraint_name, 0.)
            bounds = net_bounds.get(transporter, 0.)
            logger.info('Setting net %s bounds to %s' % (transporter, str(bounds)))
            self.fitting_model.set_bound(net_variable, bounds)
        self.fitting_model.compile()
        logger.info('Adding phloem interaction constraints finished.')

    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        phloem_transporters = self.phloem_transporters
        old_bounds = {r: self.template_model.get_bounds(r) for r in
                      phloem_transporters}

        if use_soln:
            local_transport_rates = {r: self.fitting_model.soln_by_image[i][r] for
                                     r in phloem_transporters}
            self.template_model.set_bounds(local_transport_rates)

        result = FlexibleReplica.solve_one_segment(self, i, use_soln=use_soln,
                                                   skip_signs=skip_signs)

        self.template_model.set_bounds(old_bounds)
        return result



class FlexibleSucrosePoolModel(FlexibleReplica):
    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        old_bound = self.template_model.get_bounds('bs_tx_SUCROSE')
        default_bound = getattr(self,'per_point_sucrose_bounds', (None, 50.))
        if use_soln:
            local_sucrose = self.fitting_model.soln_by_image[i]['bs_tx_SUCROSE']
            self.template_model.set_bound('bs_tx_SUCROSE', local_sucrose)
        else:
            self.template_model.set_bound('bs_tx_SUCROSE', default_bound)
        result = FlexibleReplica.solve_one_segment(self, i, use_soln=use_soln,
                                                   skip_signs=skip_signs)
        self.template_model.set_bound('bs_tx_SUCROSE', old_bound)
        return result

    def _setup_fit_infrastructure(self, **kwargs):
        FlexibleReplica._setup_fit_infrastructure(self, **kwargs)
        logger.info('Adding net sucrose consumption constraint begins')
        coefficients = {('image%d_bs_tx_SUCROSE' % i): 1.0 for i in xrange(self.N)}
        coefficients['net_sucrose'] = -1.
        net_sucrose = Linear(coefficients, name='net_sucrose_constraint')
        self.fitting_model.constraints.set('net_sucrose_constraint', net_sucrose)
        self.fitting_model.set_bound('net_sucrose_constraint', 0.)
        bounds = kwargs.get('net_sucrose_bounds', (None, 5.))
        logger.info('Setting net sucrose bounds to %s' % str(bounds))
        self.fitting_model.set_bound('net_sucrose', bounds)
        self.fitting_model.compile()
        logger.info('Adding net sucrose consumption constraint finishes')


class TemporaryTestModel(FlexibleReplica):
    def solve_one_segment(self, i, use_soln=False, skip_signs=False):
        old_bound = self.template_model.get_bounds('bs_sugar_tx')
        default_bound = getattr(self,'per_point_sucrose_bounds', (None, 50.))
        if use_soln:
            local_sucrose = self.fitting_model.soln_by_image[i]['bs_sugar_tx']
            self.template_model.set_bound('bs_sugar_tx', local_sucrose)
        else:
            self.template_model.set_bound('bs_sugar_tx', default_bound)
        result = FlexibleReplica.solve_one_segment(self, i, use_soln=use_soln,
                                                   skip_signs=skip_signs)
        self.template_model.set_bound('bs_sugar_tx', old_bound)
        return result

    def _setup_fit_infrastructure(self, **kwargs):
        FlexibleReplica._setup_fit_infrastructure(self, **kwargs)
        logger.info('Adding net sucrose consumption constraint begins')
        coefficients = {('image%d_bs_sugar_tx' % i): 1.0 for i in xrange(self.N)}
        coefficients['net_sucrose'] = -1.
        net_sucrose = Linear(coefficients, name='net_sucrose_constraint')
        self.fitting_model.constraints.set('net_sucrose_constraint', net_sucrose)
        self.fitting_model.set_bound('net_sucrose_constraint', 0.)
        bounds = kwargs.get('net_sucrose_bounds', (None, 5.))
        logger.info('Setting net sucrose bounds to %s' % str(bounds))
        self.fitting_model.set_bound('net_sucrose', bounds)
        self.fitting_model.compile()
        logger.info('Adding net sucrose consumption constraint finishes')

