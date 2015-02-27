""" 
Unified fitting interface for different approaches to fitting unsigned data series to models.

"""
import numpy as np
import pickle
import time

class FittingFailure(Exception):
    pass

class FitInterface():
    def __init__(self, model, data_keys, n_processes=1, random_seed=None, 
                 fva_cache=None, fva_keys=None, name=None, **kwargs):
        self.base_model = model
        self.data_keys = data_keys
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
            self.fva_keys = set(self.base_model.variables)
        self._setup_attributes()
        self.fva_cache = fva_cache
        self._setup_fit_infrastructure(**kwargs)
        self._set_type()
        self.name = name

    def _setup_fit_infrastructure(self):
        raise NotImplementedError

    def _set_type(self):
        """ Adds a description of the type of fit to self.metadata. """
        raise NotImplementedError

    def guess(self, N):
        """ Return random/deterministic guess for N variables """
        if self.random_guess:
            return self.prng.lognormal(2,2,(N,))
        else:
            return None

    def _setup_attributes(self):
        """ Initializes attributes common to the various fit interfaces. """
        self.N = None
        self.data = {} # {k: numpy array of shape (N,) for each variable k in self.data_keys}
        self.error = {} # {k: numpy array of shape (N,) for k in self.data_keys}
        self.extrema = {} # {v: np.array of shape (N,2), each row a tuple (lower_bound, 
                          # upper_bound) where either or both may be np.nan, 
                          # for v in self.fva_keys}
        self.fit_result = {} # {v: numpy array of shape (N,) for v in base_model.variables}
        self.fit_other = {} # {s: float or array of shape (N,) for the variables s, such as
                            # scale factors, introduced in the fit}
        self.fit_parameters = {} # {s: float or array of shape (N,) for the various parameters
                                 # introduced in the process of fitting, ex. signs, priors
                                 # on scale factors}
        self.fit_overall_value = None # float, total fitting cost
        self.fit_value_by_image = None # array of shape (N,) giving cost of best fit by 
                                       # image (note need not sum to fit_overall_value,
                                       # as the per-image values exclude eg priors on 
                                       # scale factors)
        self._setup_metadata()

    def _setup_metadata(self):
        self.metadata = {'random_seed': self.random_seed}

    def _get_ipopt_options(self):
        """ Read IPOPT options file, or '' if no file.

        TODO: set all options interactively and record their values
        as they are set, rather than from a file.

        """
        try:
            with open('ipopt.opt') as f:
                options = f.read()
        except IOError:
            options = ''
        return options

    def fit(self):
        assert set(self.data.keys()) == set(self.data_keys)
        # In some cases we may have data to which no error 
        # is relevant, ie enzyme activity levels imposed 
        # purely as upper bounds. Probably we should 
        # either drop both these checks or set up 
        # a self.error_keys attribute tracking variables
        # for which uncertainty information should be supplied,
        # but for now we'll just skip this check:
#        assert set(self.error.keys()) == set(self.data_keys)
        
        self.N = len(self.data.values()[0])
        self.metadata['fit_options'] = self._get_ipopt_options()
        self.metadata['fit_start'] = time.ctime()
        self._fit()
        self.metadata['fit_stop'] = time.ctime()

    def _fit(self):
        raise NotImplementedError

    def objective_constrained_fva(self, objective_offset):
        self.metadata['ofva_objective_offset'] = objective_offset
        self.metadata['ofva_options'] = self._get_ipopt_options()
        self.metadata['ofva_start'] = time.ctime()
        self._objective_constrained_fva(objective_offset)
        self.metadata['ofva_stop'] = time.ctime()

    def _objective_constrained_fva(self, objective_offset):
        raise NotImplementedError

    def save_results(self, filename):
        results = {'data': self.data,
                   'error': self.error,
                   'extrema': self.extrema,
                   'fit_result': self.fit_result,
                   'fit_other': self.fit_other,
                   'fit_values': self.fit_values,
                   'fit_parameters': self.fit_parameters,
                   'metadata': self.metadata,
                   'overall_value': self.fit_overall_value,
                   'value_by_image': self.fit_value_by_image}
        with open(filename) as f:
            pickle.dump(results, f)
