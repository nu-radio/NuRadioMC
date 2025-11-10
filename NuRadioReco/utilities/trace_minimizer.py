import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


class TraceMinimizer:
    """
    Class for radio signal reconstruction.

    This class unifies the interfaces of different minimization algorithms
    like SciPy and Minuit, specifically for the task of fitting radio signals
    to traces. The class needs a signal_function, which takes a number of
    parameters and returns the expected signal in a number of channels, and an
    objective function which takes an array of data traces and the signal_function
    output and compares it and returns a number. When a minimization is run, the
    objective function is minimized with respect to the parameters of the
    signal_function. This class implements additional functionality, such as
    user-defined scaling of the parameters, which can improve the stability
    of the minimization process.

    Parameters
    ----------
        signal_function : function
            Function which takes a list of parameter values as input and returns a signal in a number of antennas with dimensions [n_antennas, n_samples]
        objective_function : function
            Function which takes the data and the signal as input and returns the objective value to minimize, e.g., a minus two log likelihood or chi-square
        parameters_initial : numpy.ndarray
            Values of parameters for initialization of the minimization with dimensions [n_parameters]
        parameters_bounds : numpy.ndarray
            Upper and lower bounds on parameters in the minimization. Should have dimensions [2,n_parameters]
        save_history : bool
            Whether to save the history the parameters in the minimization process. This should only be used for debugging as it can create very large arrays
            and make the minimization slow.
        debug : bool
            Whether to print debug information during the minimization process.

    """

    def __init__(self, signal_function, objective_function, parameters_initial = None, parameters_bounds = None, save_history=False, debug=False):
        self.data = None
        self.signal_function = signal_function
        self.objective_function = objective_function
        self.parameters_initial = parameters_initial
        self.n_parameters = len(parameters_initial)
        self.parameters_bounds = parameters_bounds
        self.fixed = np.zeros(self.n_parameters,dtype=bool)
        self.scaling = np.ones(self.n_parameters)
        self.n_function_calls = 0
        self.save_history = save_history
        if self.save_history: self.history = []
        self.debug = debug

        # Reconstruction results:
        self.result = None
        self.parameters = None
        self.covariance_matrix = None
        self.history = np.array([parameters_initial])
        self.success = None
        self.parameters_array = None
        self.results_array = None

    def _function_to_minimize(self, parameters):

        signal = self.signal_function(parameters * self.scaling**-1)
        self.n_function_calls += 1
        result = self.objective_function(self.data, signal)

        if self.save_history:
            self.history = np.append(self.history, [parameters], axis = 0)

        if self.debug:
            print(f"Function call {self.n_function_calls}: parameters={parameters}, result={result}")

        return result

    def fix_parameters(self, fixed):
        assert len(fixed) == self.n_parameters, "Fixed parameters should match number of parameters"
        self.fixed = fixed

    def set_scaling(self, scaling):
        """
        Set scaling of parameters to use when fitting. This can help with minimizer stability.

        Parameters
        ----------
            scaling : numpy.ndarray
                Factors to scale each parameter with. Should be a numpy array with length n_parameters.

        """
        self.scaling = scaling
        
    def run_minimization(self, data, method, **method_kwargs):
        """
        Run minimization algorithm for a dataset containing a neutrino signal

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with neutrino signal to reconstruct. Has dimensions [n_Antennas,n_samples]
            method : str
                Name of the method used to run the minimization
        """
        self.data = data

        if method == "scipy":
            result_object = self._scipy_minimization(**method_kwargs)
        if method == "minuit":
            result_object = self._minuit_minimization(**method_kwargs)
        if method == "noisyopt":
            result_object = self._noisyopt_minimization(**method_kwargs)
        if method == "skopt":
            result_object = self._skopt_minimization(**method_kwargs)
        if method == "simple_minimizer":
            result_object = self._simple_minimizer(**method_kwargs)

        return result_object

    def run_many_minimizations(self, datasets, method, signal_true = None, **method_kwargs):
        """
        Run minimization algorithm for many datasets containing a signal. Can be used for bootstrapping of one true signal with many different realizations of noise.

        Parameters
        ----------
            datasets : numpy.ndarray
                Array containing many datasets with signals to reconstruct. Has dimensions [n_events,n_Antennas,n_samples]
            method : str
                Name of the method used to run the minimization
            signal_true : numpy.ndarray, optional
                For simulated data, the likelihood for the true signal (without noise) can be calculated. This can later be used to plot the llh distribution.

        """
        n_events = len(datasets)

        # Initialize arrays:
        self.parameters_array = np.zeros([n_events, self.n_parameters])
        self.results_array = np.zeros([n_events])
        self.results_true_array = np.zeros([n_events])

        # Loop over events:
        for i in range(n_events):
            try:
                self.run_minimization(data = datasets[i, :, :], method=method, **method_kwargs)
            except:
                pass
            self.parameters_array[i, :] = self.parameters
            self.results_array[i] = self.result
            if signal_true is not None:
                self.results_true_array[i] = self.objective_function(datasets[i, :, :], signal_true)

    def profile_likelihood_1D(self, data, method, parameter_x, parameter_grid_x, true_value = None, plot = True, **method_kwargs):
        
        n_x = len(parameter_grid_x)

        llh_values = np.zeros(n_x)

        # Get best fit point:
        self.reconstruct_event(data = data, method=method, **method_kwargs)
        best_fit_x = self.parameters[parameter_x]
        best_fit_llh = self.result

        # Now fix parameters which are being scanned:
        fixed = np.zeros(self.n_parameters, dtype=bool)
        fixed[parameter_x] = True
        self.fix_parameters(fixed)

        for i in range(n_x):
            self.parameters_initial[parameter_x] = parameter_grid_x[i]
            self.reconstruct_event(data = data, method=method, **method_kwargs)
            llh_values[i] = self.result

        if plot:
            plt.figure(figsize=[4,3])
            plt.plot(parameter_grid_x, llh_values-best_fit_llh, "b-", label=r"$-2 \Delta LLH$")
            axis = plt.axis()
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [2,2], ":", label=f"$1\sigma$")
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [4,4], ":", label=f"$2\sigma$")
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [6,6], ":", label=f"$3\sigma$")
            plt.plot([best_fit_x,best_fit_x],[0,100],"y--", label="Fit")
            if true_value is not None: plt.plot([true_value,true_value],[0,100],"r--", label="True")
            plt.axis([parameter_grid_x[0],parameter_grid_x[-1],0,axis[3]*1.2])
            plt.xlabel(r"Parameter [au]")
            plt.ylabel(r"Result")
            plt.legend()
            plt.tight_layout()

    def profile_likelihood_2D(self, data, method, parameter_x, parameter_y, parameter_grid_x, parameter_grid_y, profile = True, true_values = None, plot = True, cmap="Blues_r", vmax=60, **method_kwargs):
        
        n_x = len(parameter_grid_x)
        n_y = len(parameter_grid_y)

        llh_values = np.zeros([n_x,n_y])

        # Get best fit point:
        self.reconstruct_event(data = data, method=method, **method_kwargs)
        best_fit_x = self.parameters[parameter_x]
        best_fit_y = self.parameters[parameter_y]
        best_fit_llh = self.result

        # Now fix parameters which are being scanned:
        fixed_initial = np.copy(self.fixed)
        self.fixed[parameter_x] = True
        self.fixed[parameter_y] = True
        if not profile: self.fixed[:] = True

        for i in range(n_x):
            for j in range(n_y):
                self.parameters_initial[parameter_x] = parameter_grid_x[i]
                self.parameters_initial[parameter_y] = parameter_grid_y[j]
                self.reconstruct_event(data = data, method=method, **method_kwargs)
                llh_values[i,j] = self.result

        if plot:
            plt.figure(figsize=[4.2,3])
            plt.pcolormesh(parameter_grid_x, parameter_grid_y, llh_values.T-best_fit_llh, cmap=cmap, vmax=vmax)
            plt.colorbar(label=f"$-2\Delta LLH$")
            CS = plt.contour(parameter_grid_x, parameter_grid_y, llh_values.T-best_fit_llh, levels=[1.15*2,3.09*2,5.91*2])
            if true_values is not None: plt.plot(true_values[0], true_values[1], "r*",label="True")
            plt.plot(best_fit_x, best_fit_y, "g*", label="Fit")
            plt.legend()
            plt.xlabel(r"Parameter 1 [au]")
            plt.ylabel(r"Parameter 2 [au]")
            plt.tight_layout()

            # Contour labels:
            fmt = {}
            strs = ['$1\sigma$', '$2\sigma$', '$3\sigma$']
            for l, s in zip(CS.levels, strs):
                fmt[l] = s

            # Label every other level using strings
            plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=10)

        # Set fixed parameters back to initial:
        self.fixed = fixed_initial

    ### Methods: ###
    
    def _scipy_minimization(self, tol = 1e-3, scipy_method = "L-BFGS-B", options={}):
        import scipy.optimize as opt
        
        # Fix parameters:
        bounds_scipy = np.copy(self.parameters_bounds) * np.array([self.scaling, self.scaling]).T
        bounds_scipy[self.fixed,0] = self.parameters_initial[self.fixed]
        bounds_scipy[self.fixed,1] = self.parameters_initial[self.fixed]

        # Perform minimization:
        result = opt.optimize.minimize(
            self._function_to_minimize,
            x0 = self.parameters_initial * self.scaling,
            tol = tol,
            bounds = bounds_scipy,
            method = scipy_method,
            options=options
        )

        # Save results:
        self.success = None
        self.result = result.fun
        self.parameters = result.x * self.scaling**-1

        return result

    def _minuit_minimization(self, tolerance = 1e-3, minuit_method = "migrad"):
        from iminuit import Minuit
        
        # Initialze minimizer:
        m = Minuit(
            self._function_to_minimize,
            self.parameters_initial * self.scaling
        )
        
        # Set bounds:
        m.limits = self.parameters_bounds * np.array([self.scaling, self.scaling]).T
        
        # Fix parameters:
        for i in range(self.n_parameters):
            if self.fixed[i]:
                m.fixed[i] = True
        
        # Run minimization:
        m.migrad()
        
        # Save results:
        self.success = None
        self.result = m.fval
        self.parameters = np.array(m.values) * self.scaling**-1

        return m
    
    def _noisyopt_minimization(self, deltatol = 0.1, paired = False):
        from noisyopt import minimizeCompass

        res = minimizeCompass(
            self._function_to_minimize,
            bounds = self.parameters_bounds * np.array([self.scaling, self.scaling]).T,
            x0 = self.parameters_initial * self.scaling,
            deltatol = deltatol,
            paired = paired)

        self.success = None
        self.result = res.fun
        self.parameters = np.array(res.x) * self.scaling**-1

        return res

    def _skopt_minimization(self, n_calls = 1000, n_initial_points = 20, random_state = None):
        from skopt import gp_minimize

        # Convert bounds to list of tuples
        bounds_scaled = self.parameters_bounds * np.array([self.scaling, self.scaling]).T
        dimensions = [(bounds_scaled[i, 0], bounds_scaled[i, 1]) for i in range(len(bounds_scaled))]

        # Ensure x0 is a list of scalars
        x0_scaled = (self.parameters_initial * self.scaling).tolist()

        res = gp_minimize(
            self._function_to_minimize,
            dimensions=dimensions,
            x0=x0_scaled,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state
        )

        self.success = None
        self.result = res.fun
        self.parameters = np.array(res.x) * self.scaling**-1

        return res

    def _simple_minimizer(self, initial_step_size, decrease_rate, max_calls, epsilon, tolerance=None):
        """
        This is a very simple minimizer, which has not been thoroughly tested. It can be used for debugging.
        """
        import scipy.optimize as opt

        # Initialize variables
        current_step_size = np.copy(initial_step_size)
        current_parameters = self.parameters_initial * self.scaling
        self.success = None
        old_result = 0
        best_result = None
        best_parameters = None
        best_call = None

        for call in range(max_calls):

            # Check for convergence
            if tolerance is not None and call > 0 and abs(result - old_result) < tolerance:
                self.success = True
                break

            # Get gradient
            gradient = opt.approx_fprime(current_parameters, self._function_to_minimize, epsilon=epsilon * self.scaling)
            step_direction = -gradient / np.linalg.norm(gradient)

            # Update parameters
            current_parameters += current_step_size * step_direction

            # Clip parameters to bounds
            if any(current_parameters < self.parameters_bounds[:, 0] * self.scaling) or any(current_parameters > self.parameters_bounds[:, 1] * self.scaling):
                current_parameters = np.clip(current_parameters,
                                             self.parameters_bounds[:, 0] * self.scaling,
                                             self.parameters_bounds[:, 1] * self.scaling)

            # Calculate objective function
            result = self._function_to_minimize(current_parameters)

            print(call, best_call, current_parameters, result)

            # Update step size
            current_step_size *= decrease_rate

            # save result
            #old_result = np.copy(result)
            if best_result is None or result < best_result:
                del best_result
                del best_parameters
                del best_call
                best_result = np.copy(result)
                best_parameters = np.copy(current_parameters)
                best_call = np.copy(call)

        self.result = best_result
        self.parameters = best_parameters * self.scaling**-1

        return best_result