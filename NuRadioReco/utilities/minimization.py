import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


class Minimizer:
    """
    Class for radio signal reconstruction and general minimization tasks.

    This class unifies the interfaces of different minimization algorithms
    like SciPy and Minuit. The class has two "modes". If no signal_function
    is provided, it behaves like a standard minimizer, which minimizes an
    objective function with respect to its parameters. If a signal_function
    is provided, the class behaves like a minimizer specialized for fitting
    radio signals to traces. In this case, the signal_function should take
    the parameters as input and return the expected signal in a number of
    channels. This is then compared to data traces in the objective_function,
    which should take the data traces and the signal traces as input. When a
    minimization is run, the objective function is then minimized with respect
    to the parameters of the signal_function.

    This class implements additional functionality, such as user-defined scaling
    of the parameters, which can improve the stabilityof the minimization process.

    Parameters
    ----------
        objective_function : function
            If no signal_function is provided, this is a function which takes a number of parameters (as a numpy.ndarray)
            as input and returns the objective value to minimize, e.g., a minus two log likelihood or chi-square, i.e.,
            objective_function(parameters).
            If a signal_function is provided, the objective_function takes the the output of the signal_function and the
            data as input and returns the objective value to minimize, i.e., objective_function(data, signal).
        parameters_initial : numpy.ndarray
            Values of parameters for initialization of the minimization with dimensions [n_parameters]
        parameters_bounds : numpy.ndarray
            Upper and lower bounds on parameters in the minimization. Should have dimensions [2,n_parameters]
        signal_function : function (optional)
            Function which takes a list of parameter values as input and returns a signal. The type and shape of the output
            depends on what happens in the user-defined objective_function, but it could be a numpy array with shape
            [n_antennas, n_samples] containing a radio signal in a number channels, or a list of channels.
        save_history : bool (optional)
            Whether to save the history the parameters in the minimization process. This should only be used for debugging as it can create very large arrays
            and make the minimization slow.
        debug : bool (optional)
            Whether to print debug information during the minimization process.
    """

    def __init__(self, objective_function, parameters_initial, parameters_bounds, signal_function=None, save_history=False, debug=False):
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

    def _function_to_minimize(self, parameters_scaled):

        if self.signal_function is not None:
            signal = self.signal_function(parameters_scaled * self.scaling**-1)
            self.n_function_calls += 1
            result = self.objective_function(self.data, signal)
        else:
            self.n_function_calls += 1
            result = self.objective_function(parameters_scaled * self.scaling**-1)

        if self.save_history:
            self.history = np.append(self.history, [parameters_scaled * self.scaling**-1], axis = 0)

        if self.debug:
            print(f"Function call {self.n_function_calls}: parameters={parameters_scaled * self.scaling**-1}, result={result}")

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
        
    def run_minimization(self, method, data=None, **method_kwargs):
        """
        Run minimization algorithm for the objective function. If a signal_function is provided,
        a data array must be provided which the signal is fitted using the objective_function.

        Parameters
        ----------
            method : str
                Name of the method used to run the minimization
            data : numpy.ndarray, any (optional)
                Should only be provided if a signal_function is set. The type and shape depends on the objective_function,
                but it could be a numpy array with shape [n_antennas, n_samples] containing data traces to fit the signal to,
                or a list of channels.
        """
        if self.signal_function is not None:
            assert data is not None, "Data must be provided for minimization if a signal_function is set"
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
        Only works if a signal_function is provided.

        Parameters
        ----------
            datasets : numpy.ndarray
                Array containing many datasets with signals to reconstruct. Has dimensions [n_events,n_Antennas,n_samples]
            method : str
                Name of the method used to run the minimization
            signal_true : numpy.ndarray (optional)
                For simulated data, the likelihood for the true signal (without noise) can be calculated. This can later be
                used to plot the likelihood or chi-square distribution.

        Returns
        -------
            parameters_array : numpy.ndarray
                Array containing the best fit parameters for each event. Has dimensions [n_events,n_parameters]
            results_array : numpy.ndarray
                Array containing the best fit objective function value for each event. Has dimensions [n_events]
            results_true_array : numpy.ndarray
                If signal_true is provided, this array contains the objective function value for the true signal
                (without noise) for each event. Has dimensions [n_events]
        """
        n_events = len(datasets)

        # Initialize arrays:
        parameters_array = np.zeros([n_events, self.n_parameters])
        results_array = np.zeros([n_events])
        results_true_array = np.zeros([n_events])

        # Loop over events:
        for i_event in range(n_events):
            try:
                self.run_minimization(method=method, data=datasets[i_event, :, :], **method_kwargs)
            except:
                pass
            parameters_array[i_event, :] = self.parameters
            results_array[i_event] = self.result
            if signal_true is not None:
                results_true_array[i_event] = self.objective_function(datasets[i_event, :, :], signal_true)

        return parameters_array, results_array, results_true_array

    def profile_likelihood_1D(self, method, parameter_x, parameter_grid_x, data=None, true_value = None, plot = True, **method_kwargs):
        
        n_x = len(parameter_grid_x)

        llh_values = np.zeros(n_x)

        # Get best fit point:
        self.run_minimization(method=method, data=data, **method_kwargs)
        best_fit_x = self.parameters[parameter_x]
        best_fit_llh = self.result

        # Now fix parameters which are being scanned:
        fixed = np.zeros(self.n_parameters, dtype=bool)
        fixed[parameter_x] = True
        self.fix_parameters(fixed)

        for i in range(n_x):
            self.parameters_initial[parameter_x] = parameter_grid_x[i]
            self.run_minimization(method=method, data=data, **method_kwargs)
            llh_values[i] = self.result

        if plot:
            plt.figure(figsize=[4,3])
            plt.plot(parameter_grid_x, llh_values-best_fit_llh, "b-", label=r"$-2 \Delta LLH$")
            axis = plt.axis()
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [2,2], ":", label=r"$1\sigma$")
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [4,4], ":", label=r"$2\sigma$")
            plt.plot([min(parameter_grid_x),max(parameter_grid_x)], [6,6], ":", label=r"$3\sigma$")
            plt.plot([best_fit_x,best_fit_x],[0,100],"y--", label="Fit")
            if true_value is not None: plt.plot([true_value,true_value],[0,100],"r--", label="True")
            plt.axis([parameter_grid_x[0],parameter_grid_x[-1],0,axis[3]*1.2])
            plt.xlabel(r"Parameter [au]")
            plt.ylabel(r"Result")
            plt.legend()
            plt.tight_layout()

    def profile_likelihood_2D(self, method, parameter_x, parameter_y, parameter_grid_x, parameter_grid_y, data=None, profile = True, true_values = None, plot = True, cmap="Blues_r", vmax=60, **method_kwargs):
        
        n_x = len(parameter_grid_x)
        n_y = len(parameter_grid_y)

        llh_values = np.zeros([n_x,n_y])

        # Get best fit point:
        self.run_minimization(method=method, data=data, **method_kwargs)
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
                self.run_minimization(method=method, data=data, **method_kwargs)
                llh_values[i,j] = self.result

        if plot:
            plt.figure(figsize=[4.2,3])
            plt.pcolormesh(parameter_grid_x, parameter_grid_y, llh_values.T-best_fit_llh, cmap=cmap, vmax=vmax)
            plt.colorbar(label=r"$-2\Delta LLH$")
            CS = plt.contour(parameter_grid_x, parameter_grid_y, llh_values.T-best_fit_llh, levels=[1.15*2,3.09*2,5.91*2])
            if true_values is not None: plt.plot(true_values[0], true_values[1], "r*",label="True")
            plt.plot(best_fit_x, best_fit_y, "g*", label="Fit")
            plt.legend()
            plt.xlabel(r"Parameter 1 [au]")
            plt.ylabel(r"Parameter 2 [au]")
            plt.tight_layout()

            # Contour labels:
            fmt = {}
            strs = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$']
            for l, s in zip(CS.levels, strs):
                fmt[l] = s

            # Label every other level using strings
            plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=10)

        # Set fixed parameters back to initial:
        self.fixed = fixed_initial

    ### Minimization methods: ###
    
    def _scipy_minimization(self, tol = 1e-3, scipy_method = "L-BFGS-B", options={}):
        import scipy.optimize as opt
        
        # Fix parameters:
        bounds_scipy = np.copy(self.parameters_bounds) * np.array([self.scaling, self.scaling]).T
        bounds_scipy[self.fixed,0] = self.parameters_initial[self.fixed]
        bounds_scipy[self.fixed,1] = self.parameters_initial[self.fixed]

        # Perform minimization:
        result = opt.minimize(
            self._function_to_minimize,
            x0 = self.parameters_initial * self.scaling,
            tol = tol,
            bounds = bounds_scipy,
            method = scipy_method,
            options=options
        )

        # Save results:
        self.success = result.success
        self.result = result.fun
        self.parameters = result.x * self.scaling**-1

        return result

    def _minuit_minimization(self, minuit_method = "migrad"):
        from iminuit import Minuit
        
        # Initialze minimizer:
        m = Minuit(
            self._function_to_minimize,
            self.parameters_initial * self.scaling
        )
        
        # Set bounds:
        m.limits = self.parameters_bounds * np.array([self.scaling, self.scaling]).T
        
        # Fix parameters:
        for i_param in range(self.n_parameters):
            if self.fixed[i_param]:
                m.fixed[i_param] = True
        
        # Run minimization:
        if minuit_method == "migrad":
            m.migrad()
        elif minuit_method == "simplex":
            m.simplex()
        else:
            raise ValueError(f"Minuit method {minuit_method} not recognized")
        
        # Save results:
        self.success = m.valid
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

        self.success = res.success
        self.result = res.fun
        self.parameters = np.array(res.x) * self.scaling**-1

        return res

    def _skopt_minimization(self, n_calls = 1000, n_initial_points = 20, random_state = None):
        from skopt import gp_minimize

        # Convert bounds to list of tuples
        bounds_scaled = self.parameters_bounds * np.array([self.scaling, self.scaling]).T
        dimensions = [(bounds_scaled[i_param, 0], bounds_scaled[i_param, 1]) for i_param in range(len(bounds_scaled))]

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

    def _simple_minimizer(self, initial_step_size, decrease_rate, max_calls, epsilon, tolerance=None, print_steps=False):
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
        result = np.inf

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

            if print_steps:
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