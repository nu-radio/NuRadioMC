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
    of the parameters, which can improve the stability of the minimization process.

    Parameters
    ----------
        objective_function : function
            If no signal_function is provided, this is a function which takes a number of parameters (as one numpy.ndarray)
            as input and returns the objective value to minimize, e.g., a minus two log likelihood or chi-square, i.e.,
            objective_function(parameters).
            If a signal_function is provided, the objective_function takes the the output of the signal_function and the
            data as input and returns the objective value to minimize, i.e., objective_function(data, signal).
        parameters_initial : numpy.ndarray
            Values of parameters for initialization of the minimization with dimensions [n_parameters]
        parameters_bounds : numpy.ndarray
            Upper and lower bounds on parameters in the minimization. Should have dimensions [2,n_parameters]
        signal_function : function (optional)
            Function which takes a numpy array with parameter values as input (i.e., signal_function(parameters)) and
            returns a signal. The type and shape of the output depends on what happens in the user-defined objective_function,
            but it could be a numpy array with shape [n_antennas, n_samples] containing a radio signal in a number channels,
            or a list of channels.
        normalization : numpy.ndarray (optional)
            Set normalization factors for the parameters before fitting. Each parameter is divided by this value internally,
            which can help with minimizer stability when parameters have different scales. This can, e.g., be set to the
            natural units of a problem (PeV, km, deg, ...) or to the expected/estimated uncertainties on the parameters.
            Should be a numpy array with length n_parameters. If set to None, a normalization of 1 (i.e., no normalization
            will be used).
        fixed : list(bool)
            Wheter to fix some of the parameters in the optimization. Should be a list of bools with length n_parameters.
            If left as None, all parameters will be free in the optimization.
        save_history : bool (optional)
            Whether to save the history of the parameters in the minimization process. This should only be used for debugging as it can create very large arrays
            and make the minimization slow.
        debug : bool (optional)
            Whether to print debug information during the minimization process.
    """

    def __init__(self, objective_function, parameters_initial, parameters_bounds, normalization=None, fixed=None, signal_function=None, save_history=False, debug=False):
        self.data = None
        self.signal_function = signal_function
        self.objective_function = objective_function
        self.parameters_initial = parameters_initial
        self.n_parameters = len(parameters_initial)
        self.parameters_bounds = parameters_bounds
        self.fixed = np.zeros(self.n_parameters,dtype=bool)
        self.normalization = normalization if normalization is not None else np.ones(self.n_parameters)
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

        if fixed is not None:
            self.fix_parameters(fixed)

    def _function_to_minimize(self, parameters_normalized):

        parameters_physical = parameters_normalized * self.normalization

        if self.signal_function is not None:
            signal = self.signal_function(parameters_physical)
            self.n_function_calls += 1
            result = self.objective_function(self.data, signal)
        else:
            self.n_function_calls += 1
            result = self.objective_function(parameters_physical)

        if self.save_history:
            self.history = np.append(self.history, [parameters_physical], axis = 0)

        if self.debug:
            print(f"Function call {self.n_function_calls}: parameters={parameters_physical}, result={result}")

        return result

    def fix_parameters(self, fixed):
        assert len(fixed) == self.n_parameters, "Fixed parameters should match number of parameters"
        self.fixed = fixed
        
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
        elif method == "minuit":
            result_object = self._minuit_minimization(**method_kwargs)
        elif method == "noisyopt":
            result_object = self._noisyopt_minimization(**method_kwargs)
        elif method == "skopt":
            result_object = self._skopt_minimization(**method_kwargs)
        elif method == "simple_minimizer":
            result_object = self._simple_minimizer(**method_kwargs)
        else:
            raise ValueError(f"Unknown minimizer: {method}")

        return result_object

    def profile_scan_1D(self, method, i_parameter_scan, parameter_grid_x, parameters_best=None, data=None, profile=True, true_value=None, plot=True, **method_kwargs):
        """
        Perform a profiled (likelihood) scan around the minimum of the objective function for the i_parameter_scan'th parameter.

        If parameters_best parameters is provided, the scan will be performed around this point. Otherwise, the minimum will be found first by running a minimization.

        If profile is set to false, the scan will be performed without profiling over the other parameters, i.e.,
        they will be fixed to their best fit values. If profile is set to true, the other parameters will be optimized for each point in the scan.
        """

        parameters_initial_copy = np.copy(self.parameters_initial)
        fixed_copy = np.copy(self.fixed)

        # Get best fit point:
        if parameters_best is None:
            self.run_minimization(method=method, data=data, **method_kwargs)
            best_fit_parameters = self.parameters
            best_fit_x = self.parameters[i_parameter_scan]
            best_fit_value = self.result
        else:
            best_fit_parameters = np.copy(parameters_best)
            best_fit_x = best_fit_parameters[i_parameter_scan]
            best_fit_value = self.objective_function(data, self.signal_function(best_fit_parameters)) if self.signal_function is not None else self.objective_function(best_fit_parameters)

        # Now fix parameters which are being scanned:
        n_x = len(parameter_grid_x)
        objective_values = np.zeros(n_x)
        if profile:
            self.fixed[i_parameter_scan] = True

            for i_x in range(n_x):
                self.parameters_initial[i_parameter_scan] = parameter_grid_x[i_x]
                self.run_minimization(method=method, data=data, **method_kwargs)
                objective_values[i_x] = self.result #- best_fit_value

        else:
            for i_x in range(n_x):
                best_fit_parameters[i_parameter_scan] = parameter_grid_x[i_x]
                objective_values[i_x] = self.objective_function(data, self.signal_function(best_fit_parameters)) if self.signal_function is not None else self.objective_function(best_fit_parameters)

        if plot:
            plt.figure(figsize=[4,3])
            plt.plot(parameter_grid_x, objective_values - best_fit_value, "b-", label=r"$-2 \Delta LLH$")
            axis = plt.axis()
            plt.axhline(1, linestyle=":", color="C0", label=r"$1\sigma$")
            plt.axhline(4, linestyle=":", color="C0", label=r"$2\sigma$")
            plt.axhline(9, linestyle=":", color="C0", label=r"$3\sigma$")
            plt.axvline(best_fit_x, color="y", linestyle="--", label="Fit")
            if true_value is not None: plt.axvline(true_value, color='r', linestyle='--', label="True/Initial")
            plt.axis([parameter_grid_x[0], parameter_grid_x[-1], 0, axis[3]*1.2])
            plt.xlabel(r"Parameter [au]")
            plt.ylabel(r"Objective value")
            plt.legend()
            plt.tight_layout()

        # Set initial and fixed parameters back to initial:
        self.parameters_initial = parameters_initial_copy
        self.fixed = fixed_copy

        return objective_values

    def profile_scan_2D(self, method, i_parameter_x, i_parameter_y, parameter_grid_x, parameter_grid_y, data=None, profile=True, true_values=None, plot=True, cmap="Blues_r", vmax=60, **method_kwargs):
        """
        Perform minimization and a 2D profile (likelihood) scan around the minimum of the i_parameter_x'th and i_parameter_y'th parameters by fixing them to different values on a grid
        and optimizing the objective function with respect to the other parameters.
        """
        
        n_x = len(parameter_grid_x)
        n_y = len(parameter_grid_y)

        objective_values = np.zeros([n_x,n_y])

        parameters_initial_copy = np.copy(self.parameters_initial)

        # Get best fit point:
        self.run_minimization(method=method, data=data, **method_kwargs)
        best_fit_x = self.parameters[i_parameter_x]
        best_fit_y = self.parameters[i_parameter_y]
        best_fit_value = self.result

        # Now fix parameters which are being scanned:
        fixed_copy = np.copy(self.fixed)
        self.fixed[i_parameter_x] = True
        self.fixed[i_parameter_y] = True
        if not profile: self.fixed[:] = True

        for i in range(n_x):
            for j in range(n_y):
                self.parameters_initial[i_parameter_x] = parameter_grid_x[i]
                self.parameters_initial[i_parameter_y] = parameter_grid_y[j]
                self.run_minimization(method=method, data=data, **method_kwargs)
                objective_values[i,j] = self.result

        if plot:
            plt.figure(figsize=[4.2,3])
            plt.pcolormesh(parameter_grid_x, parameter_grid_y, objective_values.T-best_fit_value, cmap=cmap, vmax=vmax)
            plt.colorbar(label=r"Objective value")
            CS = plt.contour(parameter_grid_x, parameter_grid_y, objective_values.T-best_fit_value, levels=[1.15*2, 3.09*2, 5.91*2])
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

        # Set initial and fixed parameters back to initial:
        self.parameters_initial = parameters_initial_copy
        self.fixed = fixed_copy

        return objective_values

    ### Minimization methods: ###
    
    def _scipy_minimization(self, tol=1e-3, scipy_method="L-BFGS-B", options={}):
        import scipy.optimize as opt
        
        # Fix parameters:
        bounds_scipy = np.copy(self.parameters_bounds) / np.array([self.normalization, self.normalization]).T
        bounds_scipy[self.fixed,0] = self.parameters_initial[self.fixed] / self.normalization[self.fixed]
        bounds_scipy[self.fixed,1] = self.parameters_initial[self.fixed] / self.normalization[self.fixed]

        # Perform minimization:
        result = opt.minimize(
            self._function_to_minimize,
            x0 = self.parameters_initial / self.normalization,
            tol = tol,
            bounds = bounds_scipy,
            method = scipy_method,
            options=options
        )

        # Save results:
        self.success = result.success
        self.result = result.fun
        self.parameters = result.x * self.normalization

        return result

    def _minuit_minimization(self, minuit_method = "migrad"):
        from iminuit import Minuit
        
        # Initialze minimizer:
        m = Minuit(
            self._function_to_minimize,
            self.parameters_initial / self.normalization
        )
        
        # Set bounds:
        m.limits = self.parameters_bounds / np.array([self.normalization, self.normalization]).T
        
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
        self.parameters = np.array(m.values) * self.normalization

        return m
    
    def _noisyopt_minimization(self, deltatol=0.1, paired=False):
        from noisyopt import minimizeCompass

        res = minimizeCompass(
            self._function_to_minimize,
            bounds = self.parameters_bounds / np.array([self.normalization, self.normalization]).T,
            x0 = self.parameters_initial / self.normalization,
            deltatol = deltatol,
            paired = paired)

        self.success = res.success
        self.result = res.fun
        self.parameters = np.array(res.x) * self.normalization

        return res

    def _skopt_minimization(self, n_calls=1000, n_initial_points=20, random_state=None):
        from skopt import gp_minimize

        # Convert bounds to list of tuples
        bounds_scaled = self.parameters_bounds / np.array([self.normalization, self.normalization]).T
        dimensions = [(bounds_scaled[i_param, 0], bounds_scaled[i_param, 1]) for i_param in range(len(bounds_scaled))]

        # Ensure x0 is a list of scalars
        x0_scaled = (self.parameters_initial / self.normalization).tolist()

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
        self.parameters = np.array(res.x) * self.normalization

        return res

    def _simple_minimizer(self, initial_step_size, decrease_rate, max_calls, epsilon, tolerance=None, print_steps=False):
        """
        This is a very simple minimizer, which has not been thoroughly tested. It can be used for debugging.
        """
        import scipy.optimize as opt

        # Initialize variables
        current_step_size = np.copy(initial_step_size)
        current_parameters = self.parameters_initial / self.normalization
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
            gradient = opt.approx_fprime(current_parameters, self._function_to_minimize, epsilon=epsilon / self.normalization)
            step_direction = -gradient / np.linalg.norm(gradient)

            # Update parameters
            current_parameters += current_step_size * step_direction

            # Clip parameters to bounds
            if any(current_parameters < self.parameters_bounds[:, 0] / self.normalization) or any(current_parameters > self.parameters_bounds[:, 1] / self.normalization):
                current_parameters = np.clip(current_parameters,
                                             self.parameters_bounds[:, 0] / self.normalization,
                                             self.parameters_bounds[:, 1] / self.normalization)

            # Calculate objective function
            result = self._function_to_minimize(current_parameters)

            if print_steps:
                print(call, best_call, current_parameters, result)

            # Update step size
            current_step_size *= decrease_rate

            # save result
            #old_result = np.copy(result)
            if best_result is None or result < best_result:
                best_result = np.copy(result)
                best_parameters = np.copy(current_parameters)
                best_call = np.copy(call)

        self.result = best_result
        self.parameters = best_parameters * self.normalization

        return best_result
