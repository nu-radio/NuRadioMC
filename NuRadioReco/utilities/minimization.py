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
        self._result = None
        self._parameters = None
        self.covariance_matrix = None
        self.history = np.array([parameters_initial])
        self.success = None

        if fixed is not None:
            self.fix_parameters(fixed)

    @property
    def parameters(self):
        """ Get the best-fit parameters from the minimization. """
        if self._parameters is None:
            return None
        return np.copy(self._parameters)

    @property
    def result(self):
        """ Get best-fit objective function value. """
        return self._result

    def fix_parameters(self, fixed):
        assert len(fixed) == self.n_parameters, "Fixed parameters should match number of parameters"
        self.fixed = fixed

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

    def run_minimization(self, method, data=None, **method_kwargs):
        """
        Run minimization algorithm for the objective function. If a signal_function is provided,
        a data array must be provided which the signal is fitted using the objective_function.

        Fitted parameters are stored in self.parameters and the best-fit objective function value
        is stored in self.result.

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

        self.n_function_calls = 0

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

    def parameter_scan_1D(
            self, parameter_index, parameter_values, parameters_start=None,
            data=None, profile=False, method="minuit", plot=False, **method_kwargs):
        """
        Perform a one-dimensional scan of the objective function.

        The parameter specified by parameter_index is fixed to each value
        in parameter_values, and the objective function is evaluated.

        If profile is False, all other parameters are kept fixed. If
        profile is True, the other free parameters are minimized at each
        scan point, resulting in a profile scan.

        The starting parameter values are determined by parameters_start.
        If None, a minimization is performed to determine the best-fit
        parameters. Otherwise, the provided values are used.

        Parameters
        ----------
            parameter_index : int
                Index of the parameter to scan over
            parameter_values : numpy.ndarray
                Values of the scanned parameter at which to evaluate the objective function
            parameters_start : numpy.ndarray (optional)
                Initial parameter values
                If None, a minimization is performed to determine the best-fit parameters.
                If profile is False, all parameters except the scanned parameter are set to these values.
                If profile is True, these values are used as starting values for the minimization at each
                scan point. Parameters already marked as fixed in self.fixed remain fixed.
            data : numpy.ndarray (optional)
                Data to provide to the objective function, if self.signal_function is set
            profile : bool (optional)
                If False, the objective function is evaluated with all parameters except the scanned parameter fixed.
                If True, the objective function is minimized with respect to the other free parameters at each scan point.
            method : str
                Minimization method used for the profile scan, default is "minuit". Only used if profile is True or if parameters_start is None.
            plot : bool (optional)
                If True, plot the objective function relative to its value at the reference parameter point.

        Returns
        -------
            objective_values : numpy.ndarray
                Objective function values corresponding to parameter_values.
        """

        # Save the initial parameter configuration:
        parameters_initial_copy = np.copy(self.parameters_initial)
        fixed_copy = np.copy(self.fixed)

        n_x = len(parameter_values)
        objective_values = np.zeros(n_x)

        if parameters_start is None:
            # Get best fit point:
            self.run_minimization(method=method, data=data, **method_kwargs)
            parameters_reference = self.parameters
        else:
            parameters_reference = np.copy(parameters_start)

        # Evaluate the objective function at the reference point:
        if self.signal_function is not None:
            objective_function_reference = self.objective_function(data, self.signal_function(parameters_reference))
        else:
            objective_function_reference = self.objective_function(parameters_reference)

        if profile:
            # Fix parameters which are being scanned:
            self.fixed[parameter_index] = True

            # Run minimization for each value of the scanned parameter:
            for i_x in range(n_x):
                self.parameters_initial = np.copy(parameters_reference)
                self.parameters_initial[parameter_index] = parameter_values[i_x]
                self.run_minimization(method=method, data=data, **method_kwargs)
                objective_values[i_x] = self.result

        else:
            # Use reference parameters for all parameters except the scanned parameter:
            parameters = np.copy(parameters_reference)

            # Evaluate the objective function for each value of the scanned parameter:
            for i_x in range(n_x):
                parameters[parameter_index] = parameter_values[i_x]
                if self.signal_function is not None:
                    objective_values[i_x] = self.objective_function(data, self.signal_function(parameters))
                else:
                    objective_values[i_x] = self.objective_function(parameters)

        if plot:
            plt.figure(figsize=[4,3])
            plt.plot(parameter_values, objective_values - objective_function_reference, "b-", label="Parameter scan")
            axis = plt.axis()
            plt.axvline(
                parameters_reference[parameter_index], color="y", linestyle="--",
                label="Best-fit value" if parameters_start is None else "Reference value"
            )
            plt.axis([parameter_values[0], parameter_values[-1], axis[2], axis[3]*1.2])
            plt.xlabel(f"Parameter {parameter_index} [au]")
            plt.ylabel(r"Delta objective value")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Set initial and fixed parameters back to initial:
        self.parameters_initial = parameters_initial_copy
        self.fixed = fixed_copy

        return objective_values

    def parameter_scan_2D(
            self, parameter_index_x, parameter_index_y, parameter_values_x, parameter_values_y,
            parameters_start=None, data=None, profile=True, method="minuit", plot=True, **method_kwargs):
        """
        Perform a two-dimensional scan of the objective function.

        The two parameters specified by parameter_index_x and
        parameter_index_y are fixed to each combination of values in
        parameter_values_x and parameter_values_y, and the objective
        function is evaluated.

        If profile is False, all other parameters are kept fixed. If
        profile is True, the other free parameters are minimized at each
        grid point, resulting in a profile scan.

        The starting parameter values are determined by parameters_start.
        If None, a minimization is performed to determine the best-fit
        parameters. Otherwise, the provided values are used.

        Parameters
        ----------
            parameter_index_x : int
                Index of the parameter scanned along the x-axis.
            parameter_index_y : int
                Index of the parameter scanned along the y-axis.
            parameter_values_x : numpy.ndarray
                Values of the x-axis parameter to scan over.
            parameter_values_y : numpy.ndarray
                Values of the y-axis parameter to scan over.
            parameters_start : numpy.ndarray (optional)
                Initial parameter values
                If None, a minimization is performed to determine the best-fit parameters.
                If profile is False, all parameters except the scanned parameters are set to these values.
                If profile is True, these values are used as starting values for the minimization at each
                scan point. Parameters already marked as fixed in self.fixed remain fixed.
            data : numpy.ndarray (optional)
                Data to provide to the objective function, if self.signal_function is set
            profile : bool (optional)
                If False, the objective function is evaluated with all parameters except the scanned parameters fixed.
                If True, the objective function is minimized with respect to the other free parameters at each scan point.
            method : str
                Minimization method used for the profile scan, default is "minuit". Only used if profile is True or if parameters_start is None.
            plot : bool (optional)
                If True, plot the objective function relative to its value at the reference parameter point.

        Returns
        -------
            objective_values : numpy.ndarray
                The values of the objective function for each point in the 2D parameter grid,
                with shape [len(parameter_values_x), len(parameter_values_y)]
        """
        
        parameters_initial_copy = np.copy(self.parameters_initial)
        fixed_copy = np.copy(self.fixed)

        n_x = len(parameter_values_x)
        n_y = len(parameter_values_y)
        objective_values = np.zeros([n_x, n_y])

        if parameters_start is None:
            # Get best fit point:
            self.run_minimization(method=method, data=data, **method_kwargs)
            parameters_reference = self.parameters
        else:
            parameters_reference = np.copy(parameters_start)

        # Evaluate the objective function at the reference point:
        if self.signal_function is not None:
            objective_function_reference = self.objective_function(data, self.signal_function(parameters_reference))
        else:
            objective_function_reference = self.objective_function(parameters_reference)

        if profile:
            # Fix the parameters being scanned:
            self.fixed[parameter_index_x] = True
            self.fixed[parameter_index_y] = True

            # Run minimization for all values of the scanned parameters:
            for i_x in range(n_x):
                for i_y in range(n_y):
                    self.parameters_initial = np.copy(parameters_reference)
                    self.parameters_initial[parameter_index_x] = parameter_values_x[i_x]
                    self.parameters_initial[parameter_index_y] = parameter_values_y[i_y]
                    self.run_minimization(method=method, data=data, **method_kwargs)
                    objective_values[i_x, i_y] = self.result

        else:
            # Use reference parameters for all parameters except the scanned parameters:
            parameters = np.copy(parameters_reference)

            # Evaluate the objective function for all values of the scanned parameters:
            for i_x in range(n_x):
                for i_y in range(n_y):
                    parameters[parameter_index_x] = parameter_values_x[i_x]
                    parameters[parameter_index_y] = parameter_values_y[i_y]
                    if self.signal_function is not None:
                        objective_values[i_x, i_y] = self.objective_function(data, self.signal_function(parameters))
                    else:
                        objective_values[i_x, i_y] = self.objective_function(parameters)

        if plot:
            plt.figure(figsize=[4.2,3])
            plt.pcolormesh(parameter_values_x, parameter_values_y, objective_values.T - objective_function_reference, cmap="Blues_r")
            plt.colorbar(label=r"Delta objective value")
            plt.plot(
                parameters_reference[parameter_index_x], parameters_reference[parameter_index_y], "y.",
                label="Best-fit value" if parameters_start is None else "Reference value"
            )
            plt.legend()
            plt.xlabel(f"Parameter {parameter_index_x} [au]")
            plt.ylabel(f"Parameter {parameter_index_y} [au]")
            plt.tight_layout()
            plt.show()

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
        self._result = result.fun
        self._parameters = result.x * self.normalization

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
        self._result = m.fval
        self._parameters = np.array(m.values) * self.normalization

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
        self._result = res.fun
        self._parameters = np.array(res.x) * self.normalization

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
        self._result = res.fun
        self._parameters = np.array(res.x) * self.normalization

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

        self._result = best_result
        self._parameters = best_parameters * self.normalization

        return best_result
