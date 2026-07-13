import os
import sys
import importlib
import json
import numpy as np
import logging
import torch
from radiotools import helper as hp

from NuRadioReco.utilities import units, geometryUtilities
from NuRadioMC.utilities import attenuation as attenuation_util
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert
from NuRadioMC.SignalProp.analyticraytracing import ray_tracing_2D
from NuRadioReco.framework.parameters import electricFieldParameters as efp


ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))


class DeepLearningRayTracing(ray_tracing_base):
    """
    Deep learning ray tracing class.
    """
    def __init__(self, medium, attenuation_model=None, log_level=logging.NOTSET,
                 n_frequencies_integration=None, n_reflections=None, config=None,
                 detector=None, model_name=None, device=None):
        """
        class initilization

        Parameters
        ----------
        model_name: str
            Name of the deep learning model to load.
        medium: medium class
            class describing the index-of-refraction profile
        """
        super().__init__(
            medium=medium,
            attenuation_model=attenuation_model,
            log_level=log_level,
            n_frequencies_integration=n_frequencies_integration,
            n_reflections=n_reflections,
            config=config,
            detector=detector
        )

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = self._load_deep_learning_model(model_name=model_name)

        # Use ray_tracing_2D as a helper class:
        self.ray_tracing_2D = ray_tracing_2D(self._medium, self._attenuation_model, log_level=log_level,
                            n_frequencies_integration=self._n_frequencies_integration, use_cpp=False, compile_numba=False)

    def _load_deep_learning_model(self, model_name):
        """
        Load the deep learning model for ray tracing.

        Parameters
        ----------
        model_name: str
            Name of the deep learning model to load.
        """
        if model_name is None:
            model_name = "model_shallow_7" #"ray_tracing_model"  # Default model name
        
        model_folder_path = os.path.join(ABS_PATH_HERE, "deep_learning_models", model_name)

        spec = importlib.util.spec_from_file_location("model", os.path.join(model_folder_path, "model.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with open(os.path.join(model_folder_path, "settings.json"), "r", encoding="utf-8") as settings_file:
            settings_dict = json.load(settings_file)

        model = module.Model(
            settings_dict["n_features"],
            settings_dict["n_nodes"],
            settings_dict["n_labels"],
            settings_dict["x_normalization"],
            settings_dict["y_normalization"],
        ).to(self.device)
        model.load_state_dict(
            torch.load(
                os.path.join(model_folder_path, f"{model_name}.pth"),
                weights_only=True,
                map_location=self.device,
            )
        )
        model.eval()

        return model

    def set_start_and_end_point(self, x1, x2):
        """
        Set the start and end points between which raytracing solutions shall be found.

        If the start and end points are identical to those of the previous ray tracing,
        the existing solutions are kept and `find_solutions` will not recalculate them
        (relevant e.g. if several showers are simulated at the same position).
        Call `reset_solutions` before this function to force a recalculation.

        Parameters
        ----------
        x1: np.array of shape (3,), default unit
            start point of the ray
        x2: np.array of shape (3,), default unit
            stop point of the ray

        Returns
        -------
        geometry_changed: bool
            False if the start and end points are unchanged with respect to the previous
            ray tracing (in which case the existing solutions are kept), True otherwise.
        """
        assert x2 in np.array([[3, 0, -3], [-3, 0, -3], [0, 3, -3], [0, -3, -3], [0, 0, -10.0]]), "Current surrogate model has only been trained for shallow Gen2 station"

        geometry_changed = super().set_start_and_end_point(x1, x2)

        return geometry_changed



    def find_solutions(self):
        """
        find all solutions between x1 and x2
        """
        n_solutions, launch_vectors, receive_vectors, theta_receives, phi_receives, path_lengths, travel_times, reference_time = self._trace(self._X1, self._X2)

        self._results = []
        for iS in range(n_solutions):
            self._results.append({
                'type': solution_types_revert["direct"] if iS == 0 else solution_types_revert["reflected"],  # Assuming the first solution is direct and the second is reflected
                'launch_vector': launch_vectors[iS, :],
                'receive_vector': receive_vectors[iS, :],
                'theta_receive': theta_receives[iS, 0],
                'phi_receive': phi_receives[iS, 0],
                'path_length': path_lengths[iS, 0],
                'travel_time': travel_times[iS, 0],
                'reference_time': reference_time
            })

    def _trace(self, xyz_start, xyz_end):
        xyz_start = torch.as_tensor(xyz_start, dtype=torch.float32, device=self.device).reshape(1, 3)
        xyz_end = torch.as_tensor(xyz_end, dtype=torch.float32, device=self.device).reshape(1, 3)

        z_det = xyz_end[:, 2:3]

        # Compute distance in the horizontal plane for the single start/end pair.
        start_xy = xyz_start[:, :2]
        end_xy = xyz_end[:, :2]
        delta_xy = torch.linalg.norm(end_xy - start_xy, dim=-1, keepdim=True)

        z_vertex = xyz_start[:, 2:3]

        ray_tracing_input = torch.cat([z_det, delta_xy, z_vertex], dim=-1)

        result = self.model.predict(ray_tracing_input)
        result = result.reshape(1, 1, 8)

        n_solutions = 2
        path_lengths = torch.stack([result[0, 0, 0], result[0, 0, 4]], dim=0).unsqueeze(-1) * units.m
        travel_times = torch.stack([result[0, 0, 1], result[0, 0, 5]], dim=0).unsqueeze(-1) * units.ns
        theta_launch = torch.stack([result[0, 0, 2], result[0, 0, 6]], dim=0).abs().unsqueeze(-1) * units.deg
        theta_receives = torch.stack([result[0, 0, 3], result[0, 0, 7]], dim=0).unsqueeze(-1) * units.deg
        antenna_to_vertex = (xyz_start - xyz_end).detach().cpu().numpy()
        phi_receives = torch.as_tensor(
            hp.cartesian_to_spherical(
                antenna_to_vertex[:, 0], antenna_to_vertex[:, 1], antenna_to_vertex[:, 2]
            )[1],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1).repeat(n_solutions, 1)
        receive_vectors = hp.spherical_to_cartesian(
            theta_receives.detach().cpu().numpy().reshape(-1),
            phi_receives.detach().cpu().numpy().reshape(-1)
        )
        phi_launch = phi_receives + np.pi
        launch_vectors = hp.spherical_to_cartesian(
            theta_launch.detach().cpu().numpy().reshape(-1),
            phi_launch.detach().cpu().numpy().reshape(-1),
        )
        theta_receives = theta_receives.detach().cpu().numpy()
        phi_receives = phi_receives.detach().cpu().numpy()
        path_lengths = path_lengths.detach().cpu().numpy()
        travel_times = travel_times.detach().cpu().numpy()
        reference_time = travel_times[0, 0]

        return n_solutions, launch_vectors, receive_vectors, theta_receives, phi_receives, path_lengths, travel_times, reference_time

    def get_solution_type(self, iS):
        """ returns the type of the solution

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        solution_type: int
            integer corresponding to the types in the dictionary solution_types
        """
        return self._results[iS]['type']

    def get_path(self, iS, n_points=1000):
        """
        This is not needed for the deep learning ray tracing?
        """
        self.__logger.error('function not defined')
        raise NotImplementedError

    def get_launch_vector(self, iS):
        """
        calculates the launch vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        launch_vector: 3dim np.array
            the launch vector
        """
        return self._results[iS]['launch_vector']

    def get_receive_vector(self, iS):
        """
        calculates the receive vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the receive vector, counting
            starts at zero

        Returns
        -------
        receive_vector: 3dim np.array
            the receive vector
        """
        return self._results[iS]['receive_vector']

    def get_reflection_angle(self, iS):
        """
        This is not needed for the deep learning ray tracing?
        """
        self.__logger.error('function not defined')
        raise NotImplementedError

    def get_path_length(self, iS, analytic=True):
        """
        calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the path length, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        distance: float
            distance from x1 to x2 along the ray path
        """
        return self._results[iS]['path_length']

    def get_travel_time(self, iS, analytic=True):
        """
        calculates the travel time of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        time: float
            travel time
        """
        return self._results[iS]['travel_time']
        

    def get_attenuation(self, iS, frequency, max_detector_freq=None):
        """
        This is a simple approximation of the attenuation along the path. The neural network
        does not predict the path, so instead we calculate the attenuation along a straight
        line between the start and end points.

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        frequency: array of floats
            the frequencies for which the attenuation is calculated

        max_detector_freq: float or None
            the maximum frequency of the final detector sampling
            (the simulation is internally run with a higher sampling rate, but the relevant part of the attenuation length
            calculation is the frequency interval visible by the detector, hence a finer calculation is more important)

        Returns
        -------
        attenuation: array of floats
            the fraction of the signal that reaches the observer
            (only ice attenuation, the 1/R signal falloff not considered here)
        """

        n_steps = 25
        z_0 = self._X1[2]
        z_1 = self._X2[2]
        z = np.linspace(z_0, z_1, n_steps)

        freqs = frequency #self.ray_tracing_2D._ray_tracing_2D__get_frequencies_for_attenuation(frequency, max_detector_freq)

        attenuation_lengths = np.zeros([n_steps, len(freqs)])
        mask = freqs > 0
        for i in range(n_steps):
            attenuation_lengths[i, mask] = attenuation_util.get_attenuation_length(z[i], freqs[mask], self._attenuation_model)
        mean_attenuation_lengths = np.mean(attenuation_lengths, axis=0)

        attenuation = np.ones_like(freqs)
        attenuation[mask] = np.exp(-self.get_path_length(iS) / mean_attenuation_lengths[mask])

        return attenuation


    def apply_propagation_effects(self, efield, i_solution):
        """
        Apply propagation effects to the electric field
        Note that the 1/r weakening of the electric field is already accounted for in the signal generation

        Parameters
        ----------
        efield: ElectricField object
            The electric field that the effects should be applied to
        i_solution: int
            Index of the raytracing solution the propagation effects should be based on

        Returns
        -------
        efield: ElectricField object
            The modified ElectricField object
        """
        s_rate = efield.get_sampling_rate()
        frequencies = efield.get_frequencies()
        spec = efield.get_frequency_spectrum()

        if self._max_detector_frequency is None:
            max_freq = np.max(efield.get_frequencies())
        else:
            max_freq = self._max_detector_frequency

        attenuation = self.get_attenuation(i_solution, frequencies, max_freq)

        spec *= attenuation

        if i_solution == 1:
            zenith_reflection = self._results[i_solution]['theta_receive'] # For a shallow station the zenith angle of the reflection is to 1st order the reflection angle
            r_theta = geometryUtilities.get_fresnel_r_p(
                zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
            r_phi = geometryUtilities.get_fresnel_r_s(
                zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
            efield[efp.reflection_coefficient_theta] = r_theta
            efield[efp.reflection_coefficient_phi] = r_phi
            spec[1] *= r_theta
            spec[2] *= r_phi

        # apply the focusing effect
        # NOT IMPLENTED

        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        return efield



    def get_output_parameters(self):
        """
        Returns a list with information about parameters to include in the output data structure that are specific
        to this raytracer

        ! be sure that the first entry is specific to your raytracer !

        Returns
        -------
        list with entries of form [{'name': str, 'ndim': int}]
            ! be sure that the first entry is specific to your raytracer !
            'name': Name of the new parameter to include in the data structure
            'ndim': Dimension of the data structure for the parameter
        """
        self.__logger.error('function not defined')
        raise NotImplementedError

    def get_raytracing_output(self, i_solution):
        """
        Write parameters that are specific to this raytracer into the output data.

        Parameters
        ----------
        i_solution: int
            The index of the raytracing solution

        Returns
        -------
        dictionary with the keys matching the parameter names specified in get_output_parameters and the values being
        the results from the raytracing
        """
        return self._results[i_solution]
