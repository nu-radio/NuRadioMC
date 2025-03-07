import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from radiotools import helper as hp

import NuRadioReco.framework.radio_shower
from NuRadioReco.utilities import units, geometryUtilities
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp

import cr_pulse_interpolator.interpolation_fourier
import cr_pulse_interpolator.signal_interpolation_fourier

import logging
logger = logging.getLogger('NuRadioReco.coreasInterpolator')


class coreasInterpolator:
    """
    Interface to interpolate the electric field traces, as for example provided by CoREAS.

    For the best results, ensure that the electric fields are in on-sky coordinates (this is
    the case when using the `NuRadioReco.modules.io.coreas.coreas.read_CORSIKA7` function).

    After having created the interpolator, you can either choose to interpolate the electric field traces
    or either the fluence (or both, but be careful to not mix them up). In order to do this, you first need
    to initialize the corresponding interpolator with the desired settings. Refer to the documentation of
    `initialize_efield_interpolator` and `initialize_fluence_interpolator` for more information. After
    initialization, you can call the `interpolate_efield` and `interpolate_fluence` functions to get the
    interpolated values.

    Note that when trying to interpolate the electric fields of an air shower with a geomagnetic angle
    smaller than 15 degrees, the interpolator will fall back to using the closest observer position
    instead of performing the Fourier interpolation. Also, when attempting to extrapolate (ie get a value
    for a position that is outside the star shape pattern), the default behaviour is to return zeros when
    r > r_max and return a constant value when r <= r_min. This behaviour can be changed using the
    ``allow_extrapolation`` keyword argument when initialising the interpolator.

    Parameters
    ----------
    corsika_evt : Event
        An Event object containing the CoREAS output, from read_CORSIKA7()

    Notes
    -----
    The interpolation method is based on the Fourier interpolation method described in
    Corstanje et al. (2023), JINST 18 P09005. The implementation lives in a separate package
    called ``cr-pulse-interpolator`` (this package is a optional dependency of NuRadioReco).

    In short, the interpolation method works as follows: everything is done in the showerplane,
    where we expect circular symmetries due to the emission mechanisms. In case of fluence interpolation,
    we are working with a single value per observer position. The positions are expected to be in a
    starshape pattern, such that for each ring a Fourier series can be constructed for the fluence values.
    The components of the Fourier series are then radially interpolated using a cubic spline.

    When interpolating electric fields, the traces are first transformed to frequency spectra. The amplitude
    in each frequency bin is then interpolated as a single value, as described above. The phases are also
    interpolated, making the relative timings consistent. To also get an absolute timing, one can provide
    the trace start times to also be interpolated (this is done in this implementation).
    """

    def __init__(self, corsika_evt):
        # These are set by self.initialize_star_shape()
        self.sampling_rate = None
        self.electric_field_on_sky = None
        self.efield_times = None
        self.__efields_rotation_angle = 0

        self.obs_positions_ground = None
        self.obs_positions_showerplane = None

        # Store the SimStation and SimShower objects
        self.sim_station = corsika_evt.get_station(0).get_sim_station()
        self.shower: NuRadioReco.framework.radio_shower.RadioShower = corsika_evt.get_first_sim_shower()  # there should only be one simulated shower
        self.cs = self.shower.get_coordinatesystem()

        # Flags to check whether interpolator is initialized
        self.star_shape_initialized = False
        self.efield_interpolator_initialized = False
        self.fluence_interpolator_initialized = False

        # Interpolator objects
        self.interp_lowfreq = None
        self.interp_highfreq = None
        self.efield_interpolator = None
        self.fluence_interpolator = None

        self.initialize_star_shape()

        logger.info(
            f'Initialised star shape pattern for interpolation. '
            f'The shower arrives from zenith={self.zenith / units.deg:.1f}deg, '
            f'azimuth={self.azimuth / units.deg:.1f}deg '
            f'The starshape has radius {self.starshape_radius:.0f}m in the shower plane '
            f'and {self.starshape_radius_ground:.0f}m on ground. '
        )

    def initialize_star_shape(self):
        """
        Initializes the star shape pattern for interpolation, e.g. creates the arrays with the observer positions
        in the shower plane and the electric field.
        """
        if not np.all(self.shower.get_parameter(shp.core)[:2] == 0):
            logger.status(
                "The shower core is not at the origin. "
                "Be careful to adjust antenna positions accordingly when retrieving interpolated values."
            )

        obs_positions = []
        electric_field_on_sky = []
        efield_times = []

        for j_obs, efield in enumerate(self.sim_station.get_electric_fields()):
            obs_positions.append(efield.get_position())
            electric_field_on_sky.append(efield.get_trace().T)
            efield_times.append(efield.get_times())

        self.electric_field_on_sky = np.array(electric_field_on_sky)  # shape: (n_observers, n_samples, (eR, eTheta, ePhi))
        self.efield_times = np.array(efield_times)
        self.sampling_rate = 1. / (self.efield_times[0][1] - self.efield_times[0][0])

        self.obs_positions_ground = np.array(obs_positions)  # (n_observers, 3)
        self.obs_positions_showerplane = self.cs.transform_to_vxB_vxvxB(
            self.obs_positions_ground, core=self.shower.get_parameter(shp.core)
        )

        self.star_shape_initialized = True

    @property
    def zenith(self):
        """The zenith of the shower axis"""
        return self.shower.get_parameter(shp.zenith)

    @property
    def azimuth(self):
        """The azimuth of the shower axis"""
        return self.shower.get_parameter(shp.azimuth)

    @property
    def magnetic_field_vector(self):
        """The magnetic field vector stored inside the shower"""
        return self.shower.get_parameter(shp.magnetic_field_vector)

    @property
    def starshape_radius(self):
        """
        returns the maximal radius of the star shape pattern in the shower plane
        """
        if not self.star_shape_initialized:
            logger.error('The interpolator was not initialized, call initialize_star_shape first')
            return None
        else:
            return np.max(np.linalg.norm(self.obs_positions_showerplane[:, :-1], axis=-1))

    @property
    def starshape_radius_ground(self):
        """
        returns the maximal radius of the star shape pattern on ground
        """
        if not self.star_shape_initialized:
            logger.error('The interpolator was not initialized, call initialize_star_shape first')
            return None
        else:
            return np.max(np.linalg.norm(self.obs_positions_ground[:, :-1], axis=-1))

    @property
    def efields_rotation_angle(self):
        """
        The angle with which the e_theta and e_phi components of the electric field were rotated
        """
        return self.__efields_rotation_angle

    def get_empty_efield(self):
        """
        Get an array of zeros in the shape of the electric field on the sky
        """
        if not self.star_shape_initialized:
            logger.error('The interpolator was not initialized, call initialize_star_shape first')
            return None
        else:
            return np.zeros_like(self.electric_field_on_sky[0, :, :])

    def set_fluence_of_efields(self, function, quantity=efp.signal_energy_fluence):
        """
        Set the fluence quantity of all electric fields in the SimStation of the interpolator.

        Helper function to set the fluence of electric fields. These values can then be used for fluence interpolation.

        One option to use as ``function`` is `NuRadioReco.utilities.trace_utilities.get_electric_field_energy_fluence`.

        Parameters
        ----------
        function: callable
            The function to apply to the traces in order to calculate the fluence. Should take in a (3, n_samples) shaped
            array and return a float (or an array with 3 elements if you want the fluence per polarisation).
        quantity: electric field parameter, default=efp.signal_energy_fluence
            The parameter where to store the result of the fluence calculation

        See Also
        --------
        NuRadioReco.utilities.trace_utilities.get_electric_field_energy_fluence
            Function that can be passed as ``function`` to obtain the energy fluence of
            the electric fields in the SimStation.
        """
        coreas.set_fluence_of_efields(function, self.sim_station, quantity)

    def __rotate_efield_polarizations(self):
        """
        Rotate the electric field polarizations to make sure they do not align with vxB and vxvxB axes.
        Otherwise the amplitudes will have zeros along the circles in the footprint, which makes the
        interpolation unstable.

        Notes
        -----
        Currently the rotation is hardcoded to 45 degrees when the on-sky axes align too closely with
        the vxB or vxvxB axes. However, it could be possible to make this more general by looking at how
        a total mix of vxB and vxvxB (ie sum them and normalize) maps to the on-sky CS and then rotate the
        electric fields to align with this vector. This would (probably) ensure that both components always
        have a reasonable amplitude.
        """
        magnetic_field_normalized = self.magnetic_field_vector / np.linalg.norm(self.magnetic_field_vector)
        v = -1 * hp.spherical_to_cartesian(
            self.zenith, self.azimuth
        )
        vxB = np.cross(v, magnetic_field_normalized)

        # Check if vxB align with e_theta or e_phi axes in on-sky CS
        cs = self.shower.get_coordinatesystem()
        vxB_on_sky = cs.transform_from_ground_to_onsky(vxB)
        vxB_on_sky /= np.linalg.norm(vxB_on_sky)

        # Ensure the on-sky polarisations contain at least 1% from both vxB and vxvxB components
        # These atol/rtol settings should catch all showers which are within 6 degrees in azimuth
        # from the N-S axis or 2 degrees away from zenith
        if (np.allclose(np.abs(vxB_on_sky), [0, 0, 1], atol=1e-1, rtol=0) or
                np.allclose(np.abs(vxB_on_sky), [0, 1, 0], atol=1e-1, rtol=0)):
            logger.info(
                "The coordinate axes in the on-sky coordinate system align too close with the vxB or vxvxB axes. "
                "This can cause problems in interpolation, so I will rotate the electric fields polarisations... "
            )

            self.__efields_rotation_angle = 45 * units.deg
            rotated_efields = np.matmul(
                self.electric_field_on_sky, geometryUtilities.rot_x(self.__efields_rotation_angle / units.rad)
            )
            self.electric_field_on_sky = rotated_efields

    def initialize_efield_interpolator(self, interp_lowfreq, interp_highfreq, **kwargs):
        """
        Initialise the efield signal interpolator

        If initialised correctly, this method returns the resulting efield interpolator (it is also stored
        as the ``efield_interpolator`` attribute of the class). If the latter, the ``efield_interpolator_initialized``
        is also set to ``True``.

        If the geomagnetic angle is smaller than 15deg, the interpolator switches to using the closest
        observer position instead. In this case, no interpolator object is returned and the
        ``efield_interpolator`` attribute is set to a Fourier interpolator which interpolates
        the arrival times. To distinguish between the two cases, you can check the
        ``efield_interpolator_initialized`` attribute, which is ``True`` when the signal interpolator is
        initialised and ``False`` when the closest observer is used.

        To adjust the parameters of the signal interpolation, the options for the `interp2d_signal` class
        can be passed as keyword arguments. Please refer to the ``cr-pulse-interpolator`` documentation
        for all available options.

        Parameters
        ----------
        interp_lowfreq : float
            Lower frequency for the bandpass filter in interpolation (in internal units)
        interp_highfreq : float
            Upper frequency for the bandpass filter in interpolation (in internal units)
        **kwargs : options to pass on to the `interp2d_signal` class
            Default values are:

            - phase_method : 'phasor'
            - radial_method : 'cubic'
            - upsample_factor : 5
            - coherency_cutoff_threshold : 0.9
            - allow_extrapolation : False
            - ignore_cutoff_freq_in_timing : False
            - verbose : False

        Returns
        -------
        efield_interpolator : interpolator object

        """
        self.interp_lowfreq = interp_lowfreq
        self.interp_highfreq = interp_highfreq
        interp_options_default = {
            "phase_method" : "phasor",
            "radial_method" : 'cubic',
            "upsample_factor" : 5,
            "coherency_cutoff_threshold" : 0.9,
            "allow_extrapolation" : False,
            "ignore_cutoff_freq_in_timing" : False,
            "verbose" : False
        }
        interp_options = {**kwargs, **interp_options_default}  # merge the dicts, making sure kwargs overwrite defaults

        geomagnetic_angle = coreas.get_geomagnetic_angle(self.zenith, self.azimuth, self.magnetic_field_vector)

        # Use closest observer when geomagnetic angle is smaller than 15deg
        if geomagnetic_angle < 15 * units.deg:
            logger.warning(
                f'The geomagnetic angle is {geomagnetic_angle / units.deg:.2f} deg, '
                f'which is smaller than 15deg, which is the lower limit for the signal interpolation. '
                f'The closest observer is used instead.'
            )

            self.efield_interpolator = cr_pulse_interpolator.interpolation_fourier.interp2d_fourier(
                self.obs_positions_showerplane[:, 0], self.obs_positions_showerplane[:, 1], self.efield_times
            )

            return None

        logger.info(
            f'Initialising electric field interpolator with lowfreq {interp_lowfreq / units.MHz} MHz '
            f'and highfreq {interp_highfreq / units.MHz} MHz'
        )
        logger.debug(
            f'The following interpolation settings are used: {interp_options}'
        )

        # Rotate the electric field polarizations if necessary
        self.__rotate_efield_polarizations()

        # Construct the interpolator with the required settings
        self.efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
            self.obs_positions_showerplane[:, 0],
            self.obs_positions_showerplane[:, 1],
            self.electric_field_on_sky,
            signals_start_times=self.efield_times[:, 0] / units.s,
            sampling_period=1 / self.sampling_rate / units.s,  # interpolator wants sampling period in seconds
            lowfreq=interp_lowfreq / units.MHz,
            highfreq=interp_highfreq / units.MHz,
            **interp_options
        )

        self.efield_interpolator_initialized = True

        return self.efield_interpolator

    def initialize_fluence_interpolator(self, quantity=efp.signal_energy_fluence, **kwargs):
        """
        Initialise fluence interpolator.

        Initialize the fluence interpolator using the values stored in the ``quantity`` parameter of the electric
        fields.

        Parameters
        ----------
        quantity : electric field parameter, default=efp.signal_energy_fluence
            The quantity to get the values from which are fed to the interpolator. It needs to be available
            as parameter in the electric field object! You can use the `set_fluence_of_efields()` function to
            set this value for all electric fields.
        **kwargs: options to pass on to the `interp2d_fourier` class
            Default values are:

            - radial_method : 'cubic'
            - fill_value : None
            - recover_concentric_rings : False

        Returns
        -------
        fluence_interpolator : interpolator object
        """
        interp_options_default = {
            "radial_method" : 'cubic',
            "fill_value" : None,
            "recover_concentric_rings" : False,
        }
        interp_options = {**kwargs, **interp_options_default}

        fluence_per_position = [
            np.sum(efield[quantity]) for efield in self.sim_station.get_electric_fields()
        ]  # the fluence is calculated per polarization, so we need to sum them up

        logger.info(f'Initialising fluence interpolator')
        logger.debug(f'The following interpolation settings are used: {interp_options}')

        self.fluence_interpolator = cr_pulse_interpolator.interpolation_fourier.interp2d_fourier(
            self.obs_positions_showerplane[:, 0],
            self.obs_positions_showerplane[:, 1],
            fluence_per_position,
            **interp_options
        )
        self.fluence_interpolator_initialized = True

        return self.fluence_interpolator

    def get_position_showerplane(self, position_on_ground):
        """
        Transform the position of the antenna on ground to the shower plane.

        Parameters
        ----------
        position_on_ground : np.ndarray
            Position of the antenna on ground
            This value can either be a 2D array (x, y) or a 3D array (x, y, z). If the z-coordinate is missing, the
            z-coordinate is automatically set to the observation level of the simulation.
        """
        core = self.shower.get_parameter(shp.core)

        if len(position_on_ground) == 2:
            position_on_ground = np.append(position_on_ground, core[2])
            logger.info(
                f"The antenna position is given in 2D, assuming the antenna is on the ground. "
                f"The z-coordinate is set to the observation level {core[2] / units.m:.2f}m"
            )
        elif abs(position_on_ground[2] - core[2]) > 5 * units.cm:
            logger.warning(
                f"The antenna z-coordinate {position_on_ground[2]} differs significantly from "
                f"the observation level {core[2]}. This behaviour is not tested, so only proceed "
                f"if you know what you are doing."
            )

        antenna_pos_showerplane = self.cs.transform_to_vxB_vxvxB(position_on_ground, core=core)

        return antenna_pos_showerplane

    def get_interp_efield_value(self, position_on_ground, cs_transform='no_transform'):
        """
        Calculate the interpolated electric field given an antenna position on ground.

        If the geomagnetic angle is smaller than 15deg,
        the electric field of the closest observer position is returned instead.

        Note that `position_on_ground` is in absolute coordinates, not relative to the core position.
        Extrapolation outside of the starshape is handled by the ``cr-pulse-interpolator`` package (see
        also the description of ``kwargs`` in `initialize_efield_interpolator`).

        Parameters
        ----------
        position_on_ground : np.ndarray
            Position of the antenna on ground
            This value can either be a 2D array (x, y) or a 3D array (x, y, z). If the z-coordinate is missing, the
            z-coordinate is automatically set to the observation level of the simulation.
        cs_transform: {'no_transform', 'sky_to_ground'}, default='no_transform'
            Optional coordinate system transformation to apply to the interpolated electric field. If 'no_transform',
            the default, the electric field is returned in the same coordinate system as it was stored in the Event
            used for initialisation (usually on-sky).
            If 'sky_to_ground', the electric field is transformed to the ground coordinate system using the
            `cstrafo.transform_from_onsky_to_ground()` function.

        Returns
        -------
        efield_interp : float
            Interpolated efield value
        trace_start_time : float
            Start time of the trace
        """
        logger.debug(
            f"Getting interpolated efield for antenna position {position_on_ground} on ground"
        )

        antenna_pos_showerplane = self.get_position_showerplane(position_on_ground)
        logger.debug(f"The antenna position in shower plane is {antenna_pos_showerplane}")

        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if not self.efield_interpolator_initialized:
            # Get electric field of closest observer position and associated time (which is already in internal units)
            efield_interp, trace_start_time = self.get_closest_observer_efield(antenna_pos_showerplane)
        else:
            efield_interp, trace_start_time, _, _ = self.efield_interpolator(
                antenna_pos_showerplane[0], antenna_pos_showerplane[1],
                lowfreq=self.interp_lowfreq / units.MHz,
                highfreq=self.interp_highfreq / units.MHz,
                filter_up_to_cutoff=False,
                account_for_timing=True,
                pulse_centered=True,
                full_output=True
            )

            trace_start_time *= units.second  # interpolator returns start time in seconds

            # Rotate the electric field back to the normal on-sky coordinate system
            efield_interp = np.matmul(
                efield_interp, geometryUtilities.rot_x(-1 * self.__efields_rotation_angle / units.rad)
            )

        if cs_transform == 'sky_to_ground':
            efield_interp = self.cs.transform_from_onsky_to_ground(efield_interp)

        return efield_interp, trace_start_time

    def get_interp_fluence_value(self, position_on_ground):
        """
        Calculate the interpolated fluence for a given position on the ground.

        Note that ``position_on_ground`` is in absolute coordinates, not relative to the core position.
        Extrapolation outside of the starshape is handled by the ``cr-pulse-interpolator`` package (see
        also the description of ``kwargs`` in `initialize_fluence_interpolator`).

        Parameters
        ----------
        position_on_ground : np.ndarray
            Position of the antenna on ground
            This value can either be a 2D array (x, y) or a 3D array (x, y, z). If the z-coordinate is missing, the
            z-coordinate is automatically set to the observation level of the simulation.

        Returns
        -------
        fluence_interp : float
            interpolated fluence value
        """
        logger.debug(
            f"Getting interpolated fluence for antenna position {position_on_ground} on ground"
        )

        antenna_pos_showerplane = self.get_position_showerplane(position_on_ground)
        logger.debug(f"The antenna position in shower plane is {antenna_pos_showerplane}")

        # interpolate fluence at antenna position, letting interpolator handle extrapolation
        fluence_interp = self.fluence_interpolator(antenna_pos_showerplane[0], antenna_pos_showerplane[1])

        return fluence_interp

    def get_closest_observer_efield(self, antenna_pos_showerplane):
        """
        Returns the electric field of the closest observer position for an antenna position in the shower plane.

        The start time of the trace is also returned, by interpolating the start times of the electric field traces.

        This function is not meant to be called directly, but only from `get_interp_efield_value` as it assumes
        the ``efield_interpolator`` attribute is set to the start time interpolator, which is done in
        `initialize_efield_interpolator`.

        Parameters
        ----------
        antenna_pos_showerplane : np.ndarray
            antenna position in the shower plane

        Returns
        -------
        efield: np.ndarray
            electric field, as an array shaped
        efield_start_time: float
            The start time of the selected electric field trace (in internal units)
        """
        distances = np.linalg.norm(antenna_pos_showerplane[:2] - self.obs_positions_showerplane[:, :2], axis=1)
        index = np.argmin(distances)
        efield = self.electric_field_on_sky[index, :, :]
        # The interpolated values were the start times which were already in internal units, so what comes out
        # is also in internal units, ie no need to multiply with units.second here
        efield_start_time = self.efield_interpolator(*antenna_pos_showerplane[:2])
        logger.debug(
            f'Returning the electric field of the closest observer position, '
            f'which is {distances[index] / units.m:.2f}m away from the antenna, the time is interpolated'
        )
        return efield, efield_start_time

    def plot_fluence_footprint(self, radius=300):
        """
        plots the interpolated values of the fluence in the shower plane

        Parameters
        ----------
        radius : float
            radius around shower core which should be plotted

        Returns
        -------
        fig : figure object
        ax : axis object
        """
        if not self.fluence_interpolator_initialized:
            logger.error('The fluence interpolator was not initialized, call initialize_fluence_interpolator first')
            return None

        # Make color plot of f(x, y), using a meshgrid
        ti = np.linspace(-radius, radius, 500)
        XI, YI = np.meshgrid(ti, ti)

        # Get interpolated values at each grid point, calling the instance of interp2d_fourier
        ZI = self.fluence_interpolator(XI, YI)

        # And plot it
        maxp = np.max(ZI)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolor(XI, YI, ZI, vmax=maxp, vmin=0, cmap=cm.gnuplot2_r)
        mm = cm.ScalarMappable(cmap=cm.gnuplot2_r)
        mm.set_array([0.0, maxp])
        cbar = plt.colorbar(mm, ax=ax)
        cbar.set_label(r'energy fluence [eV/m^2]', fontsize=14)
        ax.set_xlabel(r'$\vec{v} \times \vec{B} [m]', fontsize=16)
        ax.set_ylabel(r'$\vec{v} \times (\vec{v} \times \vec{B})$ [m]', fontsize=16)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect('equal')
        return fig, ax
