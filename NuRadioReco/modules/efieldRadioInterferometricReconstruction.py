from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, interferometry
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp

from radiotools import helper as hp, coordinatesystems
from radiotools.atmosphere import models, refractivity

from collections import defaultdict
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import logging
logger = logging.getLogger('efieldRadioInterferometricReconstruction')

""" 
This module hosts to classes
    - efieldInterferometricDepthReco
    - efieldInterferometricAxisReco (TODO: Not yet implemented)

The radio-interferometric reconstruction (RIT) was proposed in [1]. 
The implementation here is based on work published in [2].

[1]: H. Schoorlemmer, W. R. Carvalho Jr., arXiv:2006.10348
[2]: F. Schlueter, T. Huege, doi:10.1088/1748-0221/16/07/P07048

"""

class efieldInterferometricDepthReco:
    """
    This class reconstructs the depth of the maximum of the longitudinal profile of the 
    beam-formed radio emission: X_RIT along a given axis. X_RIT is found to correlate with X_max. 
    This correlation may depend on the zenith angle, the frequency band, and detector layout
    (if the detector is very irregular).

    For the reconstruction, a shower axis is needed. The radio emission in the vxB polarisation is used 
    (and thus the arrival direction is needed).

    """

    def __init__(self):
        self._debug = None
        self._at = None
        self._tab = None
        self._interpolation = None
        self._signal_kind = None
        pass

    def begin(self, interpolation=True, signal_kind="power", debug=False):
        """
        Set module config.

        Parameters
        ----------

        interpolation : bool
            If true, use a linear interpolation to match sampling of the beamformed signal trace and the individual time-shifted antenna traces.
            Default is True, False is not yet implemented

        signal_kind : str
            Define which signal "metric" is used on the beamformed traces. Default "power" : sum over the squared amplitudes in a 100 ns window around the peak.
            Other options are "amplitude" or "hilbert_sum" 

        debug : bool
            If true, show some debug plots (Default: False).
        """
        self._debug = debug
        self._interpolation = interpolation
        self._signal_kind = signal_kind

        self._data = defaultdict(list)
        pass


    def sample_longitudinal_profile(
            self, traces, times, station_positions,
            shower_axis, core, depths=None, distances=None):
        """
        Returns the longitudinal profile of the interferometic signal sampled along the shower axis. 

        Parameters
        ----------

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis : array(3,)
            Axis/direction along which the interferometric signal is sampled. Anchor is "core".

        core : array(3,)
            Shower core. Keep in mind that the altitudes (z-coordinate) matters.

        depths : array (optinal)
            Define the positions (slant depth along the axis) at which the interferometric signal is sampled. 
            Instead of "depths" you can provide "distances".

        distances : array (optinal)
            Define the positions (geometrical distance from core along the axis) at which the interferometric signal is sampled.
            Instead of "distances" you can provide "depths".

        Returns
        -------
 
        signals : array
            Interferometric singals sampled along the given axis
        """

        zenith = hp.get_angle(np.array([0, 0, 1]), shower_axis)
        tstep = times[0, 1] - times[0, 0]

        if depths is not None:
            signals = np.zeros(len(depths))
            depths_or_distances = depths
        else:
            if distances is None:
                sys.exit("depths or distances has to be not None!")
            signals = np.zeros(len(distances))
            depths_or_distances = distances

        for idx, dod in enumerate(depths_or_distances):
            if depths is not None:
                try:
                    # here z coordinate of core has to be the altitude of the observation_level
                    dist = self._at.get_distance_xmax_geometric(
                        zenith, dod, observation_level=core[-1])
                except ValueError:
                    signals[idx] = 0
                    continue
            else:
                dist = dod

            if dist < 0:
                signals[idx] = 0
                continue

            point_on_axis = shower_axis * dist + core
            if self._interpolation:
                sum_trace = interferometry.interfere_traces_interpolation(
                    point_on_axis, station_positions, traces, times, tab=self._tab)
            else:
                # sum_trace = interferometry.interfere_traces_padding(
                #     point_on_axis, station_positions, core, traces, times, tab=self._tab)
                sys.exit("Not implemented")

            # plt.title(dod)
            # plt.plot(sum_trace)
            # plt.show()

            signal = interferometry.get_signal(sum_trace, tstep, kind=self._signal_kind)
            signals[idx] = signal

        return signals


    def reconstruct_interferometric_depth(
            self, traces, times, station_positions, shower_axis, core,
            lower_depth=400, upper_depth=800, bin_size=100, return_profile=False):
        """
        Returns Gauss-parameters fitted to the "peak" of the interferometic 
        longitudinal profile along the shower axis.
        
        A initial samping range and size in defined by "lower_depth", "upper_depth", "bin_size".
        However if the "peak", i.e., maximum signal is found at an edge the sampling range in
        continually increased (with a min/max depth of 0/2000 g/cm^2). The Gauss is fitted around the
        found peak with a refined sampling (use 20 samples in this narrow range).

        Parameters
        ----------

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis : array(3,)
            Axis/direction along which the interferometric signal is sampled. Anchor is "core".

        core : array(3,)
            Shower core. Keep in mind that the altitudes (z-coordinate) matters.

        lower_depth : float
            Define the lower edge for the inital sampling (default: 400 g/cm2).

        upper_depth : float
            Define the upper edge for the inital sampling (default: 800 g/cm2).

        bin_size : float
            Define the step size pf the inital sampling (default: 100 g/cm2).
            The refined sampling around the peak region is / 10 this value.

        return_profile : bool 
            If true return the sampled profile in addition to the Gauss parameter (default: False).

        Returns
        -------
        
        If return_profile is True
            
            depths_corse : np.array
                Depths along shower axis coarsely sampled
            
            depths_fine : np.array
                Depths along shower axis finely sampled (used in fitting)
            
            signals_corese : np.array
                Beamformed signals along shower axis coarsely sampled
            
            signals_fine : np.array
                Beamformed signals along shower axis finely sampled (used in fitting)
                    
            popt : list
                List of fitted Gauss parameters (amplitude, position, width)

        If return_profile is False:

            popt : list
                List of fitted Gauss parameters (amplitude, position, width)
        
        """

        depths = np.arange(lower_depth, upper_depth, bin_size)
        signals_tmp = self.sample_longitudinal_profile(
            traces, times, station_positions, shower_axis, core, depths=depths)

        # if max signal is at the upper edge add points there
        if np.argmax(signals_tmp) == len(depths) - 1:
            while True:
                depth_add = np.amax(depths) + bin_size
                signal_add = self.sample_longitudinal_profile(
                    traces, times, station_positions, shower_axis, core, depths=[depth_add])
                depths = np.append(depths, depth_add)
                signals_tmp = np.append(signals_tmp, signal_add)

                if not np.argmax(signals_tmp) == len(depths) - 1 or depth_add > 2000:
                    break

        # if max signal is at the lower edge add points there
        elif np.argmax(signals_tmp) == 0:
            while True:
                depth_add = np.amin(depths) - bin_size
                signal_add = self.sample_longitudinal_profile(
                    traces, times, station_positions, shower_axis, core, depths=[depth_add])
                depths = np.append(depth_add, depths)
                signals_tmp = np.append(signal_add, signals_tmp)

                if not np.argmax(signals_tmp) == 0 or depth_add <= 0:
                    break

        idx_max = np.argmax(signals_tmp)
        depths_final = np.linspace(
            depths[idx_max - 1], depths[idx_max + 1], 20)  # 10 g/cm2 bins
        signals_final = self.sample_longitudinal_profile(
            traces, times, station_positions, shower_axis, core, depths=depths_final)

        def normal(x, A, x0, sigma):
            """ Gauss curve """
            return A / np.sqrt(2 * np.pi * sigma ** 2) \
                * np.exp(-1 / 2 * ((x - x0) / sigma) ** 2)

        popt, pkov = curve_fit(normal, depths_final, signals_final, p0=[np.amax(
            signals_final), depths_final[np.argmax(signals_final)], 100], maxfev=1000)

        if return_profile:
            return depths, depths_final, signals_tmp, signals_final, popt

        return popt


    def update_atmospheric_model_and_refractivity_table(self, shower):
        """ 
        Updates model of the atmosphere and tabulated, integrated refractive index according to shower properties.

        Parameters
        ----------

        shower : BaseShower
        """

        if self._at is None:
            self._at = models.Atmosphere(shower[shp.atmospheric_model])
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=True)

        elif self._at.model != shower[shp.atmospheric_model]:
            self._at = models.Atmosphere(shower[shp.atmospheric_model])
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=True)
        
        elif self._tab._refractivity_at_sea_level != shower[shp.refractive_index_at_ground] - 1:
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=True)
        
        else:
            pass


    @register_run()
    def run(self, evt, det, use_MC_geometry=True, use_MC_pulses=True):
        """ 
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------

        evt : Event
            Event to run the module on.
        
        det : Detector
            Detector description

        use_MC_geometry : bool
            if true, take geometry from sim_shower. Results will than also be stored in sim_shower

        use_MC_pulses : bool
            if true, take electric field trace from sim_station
        """

        # TODO: Mimic imperfect time syncronasation by adding a time jitter here? 

        # TODO: Make it more flexible. Choose shower from which the geometry and atmospheric properties are taken.
        # Also store xrit in this shower.
        if use_MC_geometry:
            shower = evt.get_first_sim_shower()
        else:
            shower = evt.get_first_shower()
        
        self.update_atmospheric_model_and_refractivity_table(shower)
        core, shower_axis, cs = get_geometry_and_transformation(shower)

        traces_vxB, times, pos = get_station_data(
            evt, det, cs, use_MC_pulses, n_sampling=256)

        if self._debug:
            depths, depths_final, signals_tmp, signals_final, rit_parameters = \
                self.reconstruct_interferometric_depth(
                    traces_vxB, times, pos, shower_axis, core, return_profile=True)

            fig, ax = plt.subplots(1)
            ax.plot(depths, signals_tmp, "o")
            ax.plot(depths_final, signals_final, "o")
            ax.plot(depths_final, interferometry.gaus(
                depths_final, *rit_parameters[:-1]))
            ax.axvline(rit_parameters[1])
            plt.show()
        else:
            rit_parameters = self.reconstruct_interferometric_depth(
                traces_vxB, times, pos, shower_axis, core)

        xrit = rit_parameters[1]
        shower.set_parameter(shp.interferometric_shower_maximum, xrit * units.g / units.cm2)

        #TODO: Add calibration Xmax(Xrit, theta, ...)?
        
        # for plotting
        self._data["xrit"].append(xrit)
        self._data["xmax"].append(shower[shp.shower_maximum] / (units.g / units.cm2))
        self._data["zenith"].append(shower[shp.zenith])

    def end(self):
        """
        Plot reconstructed depth vs true depth of shower maximum (Xmax).
        """

        fig, ax = plt.subplots(1)
        sct = ax.scatter(
            self._data["xmax"], self._data["xrit"], s=200, c=np.rad2deg(self._data["zenith"]))
        cbi = plt.colorbar(sct, pad=0.02)
        cbi.set_label("zenith angle / deg")
        ax.set_xlabel(r"$X_\mathrm{max}$ / g$\,$cm$^{-2}$")
        ax.set_ylabel(r"$X_\mathrm{RIT}$ / g$\,$cm$^{-2}$")
        fig.tight_layout()
        plt.show()
        pass


class efieldInterferometricAxisReco(efieldInterferometricDepthReco):
    """
    Class to reconstruct the shower axis with beamforming.     
    """
    def __init__(self):
        super().__init__()


    def find_maximum_in_plane(self, xs, ys, p_axis, station_positions, traces, times, cs):
        """
        Sample interferometric signals in 2-d plane (vxB-vxvxB) perpendicular to a given axis on a rectangular/quadratic grid.
        The orientation of the plane is defined by the radiotools.coordinatesytem.cstrafo argument. 

        Parameters
        ----------

        xs : array
            x-coordinates defining the sampling positions.

        ys : array
            y-coordinates defining the sampling positions.

        p_axis : array(3,)
            Origin of the 2-d plane along the axis.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        cs : radiotools.coordinatesytem.cstrafo

        Returns
        -------

        idx : int
            Index of the entry with the largest signal (np.argmax(signals))

        signals : array(len(xs), len(ys))
            Interferometric signal

        """
        signals = np.zeros((len(xs), len(ys)))
        tstep = times[0, 1] - times[0, 0]

        for xdx, x in enumerate(xs):
            for ydx, y in enumerate(ys):
                p = p_axis + cs.transform_from_vxB_vxvxB(np.array([x, y, 0]))

                sum_trace = interferometry.interfere_traces_interpolation(
                    p, station_positions, traces, times, tab=self._tab)
                signal = interferometry.get_signal(
                    sum_trace, tstep, kind=self._signal_kind)
                signals[xdx, ydx] = signal

        idx = np.argmax(signals)
        return idx, signals


    def sample_lateral_cross_section(
            self,
            traces, times, station_positions,
            shower_axis_inital, core, depth, cs,
            shower_axis_mc, core_mc,
            relative=False, initial_grid_spacing=100, centered_around_truth=True,
            cross_section_size=1000, deg_resolution=np.deg2rad(0.005)):
        """
        Sampling the "cross section", i.e., 2d-lateral distribution of the beam formed signal for a slice in the atmosphere. 
        It is looking for the maximum in the lateral distribution with an (stupid) iterative grid search. 
        
        Returns the position and the strenght of the maximum signal.

        Parameters
        ----------

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis_inital : array(3,)
            Axis/direction which is used as initial guess for the true shower axis.
            Around this axis the interferometric signals are sample on 2-d planes.

        core : array(3,)
            Shower core which is used as initial guess. Keep in mind that the altitudes (z-coordinate) matters.

        depth : np.array

        cs : radiotools.coordinatesytem.cstrafo

        shower_axis_mc : np.array(3,)
        
        core_mc : np.array(3,)

        relative : bool
            False
            
        initial_grid_spacing : int
            100
        
        centered_around_truth : bool
            True

        cross_section_size : int
            1000
            
        deg_resolution : float
            np.deg2rad(0.005))
        
        Returns
        -------
        
        point_found : np.array(3,)
            Position of the found maximum

        weight : float
            Amplitude/Strengt of the maximum
        
        """

        zenith_inital, _ = hp.cartesian_to_spherical(
            *np.split(shower_axis_inital, 3))
        dist = self._at.get_distance_xmax_geometric(
            zenith_inital, depth, observation_level=core[-1])
        p_axis = shower_axis_inital * dist + core

        # we use the true core to make sure that it is within the inital search gri
        mc_at_plane = interferometry.get_intersection_between_line_and_plane(
            shower_axis_inital, p_axis, shower_axis_mc, core_mc)
        mc_vB = cs.transform_to_vxB_vxvxB(mc_at_plane, core=p_axis)
        # mc_points.append(mc_at_plane + core)

        dr_ref_target = np.tan(deg_resolution) * dist
        if relative:
            # ensure that true xmax is within the search grid but not on a grid point
            xlims = (
                np.array([-1.2, 1.2]) + np.random.uniform(0, .1, 2)) * np.abs(mc_vB[0])
            ylims = (
                np.array([-1.2, 1.2]) + np.random.uniform(0, .1, 2)) * np.abs(mc_vB[1])
            xs = np.linspace(*xlims, 20)
            ys = np.linspace(*ylims, 20)

        else:
            if centered_around_truth:
                xs = np.arange(mc_vB[0] - cross_section_size / 2 -
                            np.random.uniform(0, initial_grid_spacing, 1),
                            mc_vB[0] + cross_section_size / 2, initial_grid_spacing)
                ys = np.arange(mc_vB[1] - cross_section_size / 2 -
                            np.random.uniform(0, initial_grid_spacing, 1),
                            mc_vB[1] + cross_section_size / 2, initial_grid_spacing)

            else:
                # quadratic grid with true intersection contained
                max_dist = np.amax(
                    [np.abs(mc_vB[0]), np.abs(mc_vB[1]), 3 * initial_grid_spacing]) + initial_grid_spacing
                xlims = np.array([-max_dist, max_dist]) + \
                    np.random.uniform(-0.1 * initial_grid_spacing,
                                    0.1 * initial_grid_spacing, 2)
                ylims = np.array([-max_dist, max_dist]) + \
                    np.random.uniform(-0.1 * initial_grid_spacing,
                                    0.1 * initial_grid_spacing, 2)
                xs = np.arange(xlims[0], xlims[1] +
                            initial_grid_spacing, initial_grid_spacing)
                ys = np.arange(ylims[0], ylims[1] +
                            initial_grid_spacing, initial_grid_spacing)

        iloop = 0
        while True:

            idx, signals = self.find_maximum_in_plane(
                xs, ys, p_axis, station_positions, traces, times, cs=cs)

            if self._debug:
                # from AERAutilities import pyplots, pyplots_utils # for mpl rc
                plot_lateral_cross_section(
                    xs, ys, signals, mc_vB, title=r"%.1f$\,$g$\,$cm$^{-2}$" % depth)

            iloop += 1
            dr = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
            if iloop == 10 or dr < dr_ref_target:
                break

            # maximum
            x_max = xs[int(idx // len(ys))]
            y_max = ys[int(idx % len(ys))]

            # update range / grid
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            if iloop >= 2:
                dx /= 2
                dy /= 2
            xs = np.linspace(x_max - dx, x_max + dx, 5)
            ys = np.linspace(y_max - dy, y_max + dy, 5)

        weight = np.amax(signals)

        xfound = xs[int(idx // len(ys))]
        yfound = ys[int(idx % len(ys))]
        point_found = p_axis + \
            cs.transform_from_vxB_vxvxB(np.array([xfound, yfound, 0]))

        return point_found, weight


    def reconstruct_shower_axis(
            self, 
            traces, times, station_positions,
            shower_axis, core,
            magnetic_field_vector,
            is_mc=True,
            initial_grid_spacing=60,
            cross_section_size=1000):
        """
        Run interferometric reconstruction of the shower axis. Find the maxima of the interferometric signals 
        within 2-d plane (slices) along a given axis (initial guess). Through those maxima (their position in the
        atmosphere) a straight line is fitted to reconstruct the shower axis.

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis_inital : array(3,)
            Axis/direction which is used as initial guess for the true shower axis.
            Around this axis the interferometric signals are sample on 2-d planes.

        core : array(3,)
            Shower core which is used as initial guess. Keep in mind that the altitudes (z-coordinate) matters.

        magnetic_field_vector : array(3,)
            Magnetic field vector of the site you are using.

        is_mc : bool
            If true, interprete the provided shower axis as truth and add some gaussian smearing to optain an
            inperfect initial guess for the shower axis (Default: True). 
        
        initial_grid_spacing : double
            Spacing of your grid points in meters (Default: 60m)

        cross_section_size : double
            Side length on the 2-d planes (slice) along which the maximum around the initial axis is sampled in meters
            (Default: 1000m).

        """
        

        if is_mc:
            zenith_mc, azimuth_mc = hp.cartesian_to_spherical(*shower_axis)

            # smeare mc axis.
            zenith_inital = zenith_mc + np.deg2rad(np.random.normal(0, 0.5))
            azimuth_inital = azimuth_mc + np.deg2rad(np.random.normal(0, 0.5))
            shower_axis_inital = hp.spherical_to_cartesian(
                zenith=zenith_inital, azimuth=azimuth_inital)

        else:
            shower_axis_inital = shower_axis 
            core_inital = core 
            zenith_inital, azimuth_inital = hp.cartesian_to_spherical(
                *shower_axis)

            shower_axis, core = None, None

            raise ValueError("is_mc=False is not yet properly implemented!")

        cs = coordinatesystems.cstrafo(
            zenith_inital, azimuth_inital, magnetic_field_vector=magnetic_field_vector)
        
        if is_mc:
            core_inital = cs.transform_from_vxB_vxvxB_2D(
                np.array([np.random.normal(0, 100), np.random.normal(0, 100), 0]), core)

        depths = [500, 600, 700, 800, 900, 1000]
        deg_resolution = np.deg2rad(0.005)

        found_points = []
        weights = []

        relative = False
        centered_around_truth = True

        def sample_lateral_cross_section_placeholder(dep):
            """ 
            Run sample_lateral_cross_section for a particular depth.
            
            Parameters
            ----------

            dep : double
                Depth along the axis at which the cross section is sampled in g/cm2.
            
            """
            return self.sample_lateral_cross_section(
                traces, times, station_positions,
                shower_axis_inital, core_inital, dep, cs,
                shower_axis, core,
                relative=relative, initial_grid_spacing=initial_grid_spacing,
                centered_around_truth=centered_around_truth,
                cross_section_size=cross_section_size, deg_resolution=deg_resolution)

        for depth in depths:
            found_point, weight = sample_lateral_cross_section_placeholder(depth)

            found_points.append(found_point)
            weights.append(weight)

        if 0:
            while True:
                if np.argmax(weights) != 0:
                    break

                new_depth = depths[0] - (depths[1] - depths[0])
                print("extend to", new_depth)
                found_point, weight = sample_lateral_cross_section_placeholder(
                    new_depth)

                depths = [new_depth] + depths
                found_points = [found_point] + found_points
                weights = [weight] + weights

            while True:
                if np.argmax(weights) != len(weights):
                    break

                new_depth = depths[-1] + (depths[1] - depths[0])
                print("extend to", new_depth)
                found_point, weight = sample_lateral_cross_section_placeholder(
                    new_depth)

                depths.append(new_depth)
                found_points.append(found_point)
                weights.append(weight)

        found_points = np.array(found_points)
        weights = np.array(weights)

        popt, pcov = curve_fit(interferometry.fit_axis, found_points[:, -1], found_points.flatten(),
                            sigma=np.amax(weights) / np.repeat(weights, 3), p0=[zenith_inital, azimuth_inital, 0, 0])
        direction_rec = hp.spherical_to_cartesian(*popt[:2])
        core_rec = interferometry.fit_axis(np.array([core[-1]]), *popt)

        return direction_rec, core_rec

    @register_run()
    def run(self, evt, det, use_MC_geometry=True, use_MC_pulses=True):
        """ 
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------

        evt : Event
            Event to run the module on.
        
        det : Detector
            Detector description

        use_MC_geometry : bool
            if true, take geometry from sim_shower. Results will than also be stored in sim_shower

        use_MC_pulses : bool
            if true, take electric field trace from sim_station
        """

        # TODO: Mimic imperfect time syncronasation by adding a time jitter here?

        # TODO: Make it more flexible. Choose shower from which the geometry and atmospheric properties are taken.
        # Also store xrit in this shower.
        if use_MC_geometry:
            shower = evt.get_first_sim_shower()
        else:
            shower = evt.get_first_shower()

        self.update_atmospheric_model_and_refractivity_table(shower)
        core, shower_axis, cs = get_geometry_and_transformation(shower)

        traces_vxB, times, pos = get_station_data(
            evt, det, cs, use_MC_pulses, n_sampling=256)

        direction_rec, core_rec = self.reconstruct_shower_axis(
            traces_vxB, times, pos, shower_axis, core, is_mc=True, magnetic_field_vector=shower[shp.magnetic_field_vector])

        shower.set_parameter(shp.interferometric_shower_axis, direction_rec)
        shower.set_parameter(shp.interferometric_core, core_rec)

    def end(self):
        pass


def get_geometry_and_transformation(shower):
    """ 
    Returns core (def. as intersection between shower axis and observation plane,
    shower axis, and radiotools.coordinatesytem for given shower.

    Parameters
    ----------

    shower : BaseShower
    """

    observation_level = shower[shp.observation_level]
    core = shower[shp.core]

    if core[-1] != observation_level:
        sys.exit("Code down the road expect that to be equal!")

    zenith = shower[shp.zenith]
    azimuth = shower[shp.azimuth]
    magnetic_field_vector = shower[shp.magnetic_field_vector]

    shower_axis = hp.spherical_to_cartesian(zenith, azimuth)

    cs = coordinatesystems.cstrafo(
        zenith, azimuth, magnetic_field_vector=magnetic_field_vector)

    return core, shower_axis, cs


def get_station_data(evt, det, cs, use_MC_pulses, n_sampling=None):
    """ 
    Returns station data in a proper format
    
    Parameters
    ----------

    evt : Event

    det : Detector

    cs : radiotools.coordinatesystems.cstrafo

    use_MC_pulses : bool
        if true take electric field trace from sim_station

    n_sampling : int
        if not None clip trace with n_sampling // 2 around np.argmax(np.abs(trace))

    Returns
    -------
    
    traces_vxB : np.array
        The electric field traces in the vxB polarisation (takes first electric field stored in a station) for all stations/observers.

    times : mp.array  
        The electric field traces time series for all stations/observers.
        
    pos : np.array
        Positions for all stations/observers. 
    """

    traces_vxB = []
    times = []
    pos = []

    for station in evt.get_stations():

        if use_MC_pulses:
            station = station.get_sim_station()

        for electric_field in station.get_electric_fields():
            traces = cs.transform_to_vxB_vxvxB(
                cs.transform_from_onsky_to_ground(electric_field.get_trace()))
            trace_vxB = traces[0]
            time = copy.copy(electric_field.get_times())

            if n_sampling is not None:
                hw = n_sampling // 2
                m = np.argmax(np.abs(trace_vxB))

                if m < hw: 
                    m = hw
                if m > len(trace_vxB) - hw:
                    m = len(trace_vxB) - hw
                
                trace_vxB = trace_vxB[m-hw:m+hw]
                time = time[m-hw:m+hw]

            traces_vxB.append(trace_vxB)
            times.append(time)
            break  # just take the first efield. TODO: Improve this

        pos.append(det.get_absolute_position(station.get_id()))
    
    traces_vxB = np.array(traces_vxB)
    times = np.array(times)
    pos = np.array(pos)

    return traces_vxB, times, pos


def plot_lateral_cross_section(xs, ys, signals, mc_pos=None, fname=None, title=None):
    """ 
    Plot the lateral distribution of the beamformed singal (in the vxB, vxvxB directions).

    Parameters
    ----------

    xs : np.array
        Positions on x-axis (vxB) at which the signal is sampled (on a 2d grid)

    ys : np.array
        Positions on y-axis (vxvxB) at which the signal is sampled (on a 2d grid)   

    signals : np.array
        Signals sampled on the 2d grid defined by xs and ys.

    mc_pos : np.array(2,)
        Intersection of the (MC-)axis with the "slice" of the lateral distribution plotted.

    fname : str
        Name of the figure. If given the figure is saved, if fname is None the fiture is shown.

    title : str
        Title of the figure (Default: None)
    """

    yy, xx = np.meshgrid(ys, xs)

    fig, ax = plt.subplots(1)
    pcm = ax.pcolormesh(xx, yy, signals, shading='gouraud')
    ax.plot(xx, yy, "ko", markersize=3)
    cbi = plt.colorbar(pcm, pad=0.02)
    cbi.set_label(r"$f_{B_{j}}$ / eV$\,$m$^{-2}$")

    idx = np.argmax(signals)
    xfound = xs[int(idx // len(ys))]
    yfound = ys[int(idx % len(ys))]

    ax.plot(xfound, yfound, ls=None, marker="o", color="C1",
            markersize=10, label="found maximum")
    if mc_pos is not None:
        ax.plot(mc_pos[0], mc_pos[1], ls=None, marker="*", color="r",
                markersize=10, label=r"intersection with $\hat{a}_\mathrm{MC}$")

    ax.legend()
    ax.set_ylabel(
        r"$\vec{v} \times \vec{v} \times \vec{B}$ / m")
    ax.set_xlabel(r"$\vec{v} \times \vec{B}$ / m")
    if title is not None:
        ax.set_title(title)  # r"slant depth = %d g / cm$^2$" % depth)

    plt.tight_layout()
    if fname is not None:
        # "rit_xy_%d_%d_%s.png" % (depth, iloop, args.label))
        plt.savefig(fname)
    else:
        plt.show()
