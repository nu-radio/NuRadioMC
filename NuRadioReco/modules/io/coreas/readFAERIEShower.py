import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station

from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
from radiotools import helper as hp

from matplotlib import pyplot as plt
import numpy as np
import time
import h5py
import os

import logging
logger = logging.getLogger('readFAERIEShower')

class readFAERIEShower:

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = None
        self.__current_input_file = None
        self.__ascending_run_and_event_number = None

    def begin(self, input_files, logger_level=logging.NOTSET, set_ascending_run_and_event_number=False, skip_raytracing=False):
        """
        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        logger_level: string or logging variable
            Set verbosity level for logger (default: logging.NOTSET)
        set_ascending_run_and_event_number: bool
            If set to True the run number and event id is set to
            self.__ascending_run_and_event_number instead of beeing taken
            from the simulation file. The value is increases monoton.
            This can be used to avoid ambiguities values (default: False)
        """
        self.__input_files = input_files
        self.__current_input_file = 0
        logger.setLevel(logger_level)

        self.__ascending_run_and_event_number = 1 if set_ascending_run_and_event_number else 0
        self._skip_raytracing = skip_raytracing

        if not self._skip_raytracing:
            prop = propagation.get_propagation_module("analytic")
            ref_index_model = 'greenland_simple'
            ice = medium.get_ice_model(ref_index_model)

            # This function creates a ray tracing instance refracted index, attenuation model,
            # number of frequencies # used for integrating the attenuation and interpolate afterwards,
            # and the number of allowed reflections.

            self._rays = prop(ice, "GL1",
                        n_frequencies_integration=25,
                        n_reflections=0, use_cpp=False)  # use_ccp has to be False to have air-ice tracing

    @register_run()
    def run(self, depth=None):
        """
        Reads in a CoREAS file and returns one event containing all simulated observer positions as stations.
        """
        while self.__current_input_file < len(self.__input_files):
            t = time.time()
            t_per_event = time.time()

            filesize = os.path.getsize(
                self.__input_files[self.__current_input_file])
            if filesize < 18456 * 2:  # based on the observation that a file with such a small filesize is corrupt
                logger.warning(
                    "file {} seems to be corrupt, skipping to next file".format(
                        self.__input_files[self.__current_input_file]
                    )
                )
                self.__current_input_file += 1
                continue

            logger.info('Reading %s ...' %
                        self.__input_files[self.__current_input_file])

            corsika = h5py.File(
                self.__input_files[self.__current_input_file], "r")
            logger.info("using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                self.__input_files[self.__current_input_file], corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                corsika['inputs'].attrs["THETAP"][0]))

            f_coreas = corsika["CoREAS"]
            if self.__ascending_run_and_event_number:
                evt = NuRadioReco.framework.event.Event(
                    self.__ascending_run_and_event_number, self.__ascending_run_and_event_number)
                self.__ascending_run_and_event_number += 1
            else:
                evt = NuRadioReco.framework.event.Event(
                    corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])

            evt.set_event_time(corsika['CoREAS'].attrs['GPSSecs'], format="gps")

            # create sim shower, no core is set since no external detector description is given
            sim_shower = coreas.create_sim_shower_from_hdf5(corsika)
            evt.add_sim_shower(sim_shower)
            shower_core = sim_shower.get_parameter(shp.core)

            debug_plot_ice = False
            debug_plot_air = False

            # Position relative to shower core (at z=0)
            dmax = sim_shower.get_parameter(shp.distance_shower_maximum_geometric)
            xmax_above_ice = dmax > 0
            if dmax <= 0:
                logger.warning("`distance_shower_maximum_geometric` is not positive. Either the calculation within CoREAS failed and/or the maximum is below the obslevel.")

            # Define position of in-ice and in-air emission within the NuRadio CS where the ice is at z=0 (as is the core).
            position_of_xmax = sim_shower.get_axis() * sim_shower.get_parameter(shp.distance_shower_maximum_geometric)
            shower_inice_position = np.array([0, 0, -2]) * units.m

            # get (unique) observer names form either or both observer groups
            observer_names = np.unique([list(f_coreas[key].keys()) for key in ["observers", "observers_geant"] if key in f_coreas])
            assert len(observer_names) > 0, "No observer found in hdf5 file"

            has_in_air_emission = "observers" in f_coreas
            has_in_ice_emission = "observers_geant" in f_coreas

            if not has_in_air_emission:
                logger.warning("No in-air emission found in hdf5 file")

            if not has_in_ice_emission:
                logger.warning("No in-ice emission found in hdf5 file")

            # Intentionally an empty sim station is created (and not created via coreas.create_sim_station)
            sim_station = NuRadioReco.framework.sim_station.SimStation(0)  # set sim station id to 0
            sim_station.set_is_neutrino() # HACK: Since the sim. efields are always at the exact positions as the antenna(channels).

            station = NuRadioReco.framework.station.Station(0)
            station.set_sim_station(sim_station)
            evt.set_station(station)

            # add simulated pulses as sim station
            for idx, observer_name in enumerate(observer_names):
                # if both in-air and in-ice emission are present, we assure that the observer positions are the same
                if has_in_air_emission and has_in_ice_emission:
                    if not np.all(f_coreas["observers"][observer_name].attrs['position'] ==
                                  f_coreas["observers_geant"][observer_name].attrs['position']):
                        logger.error("Try to read both, in-air and in-ice emission, but observer positions do not match")
                        raise ValueError("Try to read both, in-air and in-ice emission, but observer positions do not match")

                # get observer position
                if has_in_air_emission:
                    position = f_coreas["observers"][observer_name].attrs['position']
                else:
                    position = f_coreas["observers_geant"][observer_name].attrs['position']

                position = coreas.convert_obs_positions_to_nuradio_on_ground(
                    position, sim_shower.get_parameter(shp.zenith), sim_shower.get_parameter(shp.azimuth),
                    sim_shower.get_parameter(shp.magnetic_field_vector)
                )

                if position[2] < 100:
                    # logger.warning("Observer position is < 100m. This is probably a unit issue. Multiply position by 100 (converting back to meter)")
                    position *= 100

                if depth is not None:
                    antenna_depth = shower_core[2] - position[2]
                    if depth < 0:
                        raise ValueError("Depth is positivly defined, you can not select antennas which are above the ice.")

                    if antenna_depth != depth:
                        continue

                # convert antenna position from z above sea level to z above ice (i.e. ice surface is at z=0)
                antenna_position_in_ice = position - shower_core

                if has_in_ice_emission:
                    # No point in calculating the ray tracing solution if the electric field is zero (= ray tracing in faerie did not give result)
                    if np.any(f_coreas["observers_geant"][observer_name][:, 1:]):
                        if not self._skip_raytracing:
                            # find ray tracing solution for in-ice signal

                            self._rays.set_start_and_end_point(shower_inice_position, antenna_position_in_ice)
                            self._rays.find_solutions()
                            has_in_ice_solution = self._rays.get_number_of_solutions() > 0

                            if has_in_ice_solution:
                                if debug_plot_ice:
                                    debug_plot(self._rays)

                                zenith, azimuth, ray_tracing_id = get_signal_direction(self._rays)

                                add_efield_to_sim_station(
                                    sim_station, f_coreas["observers_geant"][observer_name], antenna_position_in_ice,
                                    zenith, azimuth, sim_shower.get_parameter(shp.magnetic_field_vector),
                                    channel_id=idx, ray_tracing_id=ray_tracing_id, shower_id=1)
                        else:
                            zenith = sim_shower.get_parameter(shp.zenith)
                            azimuth = sim_shower.get_parameter(shp.azimuth)
                            add_efield_to_sim_station(
                                sim_station, f_coreas["observers_geant"][observer_name], antenna_position_in_ice,
                                zenith, azimuth, sim_shower.get_parameter(shp.magnetic_field_vector),
                                channel_id=idx, ray_tracing_id=0, shower_id=1)

                    if has_in_air_emission:

                        if antenna_position_in_ice[2] < 0 and not self._skip_raytracing:
                            self._rays.set_start_and_end_point(position_of_xmax, antenna_position_in_ice)
                            self._rays.find_solutions()
                            has_in_air_solution = self._rays.get_number_of_solutions() > 0

                            if not has_in_air_solution and xmax_above_ice:
                                logger.warning("No ray tracing solution found for in-air signal. Skipping this entire antenna for now (including in-ice signal) ..")

                            if has_in_air_solution:
                                if debug_plot_air:
                                    debug_plot(self._rays)

                                zenith, azimuth, ray_tracing_id = get_signal_direction(self._rays)

                        else:
                            has_in_air_solution = True
                            zenith = sim_shower.get_parameter(shp.zenith)
                            azimuth = sim_shower.get_parameter(shp.azimuth)
                            ray_tracing_id = 0

                        if has_in_air_solution:
                            add_efield_to_sim_station(
                                sim_station, f_coreas["observers"][observer_name], antenna_position_in_ice,
                                zenith, azimuth, sim_shower.get_parameter(shp.magnetic_field_vector),
                                channel_id=idx, ray_tracing_id=ray_tracing_id, shower_id=0)

            self.__current_input_file += 1
            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t
            yield evt

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(
            timedelta(seconds=self.__t_event_structure)))
        logger.info("per event {}".format(
            timedelta(seconds=self.__t_per_event)))
        return dt


def debug_plot(rays):
    fig, ax = plt.subplots(1, 1)
    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = propagation.solution_types[solution_int]

        path = rays.get_path(i_solution)
        # We can calculate the azimuthal angle phi to rotate the
        # 3D path into the 2D plane of the points. This is only
        # necessary if we are not working in the y=0 plane
        launch_vector = rays.get_launch_vector(i_solution)
        phi = np.arctan(launch_vector[1]/launch_vector[0])
        ax.plot(
            path[:, 0] / np.cos(phi), path[:, 2],
            label=solution_type
        )

    ax.legend()
    ax.set_xlabel('xy [m]')
    ax.set_ylabel('z [m]')
    plt.show()


def get_signal_direction(rays):
    # We can also get the 3D receiving vector at the observer position, for instance
    receive_vector = rays.get_receive_vector(0)  # HACK: For the moment we only take the first solution

    zenith = hp.get_angle(receive_vector, np.array([0, 0, 1]))
    azimuth = np.arctan2(receive_vector[1], receive_vector[0])

    return zenith, azimuth, 0

def add_efield_to_sim_station(sim_station, observer, position, zenith, azimuth, magnetic_field_vector, channel_id, ray_tracing_id, shower_id):
    efields_traces, efield_times = coreas.convert_obs_to_nuradio_efield(
        observer, zenith=zenith, azimuth=azimuth, magnetic_field_vector=magnetic_field_vector)

    sampling_rate = 1 / (efield_times[1] - efield_times[0])
    coreas.add_electric_field_to_sim_station(
        sim_station, [channel_id],
        efields_traces, efield_times[0],
        zenith, azimuth,
        sampling_rate,
        efield_position=position,
        shower_id=shower_id, ray_tracing_id=ray_tracing_id,
    )


class FAERIEDetector():

    def __init__(self):
        pass

    def set_event(self, evt):
        self.event = evt

    def get_station_ids(self):
        """ Returns all station ids """
        return self.event.get_station_ids()

    def get_channel_ids(self, station_id):
        """ Returns all channel ids of one station (sorted) """
        station = self.event.get_station(station_id)
        return np.unique([efield.get_channel_ids() for efield in station.get_sim_station().get_electric_fields()])

    def get_relative_position(self, station_id, channel_id):
        """ Return the relative position of the antenna in the station (relative to station position) """
        sim_station = self.event.get_station(station_id).get_sim_station()

        efield_position = np.unique([efield.get_position() for efield in sim_station.get_electric_fields_for_channels([channel_id])], axis=0)
        assert len(efield_position) == 1, "There should be only one unique position for each channel"

        return efield_position[0]

    ### Constant Returns ###

    def get_absolute_position(self, station_id):
        """ Return the station position """
        return np.array([0, 0, 0])

    def get_cable_delay(self, station_id=None, channel_id=None):
        return 0

    def get_site(self, station_id=None):
        return "summit"

    def get_antenna_model(self, station_id=None, channel_id=None, zenith_antenna=None):
        """ Returns the antenna model """
        return "RNOG_vpol_4inch_center_n1.73"

    def get_antenna_orientation(self, station_id=None, channel_id=None):
        """ Returns the channel's 4 orientation angles in rad """
        return np.deg2rad([0, 0, 90, 90])

    def get_site_coordinates(self, station_id=None):
        """ Returns latitude and longitude of SKA in degrees """
        return 72.57, -38.46

    def get_number_of_samples(self, station_id=None, channel_id=None):
        return 2048

    def get_sampling_frequency(self, station_id=None, channel_id=None):
        return 3.2 * units.GHz
