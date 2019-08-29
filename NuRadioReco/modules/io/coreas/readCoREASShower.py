import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.shower
from radiotools import helper as hp
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtP
from NuRadioReco.framework.parameters import showerParameters as shP
from NuRadioReco.framework.parameters import array_stationParameters as astP
import numpy as np
import logging
import time
import re
logger = logging.getLogger('readCoREASShower')


class readCoREASShower:

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0

    def begin(self, input_files):
        """
        begin method

        initialize readCoREASShower module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        """
        self.__input_files = input_files
        self.__current_input_file = 0

    def run(self):
        """
        read in a full CoREAS simulation

        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()

            # filesize = os.path.getsize(self.__input_files[self.__current_input_file])
            # if(filesize < 18456 * 2):
            #     logger.warning("file {} seems to be corrupt, skipping to next file".format(
            #            self.__input_files[self.__current_input_file]))
            #     self.__current_input_file += 1
            #     continue

            corsika = h5py.File(self.__input_files[self.__current_input_file], "r")
            logger.info("using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                self.__input_files[self.__current_input_file], corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                corsika['inputs'].attrs["THETAP"][0]))

            f_coreas = corsika["CoREAS"]

            evt_run_number = f_coreas.attrs["RunNumber"]
            evt = NuRadioReco.framework.event.Event(evt_run_number, 1)  # create empty event
            evt.__event_time = f_coreas.attrs["GPSSecs"]

            sim_shower = NuRadioReco.framework.shower.Shower(evtP.sim_shower)

            try:
                sim_shower.set_parameter(shP.primary_particle, f_coreas.attrs["PrimaryParticleType"])
            except KeyError:
                logger.info("no primary:", self.__input_files[self.__current_input_file])

            sim_shower.set_parameter(shP.refractive_index_at_ground, f_coreas.attrs["GroundLevelRefractiveIndex"])
            sim_shower.set_parameter(shP.magnetic_field_rotation, f_coreas.attrs["RotationAngleForMagfieldDeclination"] * units.degree)
            sim_shower.set_parameter(shP.distance_shower_maximum_geometric, f_coreas.attrs["DistanceOfShowerMaximum"] * units.cm)

            if "inputs" in corsika:
                f_input = corsika["inputs"]

                sim_shower.set_parameter(shP.observation_level, f_input.attrs["OBSLEV"] * units.cm)
                try:
                    evt.set_parameter(shP.atmospheric_model, f_input.attrs["ATMOD"])
                except KeyError:
                    pass

            # at the moment just reads highlevel file
            try:
                f_highlevel = corsika["highlevel"]
            except KeyError:
                logger.KeyError("No highlevel group")
                raise NotImplementedError

            evt.set_parameter(evtP.magnetic_field_inclination, f_highlevel.attrs["magnetic_field_inclination"] * units.degree)
            evt.set_parameter(evtP.magnetic_field_declination, f_highlevel.attrs["magnetic_field_declination"] * units.degree)
            evt.set_parameter(evtP.magnetic_field_strength, f_highlevel.attrs["magnetic_field_strength"])

            # shower
            sim_shower.set_parameter(shP.azimuth, hp.get_normalized_angle(f_highlevel.attrs["azimuth"]))
            sim_shower.set_parameter(shP.zenith, f_highlevel.attrs["zenith"])

            sim_shower.set_parameter(shP.energy, f_highlevel.attrs["energy"] * units.eV)
            sim_shower.set_parameter(shP.electromagnetic_energy, f_highlevel.attrs["Eem"] * units.eV)
            # sim_shower.set_parameter("invisible_energy", f_highlevel.attrs["Einv"])

            # sim_shower.set_parameter("hillas", np.array(f_highlevel.attrs["gaisser_hillas_dEdX"]))
            sim_shower.set_parameter(shP.shower_maximum, f_highlevel.attrs["gaisser_hillas_dEdX"][2] * units.g / units.cm2)

            # choose first one, rest is read latter
            planes = list(f_highlevel.keys())
            obs_plane = f_highlevel[planes[0]]
            evt.set_parameter("observation_plane", planes[0])

            sim_shower.set_parameter(shP.core, np.array(obs_plane["core"]))

            # in case simulation is not a star shaped
            try:
                # evt.set_parameter("radiation_energy_1D", obs_plane.attrs["radiation_energy_1D"] * units.eV)
                evt.set_parameter(shP.radiation_energy, obs_plane.attrs["radiation_energy"] * units.eV)
            except KeyError:
                pass

            # later parse station id from antenna name if possible
            for idx, name in enumerate(np.array(obs_plane["antenna_names"])):

                station_id = antenna_id(name, idx)
                station = NuRadioReco.framework.station.Station(station_id)
                station.set_parameter(astP.name, name)

                sim_station = NuRadioReco.framework.sim_station.SimStation(station_id, position=obs_plane["antenna_position"][idx])
                sim_station.set_parameter(astP.name, name)
                sim_station.set_parameter(astP.position_vB_vvB, obs_plane["antenna_position_vBvvB"][idx])
                sim_station.set_parameter(astP.signal_energy_fluence, obs_plane["energy_fluence"][idx])
                sim_station.set_parameter(astP.signal_energy_fluence_vector, obs_plane["energy_fluence_vector"][idx])
                sim_station.set_parameter(astP.frequency_slope, obs_plane["frequency_slope"][idx])

                station.set_sim_station(sim_station)
                evt.set_station(station)

            evt.set_parameter(evtP.sim_shower, sim_shower)

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            self.__current_input_file += 1

            yield evt

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        logger.info("\per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt


def antenna_id(antenna_name, default_id):
    """
    This function parses the antenna name given in a CoREAS simulation and tries to find an ID
    It can be extended to other name patterns
    """

    if re.match(b"AERA_", antenna_name):
        new_id = int(antenna_name.strip("AERA_"))
        return new_id
    else:
        return default_id
