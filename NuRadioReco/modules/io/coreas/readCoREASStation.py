from NuRadioReco.modules.base.module import register_run
import h5py
import numpy as np
import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('readCoREASStation')


class readCoREASStation:

    def begin(self, input_files, station_id, debug=False):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        station_id: station id
            id number of the radio station as defined in detector
        """
        self.__input_files = input_files
        self.__station_id = station_id
        self.__current_input_file = 0
        self.__current_event = 0
        self.__debug = debug

    @register_run()
    def run(self, detector):
        """
        Reads in all observers in the CoREAS files and returns a new simulated event for each observer with 
        respect to a given detector with a single station.

        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated containing one station

        """
        for input_file in self.__input_files:
            self.__current_event = 0
            with h5py.File(input_file, "r") as corsika:
                zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
                obs_positions_geo = []
                for i, (name, observer) in enumerate(corsika['CoREAS']['observers'].items()):
                    obs_positions_geo.append(coreas.convert_obs_positions_to_nuradio_on_ground(observer, zenith, azimuth, magnetic_field_vector))
                obs_positions_geo = np.array(obs_positions_geo)
                weights = coreas.calculate_simulation_weights(obs_positions_geo, zenith, azimuth, debug=self.__debug)
                if self.__debug:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    im = ax.scatter(obs_positions_geo[:, 0], obs_positions_geo[:, 1], c=weights)
                    fig.colorbar(im, ax=ax).set_label(label=r'Area $[m^2]$')
                    plt.xlabel('East [m]')
                    plt.ylabel('West [m]')
                    plt.title('Final weighting')
                    plt.gca().set_aspect('equal')
                    plt.show()

                for i, (name, observer) in enumerate(corsika['CoREAS']['observers'].items()):
                    evt = NuRadioReco.framework.event.Event(self.__current_input_file, self.__current_event)  # create empty event
                    station = NuRadioReco.framework.station.Station(self.__station_id)
                    sim_station = coreas.make_sim_station(
                        self.__station_id,
                        corsika,
                        weights[i]
                    )

                    channel_ids = detector.get_channel_ids(self.__station_id)
                    efield, efield_times = coreas.convert_obs_to_nuradio_efield(observer, zenith, azimuth, magnetic_field_vector, prepend_zeros=True)
                    coreas.add_electric_field_to_sim_station(sim_station, channel_ids, efield, efield_times, corsika)
                    station.set_sim_station(sim_station)
                    sim_shower = coreas.make_sim_shower(corsika, observer, detector, self.__station_id)
                    evt.add_sim_shower(sim_shower)
                    evt.set_station(station)
                    self.__current_event += 1
                    yield evt
            self.__current_input_file += 1

    def end(self):
        pass
