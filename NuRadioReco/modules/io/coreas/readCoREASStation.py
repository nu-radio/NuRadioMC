from NuRadioReco.modules.base.module import register_run
import numpy as np
import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.framework.parameters import stationParameters as stnp
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

            corsika_evt = coreas.read_CORSIKA7(input_file)
            coreas_sim_station = corsika_evt.get_station(0).get_sim_station()
            corsika_efields = coreas_sim_station.get_electric_fields()

            efield_pos = []
            for corsika_efield in corsika_efields:
                efield_pos.append(corsika_efield.get_position())
            efield_pos = np.array(efield_pos)

            weights = coreas.calculate_simulation_weights(efield_pos, coreas_sim_station.get_parameter(stnp.zenith), coreas_sim_station.get_parameter(stnp.azimuth), debug=self.__debug)
            if self.__debug:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                im = ax.scatter(efield_pos[:, 0], efield_pos[:, 1], c=weights)
                fig.colorbar(im, ax=ax).set_label(label=r'Area $[m^2]$')
                plt.xlabel('East [m]')
                plt.ylabel('West [m]')
                plt.title('Final weighting')
                plt.gca().set_aspect('equal')
                plt.show()

            # make one event for each observer with differen core position (set in create_sim_shower)
            for i, corsika_efield in enumerate(corsika_efields):
                evt = NuRadioReco.framework.event.Event(self.__current_input_file, self.__current_event)  # create empty event
                station = NuRadioReco.framework.station.Station(self.__station_id)
                sim_station = coreas.create_sim_station(self.__station_id, corsika_evt, weights[i])

                channel_ids = detector.get_channel_ids(self.__station_id)
                efield_trace = corsika_efield.get_trace()
                efield_sampling_rate = corsika_efield.get_sampling_rate()
                efield_times = corsika_efield.get_times()

                prepend_zeros = True # prepend zeros to not have the pulse directly at the start, heritage from old code
                if prepend_zeros:
                    n_samples_prepend = efield_trace.shape[1]
                    efield_cor = np.zeros((3, n_samples_prepend + efield_trace.shape[1]))
                    efield_cor[0] = np.append(np.zeros(n_samples_prepend), efield_trace[0])
                    efield_cor[1] = np.append(np.zeros(n_samples_prepend), efield_trace[1])
                    efield_cor[2] = np.append(np.zeros(n_samples_prepend), efield_trace[2])

                    efield_times_cor = np.arange(0, n_samples_prepend + efield_trace.shape[1]) / efield_sampling_rate

                else:
                    efield_cor = efield_trace
                    efield_times_cor = efield_times

                coreas.add_electric_field_to_sim_station(sim_station, channel_ids, efield_cor, efield_times_cor, sim_station.get_parameter(stnp.zenith), sim_station.get_parameter(stnp.azimuth), efield_sampling_rate)
                station.set_sim_station(sim_station)
                sim_shower = coreas.create_sim_shower(corsika_evt, detector, self.__station_id)
                evt.add_sim_shower(sim_shower)
                evt.set_station(station)
                self.__current_event += 1
                yield evt
            self.__current_input_file += 1

    def end(self):
        pass
