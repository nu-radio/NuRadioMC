import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.modules.io.coreas import coreas


class readCoREASStation:

    def begin(self, input_files, station_id):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        station_id: station id
            id number of the ARIANNA station
        """
        self.__input_files = input_files
        self.__station_id = station_id
        self.__current_input_file = 0

    def run(self):
        for input_file in self.__input_files:

            # read in coreas simulation and copy to event
            corsika = h5py.File(input_file, "r")
            weights = coreas.calculate_simulation_weights(corsika["highlevel"].values()[0]["antenna_position"])

            for i, (name, observer) in enumerate(corsika['CoREAS']['observers'].items()):
                evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])  # create empty event
                station = NuRadioReco.framework.station.Station(self.__station_id)
                sim_station = coreas.make_sim_station(self.__station_id, corsika, observer, weights[i])

                station.set_sim_station(sim_station)
                evt.set_station(station)
                yield evt

    def end(self):
        pass
