import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.modules.io.coreas import coreas


class readCoREASStation:

    def begin(self):
        pass

    def run(self, input_file, station_id):

        # read in coreas simulation and copy to event
        corsika = h5py.File(input_file, "r")
        weights = coreas.calculate_simulation_weights(corsika["highlevel"]["obsplane_na_na"]["antenna_position"])

        for i, (name, observer) in enumerate(corsika['CoREAS']['observers'].items()):
            evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])  # create empty event
            station = NuRadioReco.framework.station.Station(station_id)
            sim_station = coreas.make_sim_station(station_id, corsika, observer, weights[i])

            station.set_sim_station(sim_station)
            evt.set_station(station)
            yield evt

    def end(self):
        pass
