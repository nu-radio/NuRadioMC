import h5py
from NuRadioReco.framework import event, sim_event
from NuRadioReco.modules.io.coreas import coreas


class readCoREASEvent:

    def begin(self):
        pass

    def run(self, input_file):

        # read in coreas simulation and copy to event
        corsika = h5py.File(input_file, "r")

        evt = event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])  # create empty event
        sim = sim_event.SimEvent(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])

        for name, observer in corsika['CoREAS']['observers'].items():
            sim_station = coreas.make_sim_station(name, corsika, observer)
            sim.set_station(sim_station)

        evt.set_simulation(sim)
        yield evt

    def end(self):
        pass
