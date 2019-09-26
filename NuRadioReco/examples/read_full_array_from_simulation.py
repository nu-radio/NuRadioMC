import argparse
from NuRadioReco.utilities import units
import numpy as np
import NuRadioReco.modules.io.coreas.readCoREASShower
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.io.eventReader

from NuRadioReco.framework.parameters import showerParameters as shP
import NuRadioReco.modules.electricFieldBandPassFilter
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str, nargs='*',
                    help='path to NuRadioMC simulation result')

args = parser.parse_args()

# initialize modules
readCoREASShower = NuRadioReco.modules.io.coreas.readCoREASShower.readCoREASShower()
readCoREASShower.begin(args.inputfilename)

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
output_filename = "Full_CoREAS_event.nur"
eventWriter.begin(output_filename)

eventReader = NuRadioReco.modules.io.eventReader.eventReader()
from matplotlib import pyplot as plt
# for iE, event in enumerate(eventReader.run()):
for iE, event in enumerate(readCoREASShower.run()):
    print('Event {}'.format(event.get_id()))

    for sim_shower in event.get_sim_showers():
        print('CR energy:', sim_shower.get_parameter(shP.energy))
        print('electromagnetic energy:', sim_shower.get_parameter(shP.electromagnetic_energy))
        print('radiation energy:', sim_shower.get_parameter(shP.electromagnetic_energy))

        for stid, station in enumerate(event.get_stations()):
            sim_station = station.get_sim_station()
            electricFieldBandPassFilter.run(event, sim_station, det=None, passband=[30 * units.MHz, 80 * units.MHz], filter_type='butter', order=10)

            print(sim_station.get_id())
            ef = sim_station.get_electric_fields()[0]

            spec = ef.get_frequency_spectrum()
            freq = ef.get_frequencies() * 1e3  # GHz in MHz
            dist = np.linalg.norm(sim_station.get_position() - sim_shower.get_parameter(shP.core))
            plt.plot(freq, np.abs(spec[0]))
            plt.plot(freq, np.abs(spec[1]))
            # plt.plot(freq, spec[2])
            plt.xlim(0, 100)
            plt.title(dist)
            plt.show()

        eventWriter.run(event)

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))

eventReader.begin("Full_CoREAS_event.nur")
for iE, event in enumerate(eventReader.run()):
    print('Event {}'.format(event.get_id()))

    for sim_shower in event.get_sim_showers():
        print('CR energy:', sim_shower.get_parameter(shP.energy))
        print('electromagnetic energy:', sim_shower.get_parameter(shP.electromagnetic_energy))
        print('radiation energy:', sim_shower.get_parameter(shP.electromagnetic_energy))

        for stid, station in enumerate(event.get_stations()):
            sim_station = station.get_sim_station()

            print(sim_station.get_id())

# print(eventReader.run())
eventReader.end()
