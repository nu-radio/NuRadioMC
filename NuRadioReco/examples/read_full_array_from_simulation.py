import argparse

import NuRadioReco.modules.io.coreas.readCoREASShower
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.io.eventReader


from NuRadioReco.framework.parameters import showerTypes as shT
from NuRadioReco.framework.parameters import showerParameters as shP
from NuRadioReco.framework.parameters import array_stationParameters as astP

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

# for iE, event in enumerate(eventReader.run()):
for iE, event in enumerate(readCoREASShower.run()):
    print('Event {}'.format(event.get_id()))

    sim_shower = event.get_shower(shT.sim_shower)
    print('CR energy:', sim_shower.get_parameter(shP.energy))
    print('electromagnetic energy:', sim_shower.get_parameter(shP.electromagnetic_energy))
    print('radiation energy:', sim_shower.get_parameter(shP.electromagnetic_energy))

    for stid, station in enumerate(event.get_stations()):
        sim_station = station.get_sim_station()

        print(sim_station.get_axis_distance(), sim_station[astP.signal_energy_fluence])

    eventWriter.run(event)

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))

eventReader.begin("Full_CoREAS_event.nur")
for iE, event in enumerate(eventReader.run()):
    print('Event {}'.format(event.get_id()))

    sim_shower = event.get_shower(shT.sim_shower)
    print('CR energy:', sim_shower.get_parameter(shP.energy))
    print('electromagnetic energy:', sim_shower.get_parameter(shP.electromagnetic_energy))
    print('radiation energy:', sim_shower.get_parameter(shP.electromagnetic_energy))

    for stid, station in enumerate(event.get_stations()):
        sim_station = station.get_sim_station()

        print(sim_station.get_axis_distance(), sim_station[astP.signal_energy_fluence])

# print(eventReader.run())
eventReader.end()