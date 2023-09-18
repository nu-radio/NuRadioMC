from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioMC.utilities import Veff
# import detector simulation modules
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation

import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

###-----------------------------------------
#   EXAMPLE: Script to calculate the effects of birefringence on the effective volume. 
#            A full study of this calculation was published here: DOI: https://doi.org/10.22323/1.444.1101
###-----------------------------------------

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):

        simpleThreshold.run(evt, station, det,

                             threshold=3.0 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold1')  # the name of the trigger
        
energies = np.array([19])

Veff_birefringence = []
Veff_no_birefringence = []

for i in energies:

    inputfilename = 'extra_files/1e' + str(i) + '_n1e3.hdf5'
    outputfilename_no_birefringence = 'Veff_' + str(i) + '_no_birefringence.hdf5'
    outputfilename_birefringence = 'Veff_' + str(i) + '_birefringence.hdf5'
    detectordescription = 'extra_files/detector.json'
    config_no_birefringence = 'extra_files/config_no_birefringence.yaml'
    config_birefringence = 'extra_files/config_birefringence.yaml'


    if __name__ == "__main__":
        sim = mySimulation(inputfilename=inputfilename,
                                    outputfilename=outputfilename_no_birefringence,
                                    detectorfile=detectordescription,
                                    config_file=config_no_birefringence,
                                    file_overwrite=True)
        sim.run()

    if __name__ == "__main__":
        sim = mySimulation(inputfilename=inputfilename,
                                    outputfilename=outputfilename_birefringence,
                                    detectorfile=detectordescription,
                                    config_file=config_birefringence,
                                    file_overwrite=True)
        sim.run()

    data_birefringence = Veff.get_Veff_Aeff(outputfilename_birefringence)
    data_no_birefringence = Veff.get_Veff_Aeff(outputfilename_no_birefringence)

    Veff_birefringence.append(data_birefringence[0]['veff']['all_triggers'][0])
    Veff_no_birefringence.append(data_no_birefringence[0]['veff']['all_triggers'][0])

plt.scatter(energies, 100 * np.array(Veff_birefringence) / np.array(Veff_no_birefringence))
plt.ylabel(r'V_{eff._bir.} / V_{eff._iso.} [%]')
plt.xlabel(r'$\log (E)$')
plt.tight_layout()
plt.savefig('05_Veff_comp_plot.png', dpi=400)

