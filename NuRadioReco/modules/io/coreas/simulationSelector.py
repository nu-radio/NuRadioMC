from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
import time

from NuRadioReco.utilities import units


class simulationSelector:

    '''
    Module that let's you select CoREAS simulations
    based on certain criteria, e.g. signal in a relevant band
    certain arrival directions, energies, etc.
    '''

    def __init__(self):
        self.__t = 0
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.simulationSelector')

    def begin(self, debug=False):
        pass

    @register_run()
    def run(self, evt, sim_station, det, frequency_window=[100 * units.MHz, 500 * units.MHz], n_std=8):

        """
        run method, selects CoREAS simulations that have any signal in
        desired frequency_window.
        Crude approximation with n_std sigma * noise

        Parameters
        ------------
        evt: Event
        sim_station: sim_station
            CoREAS simulated efields
        det: Detector
        frequency_window: list
            [lower, upper] frequencies that will be used for analysis
        n_std: int
            number of std deviations needed, can make cut stricter, if needed

        Returns
        ------------
        selected_sim: bool
            if True then simulation has signal in desired range

        """
        t = time.time()
        efields = sim_station.get_electric_fields()
        selected_sim = False
        for efield in efields:
            fft = np.abs(efield.get_frequency_spectrum())
            freq = efield.get_frequencies()

            # identify the largest polarization
            max_pol = 0
            max_ = 0
            for i in range(3):
                if np.sum(fft[i]) > max_:
                    max_pol = i
                    max_ = np.sum(fft[i])

            # Find a n_std sigma excess above the noise
            # Seems to be a reasonably well-working number

            noise_region = fft[max_pol][np.where(freq > 1.5 * units.GHz)]
            noise = np.mean(noise_region)
            if noise == 0:
                self.logger.warning("Trace seems to have bee upsampled beyong 1.5 GHz, using lower noise window")
                noise_region = fft[max_pol][np.where(freq > 1. * units.GHz)]
                noise = np.mean(noise_region)
            if noise == 0:
                self.logger.warning("Trace seems to have bee upsampled beyong 1. GHz, using lower noise window")
                noise_region = fft[max_pol][np.where(freq > 800. * units.MHz)]
                noise = np.mean(noise_region)
            if noise == 0:
                self.logger.error("Trace seems to have bee upsampled beyong 800 MHz, unsuitable simulations")

            noise_std = np.std(noise_region)

            noise += n_std * noise_std

            mask = np.array((freq >= np.min(frequency_window)) & (freq <= np.max(frequency_window)))

            if(np.any(fft[:, mask] > noise)):
                selected_sim = True
                break

        self.__t += time.time() - t
        return selected_sim

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
