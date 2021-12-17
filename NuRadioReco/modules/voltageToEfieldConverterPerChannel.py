from NuRadioReco.modules.base.module import register_run
import numpy as np
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework import electric_field as ef
import logging


class voltageToEfieldConverterPerChannel:
    """
    Converts voltage trace to electric field per channel

    Assumes either co-pole or cross-pole, then finds the electric field per channel by E = V/H in fourier space.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.voltageToEfieldConverterPerChannel')
        self.antenna_provider = None
        self.__counter = 0
        self.begin()

    def begin(self):
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    @register_run()
    def run(self, evt, station, det, pol=0):
        """
        Performs computation for voltage trace to electric field per channel

        Will provide a deconvoluted (electric field) trace for each channel from the stations input voltage traces

        Parameters
        ----------
        evt: event data structure
            the event data structure
        station: station data structure
            the station data structure
        det: detector object
            the detector object
        pol: polarization
            0 = eTheta polarized, 1 = ePhi polarized
        """
        self.__counter += 1
        station_id = station.get_id()
        self.logger.debug("event {}, station {}".format(evt.get_id(), station_id))
        if station.get_sim_station() is not None and station.get_sim_station().has_parameter(stnp.zenith):
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
        else:
            self.logger.debug("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]

        frequencies = station.get_channel(0).get_frequencies()  # assuming that all channels have the  same sampling rate and length
        use_channels = det.get_channel_ids(station.get_id())
        efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, frequencies, use_channels, det,
                                                                          zenith, azimuth, self.antenna_provider)

        sampling_rate = station.get_channel(0).get_sampling_rate()

        for iCh, channel in enumerate(station.iter_channels()):
            efield = ef.ElectricField([iCh])
            trace = channel.get_frequency_spectrum()
            mask1 = np.abs(efield_antenna_factor[iCh][0]) != 0
            mask2 = np.abs(efield_antenna_factor[iCh][1]) != 0
            efield_spectrum = np.zeros((3, len(trace)), dtype=np.complex)
            efield_spectrum[1][mask1] = (1.0 - pol) ** 2 * trace[mask1] / efield_antenna_factor[iCh][0][mask1]
            efield_spectrum[2][mask2] = pol ** 2 * trace[mask2] / efield_antenna_factor[iCh][1][mask2]
            efield.set_frequency_spectrum(efield_spectrum, sampling_rate)
            efield.set_trace_start_time(channel.get_trace_start_time())
            efield[efp.zenith] = zenith
            efield[efp.azimuth] = azimuth
            station.add_electric_field(efield)

    def end(self):
        pass
