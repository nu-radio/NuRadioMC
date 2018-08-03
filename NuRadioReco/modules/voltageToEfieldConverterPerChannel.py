from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from scipy import signal
from NuRadioReco.detector import antennapattern
from radiotools import plthelpers as php
from numpy.polynomial import polynomial as poly
from scipy.signal import correlate
from scipy import optimize as opt
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels
import NuRadioReco.framework.base_trace
import time
import matplotlib.pyplot as plt
from radiotools import helper as hp
import logging
logger = logging.getLogger('voltageToEfieldConverterPerChannel')


class voltageToEfieldConverterPerChannel:
    """
    Converts voltage trace to electric field per channel

    Assumes either co-pole or cross-pole, then finds the electric field per channel by E = V/H in fourier space.
    """

    def __init__(self):
        self.__counter = 0
        self.begin()

    def begin(self):
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

    def run(self, evt, station, det, pol=0, debug=True):
        """
        Performs computation for voltage trace to electric field per channel

        Will provide a deconvoluted (electric field) trace for each channel from the stations input voltage traces

        Paramters
        ---------
        evt: event data structure
            the event data structure
        station: station data structure
            the station data structure
        det: detector object
            the detector object
        pol: polarization
            0 = eTheta polarized, 1 = ePhi polarized
        debug: bool
            if True additional debug plot(s) are displayed - currently unused
        """
        self.__counter += 1
        event_time = station.get_station_time()
        station_id = station.get_id()
        logger.info("event {}, station {}".format(evt.get_id(), station_id))
        if station.get_sim_station() is not None:
            zenith = station.get_sim_station()['zenith']
            azimuth = station.get_sim_station()['azimuth']
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station['zenith']
            azimuth = station['azimuth']
            sim_present = False

        channels = station.get_channels()
        efield_antenna_factor, V = get_array_of_channels(station, range(len(channels)),
                                                                       det, zenith, azimuth, self.antenna_provider)

        sampling_rate = channels[0].get_sampling_rate()

        for iCh, channel in enumerate(channels):
            mask = np.abs(efield_antenna_factor[iCh][pol]) != 0
            efield_spectrum = np.zeros_like(V[iCh])
            efield_spectrum[mask] = V[iCh][mask] / efield_antenna_factor[iCh][pol][mask]
            base_trace = NuRadioReco.framework.base_trace.BaseTrace()
            base_trace.set_frequency_spectrum(efield_spectrum, sampling_rate)
            channel.set_electric_field(base_trace)

    def end(self):
        pass
