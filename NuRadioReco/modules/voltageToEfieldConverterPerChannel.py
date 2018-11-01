from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import signal
from scipy.signal import correlate
from scipy import optimize as opt
import matplotlib.pyplot as plt
import time

from radiotools import helper as hp
from radiotools import plthelpers as php

from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.detector import antennapattern
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels
import NuRadioReco.framework.base_trace

from NuRadioReco.framework.parameters import stationParameters as stnp

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
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]
            sim_present = False

        efield_antenna_factor, V = get_array_of_channels(station, range(station.get_number_of_channels()),
                                                                       det, zenith, azimuth, self.antenna_provider)

        sampling_rate = station.get_channel(0).get_sampling_rate()

        for iCh, channel in enumerate(station.iter_channels()):
            mask = np.abs(efield_antenna_factor[iCh][pol]) != 0
            efield_spectrum = np.zeros_like(V[iCh])
            efield_spectrum[mask] = V[iCh][mask] / efield_antenna_factor[iCh][pol][mask]
            base_trace = NuRadioReco.framework.base_trace.BaseTrace()
            base_trace.set_frequency_spectrum(efield_spectrum, sampling_rate)
            channel.set_electric_field(base_trace)

    def end(self):
        pass
