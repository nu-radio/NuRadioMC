from NuRadioReco.modules.base.module import register_run
import numpy as np
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework import electric_field as ef
from NuRadioReco.modules.io.coreas.readCoREASDetector import select_channels_per_station
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels
import logging


class voltageToEfieldConverterPerChannelGroup:
    """
    This module is intended to reconstruct the electric field from dual-polarized antennas, 
    i.e., two antennas with orthogonal polarizations combined in one mechanical structure. 
    This is the typical case for air-shower detectors such as Auger and LOFAR. 

    Converts voltage trace to electric field per channel group.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.voltageToEfieldConverterPerChannelGroup')
        self.antenna_provider = None
        self.__counter = 0
        self.begin()

    def begin(self, use_MC_direction=False):
        """
        Initializes the module

        Parameters
        ----------
        use_MC_direction: bool
            If True, the MC direction is used for the reconstruction. If False, the reconstructed angles are used.
        """
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self.__use_MC_direction = use_MC_direction

    @register_run()
    def run(self, evt, station, det):
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
        """
        if self.__use_MC_direction:
            if station.get_sim_station() is not None and station.get_sim_station().has_parameter(stnp.zenith):
                zenith = station.get_sim_station()[stnp.zenith]
                azimuth = station.get_sim_station()[stnp.azimuth]
            else:
                self.logger.error(f"MC direction requested but no simulation present in station {station.get_id()}")
                raise ValueError("MC direction requested but no simulation present")
        else:
            self.logger.debug("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]

        use_channels = det.get_channel_ids(station.get_id())
        frequencies = station.get_channel(use_channels[0]).get_frequencies()  # assuming that all channels have the  same sampling rate and length

        sampling_rate = station.get_channel(use_channels[0]).get_sampling_rate()

        group_ids = select_channels_per_station(det, station.get_id(), station.get_channel_ids())
        for gid, use_channels in group_ids.items():
            efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, frequencies, use_channels, det,
                                                                            zenith, azimuth, self.antenna_provider)
            V = np.zeros((len(use_channels), len(frequencies)), dtype=complex)
            for i_ch, channel_id in enumerate(use_channels):
                V[i_ch] = station.get_channel(channel_id).get_frequency_spectrum()
            denom = (efield_antenna_factor[0][0] * efield_antenna_factor[1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[1][0])
            mask = np.abs(denom) != 0
            # solving for electric field using just two orthorgonal antennas
            E1 = np.zeros_like(V[0], dtype=complex)
            E2 = np.zeros_like(V[0], dtype=complex)
            E1[mask] = (V[0] * efield_antenna_factor[1][1] - V[1] * efield_antenna_factor[0][1])[mask] / denom[mask]
            E2[mask] = (V[1] - efield_antenna_factor[1][0] * E1)[mask] / efield_antenna_factor[1][1][mask]
            denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
            mask = np.abs(denom) != 0
            E1[mask] = (V[0] * efield_antenna_factor[-1][1] - V[-1] * efield_antenna_factor[0][1])[mask] / denom[mask]
            E2[mask] = (V[-1] - efield_antenna_factor[-1][0] * E1)[mask] / efield_antenna_factor[-1][1][mask]

            efield = ef.ElectricField(use_channels)
            efield.set_frequency_spectrum(np.array([np.zeros_like(E1), E1, E2]), sampling_rate)

            efield.set_trace_start_time(station.get_channel(use_channels[0]).get_trace_start_time())
            efield[efp.zenith] = zenith
            efield[efp.azimuth] = azimuth
            station.add_electric_field(efield)

    def end(self):
        pass
