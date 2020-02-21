import time
import copy
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import logging

from radiotools import helper as hp
from radiotools import plthelpers as php

import NuRadioReco.framework.base_trace
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.utilities import ice
from NuRadioReco.detector import antennapattern

from NuRadioReco.framework import electric_field as ef
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.electricFieldBandPassFilter



electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
voltageToEfieldConverterPerChannel = NuRadioReco.modules.voltageToEfieldConverterPerChannel.voltageToEfieldConverterPerChannel()
voltageToEfieldConverterPerChannel.begin()

def get_array_of_channels(station, det, zenith, azimuth, polarization):
    """
    Returns an array of the channel traces that is cut to the physical overlapping time

    Parameters
    ----------
    zenith: float
        Arrival zenith angle at antenna
    azimuth: float
        Arrival azimuth angle at antenna
    polarization: int
        0: eTheta
        1: ePhi
    """
    time_shifts = np.zeros(8)
    t_geos = np.zeros(8)

    sampling_rate = station.get_channel(0).get_sampling_rate()
    station_id = station.get_id()
    site = det.get_site(station_id)
    for iCh, channel in enumerate(station.get_electric_fields()):
        channel_id = channel.get_channel_ids()[0]
        antenna_position = det.get_relative_position(station_id, channel_id)
        # determine refractive index of signal propagation speed between antennas
        refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        if(zenith > 0.5 * np.pi):
            refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position
        refractive_index = 1.353
        time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
        t_geos[iCh] = time_shift
        time_shift += channel.get_trace_start_time()
        time_shifts[iCh] = time_shift

    delta_t = time_shifts.max() - time_shifts.min()
    tmin = time_shifts.min()
    tmax = time_shifts.max()
    trace_length = station.get_electric_fields()[0].get_times()[-1] - station.get_electric_fields()[0].get_times()[0]

    traces = []
    n_samples = None
    for iCh, channel in enumerate(station.get_electric_fields()):
        tstart = delta_t - (time_shifts[iCh] - tmin)
        tstop = tmax - time_shifts[iCh] - delta_t + trace_length
        iStart = int(round(tstart * sampling_rate))
        iStop = int(round(tstop * sampling_rate))
        if(n_samples is None):
            n_samples = iStop - iStart
            if(n_samples % 2):
                n_samples -= 1

        trace = copy.copy(channel.get_trace()[polarization])  # copy to not modify data structure
        trace = trace[iStart:(iStart + n_samples)]
        base_trace = NuRadioReco.framework.base_trace.BaseTrace()  # create base trace class to do the fft with correct normalization etc.
        base_trace.set_trace(trace, sampling_rate)
        traces.append(base_trace)

    return traces

class beamFormingDirectionFitter:
    """
    Fits the direction using interferometry between desired channels.
    """

    def __init__(self):
        self.__zenith = []
        self.__azimuth = []
        self.__delta_zenith = []
        self.__delta_azimuth = []
        self.begin()
        self.logger = logging.getLogger("NuRadioReco.beamFormingDirectionFitter")

    def begin(self, debug=False, log_level=None):
        if(log_level is not None):
            self.logger.setLevel(log_level)
        self.__debug = debug

    def run(self, evt, station, det, polarization, n_index=None,channels=[4,5,6,7], ZenLim=[90 * units.deg, 180 * units.deg],
            AziLim=[0 * units.deg, 360 * units.deg]):
        """
        reconstruct signal arrival direction for all events through beam forming.
        https://arxiv.org/pdf/1009.0345.pdf

        Parameters
        ----------
        polarization: int
            0: eTheta
            1: ePhi
        n_index: float
            the index of refraction
        channels: list
            the channel ids
        ZenLim: 2-dim array/list of floats
            the zenith angle limits for the fit
            default if 0-90deg (upward coming signal)
        AziLim: 2-dim array/list of floats
            the azimuth angle limits for the fit
            default is 0-360deg
        """

        def ll_regular_station(angles, evt, station, det, polarization, sampling_rate, positions, channels):
            """
            Likelihood function for a four antenna ARIANNA station, using correction.
            Using correlation, has no built in wrap around, pulse needs to be in the middle
            """

            zenith = angles[0]
            azimuth = angles[1]

            station.set_parameter(stnp.zenith,zenith)
            station.set_parameter(stnp.azimuth,azimuth)
            station.set_electric_fields([]) # resets EFields, necessary
            voltageToEfieldConverterPerChannel.run(evt, station, det,pol=polarization,debug=False)  # Antenna response
            electricFieldBandPassFilter.run(evt, station, det,passband=[120 * units.MHz, 300 * units.MHz], filter_type='butterabs')

            Efields_object = get_array_of_channels(station, det, zenith, azimuth, polarization+1)
            Efields = []
            Efield_Times = []
            maximum = 0 # location at maximum among all traces
            for chan in channels:
                efield = Efields_object[chan]
                Efield_Times.append(efield.get_times())
                Efield = efield.get_trace()
                if max(Efield) > maximum:
                    maximum = max(Efield)
                Efields.append(Efield)#np.pad(Efield, (200,200), 'constant', constant_values=(0, 0)))

            Efields[0] = Efields[0]/maximum # normalize all traces to remove small antenna responses, see last line of next for loop
            for i in range(len(Efields)-1):
                Efields[i+1] = Efields[i+1]/maximum

            N = len(Efields)
            N_pairs = 0.5*np.math.factorial(N)/np.math.factorial(N-2)
            cc = np.zeros(len(Efields[0]))
            for i in range(N-1): # finds the cc-beam taken from referenced paper above
                for j in range(N-1-i):
                    cc = cc + Efields[i]*Efields[j+1+i]
            cc = cc/N_pairs
            cc = np.sign(cc)*np.sqrt(np.abs(cc))

            cc = np.abs(cc)
            ave_cc = np.zeros_like(cc)
            n_bins = 2000
            ave_cc = np.convolve(cc, np.ones((n_bins))/float(n_bins),mode='same')

            likelihood = -1*np.max(ave_cc)
            return likelihood


        station_id = station.get_id()
        positions_all = det.get_relative_positions(station_id)
        positions = []
        for chan in channels:
            positions.append(positions_all[chan])
        sampling_rate = station.get_channel(0).get_sampling_rate()

        ll = opt.brute(ll_regular_station, ranges=(slice(ZenLim[0], ZenLim[1], 1.0*units.deg),
                                                   slice(AziLim[0], AziLim[1], 1.0*units.deg)),
                        args=(evt, station, det, polarization, sampling_rate, positions, channels),
                        full_output=True, finish=opt.fmin)  # slow but does the trick

        station[stnp.zenith] = max(ZenLim[0], min(ZenLim[1], ll[0][0]))
        station[stnp.azimuth] = ll[0][1]

        if self.__debug:
            import peakutils

            # Show fit space
            zen = np.arange(ZenLim[0], ZenLim[1], 1*units.deg)
            az = np.arange(AziLim[0], AziLim[1], 1*units.deg)

            x_plot = np.zeros(zen.shape[0] * az.shape[0])
            y_plot = np.zeros(zen.shape[0] * az.shape[0])
            z_plot = np.zeros(zen.shape[0] * az.shape[0])
            i = 0
            for z in zen:
                for a in az:
                    # Evaluate fit function for grid
                    z_plot[i] = ll_regular_station([z, a], evt, station, det, polarization, sampling_rate, positions, channels)
                    x_plot[i] = a
                    y_plot[i] = z
                    i += 1

            fig, ax = plt.subplots(1, 1)
            ax.scatter(np.asarray(x_plot)/units.deg, np.asarray(y_plot)/units.deg, c=z_plot, cmap='gnuplot2_r', lw=0)
            ax.scatter(np.rad2deg(ll[0][1]), np.rad2deg(ll[0][0]), marker='o', label='Fit')
            ax.colorbar(label='Fit parameter')
            ax.set_ylabel('Zenith [rad]')
            ax.set_xlabel('Azimuth [rad]')
            plt.tight_layout()
            plt.legend()
            plt.show()

    def end(self):
        pass
