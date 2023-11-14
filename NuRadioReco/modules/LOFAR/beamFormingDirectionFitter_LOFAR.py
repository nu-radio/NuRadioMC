import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy import constants

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.beamforming_utilities import mini_beamformer


logger = module.setup_logger(level=logging.WARNING)


lightspeed=constants.c * units.m / units.s


class beamFormingDirectionFitter:
    """
    Fits the direction using interferometry between desired channels.
    """

    def __init__(self):
        self.__zenith = []
        self.__azimuth = []
        self.__delta_zenith = []
        self.__delta_azimuth = []
        self.__debug = None
        self.begin()
        #self.logger = logging.getLogger("NuRadioReco.beamFormingDirectionFitter")

    def begin(self, debug=False, log_level=None):
        if(log_level is not None):
            self.logger.setLevel(log_level)
        self.__debug = debug

    @register_run
    def run(self, evt, station, det, polarisation):
        """
        reconstruct signal arrival direction for all events through beam forming.
        https://arxiv.org/pdf/1009.0345.pdf

        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector description
        polarization: int
            0: pol0
            1: pol1
        """
        ## replace this with estimate from LORA:
        zenith=0.789*units.rad
        azimuth=2.54*units.rad+np.pi    # are these the correct values for 92380604? 
        direction=np.asarray([zenith,azimuth])

        fft_array=[]        
        position_array=[]

        fig=plt.figure(facecolor='white')
        ax1=fig.add_subplot(1,1,1)          
        ax1.set_xlim([33200,33500])


        for channel in station.iter_channel_group(polarisation):
            ax1.plot(channel.get_trace(),color='blue',linewidth=0.5,alpha=0.3)
            freqs = channel.get_frequencies()
            fft_array.append(fft.time2freq(channel.get_trace(), channel.get_sampling_rate()))            
            position_array.append(det.get_absolute_position(station.get_id())+det.get_relative_position(station.get_id(), channel.get_id()))
        plt.show()

        fft_array=np.asarray(fft_array)
        position_array=np.asarray(position_array)

        print(fft_array.shape,position_array.shape)

        # this can be replaced by soemthing form NNR?
    
        direction_cartesian=hp.spherical_to_cartesian(direction[0],direction[1])
        print(direction_cartesian)
        
        beamed_fft=mini_beamformer(fft_array,freqs,position_array,direction_cartesian)
        print('done beamforming')
        beamformed_timeseries=np.fft.irfft(beamed_fft)


        fig=plt.figure(facecolor='white')
        ax1=fig.add_subplot(1,1,1)          
        ax1.plot(beamformed_timeseries,color='blue',linewidth=0.5,alpha=1,label='beamfformed 0')

        ax1.set_xlim([33200,33500])
        plt.show()
        '''

        maxiter=20
        fitted_direction, fitted_timeseries=directionFitBF(fft_array,freqs,position_array,direction_cartesian,maxiter)
        ax1.plot(fitted_timeseries,color='red',linewidth=0.5,alpha=1,label='beamfformed 0')

        print('start direction: ',direction)

        print('final direction: ',hp.cartesian_to_spherical(fitted_direction[0],fitted_direction[1],fitted_direction[2]))

        '''

    # steps from pycrtools

    def end(self):
        pass
