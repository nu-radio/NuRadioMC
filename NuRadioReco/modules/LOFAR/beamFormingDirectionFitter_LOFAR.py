import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy import constants
import logging
import NuRadioReco.framework.base_trace
import radiotools.helper as hp

from NuRadioReco.utilities import ice, fft
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.electricFieldBandPassFilter
from scipy.optimize import fmin_powell

lightspeed=constants.c * units.m / units.s


class beamFormer:
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
        
        beamed_fft=minibeamformer(fft_array,freqs,position_array,direction_cartesian)
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



def minibeamformer(fft_data,frequencies,positions,direction):
    #adapted from pycrtools hBeamformBlock
    nantennas=len(positions)
    nfreq=len(frequencies)
    output=np.zeros([len(frequencies)],dtype=complex)

    norm = np.sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2])
    
    for a in np.arange(nantennas):
        delay = GeometricDelayFarField(positions[a], direction, norm)

        real = 1.0 * np.cos(2*np.pi*frequencies*delay)
        imag = 1.0 * np.sin(2*np.pi*frequencies*delay)
        #de = complex(real,imag)
        de = real+1j*imag
        output=output+fft_data[a]*de
        #for j in np.arange(nfreq):
        #    real = 1.0 * np.cos(2*np.pi*frequencies[j]*delay)
        #    imag = 1.0 * np.sin(2*np.pi*frequencies[j]*delay)
        #    de=complex(real,imag)
        #    output[j]=output[j]+fft_data[a][j]*de
              #*it_out += (*it_fft) * polar(1.0, (2*np.pi)*((*it_freq) * delay));
    
    return output

def geometric_delays(antpos,sky):
    distance=np.sqrt(sky[0]**2+sky[1]**2+sky[2]**2)
    delays=(np.sqrt((sky[0]-antpos[0])**2+(sky[1]-antpos[1])**2+(sky[2]-antpos[2])**2)-distance)/lightspeed
    return delays

def GeometricDelayFarField(position, direction, length):
    delay=(direction[0]*position[0] + direction[1]*position[1]+direction[2]*position[2])/length/lightspeed
    return delay
    
    
def beamformer(fft_data,frequencies,delay):
    nantennas=len(delay)
    nfreq=len(frequencies)
    output=np.zeros([len(frequencies)],dtype=complex)

    for a in np.arange(nantennas):
        for j in np.arange(nfreq):
            real = 1.0 * np.cos(2*np.pi*frequencies[j]*delay[a])
            imag = 1.0 * np.sin(2*np.pi*frequencies[j]*delay[a])
            de=complex(real,imag)
            output[j]=output[j]+fft_data[a][j]*de
    return output

def directionFitBF(fft_data,frequencies,antpos,start_direction,maxiter):
    def negative_beamed_signal(direction):
        print('direction: ',direction)

        theta=direction[0]
        phi=direction[1]
        direction_cartesian=hp.spherical_to_cartesian(theta,phi)
        delays=geometric_delays(antpos,direction_cartesian)
        out=beamformer(fft_data,frequencies,delays)
        timeseries=np.fft.irfft(out)
        return -100*np.max(timeseries**2)
    
    
    fit_direction = fmin_powell(negative_beamed_signal, np.asarray(start_direction), maxiter=maxiter, xtol=1.0)
    
    theta=fit_direction[0]
    phi=fit_direction[1]
    direction_cartesian=hp.spherical_to_cartesian(theta,phi)
    delays=geometric_delays(antpos,direction_cartesian)
    out=beamformer(fft_data,frequencies,delays)
    timeseries=np.fft.irfft(out)
    
    return fit_direction, timeseries

