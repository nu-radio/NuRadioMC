from __future__ import print_function 

import numpy as np
from scipy import signal
from NuRadioReco.utilities import units
import scipy.signal
from NuRadioReco.detector import filterresponse
import NuRadioReco.framework.sim_station


class channelBandPassFilter:
    """
    Band pass filters the channels using different band-pass filters.
    """

    def begin(self):
        pass

    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular', order=2):
        """
        Run the filter

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        passband: list
            passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
        filter_type: string
            'rectangular': perfect straight line filter
            'butter': butterworth filter from scipy
            'butterabs': absolute of butterworth filter from scipy
            or any filter that is implemented in NuRadioReco.detector.filterresponse. In this case the
            passband parameter is ignored
            or 'FIR <type> <parameter>' - see below for FIR filter options
        order: int (optional, default 2)
            for a butterworth filter: specifies the order of the filter
        
        Added Jan-07-2018 by robert.lahmann@fau.de:
        FIR filter:
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        for window types; some windows need additional parameters; Default window is hamming
        (does not need parameters). Examples:
        filter_type='FIR' : use hamming window
        filter_type='FIR hamming': same as above
        filter_type='FIR kaiser 10' : Use Kaiser window with beta parameter 10
        filter_type='FIR kaiser' : Use Kaiser window with default beta parameter 6
        
        In principal, window names are just passed on to signal.firwin(), but if parameters
        are required, then these cases must be explicitly implemented in the code below.
        
        The for main filter types can be implemented: 
        LP: passband[0]=None, passband[1]  = f_cut
        HP: passband[0]=f_cut, passband[1] = None
        BP: passband[0]=f_cut_low, passband[1] = f_cut_high
        BS: passband[0]=f_cut_high, passband[1] = f_cut_low (i.e. passband[0] > passband[1])
        
        """
        
        for channel in station.iter_channels():
            if isinstance(station, NuRadioReco.framework.sim_station.SimStation):
                for sub_channel in channel:
                    self.__apply_filter(sub_channel, passband, filter_type, order, True)
            else:
                self.__apply_filter(channel, passband, filter_type, order, False)
            
    def __apply_filter(self, channel, passband, filter_type, order, is_efield=False):
        frequencies = channel.get_frequencies()
        trace_fft = channel.get_frequency_spectrum()
        sample_rate = channel.get_sampling_rate()
        
        # for FIR filters, it is easier to set the trace rather than the FFT to apply the
        # filter response. Eventually, I (Robert Lahmann) might figure out a way to set the
        # FFT (as is done for the IIR filters corresponding to analog filters) but for now
        # I am setting a variable to decide whether the FFT or time trace shall be used to 
        # set the trace. 

        isFIR = False
        
        if(filter_type == 'rectangular'):
            if is_efield:
                for polarization in trace_fft:
                    polarization[np.where(frequencies < passband[0])] = 0.
                    polarization[np.where(frequencies > passband[1])] = 0.
            else:
                trace_fft[np.where(frequencies < passband[0])] = 0.
                trace_fft[np.where(frequencies > passband[1])] = 0.
        elif(filter_type == 'butter'):
            mask = frequencies > 0
            b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies[mask])
            if is_efield:
                for polarization in trace_fft:
                    polarization[mask] *= h
            else:
                trace_fft[mask] *= h
        elif(filter_type == 'butterabs'):
            mask = frequencies > 0
            b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies[mask])
            if is_efield:
                for polarization in trace_fft:
                    polarization[mask] *= h
            else:
                trace_fft[mask] *= np.abs(h)
        elif(filter_type.find('FIR')>=0):
            #print('This is a FIR filter')
            firarray = filter_type.split()
            if (len(firarray)==1):
                wtype = 'hamming'
                #print('hamming window')
            else:
                wtype = firarray[1]
                if wtype.find('kaiser')>=0:
                    if len(firarray)>2:
                        beta = float(firarray[2])
                    else:
                        beta = 6.0
                    wtype = ('kaiser',beta)
            #print('window type: ', wtype)
            Nfir = order+1
            if (passband[0]==None):
                # this is a low pass filter
                pass_zero=True
                fcut=passband[1]
            elif (passband[1]==None or passband[1]/sample_rate>=0.5):
                # this is a high pass filter
                pass_zero=False
                fcut=passband[0]
            elif (passband[1]>passband[0]):
                # this is a bandpass filter
                pass_zero=False
                fcut=passband
            elif(passband[0]>passband[1]):
                # this is a bandstop filter
                pass_zero=True
                fcut=[passband[1],passband[0]]
                #print('bandstop with fcut = ',fcut)
            else:
                # something went wrong!!
                print("Error, could not define filter type")
            #print('fcut = ',fcut)
            taps = signal.firwin(Nfir, fcut, window=wtype,scale=False,pass_zero=pass_zero,fs=sample_rate)
            wfilt, hfilt = signal.freqz(taps, worN=len(frequencies))
            
            if ((Nfir//2)*2==Nfir):
                print("odd filter order, rolling is off by T_s/2")
                
            ndelay = int(0.5 * (Nfir-1))
            trace_fir = signal.lfilter(taps, 1.0, channel.get_trace())
            #print('len(trace_fir)',len(trace_fir))
            trace_fir = np.roll(trace_fir,-ndelay)
#             channel.set_trace(trace_fir, sample_rate)
#             #channel.set_trace(trace_fir, sample_rate)
#             #trace_fft = hfilt
#             trace_fft = channel.get_frequency_spectrum()
            isFIR = True
        else:
            mask = frequencies > 0
            filt = filterresponse.get_filter_response(frequencies[mask], filter_type)
            if is_efield:
                for polarization in trace_fft:
                    polarization[mask] *= filt
            else:
                trace_fft[mask] *= filt
        if isFIR:
            channel.set_trace(trace_fir, sample_rate)
            #print('set trace for fir')
        else:
            channel.set_frequency_spectrum(trace_fft, sample_rate)

    def end(self):
        pass
