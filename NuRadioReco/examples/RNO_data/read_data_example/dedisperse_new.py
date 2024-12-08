import csv
import numpy as np
from scipy import interpolate
import waveform_utilities as wu

class Dedisperse:

    def __init__(self):

        self.path = "resp_100PeV.csv"
        self.channels = [0,1,2,3,5,6,7,9,10,22,23]
        self.freq = np.arange(50, 1001) / 1000 #units GHz

    def load_phase_response_as_spline(self):
        resp = {}
        for ch in self.channels:
            resp[ch] = []

        with open(self.path) as f:
            reader = csv.reader(f)
            for row in reader:
                for i in range(1, len(row)):
                    resp[self.channels[i-1]].append(np.angle(complex((row[i]))))
    

        phase_splines = {}
        
        for ch in self.channels:
            phs_unwrapped = np.unwrap(resp[ch]) # unwrapped phase in radians

            the_phase_spline = interpolate.Akima1DInterpolator(
                self.freq, phs_unwrapped,
                method="makima",
            )
            the_phase_spline.extrapolate = False
            phase_splines[ch] = the_phase_spline

        return phase_splines

    def eval_splined_phases(self, freqs_to_evaluate):
        phase_splines_corr = {}
        phase_splines = self.load_phase_response_as_spline()

        for ch in self.channels:
            phase_spline = phase_splines[ch]
            these_phases = phase_spline(freqs_to_evaluate)
            these_phases = np.nan_to_num(these_phases) # convert nans to zeros
            phase_splines_corr[ch] = these_phases

        return phase_splines_corr

    def run(self, event, station):
        all_times = {}
        all_volts = {}

        for channel in station.iter_channels():
            if (channel.get_id() in self.channels):
                volts = channel.get_trace()
                times = channel.get_times()

                if len(times) != len(volts):
                    raise Exception("The time and volts arrays are mismatched in length. Abort.")

                # first thing to do is get the frequency domain representation of the trace

                freqs, spectrum = wu.time2freq(times, volts)

                # interpolate the *unwrapped phases* to the correct frequency base
                phased_interpolated = self.eval_splined_phases(freqs)[channel.get_id()]

                # conver these into a complex number
                phased_rewrapped = np.exp((0 + 1j)*phased_interpolated)

                # do complex division to do the dedispersion
                spectrum /= phased_rewrapped

                # back to the time domain
                times, volts = wu.freq2time(times, spectrum)
                all_times[channel.get_id()] = times
                all_volts[channel.get_id()] = volts 
        
        return all_times, all_volts


