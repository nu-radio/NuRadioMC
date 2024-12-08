import csv
import numpy as np
from scipy import interpolate
import waveform_utilities as wu
def load_arasim_phase_response_as_spline():

    """
    Load the AraSim phase response of the system to use for de-dispersion.

    Returns
    -------
    the_phase_spline : interp1d
        A spline of the unwrapped phases as a function of frequency.
        The frequency units are Ghz.
        And the phase is unwrapped, in units of radians.
    """ 

    resp = {}
    channels = [0,1,2,3,5,6,7,9,10,22,23]
    path = "resp_100PeV.csv"

    for ch in channels:
        resp[ch] = []
    
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, len(row)):
                resp[channels[i-1]].append(np.angle(complex((row[i]))))

    
    freq = np.arange(50, 1000) / 1000
    
    phase_splines = {}
    for ch in resp.keys():
        phs_unwrapped = np.unwrap(resp[ch]) # unwrapped phase in radians

        the_phase_spline = interpolate.Akima1DInterpolator(
            freq, phs_unwrapped,
            method="makima",
        )
        the_phase_spline.extrapolate = False
        phase_splines[ch] = the_phase_spline
    
    return phase_splines
    

def eval_splined_phases(phase_spline, freqs_to_evaluate):
    """"
    Just a little helper function.
    This is necessary because the Akima Interpolator will return NaN
    when called out of the range of support, but we'd rather it gave zeros.
    """
    phase_splines_corr = {}
    for ch in phase_splines.keys():
        phase_spline = phase_splines[ch]
        these_phases = phase_spline(freqs_to_evaluate)
        these_phases = np.nan_to_num(these_phases) # convert nans to zeros
        phase_splines_corr[ch] = these_phases
    
    return phase_splines_corr

def dedisperse_wave(
        times, # in nanoseconds,
        volts, # in volts,
        channel, #channel number, 
        phase_spline # the  phase spline
        ):

    """
    Fetch a specific calibrated event

    Parameters
    ----------
    times : np.ndarray(dtype=np.float64)
        A numpy array of floats containing the times for the trace,
        in nanoseconds.
    volts : np.ndarray(dtype=np.float64)
        A numpy array of floats containing the voltages for the trace,
        in volts.
    phase_spline : interp1d
        A spline of the unwrapped phase (in radians) vs frequency (in GHz).
        When the function was first written, it was meant to utilize
        the output of `dedisperse.load_arasim_phase_response_as_spline`.
        So check that function for an example of how to do it.

    Returns
    -------
    dedispersed_wave : np.ndarray(dtype=np.float64)
        The dedispersed wave

    """

    if len(times) != len(volts):
        raise Exception("The time and volts arrays are mismatched in length. Abort.")

    # first thing to do is get the frequency domain representation of the trace


    freqs, spectrum = wu.time2freq(times, volts)

    # interpolate the *unwrapped phases* to the correct frequency base
    phased_interpolated = eval_splined_phases(phase_spline, freqs)[channel]

    # conver these into a complex number
    phased_rewrapped = np.exp((0 + 1j)*phased_interpolated)

    # do complex division to do the dedispersion
    spectrum /= phased_rewrapped

    # back to the time domain
    times, volts = wu.freq2time(times, spectrum)
    return times, volts
