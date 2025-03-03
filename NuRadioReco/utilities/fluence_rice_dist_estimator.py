import numpy as np
import scipy.constants
import scipy.signal
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import fft
import logging
logger = logging.getLogger('NuRadioReco.fluence_rice_dist_estimator')

import matplotlib.pyplot as plt


# def amp_spec(trace, dt_sec=1.e-9):
# 	from scipy.fft import fft,fftfreq
# 	micro = 1e-6
# 	n = int(len(trace))
# 	amp = np.abs(fft(trace))*dt_sec/micro
# 	freq = fftfreq(n,dt_sec)*micro
# 	return np.asarray(amp[:int(n/2)]), np.asarray(freq[:int(n/2)])

#windowing is necessary to avoid artifacts when performing DFT.
def tukey_window(n_samples, relative_taper_width):
	"""
	Generate a Tukey window.

	Parameters
	----------
	n_samples : int
		Number of samples in the window.
	relative_taper_width : float
		Relative width of the taper region. 0 corresponds to a rectangular window, 1 to a Hann window.

	Returns
	-------
	np.ndarray
		Tukey window of length `n_samples`.
	"""
	number_taper_samples = int(np.floor(relative_taper_width * n_samples))
	window = np.ones(n_samples)

	# Rising taper region
	for i in range(number_taper_samples):
		window[i] = 0.5 * (1 - np.cos(np.pi * i / number_taper_samples))

    # Falling taper region
	for i in range(n_samples - number_taper_samples, n_samples):
		reverse_bin = n_samples - i - 1
		window[i] = 0.5 * (1 - np.cos(np.pi * reverse_bin / number_taper_samples))

	return window


def get_noise_fluence_estimators(trace, times, t_peak, f_low=30*units.MHz, f_high=80*units.MHz, spacing_noise_signal=20*units.ns, window_length_tot=140*units.ns, relative_taper_width=0.142857143, use_median_value=True):

	#times = np.arange(0, len(trace)*dt, dt)
	dt = times[1] - times[0]
	signal_start, signal_stop = t_peak - spacing_noise_signal - window_length_tot/2, t_peak + spacing_noise_signal + window_length_tot/2
	list_ffts_squared = []

	# Generate Tukey window
	window = tukey_window(int(window_length_tot / dt), relative_taper_width)

	#loop over the trace defining noise windows (excluding the signal window)
	noise_start = times[0]
	count = 0
	while noise_start < np.max(times):
		
		noise_stop = noise_start + window_length_tot
		if noise_stop > (len(trace) * dt):
			break

		elif (noise_stop <= signal_start and noise_start < signal_start) or (noise_stop > signal_stop and noise_start >= signal_stop):

			#clipping the noise window
			mask_time = np.all([times >= noise_start, times<noise_stop], axis=0) 
			time_trace_clipped = trace[mask_time]
			
			#applying the Tukey window
			if len(time_trace_clipped) != len(window): # This is prbabaly not the best solution, but it will run
				window = tukey_window(len(time_trace_clipped), relative_taper_width)
				print("Window length is not the same as the trace length. This is not optimal.")
			windowed_trace = time_trace_clipped * window

			#calculating the spectrum and frequencies
			frequencies_window = np.fft.rfftfreq(len(windowed_trace), d=dt)
			spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

			#masking the frequencies outside the initial frequency bandwidt
			mask_freq = np.all([frequencies_window >= f_low, frequencies_window <= f_high], axis=0)
			frequencies_window = frequencies_window[mask_freq] 
			spectrum_window = spectrum_window[mask_freq]
			list_ffts_squared.append(spectrum_window**2)
			noise_start = noise_stop

		elif noise_stop > signal_start and noise_start <= signal_start:
			noise_start = signal_stop

		else:
			print("Your peak is at zero or negative time...! ") 
			count += 1
			if count > 1000:
				break # find better solution

	list_ffts_squared = np.array(list_ffts_squared, dtype=float)
	
	if use_median_value:
		#robust estimator in presence of outliers from the noise windows  
		estimators = np.median(list_ffts_squared, axis=0) / 1.405 #from chi2 distribution
	else:
		#it works well in presence of small number of outliers 
		estimators=np.mean(list_ffts_squared, axis=0) 
	
	return estimators, frequencies_window


def get_signal_fluence_estimators(trace, times, t_peak, noise_estimators, f_low=30*units.MHz ,f_high=80*units.MHz, window_length_tot=140*units.ns, relative_taper_width=0.142857143):

	#times = np.arange(0, len(trace)*dt, dt)
	dt = times[1] - times[0]
	signal_start, signal_stop = t_peak - window_length_tot/2, t_peak + window_length_tot/2

	# Generate Tukey window
	window = tukey_window(int(window_length_tot/dt), relative_taper_width)

	#clipping the signal window around the pulse position
	mask_time = np.all([times >= signal_start, times < signal_stop], axis=0) 
	trace_clipped = trace[mask_time]

	#applying the Tukey window 
	windowed_trace = trace_clipped * window 

	#calculating the spectrum and frequencies
	frequencies_window = np.fft.rfftfreq(len(windowed_trace), d=dt)
	spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

	#masking the frequencies outside the initial frequency bandwidth
	mask_freq = np.all([frequencies_window >= f_low, frequencies_window <= f_high], axis=0)
	frequencies_window = frequencies_window[mask_freq] 
	spectrum_window = spectrum_window[mask_freq]

	#signal estimator and variance for each frequency bin 
	signal_estimators = spectrum_window**2 - noise_estimators
	signal_estimators[signal_estimators < 0] = 0
	variances = noise_estimators * (noise_estimators + 2*signal_estimators)
	
	return signal_estimators, variances