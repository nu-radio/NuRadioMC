import numpy as np
import scipy.signal
from NuRadioReco.utilities import units
from NuRadioReco.utilities import fft
import logging
logger = logging.getLogger('NuRadioReco.fluence_rice_dist_estimator')


def tukey_window(n_samples, relative_taper_width):
	"""
	Generate a Tukey window. Useful to avoid artifacts when performing DFT.

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


def get_noise_fluence_estimators(trace, times, signal_window_mask, spacing_noise_signal=20*units.ns, relative_taper_width=0.142857143, use_median_value=False):
	"""
	Estimate the noise fluence from the trace.

	Parameters
	----------
	trace : np.ndarray
		Trace to estimate the noise fluence from.
	times : np.ndarray
		Time grid for the trace.
	signal_window_mask : np.ndarray
		Boolean mask for the signal window.
	spacing_noise_signal : float
		Spacing between noise windows and signal window. Makes sure no signal leaks into the noise windows.
	relative_taper_width : float
		Width of the taper region for the Tukey window relative to the full window length.
	use_median_value : bool
		If True, the median of the squared spectra of the noise windows is used as estimator. Otherwise, the mean is used.

	Returns
	-------
	np.ndarray
		Estimators for the noise fluence.
	np.ndarray
		Frequencies corresponding to the estimators.
	"""

	dt = times[1] - times[0]
	n_samples_window = sum(signal_window_mask)
	signal_start = times[signal_window_mask][0] - spacing_noise_signal
	signal_stop = times[signal_window_mask][-1] + spacing_noise_signal
	list_ffts_squared = []

	#generate Tukey window
	window = tukey_window(n_samples_window, relative_taper_width)

	#loop over the trace defining noise windows (excluding the signal window)
	noise_start = times[0]
	while noise_start < times[-1]:

		noise_stop = noise_start + n_samples_window * dt
		if noise_stop > times[-1]:
			break

		elif (noise_stop <= signal_start and noise_start < signal_start) or (noise_stop > signal_stop and noise_start >= signal_stop):

			#clipping the noise window (rounding is needed because noise_stop = noise_start + n_samples_window * dt has numerical uncertainties)
			mask_time = np.all([np.round(times, 5) >= np.round(noise_start, 5), np.round(times, 5) < np.round(noise_stop, 5)], axis=0)
			time_trace_clipped = trace[mask_time]

			#applying the Tukey window
			windowed_trace = time_trace_clipped * window

			#calculating the spectrum and frequencies
			frequencies_window = np.fft.rfftfreq(len(windowed_trace), d=dt)
			spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

			list_ffts_squared.append(spectrum_window**2)
			noise_start = noise_stop

		elif noise_stop > signal_start and noise_start <= signal_start:
			noise_start = signal_stop

		else:
			print("Your peak is at zero or negative time...! ")
			break

	list_ffts_squared = np.array(list_ffts_squared, dtype=float)

	if use_median_value:
		#robust estimator in presence of outliers from the noise windows  
		estimators = np.median(list_ffts_squared, axis=0) / 1.405 #from chi2 distribution
	else:
		#it works well in presence of small number of outliers 
		estimators=np.mean(list_ffts_squared, axis=0) 

	return estimators, frequencies_window

def get_signal_fluence_estimators(trace, times, signal_window_mask, noise_estimators, relative_taper_width=0.142857143):
	"""
	Estimate the signal fluence from the trace.

	Parameters
	----------
	trace : np.ndarray
		Trace to estimate the signal fluence from.
	times : np.ndarray
		Time grid for the trace.
	signal_window_mask : np.ndarray
		Boolean mask for the signal window.
	noise_estimators : np.ndarray
		Estimators for the noise fluence.
	relative_taper_width : float
		Width of the taper region for the Tukey window relative to the full window length.

	Returns
	-------
	np.ndarray
		Estimators for the signal fluence.
	np.ndarray
		Variance of the signal fluence estimators.
	"""
	
	dt = times[1] - times[0]
	n_samples_window = sum(signal_window_mask)
	signal_start = times[signal_window_mask][0]
	signal_stop = times[signal_window_mask][-1] + dt

	#generate Tukey window
	window = tukey_window(n_samples_window, relative_taper_width)

	#clipping the signal window around the pulse position
	mask_time = np.all([times >= signal_start, times < signal_stop], axis=0)
	trace_clipped = trace[mask_time]

	#applying the Tukey window 
	windowed_trace = trace_clipped * window 

	#calculating the spectrum and frequencies
	spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

	#signal estimator and variance for each frequency bin 
	signal_estimators = spectrum_window**2 - noise_estimators
	signal_estimators[signal_estimators < 0] = 0
	variances = noise_estimators * (noise_estimators + 2*signal_estimators)

	return signal_estimators, variances