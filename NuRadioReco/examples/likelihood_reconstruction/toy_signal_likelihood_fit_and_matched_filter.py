
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units, fft, signal_processing, likelihood_calculator, trace_minimizer, matched_filter
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder

channelGenericNoiseAdder = channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

n_datasets = 1000
n_antennas = 2
n_samples = 1024
sampling_rate = 1.6 * units.GHz
t_array = np.arange(n_samples) * 1/sampling_rate
frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)

min_freq = 50 * units.MHz
max_freq = 500 * units.MHz
noise_amplitude = 1 * units.mV
filter = signal_processing.get_filter_response(frequencies, [min_freq, max_freq], "butter", 8)

# Generate noise:
noise_traces = np.zeros([n_datasets, n_antennas,  n_samples])
for i_data in range(n_datasets):
    for i_antenna in range(n_antennas):
        noise_traces[i_data, i_antenna, :] = channelGenericNoiseAdder.bandlimited_noise_from_spectrum(
            n_samples, sampling_rate, filter, amplitude=noise_amplitude, type='rayleigh')


def signal_model(parameters):
    """
    Simple toy pulsed signal model. Oscillation with gaussian envelope. Same signal in both antennas.
    """
    amplitude, osc_freq, width, t0 = parameters
    signal = amplitude * np.sin(2*np.pi * osc_freq * t_array) * np.exp(-(t_array - t0)**2/width**2)
    signal_fft = fft.time2freq(signal, sampling_rate)
    signal_fft *= filter
    signal = fft.freq2time(signal_fft, sampling_rate)
    return np.stack([signal for i in range(n_antennas)])

# Make true signal:
amplitude_true = 6*units.mV
osc_freq_true = 100*units.MHz
width_true = 5*units.ns
t0_true = 200*units.ns
signal_true = signal_model([amplitude_true, osc_freq_true, width_true, t0_true])

# Add signal to all noise datasets:
data_traces = noise_traces + signal_true[None, :, :]

# Plot one dataset:
fig, ax = plt.subplots(2, 1, figsize=[8,5])
i_noise = 0
for i_antenna in range(n_antennas):
    ax[i_antenna].plot(t_array, data_traces[i_noise, i_antenna], label="Signal + Noise")
    ax[i_antenna].plot(t_array, signal_true[i_antenna], label="Signal")
    ax[i_antenna].set_xlim(0, max(t_array))
    if i_antenna == 0: ax[i_antenna].legend()
    if i_antenna == n_antennas-1: ax[i_antenna].set_xlabel("Time [ns]")
    ax[i_antenna].set_ylabel("Voltage [V]")
    ax[i_antenna].set_title("Antenna "+str(i_antenna))
plt.tight_layout()
plt.show()


# Initialize likelihood calculator:
likelihood_calculator = likelihood_calculator.LikelihoodCalculator(n_antennas, n_samples, sampling_rate)
likelihood_calculator.initialize_with_spectra(abs(filter), noise_amplitude)

# Calculate likelihood for true signal for all datasets:
minus_two_llh_array = likelihood_calculator.calculate_minus_two_delta_llh(data_traces, signal_true)

# Plot distribution alongside chi2 distribution:
likelihood_calculator.plot_llh_distribution(data_traces, signal=signal_true)
plt.show()

factor = 1.1
### Fit signal to data trace ###
minimizer = trace_minimizer.TraceMinimizer(
    signal_function = signal_model,
    objective_function = likelihood_calculator.calculate_minus_two_delta_llh,
    parameters_initial = [8*units.mV, 105*units.MHz, 5.5*units.ns, t0_true+0.05], # assuming that we have good guesses for the paramters
    parameters_bounds = [[0, 10*units.mV], [10*units.MHz, 10000*units.MHz], [1*units.ns, 100*units.ns], [0, max(t_array)]]
)
minimizer.set_scaling(np.array([1/units.mV, 1/units.mHz, 1/units.ns, 1/units.ns]))
m = minimizer.run_minimization(data_traces[i_data], method="minuit")

print("Minus two delta LLH (true):", likelihood_calculator.calculate_minus_two_delta_llh(data_traces[i_data], signal_true))
print("Minus two delta LLH (fit):", minimizer.result)
print("The fitted minus two delta LLH should be slightly smaller that the minus two delta LLH of the true signal if the fit was succesful.")
print("True parameters:", [amplitude_true, osc_freq_true, width_true, t0_true])
print("Fitted paramters:", minimizer.parameters)
print("The fitted paramters should be close to the true parameters.")

# Plot fitted signal and data:
signal_fit = signal_model(minimizer.parameters)
fig, ax = plt.subplots(2, 1, figsize=[8,5])
i_noise = 0
for i_antenna in range(n_antennas):
    ax[i_antenna].plot(t_array, data_traces[i_noise, i_antenna], label="Signal + Noise")
    ax[i_antenna].plot(t_array, signal_true[i_antenna], label="Signal")
    ax[i_antenna].plot(t_array, signal_fit[i_antenna], ls="--", label="Fitted signal")
    ax[i_antenna].set_xlim(0, max(t_array))
    if i_antenna == 0: ax[i_antenna].legend()
    if i_antenna == n_antennas-1: ax[i_antenna].set_xlabel("Time [ns]")
    ax[i_antenna].set_ylabel("Voltage [V]")
    ax[i_antenna].set_title("Antenna "+str(i_antenna))
plt.tight_layout()
plt.show()


### Matched filter to find best time offset ###
signal_template = signal_model([amplitude_true, osc_freq_true, width_true, 0])
mf = matched_filter.MatchedFilter(n_samples, sampling_rate, n_antennas)
mf.set_noise_psd_from_spectra(abs(filter), noise_amplitude)
mf.set_data(data_traces[i_data])
mf.set_template(signal_template)

# Perform matched filter search:
t_best, mf_output = mf.matched_filter_search(np.arange(0, max(t_array), 0.2/sampling_rate)) # Search using 5 times finer grid than sampling
snr = mf.calculate_matched_filter_SNR()
llh = mf.calculate_matched_filter_delta_log_likelihood()
ampl = mf.calculate_matched_filter_amplitude_estimate()

print("\nMatched Filter:")
print("Best time shift:", t_best)
print("Matched filter SNR:", snr)
print("Matched filter minus two delta LLH:", -2*llh)
print("Template amplitude factor estimate:", ampl)
