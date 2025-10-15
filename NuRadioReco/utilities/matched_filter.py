import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft

def matched_filter(data, template, noise_power_spectral_density, t_array, frequencies, n_antennas, sampling_rate, threshold = 0.01, plot = False):
    """
    data = [n_antennas, n_samples]
    template = [n_antennas, n_samples]
    noise_power_spectral_density = [n_antennas, n_samples/2+1]
    t_array = [n_t]
    frequencies = [n_samples/2+1] 
    n_antennas = int
    """
    s = fft.time2freq(data, sampling_rate).flatten()
    h = fft.time2freq(template, sampling_rate).flatten()
    noise_psd = noise_power_spectral_density.flatten()
    frequencies_flattened = np.tile(frequencies, n_antennas)
    n_t = len(t_array)
    output = np.zeros(n_t)

    for i in range(n_t):
        integrand = s * h.conj() / noise_psd * np.exp(1j*2*np.pi*frequencies_flattened*t_array[i])
        output[i] = 4 * np.real(np.sum(integrand[noise_psd > np.max(noise_psd) * threshold**2] * (frequencies_flattened[1] - frequencies_flattened[0])))

    if plot:
        plt.figure(figsize=[20,3])
        plt.plot(t_array, output)
        plt.xlabel("Time [ns]")
        plt.ylabel("Matched filter output")
        plt.tight_layout()
    
    return t_array[np.argmax(output)], np.max(output)

def matched_filter_to_llh(matched_filter_output, data, template, noise_power_spectral_density, frequencies, n_antennas, sampling_rate, threshold = 0.01, fast=False):
    if not fast: 
        s = fft.time2freq(data, sampling_rate)
    h = fft.time2freq(template, sampling_rate)

    # Calculate inverse square of sigma defined in Equation 4.3 in https://arxiv.org/pdf/gr-qc/0509116:
    inv_square_sigma = 0
    for j in range(n_antennas):
        integrand_sigma = h[j,:] * h[j,:].conj() / noise_power_spectral_density[j,:]
        inv_square_sigma += 4 * np.real(np.sum(integrand_sigma[noise_power_spectral_density[j,:] > np.max(noise_power_spectral_density) * threshold**2] * (frequencies[1] - frequencies[0])))
    
    # Calculate rho defined in Equation 4.4 in https://arxiv.org/pdf/gr-qc/0509116 
    # which is the same as q defined in Eqaution 5 in https://arxiv.org/pdf/2106.03718
    s_est = matched_filter_output / inv_square_sigma
    q = s_est * np.sqrt(inv_square_sigma)

    if np.isnan(q): q=0

    # Calculate the first term in Equation 6 in https://arxiv.org/pdf/2106.03718 using Fourier transforms:
    # integrand_term_1 = (abs(s*s.conj())/noise_power_spectral_density)
    # term_1 = 4*np.trapz(integrand_term_1[noise_power_spectral_density[j,:] > np.max(noise_power_spectral_density) * threshold**2], frequencies[np.logical_and(noise_power_spectral_density>0, frequencies>0)])

    term_1 = 0
    for j in range(n_antennas):
        integrand_term_1 = s[j,:] * s[j,:].conj() / noise_power_spectral_density[j,:]
        term_1 += 4 * np.real(np.sum(integrand_term_1[noise_power_spectral_density[j,:] > np.max(noise_power_spectral_density) * threshold**2] * (frequencies[1] - frequencies[0])))

    # Return the log likelihood as defined in Equation 6 in https://arxiv.org/pdf/2106.03718:
    #return - 1/2*term_1 + 1/2 * q**2
    return - 1/2 * term_1 + 1/2 * matched_filter_output**2 / inv_square_sigma