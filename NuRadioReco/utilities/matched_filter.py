import numpy as np
import matplotlib.pyplot as plt

def matched_filter(data, template, noise_power_spectral_density, t_array, frequencies, n_antennas, plot = False):
    """
    data = [n_antennas, n_samples]
    template = [n_antennas, n_samples]
    noise_power_spectral_density = [n_antennas, n_samples/2+1]
    t_array = [n_t]
    frequencies = [n_samples/2+1] 
    n_antennas = int
    """
    s = np.fft.rfft(data, axis=1).flatten()
    h = np.fft.rfft(template, axis=1).flatten()
    noise_psd = noise_power_spectral_density.flatten()
    frequencies_flattened = np.tile(frequencies, n_antennas)
    n_t = len(t_array)
    output = np.zeros(n_t)
    
    #     plt.figure()
    #     plt.plot(np.sqrt(noise_psd / frequencies_flattened[1] / 2))
    #     plt.plot(np.abs(s))
    #     plt.plot(np.abs(h))
    #     plt.plot([0, len(frequencies_flattened)], [np.sqrt(np.max(noise_power_spectral_density)/100**2 / frequencies_flattened[1] / 2), np.sqrt(np.max(noise_power_spectral_density)/100**2 / frequencies_flattened[1] / 2)], "k--")
    #     plt.show()

    for i in range(n_t):
        integrand = s * h.conj() / noise_psd * np.exp(1j*2*np.pi*frequencies_flattened*t_array[i])
        output[i] = 4 * np.real(np.sum(integrand[noise_psd>np.max(noise_psd)/100**2] * (frequencies_flattened[1] - frequencies_flattened[0])))

    if plot:
        plt.figure(figsize=[20,3])
        plt.plot(t_array, output)
        plt.xlabel("Time [ns]")
        plt.ylabel("Matched filter output")
        plt.tight_layout()
        plt.show()
    
    return t_array[np.argmax(output)], np.max(output)

def matched_filter_to_llh(matched_filter_output, data, template, noise_power_spectral_density, frequencies, n_antennas):
    #s = np.fft.rfft(trace)
    h = np.fft.rfft(template)
    
    # Calculate sigma defined in Equation 4.3 in https://arxiv.org/pdf/gr-qc/0509116:
    sigma = 0
    for j in range(n_antennas):
        integrand_sigma = h[j,:] * h[j,:].conj() / noise_power_spectral_density[j,:]
        sigma += 4 * np.real(np.sum(integrand_sigma[noise_power_spectral_density[j,:] > np.max(noise_power_spectral_density)/100**2] * (frequencies[1] - frequencies[0])))
    
    # Calculate rho defined in Equation 4.4 in https://arxiv.org/pdf/gr-qc/0509116 
    # which is the same as q defined in Eqaution 5 in https://arxiv.org/pdf/2106.03718
    q = matched_filter_output/np.sqrt(sigma)

    if np.isnan(q): q=0

    # Calculate the first term in Equation 6 in https://arxiv.org/pdf/2106.03718 using Fourier transforms:
    #integrand_term_1 = (abs(s*s.conj())/noise_power_spectral_density)
    #term_1 = 0#4*np.trapz(integrand_term_1[np.logical_and(noise_power_spectral_density>0, frequencies>0)], frequencies[np.logical_and(noise_power_spectral_density>0, frequencies>0)])

    # Return the log likelihood as defined in Equation 6 in https://arxiv.org/pdf/2106.03718:
    return q**2 #-1/2*term_1 + 1/2 * q**2