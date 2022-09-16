import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
# from sklearn.preprocessing import MinMaxScaler


def l1_scores(frequency_spectrum, averages=5, spacing=1, min_power=40**2):
    """Calculate a windowed l1 score over a frequency spectrum by sliding over frequencies
    
    Parameters
    ----------
    frequency_spectrum: 1D array
        the frequency spectrum of a trace
    averages: int
        number of frequency values on each side of the frequency in question to average over.
    spacing: int
        number of frequency values on each side of the frequency in question to ignore.
    min_power: float
        set frequencies with power below min_power to that value

    Returns
    ----------
    l1_scores: 1D array
        the l1_scores for the frequency spectrum
    """
    
    n_samples = len(frequency_spectrum)
    first = averages+spacing
    last = n_samples-averages-spacing
    # use the power
    fsq = abs(frequency_spectrum)**2
    # put minimum
    fsq[fsq<(min_power)] = min_power
    
    # zero padding before
    l1s = list(np.zeros(averages+spacing))
    
    for i in range(first, last):
        #print(i, len(fsq[i-spacing-averages:i-spacing]), len(fsq[i+spacing:i+spacing+averages]))
        denom_sum_left = np.sum(fsq[i-spacing-averages:i-spacing])
        denom_sum_right = np.sum(fsq[i+spacing:i+spacing+averages])
        denom = (denom_sum_left + denom_sum_right)/(2*averages)
        l1 = fsq[i]/denom
        l1s.append(l1)
        #break
        
    # zero padding after
    l1s += list(np.zeros(averages+spacing))
    
    return np.array(l1s)

def vectorized_l1s(frequency_spectrum, averages=5, spacing=1, min_power=40**2):
    """Calculate l1_scores for N events, frequency spectrum is expected to be of shape (Nevents, Nfrequencies)"""
    return np.apply_along_axis(l1_scores, axis=1, arr=frequency_spectrum, averages=averages, spacing=spacing, min_power=min_power)

def l1_preprocessing(data, averages=10, spacing=1, min_power = 50**2, low_freq=80, high_freq=700, cut_off = 20):
    """
    Function that applies L1 score preprocessing on data
    See l1_scores and vectorized_l1s for details on L1 score.
    """
    # Calculate the frequency spectrum of each trace in the data
    freq_data = fft.time2freq(data, 3.2*units.GHz)

    # Pick the frequencies were we want to apply the L1 score
    test_trace = base_trace.BaseTrace()
    test_trace.set_trace(np.zeros(len(data[0])), 3.2*units.GHz)
    frequencies = test_trace.get_frequencies()
    sel = np.argwhere((frequencies>low_freq*units.MHz)&(frequencies<high_freq*units.MHz))
    freq_data = freq_data[:,sel]

    # Apply the L1 score function on the frequency data
    l1_array = vectorized_l1s(freq_data, averages=averages, spacing=spacing, min_power=min_power)

    # Pick the indices of the traces which have a maximum L1 greater than cut_off
    indices = [idx for idx,l1 in enumerate(np.amax(l1_array, axis=1)) if l1 > cut_off]

    #### Plotting ####
    # # Plot histogram of the max L1 score to find a good cutoff for 
    # plt.hist(np.amax(l1_array, axis=1), bins=500)
    # plt.title("L1 score histogram of the traces in the data")
    # plt.xlim([0,40])
    # plt.semilogy()

    # np.shape(np.amax(l1_array, axis=1)), np.shape(l1_array)
    # plt.show()

    # for idx in indices:
    #     plt.plot(abs(fft.time2freq(data[idx], 3.2*units.GHz)))
    #     #     plt.plot(l1_array(idx))
    #     plt.show()
    #### Plotting ####

    # Delete the indices found above from data
    pre_processed_data = np.delete(data, indices, axis=0)

    return pre_processed_data

def rms_preprocessing(data, sigma=5):

    """Removes the traces that have RMS that is sigma amount standard deviation from the mean RMS"""

    # Calculate the rms of the data
    rms_array = np.std(data, axis=1)

    # Calculate the mean and std of the array
    data_mean, data_std = rms_array.mean(), rms_array.std()

    # Define cut-off
    cut_off = data_std * sigma

    # Define upper and lower limit based on cutoff
    lower, upper = data_mean - cut_off, data_mean + cut_off

    # Find indices of the trace we want to remove
    indices = []
    for i, rms in enumerate(rms_array):
        if rms < lower or rms > upper:
            indices.append(i)

    # Remove traces with the indices      
    pre_processed_data = np.delete(data, indices, axis=0)

    return pre_processed_data

# def minmax_scaling(data):

#     # Create the MinMaxScaler:
#     scaler = MinMaxScaler()
    
#     # Fit the scaler to the data
#     scaler.fit(data)

#     # Scale the data
#     scaled_data = scaler.transform(data)

#     # Return both the scaled data and scaler so you can inverse scale at afterwards
#     return scaled_data, scaler

def normalize(data):
    '''Normalizes the data to 0 mean and unit variance'''

    # Calculate mean and std 
    mean, std = data.mean(), data.std()

    # Subtract the mean and divide with the std
    normalize = (data - mean)/std

    return normalize
        

def preprocess(data):
    '''Function using all the above preprocessing techniques. Returns the data and saves it to a numpy array'''

    # Shuffle
    np.random.shuffle(data)

    # RMS
    rms = rms_preprocessing(data)

    # L1
    l1 = l1_preprocessing(rms)

    # Remove DC-offset
    fft_traces = fft.time2freq(l1, 3.2*units.GHz)
    # no_offset = fft_traces[:,1:len(fft_traces[0])]
    fft_traces[:,0] = 0
    data_no_offset = fft.freq2time(fft_traces, 3.2*units.GHz)

    # Shorten the trace
    short = data_no_offset[:,0:512]
    # short = data_no_offset

    # Normalization
    normalized = normalize(short)

    # Set data to the normailzed data
    data = normalize

    # Save data to file
    np.save("data_preprocessed",data)

    return data





if __name__ == "__main__":
    data = np.load('data.npy')
    data = preprocess(data)
    plt.plot(data[0])