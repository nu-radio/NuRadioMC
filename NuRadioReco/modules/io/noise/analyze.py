
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
from sklearn.metrics import mean_squared_error
import similaritymeasures
from sklearn.decomposition import PCA



def mean_std(data, generated_signals):

    # Mean
    print(f"Mean data: {np.mean(data)}")
    print(f"Mean generated: {np.mean(generated_signals)}")
 
    print(f"Std data: {np.std(data)}")
    print(f"Std generated: {np.std(generated_signals)}")
    
def fft_mse(data, generated_signals):
    fft_data = abs(fft.time2freq(data, 3.2*units.GHz))
    fft_gen = abs(fft.time2freq(generated_signals, 3.2*units.GHz))

    mse_gen = mean_squared_error(fft_data, fft_gen)

    print(f"FFT MSE: {mse_gen}")

    return mse_gen

def avg_med_quantile_freq(data, generated_signals):

    ### Average frequency ###
    fig, (ax1, ax2,) = plt.subplots(1, 2)
    fig.set_size_inches(18, 6, forward=True)

     # Calculate trace length
    trace_length = len(data[0])

    # Get frequencies of data
    data_freq = fft.time2freq(data, 3.2*units.GHz)

    # Get frequencies of generated data
    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)

    # Get average frequencies for both
    avg_freq_data = np.mean(abs(data_freq), axis=0)
    avg_freq_generator = np.mean(abs(generator_freq), axis=0)

    ### Frechet distance for average curve ###
    # Generator data
    x_gen = list(range(0,len(avg_freq_generator+1)))
    y_gen = avg_freq_generator
    matrix_gen = np.zeros((len(avg_freq_generator), 2))
    matrix_gen[:, 0] = x_gen
    matrix_gen[:, 1] = y_gen

    # Data
    x_data = list(range(0,len(avg_freq_data+1)))
    y_data = avg_freq_data
    matrix_data = np.zeros((len(avg_freq_data), 2))
    matrix_data[:, 0] = x_data
    matrix_data[:, 1] = y_data

    # Frechet distance
    df_gen = similaritymeasures.frechet_dist(matrix_data, matrix_gen)



    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    ax1.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_data, label =f"Data")
    ax1.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_generator, label =f"Generator")
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_title(f"Average frequencies for data and the generated data\n Fréchet distance: {np.round(df_gen,3)}")
    ax1.legend()


    # Quantile calculation
    gen_15, gen_85 = np.quantile(abs(generator_freq), 0.15, axis = 0), np.quantile(abs(generator_freq), 0.85, axis = 0)
    data_15, data_85 = np.quantile(abs(data_freq), 0.15, axis = 0), np.quantile(abs(data_freq), 0.85, axis = 0)

    # Median freq
    med_freq_data = np.median(abs(data_freq), axis=0)
    med_freq_generator = np.median(abs(generator_freq), axis=0)

    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    ### Frechet distance for median curve ###
    # Generator data
    x_gen = list(range(0,len(med_freq_generator+1)))
    y_gen = med_freq_generator
    matrix_gen = np.zeros((len(med_freq_generator), 2))
    matrix_gen[:, 0] = x_gen
    matrix_gen[:, 1] = y_gen

    # Data
    x_data = list(range(0,len(med_freq_data+1)))
    y_data = med_freq_data
    matrix_data = np.zeros((len(med_freq_data), 2))
    matrix_data[:, 0] = x_data
    matrix_data[:, 1] = y_data

    # Frechet distance
    df_gen = similaritymeasures.frechet_dist(matrix_data, matrix_gen)

    

    # Plotting
    ax2.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_data, label =f"Data")
    ax2.fill_between(dummy_trace.get_frequencies()/units.MHz, data_15, data_85, alpha = 0.5)

    ax2.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_generator, label =f"Generator")
    ax2.fill_between(dummy_trace.get_frequencies()/units.MHz, gen_15, gen_85, alpha = 0.5)


    ax2.set_title(f"Median and 15-85 quantile frequencies for data and the generated data\n Fréchet distance for median: {np.round(df_gen,3)}")
    ax2.set_xlabel("Frequency [MHz]")
    ax2.legend()




    plt.show()

def plot_distributions(data, generated_signals):
    fig, (ax1, ax2,) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6, forward=True)

    # PDF
    ax1.hist(data.ravel(), bins=100, label = "Data", density=True)
    ax1.hist(generated_signals.ravel(), alpha=0.5,  bins=100, label = "Generator", density=True)
    ax1.legend()
    ax1.set_title("PDF of data and generated data")
    ax1.set_xlabel("Amplitude")
    ax1.set_ylabel("Density")

    # Create a PCA with 2 components (2D)
    pca = PCA(n_components=2)

    # Fit it to the data
    pca.fit(data)

    # Transform using PCA
    pca_data_results = pca.transform(data[0:1000])
    pca_gen_results = pca.transform(generated_signals[0:1000])  

    ax2.scatter(pca_data_results[:,0], pca_data_results[:,1],
                c = "blue", alpha = 0.6, label = "Real")
    ax2.scatter(pca_gen_results[:,0], pca_gen_results[:,1], 
                c = "red", alpha = 0.6, label = "Generator")

    ax2.legend()  
    ax2.set_title('PCA for real and generated data')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    
    plt.show()


# Previously used metrics. Most of them are included in the functions above

def plot_traces(data, generated_signals, current_noise):
    """Function plotting 1 trace from the data and number_of_traces from the Generator data in time and frequency domain"""

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6, forward=True)

    # Time domain
    ax1.plot(data[0], label = "Data")
    ax1.plot(generated_signals[0], label = "Generator")
    # ax1.plot(current_noise[0], label = "Current noise")
    ax1.legend()
    ax1.set_title("Time domain")

    # Frequency domain
    ax2.plot(abs(fft.time2freq(data[0], 3.2*units.GHz)), label = "Data")
    ax2.plot(abs(fft.time2freq(generated_signals[0], 3.2*units.GHz)),label = "Generator")
    # ax2.plot(abs(fft.time2freq(current_noise[0], 3.2*units.GHz)),label = "Current noise")
    ax2.legend()
    ax2.set_title("Frequency domain")
    plt.show()

def plot_histograms_time(data, generated_signals, current_noise):
    fig, (ax1, ax2,) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6, forward=True)

    # Histogram
    ax1.hist(data.ravel(), bins=100, label = "Data")
    ax1.hist(generated_signals.ravel(), alpha=0.5,  bins=100, label = "Generator")
    # ax1.hist(current_noise.ravel(), alpha=0.5,  bins=100, label = "Current noise")
    ax1.legend()
    ax1.set_title("Histogram of data and generated data")

    # PDF
    ax2.hist(data.ravel(), bins=100, label = "Data", density=True)
    ax2.hist(generated_signals.ravel(), alpha=0.5,  bins=100, label = "Generator", density=True)
    # ax2.hist(current_noise.ravel(), alpha=0.5,  bins=100, label = "Current noise", density=True)
    ax2.legend()
    ax2.set_title("PDF of data and generated data")
    plt.show()

def avg_frequencies(data, generated_signals, current_noise):
    
    # Calculate trace length
    trace_length = len(data[0])

    # Get frequencies of data
    data_freq = fft.time2freq(data, 3.2*units.GHz)

    # Get frequencies of generated data
    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)

    # Get frequencies of generated data
    current_noise_freq = fft.time2freq(current_noise, 3.2*units.GHz)

    # Get average frequencies for both
    avg_freq_data = np.mean(abs(data_freq), axis=0)
    avg_freq_generator = np.mean(abs(generator_freq), axis=0)
    avg_freq_current_noise = np.mean(abs(current_noise_freq), axis=0)


    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    plt.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_data, label =f"Data")
    plt.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_generator, label =f"Generator")
    # plt.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_current_noise, label =f"Current noise")
    plt.xlabel("Frequency [MHz]")
    plt.title("Average frequencies for data and the generated data")
    plt.legend()
    plt.show()

def quantile_frequencies(data, generated_signals, currrent_noise):


    # Calculate trace length
    trace_length = len(data[0])

    # FFT transform
    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)
    data_freq = fft.time2freq(data, 3.2*units.GHz)
    # current_freq = fft.time2freq(currrent_noise, 3.2*units.GHz)


    # Quantile calculation
    gen_15, gen_85 = np.quantile(abs(generator_freq), 0.15, axis = 0), np.quantile(abs(generator_freq), 0.85, axis = 0)
    data_15, data_85 = np.quantile(abs(data_freq), 0.15, axis = 0), np.quantile(abs(data_freq), 0.85, axis = 0)
    # current_15, current_85 = np.quantile(abs(current_freq), 0.15, axis = 0), np.quantile(abs(current_freq), 0.85, axis = 0)

    # Median freq
    med_freq_data = np.median(abs(data_freq), axis=0)
    med_freq_generator = np.median(abs(generator_freq), axis=0)
    # med_freq_current = np.median(abs(current_freq), axis=0)

    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    # Plotting
    plt.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_data, label =f"Data")
    plt.fill_between(dummy_trace.get_frequencies()/units.MHz, data_15, data_85, alpha = 0.5)

    plt.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_generator, label =f"Generator")
    plt.fill_between(dummy_trace.get_frequencies()/units.MHz, gen_15, gen_85, alpha = 0.5)

    # plt.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_current, label =f"Current noise")
    # plt.fill_between(dummy_trace.get_frequencies()/units.MHz, current_15, current_85, alpha = 0.2)


    plt.title("Median and 15-85 quantile frequencies for data and the generated data")
    plt.xlabel("Frequency [MHz]")
    plt.legend()
    plt.show()

def pca(data, generated_signals, current_noise):

    # Create a PCA with 2 components (2D)
    pca = PCA(n_components=2)

    # Fit it to the data
    pca.fit(data)

    # Transform using PCA
    pca_data_results = pca.transform(data[0:1000])
    pca_gen_results = pca.transform(generated_signals[0:1000])
    pca_current = pca.transform(current_noise[0:1000])

    # Plotting
    f, ax = plt.subplots(1)    

    plt.scatter(pca_data_results[:,0], pca_data_results[:,1],
                c = "blue", alpha = 0.6, label = "Real")
    plt.scatter(pca_gen_results[:,0], pca_gen_results[:,1], 
                c = "red", alpha = 0.6, label = "Generator")


    # plt.scatter(pca_current[:,0], pca_current[:,1], 
    #             c = "green", alpha = 0.6, label = "Current")

    ax.legend()  
    plt.title('PCA for real and generated data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def frdist(data, generated_signals, current_noise):

    # Get frequencies of data
    data_freq = fft.time2freq(data, 3.2*units.GHz)

    # Get frequencies of generated data
    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)

    # Get frequencies of generated data
    current_noise_freq = fft.time2freq(current_noise, 3.2*units.GHz)

    # Get average frequencies for both
    avg_freq_data = np.mean(abs(data_freq), axis=0)
    avg_freq_generator = np.mean(abs(generator_freq), axis=0)
    avg_freq_current_noise = np.mean(abs(current_noise_freq), axis=0)


    # Generator data
    x_gen = list(range(0,len(avg_freq_generator+1)))
    y_gen = avg_freq_generator
    matrix_gen = np.zeros((len(avg_freq_generator), 2))
    matrix_gen[:, 0] = x_gen
    matrix_gen[:, 1] = y_gen

    # Data
    x_data = list(range(0,len(avg_freq_data+1)))
    y_data = avg_freq_data
    matrix_data = np.zeros((len(avg_freq_data), 2))
    matrix_data[:, 0] = x_data
    matrix_data[:, 1] = y_data

    # Current
    x_current = list(range(0,len(avg_freq_current_noise+1)))
    y_current = avg_freq_current_noise
    matrix_current = np.zeros((len(avg_freq_current_noise), 2))
    matrix_current[:, 0] = x_current
    matrix_current[:, 1] = y_current

    df_gen = similaritymeasures.frechet_dist(matrix_data, matrix_gen)
    df_current = similaritymeasures.frechet_dist(matrix_data, matrix_current)

    print(f"Fréchet distance for average frequency: {df_gen}")


    return df_gen
