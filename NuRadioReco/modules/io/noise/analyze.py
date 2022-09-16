from lib2to3.pgen2.tokenize import untokenize
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace

'''File with functions that either prints or plots metrics/characteristics for the data versus the signals generated from the generator'''

def metrics(data, generated_signals):
    print(f"Mean generated: {np.mean(generated_signals)}")
    print(f"Mean data: {np.mean(data)}\n")

    print(f"Std generated: {np.std(generated_signals)}")
    print(f"Std data: {np.std(data)}\n")


def plot_traces(data, generated_signals, number_of_traces):
    """Function plotting 1 trace from the data and number_of_traces from the Generator data in time and frequency domain"""

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6, forward=True)

    # Time domain
    ax1.plot(data[0], label = "Data")
    for i in range(number_of_traces):
        ax1.plot(generated_signals[5*i],alpha=0.2)
        # plt.plot(np.random.randn(trace_length), label = 'Random noise')
    ax1.legend()
    ax1.set_title("Time domain")

    # Frequency domain
    ax2.plot(abs(fft.time2freq(data[0], 3.2*units.GHz)), label = "Data")
    for i in range(number_of_traces):
        ax2.plot(abs(fft.time2freq(generated_signals[5*i], 3.2*units.GHz)),alpha=0.2)
    # plt.plot(np.random.randn(trace_length), label = 'Random noise')
    ax2.legend()
    ax2.set_title("Frequency domain")

def plot_histograms_time(data, generated_signals):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6, forward=True)

    ax1.hist(data.ravel(), bins=100, label = "Data")
    ax1.hist(generated_signals.ravel(), alpha=0.5,  bins=100, label = "Generator")
    ax1.legend()
    ax1.set_title("Histogram of data and generated data")

    ax2.hist(data.ravel(), bins=100, label = "Data", density=True)
    ax2.hist(generated_signals.ravel(), alpha=0.5,  bins=100, label = "Generator", density=True)
    ax2.legend()
    ax2.set_title("PDF of data and generated data")

def avg_frequencies(data, generated_signals):
    
    # Calculate trace length
    trace_length = len(data[0])

    # Get frequencies of data
    data_freq = fft.time2freq(data, 3.2*units.GHz)

    # Get frequencies of generated data
    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)

    # Get average frequencies for both
    avg_freq_data = np.mean(abs(data_freq), axis=0)
    avg_freq_generator = np.mean(abs(generator_freq), axis=0)


    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    plt.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_data, label =f"Data")
    plt.plot(dummy_trace.get_frequencies()/units.MHz,avg_freq_generator, label =f"Generator")
    plt.xlabel("Frequency [MHz]")
    # plt.ylabel("Square root of power per MHZ")
    plt.title("Average frequencies for data and the generated data")
    plt.legend()

    #plt.semilogy()

def quantile_frequencies(data, generated_signals):

    # Calculate trace length
    trace_length = len(data[0])

    generator_freq = fft.time2freq(generated_signals, 3.2*units.GHz)
    data_freq = fft.time2freq(data, 3.2*units.GHz)



    gen_15, gen_85 = np.quantile(abs(generator_freq), 0.15, axis = 0), np.quantile(abs(generator_freq), 0.85, axis = 0)
    data_15, data_85 = np.quantile(abs(data_freq), 0.15, axis = 0), np.quantile(abs(data_freq), 0.85, axis = 0)

    med_freq_data = np.median(abs(data_freq), axis=0)
    med_freq_generator = np.median(abs(generator_freq), axis=0)

    # Create dummy trace to get frequencies
    dummy_trace = base_trace.BaseTrace()
    dummy_trace.set_trace(np.zeros(trace_length), sampling_rate = 3.2*units.GHz)

    # Plotting
    plt.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_data, label =f"Median")
    plt.fill_between(dummy_trace.get_frequencies()/units.MHz, data_15, data_85, label = "15-85 quantile", alpha = 0.5)
    plt.plot(dummy_trace.get_frequencies()/units.MHz,med_freq_generator, label =f"Median")
    plt.fill_between(dummy_trace.get_frequencies()/units.MHz, gen_15, gen_85, label = "15-85 quantile", alpha = 0.5)
    plt.xlabel("Frequency [MHz]")
    plt.legend()




    






