import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelAddCableDelay, channelBandPassFilter, sphericalWaveFitter
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G import readRNOGDataMattak
from datetime import datetime
import NuRadioReco.detector.detector
import astropy.time
import logging
from NuRadioReco.utilities import fft
from os import listdir
from os.path import isfile, join
from scipy.signal import correlate
from NuRadioReco.utilities.fft import time2freq
from NuRadioReco.utilities.fft import freq2time
from scipy.interpolate import interp1d
import pandas as pd
import json
from scipy.interpolate import griddata

AddCableDelay = channelAddCableDelay.channelAddCableDelay()

def read_root_file(station_number, fiber, run, selectors=[], sampling_rate=3.2):
    """
    Reads the root file of the station id, fiber and run number
    
    Parameters
    ----------
    station_number : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    run : int
        Run number. Example: 1785
    selectors : list of functions or lambdas, optional
        All functions or lambdas that return a boolean to decide if an event should be selected or not, default []
    sampling_rate : float, optional
        Sampling rate in GHz, default 3.2

    Returns
    ----------
    reader : NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData
        a reader object that can be used to read each selected event in the run from the root file
    """

    reader = readRNOGDataMattak.readRNOGData()
    file = '/home/sanyukta/astropart/data/st' + str(station_number) + '/fiber' + str(fiber) + '/st' + str(station_number) + '_run' + str(run) + '.root'
    reader.begin(file, selectors=selectors, overwrite_sampling_rate=sampling_rate)
    return reader

def read_raw_traces(detector, station_id, fiber, run, cal_pulser_threshold=None, sampling_rate=3.2):
    """
    Reads raw voltage traces of all selected events in a station, fiber and run number
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    run : int
        Run number. Example: 1785
    cal_pulser_threshold : float, optional
        Maximum interval between events sysclk to be selected in the list of events, default None
    sampling_rate : float, optional
        Sampling rate in GHz, default 3.2

    Returns
    ----------
    events : list of int
        a list of event indices
    times : dict of int to list of lists of floats
        a dictionary of channel ids mapped to a list of lists of times in which every list represents an event
    volts : dict of int to list of lists of floats
        a dictionary of channel ids mapped to a list of lists of volts in which every list represents an event
    """
    
    selectors = [lambda event_info: (event_info.sysclk - event_info.sysclkLastPPS[0]) % (2**32) <= cal_pulser_threshold]
    reader = read_root_file(station_id, fiber, run, selectors=selectors, sampling_rate=sampling_rate)
    times = {}
    volts = {}
    events = []
    channel_ids = detector.get_channel_ids(station_id)
    for ch in channel_ids:
        times[ch] = []
        volts[ch] = []
    for index, event in enumerate(reader.run()):
        station = event.get_station(station_id)
        AddCableDelay.run(event, station, detector, mode='subtract')
        events.append(index)
        for ch in channel_ids:
            channel = station.get_channel(ch)
            times[ch].append(channel.get_times())
            volts[ch].append(channel.get_trace())
    return events, times, volts

def get_vector_from_receiver(detector, station_id, fiber, rx, tx='cal_pulser'):
    """
    Gets the vector pointing from the receiver to the transmitter
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    rx : int
        Receiver channel id. Example: 3
    tx : int, optional
        Transmitter channel id, default 'cal_pulser' to signify cal pulser being the default transmitter

    Returns
    ----------
    theta_rec : float
        polar angle in radians between the vector and the z-axis
    s : float
        Euclidean distance between transmitter and receiver in metres
    """
    
    if tx == 'cal_pulser':        
        cal_pulser = detector.get_device(station_id, fiber)
        x_tx, y_tx, z_tx = cal_pulser['ant_position_x'], cal_pulser['ant_position_y'], cal_pulser['ant_position_z']
    else:
        transmitter_channel = detector.get_channel(station_id, tx)
        x_tx, y_tx, z_tx = transmitter_channel['ant_position_x'], transmitter_channel['ant_position_y'], transmitter_channel['ant_position_z']
    receiver_channel = detector.get_channel(station_id, rx)
    x_rx, y_rx, z_rx = receiver_channel['ant_position_x'], receiver_channel['ant_position_y'], receiver_channel['ant_position_z']
    theta_rec = np.arctan(np.sqrt((x_tx - x_rx)**2 + (y_tx - y_rx)**2)/(z_tx - z_rx + 1e-8))
    if theta_rec < 0:
        theta_rec = theta_rec + np.pi
    s = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2 + (z_rx - z_tx)**2)
    return theta_rec, s

def correct_for_gain(detector, station_id, fiber, zenith_data_path, volts, sampling_rate=3.2*units.GHz):
    """
    Correct voltage traces for gain
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    zenith_data_path : str
        Filepath of zenith to gain mapping. File should have column names like Zenith,Gain(50MHz),Gain(100MHz),...Phase(50MHz),Phase(100MHz),...
    volts: dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts in which every list represents an event
    sampling_rate : float, optional
        Sampling rate in GHz, default 3.2

    Returns
    ----------
    volts_corr : dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts corrected for gain based on polar angle in which every list represents an event
    """
    
    volts_corr = {}
    channel_ids = detector.get_channel_ids(station_id)
    gain_zenith_df = pd.read_csv(zenith_data_path, sep=",")
    gain_melted = pd.melt(gain_zenith_df, id_vars=['Zenith'], value_vars=[col for col in gain_zenith_df.columns if 'Gain' in col], var_name='Frequency', value_name='Gain')
    gain_melted['Zenith'] = [theta*np.pi/180 for theta in gain_melted['Zenith']]
    gain_melted['Frequency'] = [int(freq[5:-4])/1000 for freq in gain_melted['Frequency']]
    points = np.array([(row['Zenith'], row['Frequency']) for _, row in gain_melted.iterrows()])
    values = np.array(gain_melted['Gain'])
    d_frequencies = [np.min(gain_melted['Frequency']), np.max(gain_melted['Frequency'])]
    for rx_channel in channel_ids:
        exact_azimuth, _ = get_vector_from_receiver(detector, station_id, fiber, rx_channel)
        # d_gains = gains_df.loc[[col for col in gains_df.index if 'Gain' in col]].to_list()
        # f = interp1d(d_frequencies, d_gains)
        volts_corr[rx_channel] = []
        for voltage_trace in volts[rx_channel]:
            tr = NuRadioReco.framework.base_trace.BaseTrace()
            tr.set_trace(np.zeros_like(voltage_trace), sampling_rate=sampling_rate)
            frequencies = tr.get_frequencies()
            xy_pairs = np.array([(exact_azimuth, frequency) for frequency in frequencies])
            gains_db = griddata(points, values, xy_pairs, fill_value=0, method='cubic')
            gains_ratio = np.power(10, gains_db/20)
            spectrum = time2freq(voltage_trace, sampling_rate)
            spectrum = np.divide(spectrum, gains_ratio)
            volts_corr[rx_channel].append(freq2time(spectrum, sampling_rate))
    return volts_corr

def correct_for_position(detector, station_id, fiber, raw_volts):
    """
    Correct voltage traces for position
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    raw_volts: dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts in which every list represents an event

    Returns
    ----------
    volts_corr : dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts corrected for position based on distance to transmitter in which every list represents an event
    """

    volts_corr = {}
    channel_ids = detector.get_channel_ids(station_id)
    for ch in channel_ids:
        theta, r = get_vector_from_receiver(detector, station_id, fiber, ch)
        sine = np.sin(theta)
        volts_corr[ch] = np.array(raw_volts[ch])*r/(sine*sine + 1e-8)
    return volts_corr

def get_magnification_factor(detector, station_id, fiber, density, rx_channel, tx='cal_pulser'):
    """
    Calculates magnification factor based on refractive index by depth of the medium
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    density : pd.DataFrame
        Pandas DataFrame with depth and refractive_index as columns
    rx_channel: int
        Channel ID of the receiver channel for which magnification factor is to be calculated

    Returns
    ----------
    M : float
        magnification factor for the channel
    """
    
    if tx == 'cal_pulser':
        cal_pulser = detector.get_device(station_id, fiber)
        x_tx, y_tx, z_tx = cal_pulser['ant_position_x'], cal_pulser['ant_position_y'], cal_pulser['ant_position_z']
    else:
        transmitter_channel = detector.get_channel(station_id, tx)
        x_tx, y_tx, z_tx = transmitter_channel['ant_position_x'], transmitter_channel['ant_position_y'], transmitter_channel['ant_position_z']
    receiver_channel = detector.get_channel(station_id, rx_channel)
    x_rx, y_rx, z_rx = receiver_channel['ant_position_x'], receiver_channel['ant_position_y'], receiver_channel['ant_position_z']    
    theta_rec = np.arctan(np.sqrt((x_tx - x_rx)**2 + (y_tx - y_rx)**2)/(z_tx - z_rx + 1e-8))
    if theta_rec < 0:
        theta_rec = theta_rec + np.pi
    s = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2 + (z_rx - z_tx)**2)
    theta_L = theta_rec
    theta_L_del = np.arctan(np.sqrt((x_tx - x_rx)**2 + (y_tx - y_rx)**2)/(z_tx - (z_rx - 0.01) + 1e-8))
    if theta_L_del < 0:
        theta_L_del = theta_L_del + np.pi
    dtheta_L_dz = (theta_L - theta_L_del)/0.01
    depths = density['depth'].tolist()
    refractive_index = density['refractive_index'].tolist()
    depths.insert(0, 0)
    refractive_index.insert(0, 1)
    f = interp1d(np.array(depths), np.array(refractive_index))
    n_L, n_rec  = f(np.abs(z_tx)), f(np.abs(z_rx))
    sin_theta_rec = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2)/s
    M = np.sqrt(s/(sin_theta_rec + 1e-8) * np.abs(dtheta_L_dz) * n_L/(n_rec + 1e-8))
    if M <= 1e-6:
        M = n_L/n_rec
    return M

def correct_for_magnification(detector, station_id, fiber, density, volts, sampling_rate=3.2*units.GHz):
    """
    Correct voltage traces for magnification due to refractive index
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    fiber : int
        Fiber number. Example: 0
    density : pd.DataFrame
        Pandas DataFrame with depth and refractive_index as columns
    volts: dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts in which every list represents an event
    sampling_rate : float, optional
        Sampling rate in GHz, default 3.2

    Returns
    ----------
    volts_corr : dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts corrected for magnification due to refractive index in which every list represents an event
    """
    
    volts_corr = {}
    channel_ids = detector.get_channel_ids(station_id)
    for rx_channel in channel_ids:
        M = get_magnification_factor(detector, station_id, fiber, density, rx_channel)
        print(rx_channel, M)
        volts_corr[rx_channel] = []
        for voltage_trace in volts[rx_channel]:
            volts_corr[rx_channel].append(voltage_trace/(M))
    return volts_corr


def band_pass_filter(detector, station_id, volts, lower_freq, higher_freq, sampling_rate=3.2*units.GHz):
    """
    Filter out a frequency range from voltage traces of all events
    
    Parameters
    ----------
    detector : NuRadioReco.detector.detector.Detector
        A detector object to find distances to receivers and apply cable delay    
    station_id : int
        Station id. Example: 11
    volts: dict of int to list of lists
        A dictionary of channel ids mapped to a list of lists of volts in which every list represents an event
    lower_freq : float
        Lowest frequency in GHz below which all frequencies should be removed
    higher_freq : float
        Highest frequency in GHz above which all frequencies should be removed
    sampling_rate : float, optional
        Sampling rate in GHz, default 3.2

    Returns
    ----------
    volts_noise_removed : dict of int to list of lists
        a dictionary of channel ids mapped to a list of lists of volts from which noise is removed in which every list represents an event
    """
    
    volts_noise_removed = {}
    channel_ids = detector.get_channel_ids(station_id)
    for ch in channel_ids:
        volts_noise_removed[ch] = []
        for voltage_trace in volts[ch]:
            tr = NuRadioReco.framework.base_trace.BaseTrace()
            tr.set_trace(np.zeros_like(voltage_trace), sampling_rate=sampling_rate)
            frequencies = tr.get_frequencies()
            spectrum = time2freq(voltage_trace, sampling_rate)
            for i in np.where(frequencies < lower_freq)[0]:
                spectrum[i] = 0
            for i in np.where(frequencies > higher_freq)[0]:
                spectrum[i] = 0
            volts_noise_removed[ch].append(freq2time(spectrum, sampling_rate))
    return volts_noise_removed

def get_SNR(noise_volts, signal_volts, factor=1):
    """
    SNR calculation with max voltage of each event and rms averaged over all the events; rms is found using moving variance method
    
    Parameters
    ----------
    noise_volts : dict of int to list of lists
        A dictionary of channel ids mapped to a list of lists of raw voltage traces for rms calculation in which every list represents an event
    signal_volts : dict of int to list of lists
        A dictionary of channel ids mapped to a list of lists of corrected voltage traces in which every list represents an event
    factor : float, optional
        If the noise volts are higher than factor*moving variance at time t then t is the pulse start, increase factor to make pulse start more pronounced, default value 1
6
    Returns
    ----------
    SNR : dict of int to list
        a dictionary of channel ids mapped to a list of SNRs in which every value is the SNR of an event
    """    

    n_events = len(noise_volts[0])
    max_volt = {}
    sum_rms = {}
    for ch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        max_volt[ch] = []
        sum_rms[ch] = 0
    for ch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        waveforms = noise_volts[ch]
        waveforms_s = signal_volts[ch]
        for index, waveform in enumerate(waveforms):
            waveform_mean = waveform - np.mean(waveform) #dc offset
            noise_rms = None
            for i in range(2, len(waveform_mean)):
                if np.abs(waveform_mean[i]) > np.mean(waveform_mean[0:i]) + factor*np.std(waveform_mean[0:i]):
                    noise_rms = np.sqrt(np.mean(waveform_mean[0:i]**2))
                    break
            if noise_rms is None:
                noise_rms = np.sqrt(np.mean(waveform_mean**2))
            this_max = np.nanmax(waveforms_s[index])
            max_volt[ch].append(this_max)
            sum_rms[ch] = sum_rms[ch] + noise_rms
    SNR = {}
    for ch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        SNR[ch] = max_volt[ch]/(sum_rms[ch]/(n_events) + 1e-6)
    return SNR

def align_voltage_traces(time_values, voltage_traces):
    """
    Within an event, align voltage traces by time
    
    Parameters
    ----------
    time_values : list of lists
        A list of lists of times in which every list represents an event
    voltage_traces : list of lists
        A list of lists of volts in which every list represents an event

    Returns
    ----------
    aligned_traces : np.array
        A 2D array of aligned volts SNRs in which axis=0 is event and axis=1 is time
    """    

    # Calculate the cross-correlation between the first trace and the others
    reference_trace = voltage_traces[0]
    aligned_traces = [reference_trace]

    for i in range(1, len(voltage_traces)):
        cross_corr = correlate(reference_trace, voltage_traces[i], mode='full')
        shift = np.argmax(cross_corr) - len(reference_trace) + 1
        aligned_trace = np.roll(voltage_traces[i], shift)
        aligned_traces.append(aligned_trace)

    # Convert the list of aligned traces to a numpy array
    aligned_traces = np.array(aligned_traces)
    return aligned_traces

def avg_voltage_traces(voltage_traces):
    average_voltage = np.mean(voltage_traces, axis=0)
    return average_voltage