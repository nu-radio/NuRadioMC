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

#define json file to take positions from
distance_params_df = pd.DataFrame(list(json.load(open('/home/sanyukta/astropart/data/json/RNO_season_2023.json'))['channels'].values()))

def get_channel_positions(station_number):
    """
    To find the position of each channel in a given year (using json file)
    Input parameters: 
    -station number (should be a valid station in the json file)
    Returns:
    - 3 arrays of x, y, z coordinates of each channel
    """
    x, y, z = {}, {}, {}
    for ch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        x[ch] = distance_params_df.loc[(distance_params_df['station_id'] == station_number) & (distance_params_df['channel_id'] == ch), 'ant_position_x'].iloc[0]
        y[ch] = distance_params_df.loc[(distance_params_df['station_id'] == station_number) & (distance_params_df['channel_id'] == ch), 'ant_position_y'].iloc[0]
        z[ch] = distance_params_df.loc[(distance_params_df['station_id'] == station_number) & (distance_params_df['channel_id'] == ch), 'ant_position_z'].iloc[0]
    return x, y, z

def correct_for_position(x, y, z, ref, raw_volts):
    """
    Given the ref channel (pulsing channel), corrects for distance and azimuthal attenuation in the signal
    Input Parameters:
    - x, y, z: arrays of x, y and z coordinate of channels (obtained from get_channel_positions)
    - ref: ref channel (cal pulser)
    - raw volts: voltage traces of channels
    Returns:
    -corrected voltage trances for each channel
    """
    volts_corr = {}
    for ch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
        x_val = np.abs(x[ch]-x[ref])
        y_val = np.abs(y[ch]-y[ref])
        z_val = np.abs(z[ch]-z[ref])
        r = np.sqrt(x_val*x_val + y_val*y_val + z_val*z_val + 1e-6)
        sine = np.sqrt(x_val*x_val + y_val*y_val + 1e-6)/r
        volts_corr[ch] = np.array(raw_volts[ch])*r/(sine*sine)
    return volts_corr

"""
Example: 
det = detector.Detector(json_filename = 'RNO_G/RNO_season_2023.json')
det.update(datetime.now())
station_number = 11
fiber_number = 0
run_number = 1785
ref_ch = 21
factor = 1
events, times, volts_raw = read_raw_traces(station_number, fiber_number, run_number, det)
x, y, z = get_channel_positions(station_number)
volts_corr_for_pos = correct_for_position(x, y, z, ref_ch, volts_raw)
"""
