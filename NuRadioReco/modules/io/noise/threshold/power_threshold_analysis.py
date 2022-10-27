#!/usr/bin/env python
# coding: utf-8

# # LPDA trigger threshold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter

from NuRadioReco.modules.trigger import powerIntegration
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.rno_g import readRNOGData
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units

import uproot
from astropy.time import Time, TimeDelta

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("generate_current_noise.py"))))
from generate_current_noise import empty_event, get_sim_noise

#disable trigger overridden warning
import logging
logging.getLogger('BaseStation').setLevel(logging.ERROR)

from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

def get_file_names(path, station):
    # Save original directory
    owd = os.getcwd()

    # Change to the data directory
    os.chdir(path)

    # Create a list for file names
    file_names = []

    # For each root-file in the directory
    for file in glob.glob("*.root"):

        # If it belongs to the station
        if file.startswith(f"forced_triggers_station{station}"):

            # Append to file names
            file_names.append(path+file)
            
    # Change back to original directory
    os.chdir(owd)


    return file_names

def get_generated_noise(generator, dataset):
    input_noise = np.random.randn(1, 128)
    input_noise = np.expand_dims(input_noise, axis=-1) 
    noise = generator.predict_on_batch(input_noise)
    noise = noise[:,:,0]
    noise = noise*dataset.std() + dataset.mean()
    return noise[0]

def get_current_noise_scaled(dataset, trace_length):
    '''Function that generates noise used in current simulations'''

    # Get event
    evt = get_sim_noise()
    channel = evt.get_station(11).get_channel(13)
    event = channel.get_trace()[0:trace_length]

    # Scale event
    event = event * np.mean(np.abs(dataset))/np.mean(np.abs(event))

    
    return event

def get_data_event(data, dataset, count):
    '''Fetches an event from the dataset'''
    event = data[count]
    event = event*dataset.std() + dataset.mean()
    
    return event



def threshold(path_to_files, lower, higher):
    '''Function that performs the power trigger analysis'''

    file_names = get_file_names(path_to_files, 24)

    dataset = np.load('../data.npy')
    
    data_preproccessed = np.load("../data_preprocessed_512.npy")


    generator = keras.models.load_model('../kapre_lstm_generator_4')

    # Detector
    det = detector.Detector(json_filename="RNO_G/RNO_season_2021.json", antenna_by_depth=False)

    # RNOG data reader
    reader = readRNOGData.readRNOGData()

    # Power integration trigger (since diode is supposed to be ~ power)
    trigger = powerIntegration.triggerSimulator()
    trigger.begin()

    # trigger path, has a ~11 ns integration and bandpass
    trigger_integration_window = 30 * units.ns
    trigger_passband = [lower*units.MHz, higher*units.MHz]
    bandpass = channelBandPassFilter()

    # test threshold range
    thresholds= np.logspace(-3,-1,200)[::-1]


    results = []

    # which channels to trigger on
    surface_channel = 13 #[12,13,14,15,16,17,18,19,20]

    # dummy_file = "../../shallman/data/rno_g/forced_triggers/inbox/forced_triggers_station24_run117.root"


    count = 0
    
    # Looping through the datafiles like this is unnessecary since we don't use them in the analysis.
    # However, the approach of loading numpy array instead datafiles was found as a bug at the very end of
    # the project and therefore this has not been changed. It does not affect the results but is not 
    # most efficient way of doing this analysis.
    for rf_i, run_file in enumerate(file_names):
        if rf_i > 50:
            break
        reader.begin(run_file)
        for i, event in enumerate(reader.run()):
            count+=1
            # Data
            station = event.get_station(event.get_station_ids()[0])
            if not station.get_trigger('force_trigger').has_triggered():
                continue
            channel = station.get_channel(surface_channel)
#             trace = channel.get_trace()[0:512]
            trace = get_data_event(data_preproccessed, dataset, count)/1000
            channel.set_trace(trace, 3.2*units.GHz)
            
            station.add_channel(channel)
            
            bandpass.run(event, station, det, passband=trigger_passband,
                        filter_type='butter', order=10, rp=None)
        
            triggered_any = False
            triggered_threshold_any = 0
            
            for threshold in thresholds:
                if not triggered_any:
                    trigger.run(event, station, det, threshold=threshold,
                            integration_window=trigger_integration_window,
                            triggered_channels=[surface_channel],
                            trigger_name="power_integration_trigger")
                    tt = station.get_trigger('power_integration_trigger')
                    # set to triggered
                    triggered_any = tt.has_triggered()
                    if tt.has_triggered():
                        triggered_threshold_any = threshold
                else:
                    break
                
            # Generated
            station = event.get_station(event.get_station_ids()[0])
            channel = station.get_channel(surface_channel)
            gen_trace = get_generated_noise(generator, dataset)/1000
            channel.set_trace(gen_trace, 3.2*units.GHz)
            station.add_channel(channel)
            
            bandpass.run(event, station, det, passband=trigger_passband,
                        filter_type='butter', order=10, rp=None)
        
            triggered_any_gen = False
            triggered_threshold_gen = 0

            
            for threshold in thresholds:
                if not triggered_any_gen:
                    trigger.run(event, station, det, threshold=threshold,
                            integration_window=trigger_integration_window,
                            triggered_channels=[surface_channel],
                            trigger_name="power_integration_trigger")
                    tt = station.get_trigger('power_integration_trigger')
                    # set to triggered
                    triggered_any_gen = tt.has_triggered()
                    if tt.has_triggered():
                        triggered_threshold_gen = threshold
                else:
                    break

            
            # Current
            station = event.get_station(event.get_station_ids()[0])
            channel = station.get_channel(surface_channel)
            current_trace = get_current_noise_scaled(dataset, 512)/1000
            channel.set_trace(current_trace, 3.2*units.GHz)
            station.add_channel(channel)
            
            bandpass.run(event, station, det, passband=trigger_passband,
                        filter_type='butter', order=10, rp=None)
        
            triggered_any_current = False
            triggered_threshold_current = 0

            
            for threshold in thresholds:
                if not triggered_any_current:
                    trigger.run(event, station, det, threshold=threshold,
                            integration_window=trigger_integration_window,
                            triggered_channels=[surface_channel],
                            trigger_name="power_integration_trigger")
                    tt = station.get_trigger('power_integration_trigger')
                    # set to triggered
                    triggered_any_current = tt.has_triggered()
                    if tt.has_triggered():
                        triggered_threshold_current = threshold
                else:
                    break
            
            
            print(f"Data event {i}, triggered at threshold {triggered_threshold_any}")
            print(f"Generated event {i}, triggered at threshold {triggered_threshold_gen}")
            print(f"Current event {i}, triggered at threshold {triggered_threshold_current}")
            result = {"station_number": event.get_station_ids()[0],
                "run_number": event.get_run_number(),
                "event_number": event.get_id(),
                "event_i": i,
                "threshold": triggered_threshold_any, "threshold_generator": triggered_threshold_gen,
                "threshold_current": triggered_threshold_current, "triggered": triggered_any}
            results.append(result)
            print("\n")
            
    df = pd.DataFrame(results)
    df.to_hdf("power_trigger_threshold_scan.hdf5", "data")

def analyze_threshold(lower, higher):
    '''Plots the triggered threshold fraction versus trigger threshold for the power_trigger_threshold_scan.hdf5 file'''
    if lower < 1:
        lower = 0
    data = pd.read_hdf("power_trigger_threshold_scan.hdf5")

    thresholds = np.logspace(-3,-1,200)

    n_triggers = np.zeros_like(thresholds)
    n_triggers_gen = np.zeros_like(thresholds)
    n_triggers_current = np.zeros_like(thresholds) 

    for i, thres in enumerate(thresholds):
        n_triggers[i] = np.sum(data.threshold>thres)
        n_triggers_gen[i] = np.sum(data.threshold_generator>thres)
        n_triggers_current[i] = np.sum(data.threshold_current>thres)

    plt.plot(thresholds, n_triggers/len(data.threshold), label = "Data")
    plt.plot(thresholds, n_triggers_gen/len(data.threshold_generator), label = "Generated")
    plt.plot(thresholds, n_triggers_current/len(data.threshold_current), label = "Current")
    plt.xlabel("Trigger threshold")
    plt.ylabel("Triggered event fraction")
    plt.legend()
    plt.title(f"Power trigger analysis with filter ({lower}-{higher} MHz)")
    plt.semilogx()
    plt.show()
    
    # plt.savefig(f"threshold_{lower}_{higher}.png")
    # plt.savefig(f"threshold_kapre_{lower}_{higher}.png")
    

if __name__ == "__main__":
    path_to_files = "../../shallman/data/rno_g/forced_triggers/inbox/"
    lower = 0.0000001
    higher = 1600
    threshold(path_to_files, lower, higher)
    analyze_threshold(lower, higher)
    