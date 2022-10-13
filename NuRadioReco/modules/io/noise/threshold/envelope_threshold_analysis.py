#!/usr/bin/env python
# coding: utf-8

# # LPDA trigger threshold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter

from NuRadioReco.modules.trigger import powerIntegration
from NuRadioReco.modules.trigger import envelopeTrigger
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.io.rno_g import readRNOGData
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units

import uproot
from astropy.time import Time, TimeDelta

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

def threshold(path_to_files, lower, higher):

    file_names = get_file_names(path_to_files, 24)

    dataset = np.load('../data.npy')

    generator = keras.models.load_model('../kapre_lstm_generator_4')

    # Detector
    det = detector.Detector(json_filename="detector.json", antenna_by_depth=False)
    det.update(time=Time.now())

    # RNOG data reader
    reader = readRNOGData.readRNOGData()

    # Envelope trigger
    trigger = envelopeTrigger.triggerSimulator()
    trigger.begin()

    # trigger path, has a ~11 ns integration and bandpass
    trigger_integration_window = 11 * units.ns
    trigger_passband = [lower*units.MHz, higher*units.MHz]
    bandpass = channelBandPassFilter()

    # test threshold range
    thresholds= np.logspace(-3,1,200)[::-1]


    results = []

    # which channels to trigger on
    surface_channel = 13 #[12,13,14,15,16,17,18,19,20]

    # dummy_file = "../../shallman/data/rno_g/forced_triggers/inbox/forced_triggers_station24_run117.root"



    for rf_i, run_file in enumerate(file_names):
        if rf_i > 10:
            break
        reader.begin(run_file)
        for i, event in enumerate(reader.run()):
            
            # Data
            station = event.get_station(event.get_station_ids()[0])
            if not station.get_trigger('force_trigger').has_triggered():
                continue
            channel = station.get_channel(surface_channel)
            trace = channel.get_trace()[0:512]
            channel.set_trace(trace, 3.2*units.GHz)
            
            station.add_channel(channel)
            
            # bandpass.run(event, station, det, passband=trigger_passband,
            #             filter_type='butter', order=10, rp=None)
        
            triggered_any = False
            triggered_threshold_any = 0
            
            for threshold in thresholds:
                if not triggered_any:
                    trigger.run(event, station, det, passband = [lower*units.MHz, higher*units.MHz], order = 10, threshold = threshold, 
                    coinc_window = 11 * units.ns, number_coincidences=1, triggered_channels=[surface_channel], trigger_name='envelope_trigger')

                    
                    tt = station.get_trigger('envelope_trigger')
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
            
            # bandpass.run(event, station, det, passband=trigger_passband,
            #             filter_type='butter', order=10, rp=None)
        
            triggered_any_gen = False
            triggered_threshold_gen = 0

            
            for threshold in thresholds:
                if not triggered_any_gen:
                    trigger.run(event, station, det, passband = [lower*units.MHz, higher*units.MHz], order = 10, threshold = threshold, 
                    coinc_window = 11 * units.ns, number_coincidences=1, triggered_channels=[surface_channel], trigger_name='envelope_trigger')
                    tt = station.get_trigger('envelope_trigger')
                    # set to triggered
                    triggered_any_gen = tt.has_triggered()
                    if tt.has_triggered():
                        triggered_threshold_gen = threshold
                else:
                    break
            
            
            
            print(f"Data event {i}, triggered at threshold {triggered_threshold_any}")
            print(f"Generated event {i}, triggered at threshold {triggered_threshold_gen}")
            result = {"station_number": event.get_station_ids()[0],
                "run_number": event.get_run_number(),
                "event_number": event.get_id(),
                "event_i": i,
                "threshold": triggered_threshold_any, "threshold_generator": triggered_threshold_gen,
                "triggered": triggered_any}
            results.append(result)
            
    df = pd.DataFrame(results)
    df.to_hdf("envelope_trigger_threshold_scan.hdf5", "data")

def analyze_threshold(lower, higher):
    if lower < 1:
        lower = 0
    data = pd.read_hdf("envelope_trigger_threshold_scan.hdf5")

    thresholds = np.logspace(-3,1,200)

    n_triggers = np.zeros_like(thresholds)
    n_triggers_gen = np.zeros_like(thresholds)

    for i, thres in enumerate(thresholds):
        n_triggers[i] = np.sum(data.threshold>thres)
        n_triggers_gen[i] = np.sum(data.threshold_generator>thres)

    plt.plot(thresholds, n_triggers/len(data.threshold), label = "Data")
    plt.plot(thresholds, n_triggers_gen/len(data.threshold_generator), label = "Generated")
    plt.xlabel("Trigger threshold")
    plt.ylabel("Triggered event fraction")
    plt.legend()
    plt.title(f"Envelope trigger analysis with filter ({lower}-{higher} MHz)")
    plt.semilogx()
    plt.show()
    
    # plt.savefig(f"threshold_{lower}_{higher}.png")
    # plt.savefig(f"envelope_threshold_kapre_{lower}_{higher}.png")
    

if __name__ == "__main__":
    path_to_files = "../../shallman/data/rno_g/forced_triggers/inbox/"
    lower = 0.0000001
    higher = 1600
    threshold(path_to_files, lower, higher)
    analyze_threshold(lower, higher)
    