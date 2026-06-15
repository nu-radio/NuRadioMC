import sys
from pathlib import Path
import json 
import argparse
from NuRadioReco.utilities import units
import numpy as np
import NuRadioReco.detector.RNO_G.rnog_detector
from NuRadioReco.detector.RNO_G import rnog_detector
import NuRadioReco.modules.RNO_G.dataProviderRNOG_nu
import NuRadioReco.modules.interferometricReconstruction
from NuRadioReco.examples.RNOG.processing import process_event
from NuRadioReco.utilities.framework_utilities import get_averaged_channel_parameter
from NuRadioReco.framework.parameters import (
    eventParametersRNOG as ep, channelParameters as chp, showerParameters as shp,
    particleParameters as pap, generatorAttributes as gta, stationParameters as stp)
import datetime
from NuRadioReco.detector import detector
import pickle
import matplotlib.pyplot as plt
import csv
from NuRadioReco.modules.io import eventReader
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelBandPassFilter
channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelCableDelayAdder.begin()
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
channelBlockOffsetFitter = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
channelBlockOffsetFitter.begin()
import NuRadioReco.modules.RNO_G.channelGlitchDetector
channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()
channelGlitchDetector.begin()

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

import NuRadioReco.modules.channelSignalReconstructor
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
from NuRadioReco.utilities.framework_utilities import get_averaged_channel_parameter
import NuRadioReco.modules.io.eventWriter
#import zarr
import logging

from NuRadioReco.utilities import fft as fft_reco
import scipy
import snr as snr
import rpr as rpr
import hilbert as hilbert
import impulsivity as impulsivity
rpr = rpr.RPR()
snr = snr.SNR()
hilbert = hilbert.Hilbert()
impulsivity = impulsivity.Impulsivity()

csw_info = {
  "PA" : [0,1,2,3],
  "PS" : [0,1,2,3,5,6,7],
  "ALL" : [0,1,2,3,5,6,7,9,10,22,23]
}
channels_to_include = csw_info["ALL"]

def run_num(run):
    nums = []
    for char in run:
        if (char.isdigit() == True):
            nums.append(char)
    num = 0

    for i in range(len(nums)):
        num += float(nums[i])*(10**(len(nums)-i-1))
    return int(num)

def get_event_source(path, det):
    """
    Returns an iterator over events, regardless of input type.
    """
    if path.endswith(".nur"):
        reader = eventReader.eventReader()
        reader.begin(path)

        def generator():
            for evt in reader.run():
                yield evt

        return generator()

    else:
        provider = NuRadioReco.modules.RNO_G.dataProviderRNOG_nu.dataProviderRNOG()
        provider.begin(files=path, det=det)

        def generator():
            for evt, times, readout_times in provider.run():
                yield evt, times, readout_times

        return generator()

parser = argparse.ArgumentParser(description='L2')
#parser.add_argument('--input', type=str, required=True)
parser.add_argument('--chunk-file', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--station', type=int, required=True)
parser.add_argument('--events', type=int, nargs='+', default=None)
parser.add_argument('--time-delay-tables', type=str, required=True)
parser.add_argument('--detector-file', type=str, required=True)
parser.add_argument('--burn-data-source', type=str, default=None)
parser.add_argument('--sim-data-source', type=str, default=None)
parser.add_argument('--burn-sample', type=str, default=None)

args = parser.parse_args()
station_id = args.station
output_file = args.output
events = args.events

with open(args.chunk_file) as f:
    #chunk = json.load(f)
    items = [line.strip() for line in f if line.strip()]

if len(items) == 0:
    Path(output_file).touch()
    sys.exit(0)


burn_sample = None
if args.burn_sample:
    with open(args.burn_sample) as f:
        burn_sample = json.load(f)

is_sim = items and items[0].endswith(".nur")


#chunk_type = chunk["type"]
#run_events = chunk.get("runs", {})
#files = chunk.get("runs", [])

#main_path = f"{args.time_delay_tables}/station{station_id}"
main_path = f"{args.time_delay_tables}"
#main_path = "/data/i3store/users/avijai/tt_ior3_c8_full_st23_0330"
#main_path = "/data/i3store/users/avijai/tt_ior3_c8_full_0301/station11"

def load_maps_npz(path, soln):
    ttcs = {}
    for ch in channels_to_include:
        path_ch = path + "/" + f"st{station_id}_ch{ch}_{soln}_table.npz"
        ttcs[ch] = path_ch
    return ttcs

ttcs_dir = load_maps_npz(main_path, "early_full")
ttcs_refl = load_maps_npz(main_path, "late_full")

print("loaded maps")
"""
detectorpath = args.detector_filei
#detectorpath = "/data/i3store/users/avijai/detector_0330.json"
det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(detector_file = None)
det.update(datetime.datetime(2022, 10, 1))
"""
det = rnog_detector.Detector(
        detector_file=None, log_level=logging.INFO,
        always_query_entire_description=True, select_stations=args.station)
event_time = datetime.datetime(2024, 2, 3)
det.update(event_time)
print("loaded det")
#detector_file = "/data/i3home/avijai/detector_0414_11.json"
#dict_with_temps = {0: 217.19629113753282, 1: 215.20627029636574, 2: 222.61509139414372, 3: 212.90523779386774, 5: 221.08190519458805, 6: 216.96692334930526, 7: 224.054135578712, 9: 223.37765887763643, 10: 226.38776506877818, 22: 220.25936001026346, 23: 222.77790088196727, 4: 117.61594720917961, 8: 118.91603235682096, 11: 116.02361247028553, 21: 120.91470289135322, 12: 377.4768636931619, 13: 350.1865829749677, 14: 379.53681971418484, 15: 379.9537072977946, 16: 349.76426069834565, 17: 378.9143374913414, 18: 378.3784770551233, 19: 348.31582771160225, 20: 378.6600575424777}
#det = rnog_detector.Detector(detector_file = detectorpath, always_query_entire_description=True, select_stations=station_id, over_write_handset_values={"noise_temperature": dict_with_temps})
#event_time = datetime.datetime(2024, 2, 3)
#det.update(event_time)

reco_dir = NuRadioReco.modules.interferometricReconstruction.InterferometricReco(det, station_id, ttcs_dir)
reco_refl = NuRadioReco.modules.interferometricReconstruction.InterferometricReco(det, station_id, ttcs_refl)

csw = NuRadioReco.modules.interferometricReconstruction.CSW(station_id, det)
scr = NuRadioReco.modules.interferometricReconstruction.SurfaceCorr(station_id, det)

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(filename=output_file)
soln_types = ["early", "late"]

if (is_sim == False):
    print("running data")
    for run in items:
        run = f"{args.burn_data_source}/{run}"
        run_id = str(run_num(run.split("/")[-1]))
        if (run_id not in burn_sample):
            continue 
        event_list = set(burn_sample[run_id])
        count = 0
        filename = run 
        run_id = filename.split("/")[-1]
        event_source = get_event_source(filename, det)
        print("got event source")
        for event, trig_times, readout_times in event_source:
            station = event.get_station(station_id)
            time = station.get_station_time()
            triggers = station.get_triggers()
            trigger_names = [key for key in triggers]
        
            if (event.get_id() not in event_list):
                continue

            process_event(event, det)
            avg_snr = get_averaged_channel_parameter(event, chp.SNR, channels_to_include = csw_info["ALL"])
            avg_rpr = get_averaged_channel_parameter(event, chp.root_power_ratio, channels_to_include = csw_info["ALL"]) 
        
            r = 300 #change initial guess for r (spherical) based on event to be reconstructed 
        
            #change ranges and number of points according to event reconstucted and range of travel time maps used 
            #change path to where correlation map plots are saved as needed 
        
            results_dir = reco_dir.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-3000, 300), (0,3000), 250, 250, 180, 360, "plots_0326_out/early", return_reco = True, return_score = True, return_delays = True, return_maps = True)
            results_refl = reco_refl.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-3000, 300), (0,3000), 250, 250, 180, 360, "plots_0326_out/late", return_reco = True, return_score = True, return_delays = True, return_maps = True)
        
        
            max_corrs = np.array([results_dir["maxcorr"], results_refl["maxcorr"]])
            results = [results_dir, results_refl]
            max_index = np.argmax(max_corrs)
            max_results = results[max_index]["maxcorr_coord"]
            max_soln = soln_types[max_index]
            max_corr = np.max(max_corrs)
        
            #reconstruction, coherently summed waveform (CSW) and surface correlation ratio (SCR) calculation 
            print(event.get_id(), max_results, max_corr, "stuff")
            csw_rpr = {}
            csw_snr = {}
            csw_hilbert_snr = {}
            csw_impulsivity = {}
            csw_peak = {}
            csw_power = {}
            for chan_combo in csw_info.keys():
                if (max_soln == "early"):
                    csw_times, csw_values = csw.run(event, station, channels_to_include, results_dir["maps"], results_dir["maxcorr_coord"], results_dir["score"], results_dir["delays"])
                elif (max_soln == "late"):
                    csw_times, csw_values = csw.run(event, station, channels_to_include, results_refl["maps"], results_refl["maxcorr_coord"], results_refl["score"], results_refl["delays"])
            
                csw_snr[chan_combo]  = snr.get_snr_single(csw_times, csw_values)
                csw_rpr[chan_combo] = rpr.get_single_rpr(csw_times, csw_values)
                csw_hilbert_snr[chan_combo] = hilbert.hilbert_snr(csw_values)
                csw_hilbert = np.abs(scipy.signal.hilbert(csw_values))
                csw_peak[chan_combo] = max(csw_hilbert)
                csw_power[chan_combo] = csw_times[np.argmax(np.array(csw_values)**2)]
                csw_impulsivity[chan_combo] = impulsivity.calculate_impulsivity_measures(csw_values)

            #surf_corr_ratio, max_surf_corr, max_r, max_r = scr.run(max_results["reco"], results["maxcorr"])
            if (max_soln == "early"):
                surf_corr_ratio, max_surf_corr, max_r, max_z = scr.run(results_dir["reco"], max_corr)
            elif (max_soln == "late"):
                surf_corr_ratio, max_surf_corr, max_r, max_z = scr.run(results_refl["reco"], max_corr)        
            
            event.add_parameter_type(ep)
            event[ep.run_num] = run_id
            event[ep.avg_snr] = avg_snr
            event[ep.avg_rpr] = avg_rpr
            event[ep.csw_snr] = csw_snr
            event[ep.csw_rpr] = csw_rpr
            event[ep.csw_hilbert_snr] = csw_hilbert_snr
            event[ep.csw_impulsivity] = csw_impulsivity
            event[ep.csw_peak] = csw_peak
            event[ep.csw_power] = csw_power
            event[ep.max_corr_coords] = max_results 
            event[ep.max_corr] = max_corr
            event[ep.surf_corr_ratio] = surf_corr_ratio
            event[ep.max_surf_corr] = max_surf_corr
            event[ep.max_surf_corr_pos] = [(max_r, max_z)]
            event[ep.trigger_type] = trigger_names 
            event[ep.unixtime] = time
            event[ep.energy] = None
            event[ep.trigger_times] = trig_times
            event[ep.readout_times] = readout_times 
            eventWriter.run(event, det=None, mode={'Channels':False, "ElectricFields":False})

    
            print(event.get_id(), run_id, avg_snr, avg_rpr, csw_snr, csw_rpr, csw_hilbert_snr, csw_impulsivity, csw_peak, csw_power, max_results, max_corr, surf_corr_ratio, max_surf_corr, trigger_names, time)
       
        count += 1

if (is_sim == True):
    print("running sim")
    for filename in items:
        count = 0
        #filename = f"{args.sim_data_source}/{filename}"
        run_id = filename.split("/")[-2]
        event_source = get_event_source(filename, det)
        print("got event source")
        for event in event_source:
            station = event.get_station(station_id)
            showers = event.get_sim_showers()
            for shower in showers:
                energy = shower[shp.energy]
            time = station.get_station_time()
            triggers = station.get_triggers()
            trigger_names = [key for key in triggers]

            channelBlockOffsetFitter.run(event, station, det)
            channelGlitchDetector.run(event, station, det)
            channelCableDelayAdder.run(event, station, det, mode='subtract')
            event.set_id(count)

            process_event(event, det)
            avg_snr = get_averaged_channel_parameter(event, chp.SNR, channels_to_include = csw_info["ALL"])
            avg_rpr = get_averaged_channel_parameter(event, chp.root_power_ratio, channels_to_include = csw_info["ALL"])

            r = 300 #change initial guess for r (spherical) based on event to be reconstructed

            #change ranges and number of points according to event reconstucted and range of travel time maps used
            #change path to where correlation map plots are saved as needed

            results_dir = reco_dir.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-3000, 300), (0,3000), 250, 250, 180, 360, "plots_0326_out/early", return_reco = True, return_score = True, return_delays = True, return_maps = True)
            results_refl = reco_refl.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-3000, 300), (0,3000), 250, 250, 180, 360, "plots_0326_out/late", return_reco = True, return_score = True, return_delays = True, return_maps = True)


            max_corrs = np.array([results_dir["maxcorr"], results_refl["maxcorr"]])
            results = [results_dir, results_refl]
            max_index = np.argmax(max_corrs)
            max_results = results[max_index]["maxcorr_coord"]
            max_soln = soln_types[max_index]
            max_corr = np.max(max_corrs)
            print(event.get_id(), max_results, max_corr, "stuff")
            #reconstruction, coherently summed waveform (CSW) and surface correlation ratio (SCR) calculation

            csw_rpr = {}
            csw_snr = {}
            csw_hilbert_snr = {}
            csw_impulsivity = {}
            csw_peak = {}
            csw_power = {}
            for chan_combo in csw_info.keys():
                if (max_soln == "early"):
                    csw_times, csw_values = csw.run(event, station, channels_to_include, results_dir["maps"], results_dir["maxcorr_coord"], results_dir["score"], results_dir["delays"])
                elif (max_soln == "late"):
                    csw_times, csw_values = csw.run(event, station, channels_to_include, results_refl["maps"], results_refl["maxcorr_coord"], results_refl["score"], results_refl["delays"])

                csw_snr[chan_combo]  = snr.get_snr_single(csw_times, csw_values)
                csw_rpr[chan_combo] = rpr.get_single_rpr(csw_times, csw_values)
                csw_hilbert_snr[chan_combo] = hilbert.hilbert_snr(csw_values)
                csw_hilbert = np.abs(scipy.signal.hilbert(csw_values))
                csw_peak[chan_combo] = max(csw_hilbert)
                csw_power[chan_combo] = csw_times[np.argmax(np.array(csw_values)**2)]
                csw_impulsivity[chan_combo] = impulsivity.calculate_impulsivity_measures(csw_values)

            #surf_corr_ratio, max_surf_corr, max_r, max_r = scr.run(max_results["reco"], results["maxcorr"])
            if (max_soln == "early"):
                surf_corr_ratio, max_surf_corr, max_r, max_z = scr.run(results_dir["reco"], max_corr)
            elif (max_soln == "late"):
                surf_corr_ratio, max_surf_corr, max_r, max_z = scr.run(results_refl["reco"], max_corr)
            

            event.add_parameter_type(ep)
            event[ep.run_num] = run_id
            event[ep.avg_snr] = avg_snr
            event[ep.avg_rpr] = avg_rpr
            event[ep.csw_snr] = csw_snr
            event[ep.csw_rpr] = csw_rpr
            event[ep.csw_hilbert_snr] = csw_hilbert_snr
            event[ep.csw_impulsivity] = csw_impulsivity
            event[ep.csw_peak] = csw_peak
            event[ep.csw_power] = csw_power
            event[ep.max_corr_coords] = max_results
            event[ep.max_corr] = max_corr
            event[ep.surf_corr_ratio] = surf_corr_ratio
            event[ep.max_surf_corr] = max_surf_corr
            event[ep.max_surf_corr_pos] = [(max_r, max_z)]
            event[ep.trigger_type] = trigger_names
            event[ep.unixtime] = time
            event[ep.energy] = energy
            event[ep.trigger_times] = None
            event[ep.readout_times] = None

            eventWriter.run(event, det=None, mode={'Channels':False, "ElectricFields":False})


            print(event.get_id(), run_id, avg_snr, avg_rpr, csw_snr, csw_rpr, csw_hilbert_snr, csw_impulsivity, csw_peak, csw_power, max_results, max_corr, surf_corr_ratio, max_surf_corr, trigger_names, time)

            count += 1



#dataProviderRNOG.end()
eventWriter.end()


