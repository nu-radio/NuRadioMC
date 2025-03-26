"""
This file calculates the SNR curves for a given phased array configuration.
Neutrinos specified by the input file should come from a fixed arrival direction
if the SNR curve for that particular direction is required. It is convenient
that the array sees the neutrino at the Cherenkov angle, so that the signals
are clearer. The signals are then rescaled to have the appropriate SNR and
the phased trigger is run to decide whether the event is selected.

python T02RunSNR.py input_neutrino_file.hdf5 proposalcompact_50m_1.5GHz.json
config.yaml output_NuRadioMC_file.hdf5 output_SNR_file.hdf5 output_NuRadioReco_file.nur(optional)

The output_SNR_file.hdf5 contains the information on the SNR curves stored in
hdf5 format. The fields in this file are:
    - 'total_events', the total number of events that have been used
    - 'SNRs', a list with the SNR

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of tnhe phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""
from __future__ import absolute_import, division, print_function
from astropy.time import Time

import os
import datetime as dt
import argparse
import json
import logging
import copy
import numpy as np
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.efieldToVoltageConverter
from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import triggerSimulator as phasedArrayTrigger
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector import detector
from scipy import constants
import matplotlib.pyplot as plt

from NuRadioReco.modules.trigger.highLowThreshold import triggerSimulator as highLowTrigger
from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import triggerSimulator as powerTrigger
from NuRadioReco.modules.phasedarray.digitalBeamformedEnvelopeTrigger import triggerSimulator as envelopeTrigger

#pulses=np.load('../avg_pulses_0.npy')

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogHardwareResponse.begin(trigger_channels=[0,1,2,3])
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.ERROR)
rnogADCResponse.begin(clock_offset=0.0, adc_output="counts")

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--detectordescription', type=str, help='path to file containing the detector description', default='../RNO_single_station_only_PA.json')
parser.add_argument('station_id', type=int, help='station_id', default=0)
parser.add_argument('--trigger', type=str, help='trigger type', default="power")
parser.add_argument('--upsampling_factor', type=int, help='upsampling factor', default=4)
parser.add_argument('--window', type=int, help='power integration window in units of unsamping_factor*1 samples', default=24)
parser.add_argument('--step', type=int, help='power trigger step in units of upsampling_factor*1 samples', default=4)
parser.add_argument('--beam_number', type=int, help='', default=12)

parser.add_argument('inputfilename', type=str, help='path to NuRadioMC input event list')
parser.add_argument('--config', type=str, default='config.yaml', help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str, help='hdf5 output filename')
parser.add_argument('outputSNR', type=str, help='outputfilename for the snr files')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None, help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()
det_file = args.detectordescription

det_defaults = {
    "trigger_adc_sampling_frequency": 0.472,
    "trigger_adc_nbits": 8,
    "trigger_adc_noise_count": 5,
    "trigger_adc_min_voltage": -1,
    "trigger_adc_max_voltage": 1,
}


pa_channels = [0, 1, 2, 3]
main_low_angle = np.deg2rad(-60)
main_high_angle = np.deg2rad(60)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), args.beam_number))

# Get the station description from the station id.
if args.station_id>0:
    station_id=args.station_id
    print("Getting measured, database, detector description")
    det = rnog_detector.Detector(
        detector_file=None, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=[station_id],
        over_write_handset_values=det_defaults)

    det.update(dt.datetime(2023, 8, 3))
    measured_response=True
else:
    print("Using place holder station description")
    station_id=11
    det = detector.Detector(source='json',json_filename=det_file)
    det.update(dt.datetime(2023, 8, 3))
    measured_response=False

# Setup the trigger.
if args.trigger == "power":
    trigger = powerTrigger()
    trig_kwargs = triggerBoardResponse.powerTriggerKwargs
    trig_kwargs["window"] = args.window
    trig_kwargs["step"] = args.step
    trig_kwargs["upsampling_factor"] = args.upsampling_factor
    trig_kwargs["phasing_angles"] = phasing_angles
    pattern = f"{args.trigger}_{args.upsampling_factor}x_{args.window}win_{args.step}step"
    print(pattern)

elif args.trigger == "envelope":
    trigger = envelopeTrigger()
    trig_kwargs = triggerBoardResponse.envelopeTriggerKwargs
    trig_kwargs["upsampling_factor"] = args.upsampling_factor
    trig_kwargs["phasing_angles"] = phasing_angles
    pattern = f"{args.trigger}_{args.upsampling_factor}x"
    print(pattern)

elif args.trigger == "highlow":
    trigger = highLowTrigger()
    trig_kwargs=triggerBoardResponse.highLowTriggerKwargs
    pattern = f"{args.trigger}"
    print(pattern)

else:
    raise ValueError("Provide valid trigger type: power, envelope, highlow")



# SNR scan things
N = 51
SNRs = (np.linspace(0.1, 8.0, N))
SNRtriggered = np.zeros(N)

# Get the noise temperature for the manual noise adder.
min_freq = 0.0 * units.MHz
max_freq = 236.0 * units.MHz
fff = np.linspace(min_freq, max_freq, 10000)

four_filters_highres={}
if args.station_id==0:
    filt_highres=rnogHardwareResponse.get_filter(fff,11,0,det,sim_to_data=True,is_trigger=True)
    for i in range(4):
        four_filters_highres[i]=filt_highres
else:
    for i in range(4):
        four_filters_highres[i]=det.get_signal_chain_response(station_id=station_id, channel_id=i, trigger=True)(fff)


noise_temp=300
bandwidth={}
Vrms_ratio = {}
amplitude = {}
Vrms = 1
per_channel_vrms=[]
for i in range(4):
    integrated_channel_response = np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
    rel_channel_response=np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
    bandwidth[i]=integrated_channel_response
    Vrms_ratio[i] = np.sqrt(rel_channel_response / (max_freq - min_freq))    
    chan_vrms=(noise_temp * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
    per_channel_vrms.append(chan_vrms)
    amplitude[i] = chan_vrms / Vrms_ratio[i]

new_sampling_rate = np.round(472.0 * units.MHz,6)


def count_events():
    count_events.events += 1


count_events.events = 0


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        rnogHardwareResponse.run(evt,station,det,sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        # Store the trace manually, and resample down to FLOWER to get pk-pk.
        Vpps = np.zeros(4)
        filtered_signal_traces = {}
        for iChan, channel in enumerate(station.iter_trigger_channels()):
            if channel.get_id()>3: break

            original_frequency=channel.get_sampling_rate()
            original_length=len(channel.get_trace())
            #print("orig",original_frequency)
            copied=copy.deepcopy(channel)
            copied.resample(new_sampling_rate)
            trace=copied.get_trace()
            Vpps[iChan] = np.max(trace) - np.min(trace)

            filtered_signal_traces[channel.get_id()] = channel.get_trace()


        raw_noise_traces={}
        has_triggered = True

        while(has_triggered):  # search for noise traces that don't set off a trigger

            # loop over all channels (i.e. antennas) of the station
            for channel in station.iter_channels():
                if channel.get_id()>3: break

                trace = np.zeros(original_length)
                channel.set_trace(trace, sampling_rate=channel.get_sampling_rate())

            # Adding noise AFTER the SNR calculation
            channelGenericNoiseAdder.run(evt, station, det, amplitude=amplitude, excluded_channels=np.arange(4,24,1),
                                         min_freq=min_freq, max_freq=max_freq, type='rayleigh')

            # filter noise traces
            rnogHardwareResponse.run(evt,station,det,temp=293.15,sim_to_data=True)

            # If these don't trigger, store the noise traces to be added with signal later.
            for channel in station.iter_trigger_channels():
                if channel.get_id()>3: break
                raw_noise_traces[channel.get_id()]=channel.get_trace()

            # Run gain normalization and digitzation.
            ch_rms, noise_gains = rnogADCResponse.run(evt,station,det,trigger_channels=pa_channels,
                                                       apply_adc_gain=True, digitize_trace=True, vrms=per_channel_vrms)

            #Calculate all trigger thresholds.
            power_vrms=0
            threshold_high={}
            threshold_low={}
            for i in pa_channels:
                power_vrms+=ch_rms[i]**2
                threshold_high[i]=np.rint(ch_rms[i]*3.8)
                threshold_low[i]=-np.rint(ch_rms[i]*3.8)

            voltage_rms=np.sqrt(power_vrms)

            # Pick the right threshold to pass and round to an int.
            if args.trigger=="power":
                trigger_threshold=np.rint(power_vrms*9.01)
            elif args.trigger=="envelope":
                trigger_threshold=np.rint(voltage_rms*6.7)
            else: trigger_threshold=None

            # Finally run the "ambiguous" trigger.
            has_triggered=trigger.run(
                    evt,
                    station,
                    det,
                    Vrms=None,
                    threshold=trigger_threshold,
                    threshold_high=threshold_high,
                    threshold_low=threshold_high,
                    trigger_channels=pa_channels,
                    trigger_name=f"{args.trigger}",
                    **trig_kwargs
                )
            if(has_triggered):
                print('Trigger on noise... ')

        # Calculate factor to set specific SNR
        factor = np.sort(per_channel_vrms/((Vpps/2)))[1] # use second highest SNR to match other studies.
        mult_factors = factor * SNRs

        # Loop over all SNR's
        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            # Loop over channels and add together signal traces and noise traces.
            for channel in station.iter_trigger_channels():
                if channel.get_id()>3: break

                trace = filtered_signal_traces[channel.get_id()][:] * factor
                noise = raw_noise_traces[channel.get_id()]

                # Length matching for sanity. (This might be able to be removed)
                if len(trace)>len(noise):
                    channel.set_trace(trace[0:len(noise)]+noise,sampling_rate=original_frequency)
                elif len(noise)>len(trace):
                    channel.set_trace(trace+noise[0:len(trace)],sampling_rate=original_frequency)
                else:
                    channel.set_trace(trace + noise, sampling_rate=original_frequency)

            # Gain normalization (with gain values already calculated on noise) and digitiztion.
            ch_rms, gains = rnogADCResponse.run(evt, station, det, trigger_channels=pa_channels, vrms=per_channel_vrms,
                                                 apply_adc_gain=True, digitize_trace=True, gain_values=noise_gains)

            # Thresholds again (pre computed in the noise loop, can be removed).
            power_vrms=0
            threshold_high={}
            threshold_low={}
            for i in pa_channels:
                power_vrms+=ch_rms[i]**2
                threshold_high[i]=np.rint(ch_rms[i]*3.8)
                threshold_low[i]=-np.rint(ch_rms[i]*3.8)

            voltage_rms=np.sqrt(power_vrms)

            if args.trigger=="power":
                trigger_threshold=np.rint(power_vrms*9.01)
            elif args.trigger=="envelope":
                trigger_threshold=np.rint(voltage_rms*6.7)
            else: trigger_threshold=None

            # And the trigger.
            has_triggered=trigger.run(
                    evt,
                    station,
                    det,
                    Vrms=None,
                    threshold=trigger_threshold,
                    threshold_high=threshold_high,
                    threshold_low=threshold_high,
                    trigger_channels=pa_channels,
                    trigger_name=f"{args.trigger}_{iSNR}",
                    **trig_kwargs
                )
            
            
            if(has_triggered):
                SNRtriggered[iSNR] += 1

        count_events()

        if(count_events.events % 10 == 0):
            # Save every ten triggers
            output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}
            outputfile = args.outputSNR
            with open(outputfile, 'w+') as fout:
                json.dump(output, fout, sort_keys=True, indent=4)

# Kinda dumb directory checks and file checks.
if not os.path.exists(f"data/output/station{args.station_id}"):
    os.mkdir(f"data/output/station{args.station_id}")
if not os.path.exists(f"data/output/station{args.station_id}/boresight"):
    os.mkdir(f"data/output/station{args.station_id}/boresight")
if os.path.exists(args.outputfilename):
    os.remove(args.outputfilename)
if os.path.exists(args.outputSNR):
    os.remove(args.outputSNR)

sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    det=det,
    outputfilenameNuRadioReco=None,
    config_file=args.config,
    trigger_channels=[0,1,2,3],
    evt_time=dt.datetime(2023, 8, 3)
)
sim.run()

print("Total events", count_events.events)
print("SNRs: ", SNRs)
print("Triggered: ", SNRtriggered)

output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}

outputfile = args.outputSNR
with open(outputfile, 'w+') as fout:
    json.dump(output, fout, sort_keys=True, indent=4)
