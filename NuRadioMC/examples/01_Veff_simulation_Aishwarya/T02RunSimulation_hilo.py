#!/bin/env python3

import argparse
import copy
import logging
import numpy as np
import os
import secrets
import datetime as dt
from scipy import constants


from NuRadioReco.framework.channel import Channel

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units

from NuRadioReco.detector.RNO_G import rnog_detector, analog_components, rnog_detector_mod
from NuRadioReco.detector.response import Response

from NuRadioReco.modules import triggerTimeAdjuster, channelResampler
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
import NuRadioReco.modules.trigger.highLowThreshold


def get_trigger_board_analog_response(freqs, station_id=-1, channel_id=None):
    trigger_amp_response = analog_components.load_amp_response("ULP_216")

    gain = trigger_amp_response["gain"](freqs)
    complex_phase = trigger_amp_response["phase"](freqs)
    phase = np.imag(np.log(complex_phase))

    return Response(freqs, [gain, phase], ["mag", "rad"], name="ULP_216", station_id=station_id, channel_id=channel_id)


def convert_daq_to_trigger_response(resp):
    """
    All components are here are not channel or station specific... (besides the argument `resp`)
    """
    global ulp_216_resp
    global coax_cable_flower
    resp.remove("radiant_response")
    resp.remove("coax_cable")

    resp *= ulp_216_resp
    resp *= coax_cable_flower

    return resp


def RNO_G_HighLow_Thresh(lgRate_per_hz):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    # This applies for the VPol antennas
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0


#############################
## Set up trigger definitions
#############################

high_low_trigger_thresholds = {}
high_low_trigger_thresholds["10mHz"] = RNO_G_HighLow_Thresh(-2)
high_low_trigger_thresholds["100mHz"] = RNO_G_HighLow_Thresh(-1)
high_low_trigger_thresholds["1Hz"] = RNO_G_HighLow_Thresh(0)
high_low_trigger_thresholds["3Hz"] = RNO_G_HighLow_Thresh(np.log10(3))

root_seed = secrets.randbits(128)

channel_id_offset = 24  # simulation.py (2.x) can only handle consecutive channel ids
deep_trigger_channels = np.array([0, 1, 2, 3]) + channel_id_offset

highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerTimeAdjuster = triggerTimeAdjuster.triggerTimeAdjuster(log_level=logging.WARNING)
rnogHarwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.DEBUG)
rnogADCResponse.begin(adc_input_range=2 * units.volt, clock_offset=0.0, adc_output="voltage")

channel_resampler = channelResampler.channelResampler()
channel_resampler.begin()


ulp_216_resp = get_trigger_board_analog_response(np.linspace(20, 1000, 2000) * units.MHz)


class mySimulation(simulation.simulation):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eff_bandwitdth_trigband = {}
        triggerTimeAdjuster.begin(pre_trigger_time=240 * units.ns)


    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
        rnogHarwareResponse.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        vrms_input_to_adc = []
        for channel_id in deep_trigger_channels:
            vrms_input_to_adc.append(self._Vrms_per_channel[station.get_id()][channel_id])

        sampling_rate = det.get_sampling_frequency(station.get_id())
        print(f'Radiant sampling rate is {sampling_rate / units.MHz:.1f} MHz')

        # Runs the FLOWER board response
        vrms_after_gain = rnogADCResponse.run(
            evt, station, det, requested_channels=deep_trigger_channels,
            vrms=copy.copy(vrms_input_to_adc), digitize_trace=True,
            do_apply_trigger_filter=False,  # The trigger filter is already applied in the channel
        )
            

        for idx, trigger_channel in enumerate(deep_trigger_channels):
            print(
                f'Vrms = {vrms_input_to_adc[idx] / units.mV:.2f} mV / {vrms_after_gain[idx] / units.mV:.2f} mV (after gain). '
            )

                # from matplotlib import pyplot as plt
                # fig, axs = plt.subplots(1, 2, figsize=(10, 6))
                # channel = station.get_channel(trigger_channel)
                # axs[0].plot(channel.get_frequencies() / units.MHz, np.abs(channel.get_frequency_spectrum()))
                # axs[1].plot(channel.get_times(), channel.get_trace())
                # plt.show()

            # this is only returning the correct value if digitize_trace=True for rnogADCResponse.run(..)
        flower_sampling_rate = station.get_channel(deep_trigger_channels[0]).get_sampling_rate()
        print(f'Flower sampling rate is {flower_sampling_rate / units.MHz:.1f} MHz')

        for thresh_key, threshold in high_low_trigger_thresholds.items():
            
            
            threshold_high = {channel_id: threshold * vrms for channel_id, vrms in zip(deep_trigger_channels, vrms_after_gain)}
            threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms in zip(deep_trigger_channels, vrms_after_gain)}    
            
            print(threshold_high, threshold_low)

            highLowThreshold.run(
                evt,
                station,
                det,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                use_digitization=False, #the trace has already been digitized with the rnogADCResponse
                high_low_window=6 / flower_sampling_rate,
                coinc_window=20 / flower_sampling_rate,
                number_concidences=2,
                triggered_channels=deep_trigger_channels,
                trigger_name=f"deep_high_low_{thresh_key}",
            )

        # run the adjustment on the full-band waveforms
        triggerTimeAdjuster.run(evt, station, det)

"""
sim = mySimulation(
        inputfilename=input_filename,
        outputfilename=output_filename,
        det=detector,
        evt_time=dt.datetime(2023, 8, 3),
        outputfilenameNuRadioReco=output_filename.replace(".hdf5", ".nur"),
        config_file=config,
        trigger_channels=deep_trigger_channels,
    )
sim.run()
sim = mySimulation(inputfilename=args.inputfilename,
                                outputfilename=args.outputfilename,
                                det=args.detectordescription,
                                outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                                config_file=args.config,
                                evt_time=dt.datetime(2023, 8, 3),
                                trigger_channels=deep_trigger_channels,
                                file_overwrite=True)
sim.run()

"""
if __name__ == "__main__":
    
    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")

    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
    parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
    parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
    parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
    parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
    args = parser.parse_args()
    

    defaults = {
        "trigger_adc_sampling_frequency": 0.472,
        "trigger_adc_nbits": 8,
        "trigger_adc_noise_nbits": 3.321,
        "trigger_amp_type": "ULP_216",
        "is_noiseless": False,
    }
    
    station_id = 24

    if args.detectordescription != "None":
        det = rnog_detector_mod.ModDetector(
            detector_file=args.detectordescription, log_level=logging.INFO,
            always_query_entire_description=True, select_stations=station_id,
            over_write_handset_values=defaults)
    else:
        det = rnog_detector_mod.ModDetector(
            database_connection='RNOG_public', log_level=logging.INFO,
            always_query_entire_description=True, select_stations=station_id,
            over_write_handset_values=defaults)

    det.update(dt.datetime(2023, 8, 3))
    #det.update(dt.datetime(2023, 8, 2, 0, 0))
    
    det.modify_station_description(station_id, "sampling_rate", 2.4 * units.GHz)
    component_data = det.additional_data["daq_drab_flower_2024_avg"]
    coax_cable_flower = Response(
        component_data["frequencies"], [component_data["mag"], component_data["phase"]], ["dB", "deg"],
        name="coax_cable_flower", station_id=-1, channel_id=None)

    # here comes a dirty dirty HACK:
    for channel_id in deep_trigger_channels:
        orig_id = channel_id - channel_id_offset
        resp = copy.deepcopy(det.get_signal_chain_response(station_id, orig_id))
        if channel_id not in det._Detector__buffered_stations[station_id]["channels"]:
            det._Detector__buffered_stations[station_id]["channels"][int(channel_id)] = copy.deepcopy(
                det._Detector__buffered_stations[station_id]["channels"][int(orig_id)]
            )

            resp = convert_daq_to_trigger_response(resp)
            det._Detector__buffered_stations[station_id]["channels"][int(channel_id)]['id'] = channel_id
            det._Detector__buffered_stations[station_id]["channels"][int(channel_id)]["signal_chain"]["total_response"] = resp

    sim = mySimulation(inputfilename=args.inputfilename,
                                outputfilename=args.outputfilename,
                                det=det,
                                outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                                config_file=args.config,
                                evt_time=dt.datetime(2023, 8, 3),
                                trigger_channels=deep_trigger_channels,
                                file_overwrite=True)
    sim.run()
