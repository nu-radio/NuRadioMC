#!/bin/env python3

# This file is an adaptation of simulate.py, found in NuRadioMC/examples/08_RNO_G_trigger_simulation.
# This example has been adapted to the simulate a secondary in-ice cascade originated from a CR-induced EAS.
# This file also has been extended to work within a NuRadio's runner object, so that it can be run in a cluster.

import argparse
import copy
import numpy as np
import os
import secrets
import datetime as dt
from scipy import constants


from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.detector.RNO_G import rnog_detector

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold

import logging
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)

from NuRadioMC.utilities import runner

def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):

    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(temperature=temperature, response=resp)
        )

    return vrms_per_channel

# The fiducial volume has been adapted for Cosmic Rays for a Greenland station. 
# A reasonable volume is 175 x 100 m per station, regardless of energy (for now).
# The structure of this module is kept similar to its original source, 
#in case of energy-dependant fiducial volume for CR.
def get_fiducial_volume_CR(energy):
    # key: log10(E), value: radius in km
    max_radius_shallow = {
        100: 0.0175,
    }

    # key: log10(E), value: depth in km
    min_z_shallow = {
        100: 0.01,
    }

    def get_limits(dic, E):
        # find all energy bins which are higher than E
        idx = np.arange(len(dic))[E - 10 ** np.array(list(dic.keys())) * units.eV <= 0]
        assert len(idx), f"Energy {E} is too high. Max energy is {10 ** np.amax(dic.keys()):.1e}."

        # take the lowest energy bin which is higher than E
        return np.array(list(dic.values()))[np.amin(idx)] * units.km

    r_max = get_limits(max_radius_shallow, energy)
    print(f"Maximum radius {r_max}")
    z_min = get_limits(min_z_shallow, energy)
    print(f"Depth {z_min}")

    volume = {
        "fiducial_rmax": r_max,
        "fiducial_rmin": 0 * units.km,
        "fiducial_zmin": z_min,
        "fiducial_zmax": 0
    }

    return volume

# So-far, RNO-G has used/aimed for a trigger rate of 1 Hz = 3.731 SNR according to the formula below
# TO-DO maybe add exact values per station @ 1 Hz (ll are between 3.7 and 3.8)
def RNO_G_HighLow_Thresh(lgRate_per_hz):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    # This applies for the VPol antennas
    # parameterization comes from Alan: https://radio.uchicago.edu/wiki/images/e/e6/2023.10.11_Simulating_RNO-G_Trigger.pdf
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0


class mySimulation(simulation.simulation):

    def __init__(self, *args, trigger_channel_noise_vrms=None, **kwargs):

        # Check that we have the vrms
        assert trigger_channel_noise_vrms is not None, "Please provide the trigger channel noise vrms"
        self.trigger_channel_noise_vrms = trigger_channel_noise_vrms

        # this module is needed in super().__init__ to calculate the vrms
        self.rnogHarwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
#        self.rnogHarwareResponse.begin(trigger_channels=deep_trigger_channels)
        # Should be identical to
        self.rnogHardwareResponse.begin(trigger_channels=kwargs['trigger_channels'])

        super().__init__(*args, **kwargs)
        self.logger = logger
        self.deep_trigger_channels = kwargs['trigger_channels']

        self.highLowThreshold = highLowThreshold.triggerSimulator()
        self.rnogADCResponse = triggerBoardResponse.triggerBoardResponse()
#        self.rnogADCResponse.begin(adc_input_range=2 * units.volt, clock_offset=0.0, adc_output="voltage")
#       Using now counts instead of voltage to determine the ADC Response
        self.rnogADCResponse.begin(
            clock_offset=0.0, adc_output="counts")

        # future TODO: Add noise
        # self.channel_generic_noise_adder = channelGenericNoiseAdder.channelGenericNoiseAdder()
        # self.channel_generic_noise_adder.begin(seed=self._cfg['seed'])

        self.output_mode = {'Channels': self._config['output']['channel_traces'],
                            'ElectricFields': self._config['output']['electric_field_traces'],
                            'SimChannels': self._config['output']['sim_channel_traces'],
                            'SimElectricFields': self._config['output']['sim_electric_field_traces']}

        self.high_low_trigger_thresholds = {
            "10mHz": RNO_G_HighLow_Thresh(-2),
            "100mHz": RNO_G_HighLow_Thresh(-1),
            "1Hz": RNO_G_HighLow_Thresh(0),
            "3Hz": RNO_G_HighLow_Thresh(np.log10(3)),
        }


    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
        self.rnogHarwareResponse.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):

        vrms_input_to_adc = get_vrms_from_temperature_for_trigger_channels(
            det, station.get_id(), self.deep_trigger_channels, 300)

#        sampling_rate = det.get_sampling_frequency(station.get_id())
#        self.logger.info(f'Radiant sampling rate is {sampling_rate / units.MHz:.1f} MHz')

        # Runs the FLOWER board response
        vrms_after_gain = self.rnogADCResponse.run(
            evt, station, det, trigger_channels=self.deep_trigger_channels,
            vrms=self.trigger_channel_noise_vrms, digitize_trace=True,
        )
        
        for idx, trigger_channel in enumerate(self.deep_trigger_channels):
            self.logger.debug(
                'Vrms = {:.2f} mV / {:.2f} mV (after gain).'.format(
                    self.trigger_channel_noise_vrms[idx] / units.mV, vrms_after_gain[idx] / units.mV
                ))
            self._Vrms_per_trigger_channel[station.get_id()][trigger_channel] = vrms_after_gain[idx]

        # this is only returning the correct value if digitize_trace=True for self.rnogADCResponse.run(..)
        flower_sampling_rate = station.get_trigger_channel(self.deep_trigger_channels[0]).get_sampling_rate()
#        self.logger.debug(f'Flower sampling rate is {flower_sampling_rate / units.MHz:.1f} MHz')
        self.logger.debug('Flower sampling rate is {:.1f} MHz'.format(
            flower_sampling_rate / units.MHz
        ))

        for thresh_key, threshold in self.high_low_trigger_thresholds.items():

            if self.rnogACDResponse.acd_output == "voltage":
                threshold_high = {channel_id: threshold * vrms 
                        for channel_id, vrms in zip(self.deep_trigger_channels, vrms_after_gain)}
                threshold_low = {channel_id: -1 * threshold * vrms 
                        for channel_id, vrms in zip(self.deep_trigger_channels, vrms_after_gain)}
#Inherited from simulate.py, Not sure that this accurately represents the trigger threshold in adc counts
#This was switched to from previous version. 
            else:
                # We round here. This is not how an ADC works but I think this is not needed here.
                threshold_high = {channel_id: int(round(threshold * vrms)) 
                        for channel_id, vrms in zip(self.deep_trigger_channels, vrms_after_gain)}
                threshold_high = {channel_id: int(round(-1*threshold * vrms)) 
                        for channel_id, vrms in zip(self.deep_trigger_channels, vrms_after_gain)}

            self.highLowThreshold.run(
                evt, station, det,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                use_digitization=False, #the trace has already been digitized with the rnogADCResponse
                high_low_window=6 / flower_sampling_rate,
                coinc_window=20 / flower_sampling_rate,
                number_concidences=2,
                triggered_channels=self.deep_trigger_channels,
                trigger_name=f"deep_high_low_{thresh_key}",
            )


if __name__ == "__main__":

    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    default_config_path = os.path.join(ABS_PATH_HERE, "../07_RNO_G_simulation/RNO_config.yaml")

    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    
    # Sim steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None, help="Path to RNO-G detector description file. If None, query from DB")
    parser.add_argument("--station_id", type=int, default=None, help="Set station to be used for simulation", required=True)

    # Neutrino arguments
    parser.add_argument("--energy", '-e', default=1e18, type=float, help="Neutrino energy [eV]")
#    parser.add_argument("--flavor", '-f', default="all", type=str, help="the flavor")
#    parser.add_argument("--interaction_type", '-it', default="ccnc", type=str, help="interaction type cc, nc or ccnc")
    parser.add_argument("--cos_min", default=0.0, type=float, help="minimum cos zenith angle")
    parser.add_argument("--cos_max", default=0.5, type=float, help="maximum cos zenith angle")
    # maximum distances only tested up to 60 deg -> TODO is this still true?

    # File meta-variables
    parser.add_argument("--index", '-i', default=0, type=int, help="counter to create a unique data-set identifier")
#    parser.add_argument("--n_events_per_file", '-n', type=int, default=1e3, help="Number of nu-interactions per file")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="directory name where the library will be created")
#    parser.add_argument("--proposal", action="store_true", help="Use PROPOSAL for simulation")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    # Runner meta-variables
    # n_triggers = 1000 - 10000 (per Alan)
    parser.add_argument("--n_triggers", type=int, default=3000, help="Number of total events to generate")
    parser.add_argument("--n_events_per_file", type=int, default=1000, help="Number of nu-interactions per file")
    parser.add_argument("--seed", default=None, type=int, help="Random number gen seed, None for truly random")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores for parallel running")
    parser.add_argument("--max_runtime_hours", type=int, default=24, help="Max amount of hours that the node should be occupied."  )

    args = parser.parse_args()
    config = simulation.get_config(args.config)
    
    kwargs = args.__dict__
    assert args.station_id is not None, "Please specify a station id with `--station_id`"
    
    zen_min = min(np.arccos(args.cos_min), np.arccos(args.cos_max))
    zen_max = max(np.arccos(args.cos_min), np.arccos(args.cos_max))
    assert zen_max < np.deg2rad(61) is not None, "Max zenith tested is 60 degrees"

    deep_trigger_channels = np.array([0, 1, 2, 3])

    seed = args.seed
    if seed is not None:
        seed += args.index
#    else:
#        seed = secrets.randbits(128) + args.index

# Create your detector
    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id)

#TODO add time as input
    evt_time = dt.datetime(2023, 8, 3)
    det.update(evt_time)

#   In order to not have to consider inelasticity, we will force all interactions to be e-cc.
#   e-cc interaction is the least likely to have missing energy escaping in the secondary fermion.
    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16], "all": [12, 14, 16, -12, -14, -16]}
    #flavor = args.flavor
    flavor = "e"
#   Therefore, we do not need to use proposal at all
    run_proposal = False
#       run_proposal = args.proposal and ("cc" in args.interaction_type) and (args.flavor in ["mu", "tau", "all"])
#       if run_proposal:
#           print(f"Using PROPOSAL for simulation of {args.flavor} {args.interaction_type}")
 
#################################
## Define simulation task
#################################

    # A task is needed for multiprocessing (runner)
    # In order to create a task, we have to pass a the output filename as a kwarg
    def task(q, iSim, **kwargs):
        output_filename = kwargs["output_filename"]
        
        # Get the trigger channel noise vrms
        trigger_channel_noise_vrms = get_vrms_from_temperature_for_trigger_channels(
            det, args.station_id, deep_trigger_channels, config['trigger']['noise_temperature'])

        # The volume might be energy-dependent, therefore is kept in the task.
        volume = get_fiducial_volume_CR(args.energy)

        # Simulate fiducial volume around station
        pos = det.get_absolute_position(args.station_id)
        print(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")
        volume.update({"x0": pos[0], "y0": pos[1]})
   
        input_data = generator.generate_eventlist_cylinder(
            "on-the-fly",
            kwargs["n_events_per_file"],
            args.energy, args.energy,
            volume,
            thetamin=zen_min, thetamax=zen_max,
            start_event_id=args.index * args.n_events_per_file + 1,
            flavor=flavor_ids[flavor],
            n_events_per_file=None,
            deposited=False,
            proposal=run_proposal,
            proposal_config="Greenland",
            start_file_id=0,
            log_level=None,
            proposal_kwargs={},
            max_n_events_batch=args.n_events_per_file,
            write_events=False,
            seed=seed,
            interaction_type=args.interaction_type,
        )

        # Inherited from previous
        print(f"[{iSim}] generating {input_data[1]['n_events']} events...", flush=True)

        if args.nur_output:
            nur_output_filename = output_filename.replace(".hdf5", ".nur")
        else:
            nur_output_filename = None

        sim = mySimulation(
            inputfilename=input_data,
            outputfilename=output_filename,
            det=det,
            evt_time=evt_time,
            outputfilenameNuRadioReco=nur_output_filename,
            config_file=args.config,
            trigger_channels=deep_trigger_channels,
            trigger_channel_noise_vrms=trigger_channel_noise_vrms
#            log_level=logging.WARNING,
#            log_level=logging.INFO,
        )

        n_trig = sim.run()

        print(f"simulation pass {iSim} with {n_trig} events", flush=True)
        q.put(n_trig)

#################################
## END simulation task
#################################

# Path naming "A la Felix"
#   output_path = f"{args.data_dir}/station_{args.station_id}/nu_{args.flavor}_{args.interaction_type}"
#   output_filename = f"{output_path}/{args.flavor}_{args.interaction_type}_1e{np.log10(args.energy):.2f}eV_{args.index:08d}.hdf5"

# Path naming "A la Alan"
    filename = (os.path.splitext(os.path.basename(__file__))[0]).split(".")[0]
    output_path = os.path.join(
        # ABS_PATH_HERE,
        # "data",
        args.data_dir,  
        os.path.splitext(os.path.basename(args.detectordescription))[0],
        os.path.splitext(os.path.basename(args.config))[0],
        filename,
        f"lgE_{np.log10(args.energy):.2f}",
        f"cos_{args.cos_min:.2f}",
    )

    if not os.path.exists(output_path):
        print("Making dirs", output_path)
        os.makedirs(output_path, exist_ok=True)

# The runner is the object that parallelizes the task
    class myrunner(runner.NuRadioMCRunner):
        # if required override the get_outputfilename function for a custom output file
        def get_outputfilename(self):
            return os.path.join(
                self.output_path,  f"lgE_{np.log10(args.energy):.2f}-cos_{args.cos_min:.2f}-{self.i_task:06d}.hdf5"
            )

    r = myrunner(args.n_cores, task, output_path, max_runtime=3600 * args.max_runtime_hours, n_triggers_max=args.n_triggers)
    r.run()
