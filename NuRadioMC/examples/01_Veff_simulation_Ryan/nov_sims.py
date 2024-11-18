#!/bin/env python3

import argparse
import copy
import logging
import numpy as np
import os
import scipy
from scipy import constants
import secrets
import time
import matplotlib.pyplot as plt
from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioMC.utilities import runner
from NuRadioReco.utilities import units
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.detector import detector
from NuRadioReco.modules import triggerTimeAdjuster
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.modules.phasedarray.triggerSimulator import triggerSimulator as phasedArrayTrigger
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder

root_seed = secrets.randbits(128)
adc_output='voltage'
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerTimeAdjuster = triggerTimeAdjuster.triggerTimeAdjuster(log_level=logging.ERROR)
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.ERROR)
rnogADCResponse.begin(adc_input_range=2 * units.volt, clock_offset=0.0, adc_output=adc_output)
rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()

#############################
## Set up trigger definitions
#############################

pa_channels = [0, 1, 2, 3]
phasedArrayTrigger = phasedArrayTrigger(log_level=logging.WARNING)
main_low_angle = np.deg2rad(-60)
main_high_angle = np.deg2rad(60)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 12))

def task(q, iSim, energy_min, energy_max,  detectordescription, config, output_filename, flavor, interaction_type, **kwargs):
    time.sleep(0.01)
    def get_max_radius_shallow(
        E,
    ):  # estimated radius from max distances from noiseless 1 sigma triggers (for PA trigger), E = shower energy
        if E <= 10**16.25 * units.eV:
            return 1.5 * units.km
        elif E <= 10**16.5 * units.eV:
            return 2.1 * units.km
        elif E <= 10**16.75 * units.eV:
            return 2.7 * units.km
        elif E <= 10**17.0 * units.eV:
            return 3.1 * units.km
        elif E <= 10**17.25 * units.eV:
            return 3.7 * units.km
        elif E <= 10**17.5 * units.eV:
            return 3.9 * units.km
        elif E <= 10**17.75 * units.eV:
            return 4.4 * units.km
        elif E <= 10**18.00 * units.eV:
            return 4.8 * units.km
        elif E <= 10**18.25 * units.eV:
            return 5.1 * units.km
        elif E <= 10**18.50 * units.eV:
            return 5.25 * units.km
        elif E <= 10**18.75 * units.eV:
            return 5.3 * units.km
        elif E <= 10**19.0 * units.eV:
            return 5.6 * units.km
        else:
            return 6.1 * units.km

    def get_min_z_shallow(E):  # estimated min z-pos from noiseless 1 sigma triggers (for PA trigger), E = shower energy
        if E <= 10**16.25 * units.eV:
            return -0.65 * units.km
        elif E <= 10**16.50 * units.eV:
            return -0.8 * units.km
        elif E <= 10**16.75 * units.eV:
            return -1.2 * units.km
        elif E <= 10**17.00 * units.eV:
            return -1.5 * units.km
        elif E <= 10**17.25 * units.eV:
            return -1.7 * units.km
        elif E <= 10**17.50 * units.eV:
            return -2.0 * units.km
        elif E <= 10**17.75 * units.eV:
            return -2.1 * units.km
        elif E <= 10**18.00 * units.eV:
            return -2.3 * units.km
        elif E <= 10**18.25 * units.eV:
            return -2.4 * units.km
        elif E <= 10**18.50 * units.eV:
            return -2.55 * units.km
        else:
            return -2.7 * units.km
        
    class mySimulation(simulation.simulation):
        def _detector_simulation_filter_amp(self, evt, station, det):
            # apply the amplifiers and filters to get to RADIANT-level
            rnogHardwareResponse.run(evt, station, det, temp=293.15, sim_to_data=True)

        def _detector_simulation_trigger(self, evt, station, det):

            station_copy = copy.deepcopy(station)
            #fig,ax=plt.subplots(1,2)
            #for i in range(4):
            #    ax[0].plot(station_copy.get_channel(i).get_trace())
            chan_rms=rnogADCResponse.run(evt, station_copy, det, requested_channels=pa_channels, apply_adc_gain=True,vrms=None, digitize_trace=True, do_apply_trigger_filter=True )
            
            #print(chan_rms)
            #for i in range(4):
            #    ax[1].plot(station_copy.get_channel(i).get_trace())
            #plt.show()
            lVrms=0
            threshold_low={}
            threshold_high={}
            threshold=3.7
            for i,val in enumerate(chan_rms):
                lVrms+=val**2 #since the thresholds acts on the coherent sum's average power, thresholds should be on coherent sum rms.
                threshold_low[i]=-threshold*val
                threshold_high[i]=threshold*val


            pa_channel = det.get_channel(station_copy.get_id(), pa_channels[0])
            pa_sampling_rate = pa_channel["trigger_adc_sampling_frequency"] * units.GHz

            upsampling_factor = 4
            pa_window = 32
            pa_step = 8

            phasedArrayTrigger.run(
                evt,
                station_copy,
                det,
                Vrms=1,
                threshold= 9.33 * lVrms,
                triggered_channels=pa_channels,
                phasing_angles=phasing_angles,
                ref_index=1.75,
                trigger_name="PA_fir",  # the name of the trigger
                trigger_adc=False,
                adc_output=adc_output,  # output in volts
                trigger_filter=None,  # already applied
                upsampling_factor=upsampling_factor,
                window=pa_window,
                step=pa_step,
                apply_digitization=False,
                upsampling_method='fir',
                coeff_gain=128,
                rnog_like=True
            )

            highLowThreshold.run(
                evt,
                station_copy,
                det,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                high_low_window=6 / pa_sampling_rate,
                coinc_window=20 / pa_sampling_rate,
                number_concidences=2,
                triggered_channels=pa_channels,
                trigger_name="high_low"
            )


            # write the triggers back into the original copy of the station
            for trigger in station_copy.get_triggers().values():
                station.set_trigger(trigger)

            # run the adjustment on the full-band waveforms
            #triggerTimeAdjuster.begin(pre_trigger_time=100 * units.ns)
            #triggerTimeAdjuster.run(evt, station, det)

    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16]}


    r_max = get_max_radius_shallow(energy_max)
    z_min = get_min_z_shallow(energy_max)
    volume = {"fiducial_rmax": r_max, "fiducial_rmin": 0 * units.km, "fiducial_zmin": z_min, "fiducial_zmax": 0}
    print(volume)
    input_data = generator.generate_eventlist_cylinder(
        "on-the-fly",
        kwargs["n_events_per_file"],
        energy_min,
        energy_max,
        volume,
        thetamin=0.0 * units.rad,
        thetamax=np.pi * units.rad,
        phimin=0.0 * units.rad,
        phimax=2 * np.pi * units.rad,
        start_event_id=1,
        #flavor=flavor,
        n_events_per_file=None,
        spectrum="log_uniform",
        deposited=True,
        proposal=False,
        proposal_config="SouthPole",
        start_file_id=0,
        log_level=None,
        proposal_kwargs={},
        max_n_events_batch=1e4,
        write_events=False,
        seed=root_seed + iSim
        #interaction_type='ccnc',
    )

    sim = mySimulation(
        inputfilename=input_data,
        outputfilename=output_filename,
        detectorfile=detectordescription,
        evt_time=datetime.datetime(2024, 8, 2, 0, 0),
        outputfilenameNuRadioReco=None,
        config_file=config,
    )
    n_trig = sim.run()

    print(f"simulation pass {iSim} with {n_trig} events", flush=True)
    q.put(n_trig)


if __name__ == "__main__":

    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    #def_data_dir = "/data2/ryank/nuradiomc_data/nov_prop_fixed"


    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    # Sim steering arguments
    parser.add_argument("--detectordescription", type=str, default="RNO_single_station_only_PA.json",help="path to file containing the detector description")
    parser.add_argument("--config", type=str, default="config.yaml",help="NuRadioMC yaml config file")

    # Neutrino arguments
    parser.add_argument("--energy_min", type=float,default=20., help="Minimum range of neutrino energies [eV]")
    parser.add_argument("--energy_max", type=float,default=20., help="Maximum range of neutrino energies [eV]")
    parser.add_argument("--flavor", type=str,default="e", help="the flavor")
    parser.add_argument("--interaction_type", type=str, default="nc",help="interaction type cc, nc or ccnc")

    # File meta-variables
    parser.add_argument("--index", type=int, default=0,help="counter to create a unique data-set identifier")
    parser.add_argument("--n_cpus", type=int, default=8, help="Number of cores for parallel running")
    parser.add_argument("--n_triggers", type=int, default=1e2, help="Number of total events to generate")
    parser.add_argument("--n_events_per_file", type=int, default=1e3, help="Number of nu-interactions per file")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="directory name where the library will be created")
    
    args = parser.parse_args()
    args.energy_min=10**args.energy_min
    args.energy_max=10**args.energy_max
    print(args.energy_min)
    kwargs = args.__dict__
    filename = (os.path.splitext(os.path.basename(__file__))[0]).split(".")[0]
    output_path = os.path.join(
        args.data_dir,
        os.path.splitext(os.path.basename(args.detectordescription))[0],
        os.path.splitext(os.path.basename(args.config))[0],
        filename,
        f"nu_{args.flavor}_{args.interaction_type}",
        f"{args.index:06}",
        f"{np.log10(args.energy_min):.2f}eV",
    )

    if not os.path.exists(output_path):
        print("Making dirs", output_path)
        os.makedirs(output_path)

    class myrunner(runner.NuRadioMCRunner):
        def get_outputfilename(self):
            return os.path.join(
                self.output_path, f"{np.log10(args.energy_min):.2f}_{self.kwargs['index']:06d}_{self.i_task:06d}.hdf5"
            )

    # start simulating a library across the chosen number of cpus. Each CPU will only run for 1 day
    r = myrunner(args.n_cpus, task, output_path, max_runtime=3600 * 24, n_triggers_max=args.n_triggers, kwargs=kwargs)
    r.run()
