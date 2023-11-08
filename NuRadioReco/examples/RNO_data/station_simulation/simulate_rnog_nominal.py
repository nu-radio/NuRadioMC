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

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioMC.utilities import runner
from NuRadioReco.utilities import units

from NuRadioReco.modules import triggerTimeAdjuster
from NuRadioReco.modules.phasedarray.triggerSimulator import triggerSimulator as phasedArrayTrigger
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse

root_seed = secrets.randbits(128)

triggerTimeAdjuster = triggerTimeAdjuster.triggerTimeAdjuster(log_level=logging.WARNING)
rnogHarwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.WARNING)
rnogADCResponse.begin(adc_input_range=2 * units.volt, clock_offset=0.0, adc_output="voltage")

#############################
## Set up trigger definitions
#############################


def RNO_G_PA_Thresh(lgR):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    return 0.0282 * lgR**2 - 4.607 * lgR + 46.24


pa_channels = [0, 1, 2, 3]
phasedArrayTrigger = phasedArrayTrigger(log_level=logging.WARNING)
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))

pa_thresholds = {}
pa_thresholds["1Hz"] = RNO_G_PA_Thresh(0)
pa_thresholds["10Hz"] = RNO_G_PA_Thresh(1)
pa_thresholds["100Hz"] = RNO_G_PA_Thresh(2)
pa_thresholds["1kHz"] = RNO_G_PA_Thresh(3)


def task(q, iSim, energy_min, energy_max, detectordescription, config, output_filename, flavor, interaction_type, **kwargs):
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
            rnogHarwareResponse.run(evt, station, det, temp=293.15, sim_to_data=True)

        def GetTriggerBandVrmsRatio(self, channel_id, evt, station, det):
            # Because there isn't a good way to track all changes that have been made to a
            # channel, have to recalculate all filters that have been applied so far
            # to get the theoretical VRMS

            station_copy = copy.deepcopy(station)

            # Calculate the effective bandwidth of the "full band"
            channel = station_copy.get_channel(channel_id)
            spectrum = channel.get_frequency_spectrum()
            spectrum = np.ones_like(spectrum).astype(complex)
            channel.set_frequency_spectrum(spectrum, channel.get_sampling_rate())
            self._detector_simulation_filter_amp(evt, station_copy, det)
            channel = station_copy.get_channel(channel_id)
            filt_fullband = channel.get_frequency_spectrum()
            freqs_fullband = channel.get_frequencies()
            eff_bandwitdth_fullband = np.trapz(np.abs(filt_fullband) ** 2, freqs_fullband)

            # Calculate the effective bandwidth on the trigger board
            det_channel = det.get_channel(station_copy.get_id(), channel_id)
            sampling_rate = det_channel["trigger_adc_sampling_frequency"]
            mask = freqs_fullband <= sampling_rate
            channel.resample(sampling_rate)
            _, trigger_filter = rnogADCResponse.get_trigger_values(station_copy, det, requested_channels=pa_channels)
            freqs_trigband = channel.get_frequencies()
            filt_trigband = channel.get_frequency_spectrum() * trigger_filter(freqs_trigband)
            eff_bandwitdth_trigband = np.trapz(np.abs(filt_trigband) ** 2, freqs_trigband)

            # V^2 is prop. to the bandwidth
            vrms_scaling_ratio = (eff_bandwitdth_trigband / eff_bandwitdth_fullband) ** 0.5

            return vrms_scaling_ratio

        def _detector_simulation_trigger(self, evt, station, det):

            # Make a copy so that the "RADIANT waveforms" for the PA channels will be retained
            station_copy = copy.deepcopy(station)

            ratio = self.GetTriggerBandVrmsRatio(pa_channels[0], evt, station_copy, det)
            vrms_input_to_adc = self._Vrms_per_channel[station_copy.get_id()][pa_channels[0]] * ratio

            # Runs the FLOWER board response
            vrms_after_gain = rnogADCResponse.run(
                evt, station_copy, det, requested_channels=pa_channels, vrms=vrms_input_to_adc, digitize_trace=True
            )  # Digitization will be done in the trigger module

            pa_sampling_rate = station_copy.get_channel(pa_channels[0]).get_sampling_rate()

            # integration window size and stride
            # Warning: if you change these, you must also recalculate the thresholds!
            upsampling_factor = 2
            pa_window = int(16 * units.ns * pa_sampling_rate * upsampling_factor)
            pa_step = int(8 * units.ns * pa_sampling_rate * upsampling_factor)

            # Run the phased array trigger
            for thresh_key in pa_thresholds.keys():
                threshold = pa_thresholds[thresh_key]

                phasedArrayTrigger.run(
                    evt,
                    station_copy,
                    det,
                    Vrms=vrms_after_gain,
                    threshold=threshold * np.power(vrms_after_gain, 2.0),
                    triggered_channels=pa_channels,
                    phasing_angles=phasing_angles,
                    ref_index=1.75,
                    trigger_name=f"PA_{thresh_key}",  # the name of the trigger
                    trigger_adc=False,
                    adc_output=f"voltage",  # output in volts
                    trigger_filter=None,  # already applied
                    upsampling_factor=upsampling_factor,
                    window=pa_window,
                    step=pa_step,
                    apply_digitization=False,
                )

            # write the triggers back into the original copy of the station
            for trigger in station_copy.get_triggers().values():
                station.set_trigger(trigger)

            # run the adjustment on the full-band waveforms
            triggerTimeAdjuster.begin(pre_trigger_time=100 * units.ns)
            triggerTimeAdjuster.run(evt, station, det)

    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16]}
    r_max = get_max_radius_shallow(energy_max)
    z_min = get_min_z_shallow(energy_max)
    volume = {"fiducial_rmax": r_max, "fiducial_rmin": 0 * units.km, "fiducial_zmin": z_min, "fiducial_zmax": 0}

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
        flavor=flavor_ids[flavor],
        n_events_per_file=None,
        spectrum="log_uniform",
        deposited=True,
        proposal=False,
        proposal_config="SouthPole",
        start_file_id=0,
        log_level=None,
        proposal_kwargs={},
        max_n_events_batch=1e3,
        write_events=False,
        seed=root_seed + iSim,
        interaction_type=interaction_type,
    )

    sim = mySimulation(
        inputfilename=input_data,
        outputfilename=output_filename,
        detectorfile=detectordescription,
        outputfilenameNuRadioReco=output_filename.replace(".hdf5", ".nur"),
        config_file=config,
    )
    n_trig = sim.run()

    print(f"simulation pass {iSim} with {n_trig} events", flush=True)
    q.put(n_trig)


if __name__ == "__main__":

    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")

    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    # Sim steering arguments
    parser.add_argument("detectordescription", type=str, help="path to file containing the detector description")
    parser.add_argument("config", type=str, help="NuRadioMC yaml config file")

    # Neutrino arguments
    parser.add_argument("energy_min", type=float, help="Minimum range of neutrino energies [eV]")
    parser.add_argument("energy_max", type=float, help="Maximum range of neutrino energies [eV]")
    parser.add_argument("flavor", type=str, help="the flavor")
    parser.add_argument("interaction_type", type=str, help="interaction type cc, nc or ccnc")

    # File meta-variables
    parser.add_argument("index", type=int, help="counter to create a unique data-set identifier")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cores for parallel running")
    parser.add_argument("--n_triggers", type=int, default=3e4, help="Number of total events to generate")
    parser.add_argument("--n_events_per_file", type=int, default=1e3, help="Number of nu-interactions per file")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="directory name where the library will be created")

    args = parser.parse_args()
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
