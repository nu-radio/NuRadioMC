import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
import os
import time
import secrets
import argparse
from NuRadioMC.utilities import runner

"""
Example of how to use the runner class to run NuRadioMC on a cluster.
"""


root_seed = secrets.randbits(128)

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

# same filter for all channels
passband_low = [96 * units.MHz, 100 * units.GHz]
passband_high = [0 * units.MHz, 800 * units.MHz]
filter_type = 'cheby1'
order_low = 4
order_high = 7

thresholds = {
  '2/4_100Hz': 3.9498194908011524,
  '2/4_10mHz': 4.919151494949084,
  '2/6_100Hz': 4.04625348733533,
  '2/6_10mHz': 5.015585491483261,
  'fhigh': 0.15,
  'flow': 0.08
  }

passband_low = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}
# downward LPDAs for nu triggering, bandwidth limited
for channel_id in range(0, 4):
    passband_low[channel_id] = [0 * units.MHz, thresholds['fhigh']]
    passband_high[channel_id] = [thresholds['flow'], 800 * units.GHz]
    filter_type[channel_id] = 'butter'
    order_low[channel_id] = 10
    order_high[channel_id] = 5

# 16 is single noiseless dipole at 15 m
passband_low[4] = [1 * units.MHz, 220 * units.MHz]
filter_type[4] = 'cheby1'
order_low[4] = 7
passband_high[4] = [96 * units.MHz, 100 * units.GHz]
order_high[4] = 4


def task(q, iSim, nu_energy, nu_energy_max, detectordescription, config, output_filename,
         flavor, interaction_type, **kwargs):

    def get_max_radius(E):
        if(E <= 10 ** 16.6 * units.eV):
            return 3 * units.km
        elif(E <= 1e17 * units.eV):
            return 3 * units.km
        elif(E <= 10 ** 17.6 * units.eV):
            return 4 * units.km
        elif(E <= 10 ** 18.1 * units.eV):
            return 4.5 * units.km
        elif(E <= 10 ** 18.6 * units.eV):
            return 5 * units.km
        elif(E <= 10 ** 19.1 * units.eV):
            return 6 * units.km
        elif(E <= 10 ** 19.6 * units.eV):
            return 6 * units.km
        elif(E <= 10 ** 20.1 * units.eV):
            return 6 * units.km
        else:
            return 6 * units.km

    # initialize detector sim modules
    simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
    highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

    triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()

    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):
            channelBandPassFilter.run(evt, station, det,
                                  passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
            channelBandPassFilter.run(evt, station, det,
                                  passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)

        def _detector_simulation_trigger(self, evt, station, det):
            # noiseless trigger
            simpleThreshold.run(evt, station, det,
                                         threshold=2.5 * self._Vrms_per_channel[station.get_id()][4],
                                         triggered_channels=[4],  # run trigger on all channels
                                         number_concidences=1,
                                         trigger_name=f'dipole_2.5sigma')  # the name of the trigger

            simpleThreshold.run(evt, station, det,
                                         threshold=3.5 * self._Vrms_per_channel[station.get_id()][4],
                                         triggered_channels=[4],  # run trigger on all channels
                                         number_concidences=1,
                                         trigger_name=f'dipole_3.5sigma')  # the name of the trigger

            threshold_high = {}
            threshold_low = {}
            for channel_id in range(4):
                threshold_high[channel_id] = thresholds['2/4_100Hz'] * self._Vrms_per_channel[station.get_id()][channel_id]
                threshold_low[channel_id] = -thresholds['2/4_100Hz'] * self._Vrms_per_channel[station.get_id()][channel_id]
            highLowThreshold.run(evt, station, det,
                                        threshold_high=threshold_high,
                                        threshold_low=threshold_low,
                                        coinc_window=40 * units.ns,
                                        triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                        number_concidences=2,  # 2/4 majority logic
                                        trigger_name='LPDA_2of4_100Hz')

            triggerTimeAdjuster.run(evt, station, det)

    flavor_ids = {'e': [12, -12],
                 'mu': [14, -14],
                'tau': [16, -16]}

    r_max = get_max_radius(nu_energy)
    volume = {'fiducial_rmax': r_max,
            'fiducial_rmin': 0 * units.km,
            'fiducial_zmin':-2.7 * units.km,
            'fiducial_zmax': 0
            }

    n_events = 1e4

    input_data = generator.generate_eventlist_cylinder("on-the-fly", n_events, nu_energy, nu_energy_max,
                                                        volume,
                                                        thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                                        phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                                        start_event_id=1,
                                                        flavor=flavor_ids[flavor],
                                                        n_events_per_file=None,
                                                        spectrum='log_uniform',
                                                        deposited=False,
                                                        proposal=False,
                                                        proposal_config='SouthPole',
                                                        start_file_id=0,
                                                        log_level=None,
                                                        proposal_kwargs={},
                                                        max_n_events_batch=n_events,
                                                        write_events=False,
                                                        seed=root_seed + iSim,
                                                        interaction_type=interaction_type)

#     with Pool(20) as p:
#         print(p.map(tmp, input_kwargs))

    sim = mySimulation(inputfilename=input_data,
                       outputfilename=output_filename,
                       detectorfile=detectordescription,
                       outputfilenameNuRadioReco=None,
                       config_file=config,
                       default_detector_station=1,
                       default_detector_channel=0)
    n_trig = sim.run()

    print(f"simulation pass {iSim} with {n_trig} events", flush=True)
    q.put(n_trig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('energy_min', type=float,
                        help='neutrino energy')
    parser.add_argument('energy_max', type=float,
                        help='neutrino energy')
    parser.add_argument('detectordescription', type=str,
                        help='path to file containing the detector description')
    parser.add_argument('config', type=str,
                        help='NuRadioMC yaml config file')
    parser.add_argument('index', type=int,
                        help='a running index to create unitque files')
    parser.add_argument('flavor', type=str,
                        help='the flavor')
    parser.add_argument('interaction_type', type=str,
                        help='interaction type cc, nc or ccnc')

    args = parser.parse_args()

    nu_energy = args.energy_min * units.eV
    nu_energy_max = args.energy_max * units.eV
    kwargs = args.__dict__
    kwargs['nu_energy'] = args.energy_min * units.eV
    kwargs['nu_energy_max'] = args.energy_min * units.eV

    filename = os.path.splitext(os.path.basename(__file__))[0]

    output_folder = f"nu_{args.flavor}_{args.interaction_type}"
    output_path = os.path.join(output_folder, args.detectordescription, args.config, filename, f"{np.log10(nu_energy):.2f}eV", f"{args.index:06}")
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    class myrunner(runner.NuRadioMCRunner):

        # if required override the get_outputfilename function for a custom output file
        def get_outputfilename(self):
            return runner.NuRadioMCRunner.get_outputfilename(self)

    # start a simulation on two cores with a runtime of 23h
    r = myrunner(2, task, output_path, max_runtime=3600 * 23, kwargs=kwargs)
    r.run()
