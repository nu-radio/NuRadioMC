import logging
import subprocess
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s:%(message)s", datefmt="%H:%M:%S")

import NuRadioReco.modules.io.eventReader as eventReader
import NuRadioReco.modules.io.eventWriter as eventWriter
import NuRadioReco.framework.parameters as parameters
from NuRadioReco.utilities import units, fft
import NuRadioReco.detector.detector as detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.modules import channelAddCableDelay
import numpy as np
import pandas as pd
from NuRadioReco.modules.neutrinoVertexReconstructor import neutrino3DVertexReconstructor
from NuRadioReco.modules.neutrinoDirectionReconstruction import rayTypeSelecter, neutrinoDirectionReconstructor
from NuRadioReco.modules import channelBandPassFilter, channelResampler, channelGenericNoiseAdder
import NuRadioMC.utilities.medium
import NuRadioMC.SignalGen.askaryan
import NuRadioReco.framework.base_trace
import os
import time
import yaml
import argparse
from radiotools import helper as hp

stnp = parameters.stationParameters
chnp = parameters.channelParameters
evp = parameters.eventParameters
shp = parameters.showerParameters
pap = parameters.particleParameters
cabledelayadder = channelAddCableDelay.channelAddCableDelay()
resampler = channelResampler.channelResampler()
bandpassfilter = channelBandPassFilter.channelBandPassFilter()
neutrinodirectionreconstructor = neutrinoDirectionReconstructor.neutrinoDirectionReconstructor()
raytypeselecter = rayTypeSelecter.rayTypeSelecter()
antenna_provider = antennapattern.AntennaPatternProvider()
noiseadder = channelGenericNoiseAdder.channelGenericNoiseAdder()

logger = logging.getLogger('reconstruction')
logger.setLevel(logging.INFO)

### set settings for debug plots (?)
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('legend', fontsize=12, handlelength=1, borderaxespad=.3)
plt.rc('axes', labelsize=14, titlesize=14)

plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

if __name__ == "__main__":

    # Because some antennas are treated differently in the reconstruction,
    # we need to manually specify:
    # 1. the list of all antennas to use;
    # 2. the list of all hpol antennas;
    # 3. the list of phased array channels;
    # 4. the phased array cluster, i.e. the phased-array plus nearest hpol channel(s)
    # IMPORTANT! These channel ids are of course DIFFERENT for different detectors,
    # and should be adjusted accordingly by the user!
    channel_dict = {}
    channel_dict['all'] = np.arange(15)
    channel_dict['hpol_channels'] = np.array([7,8,11,14])
    channel_dict['phased_array'] = np.array([0,1,2,3])
    channel_dict['phased_array_cluster'] = np.array([0,1,2,3,7,8])

    argparser = argparse.ArgumentParser(
        "Reconstruction",
        description="Script to run neutrino reconstruction, suitable for parallelization."
    )
    argparser.add_argument("filename", type=str, help="path to .nur file to reconstruct")
    argparser.add_argument("detector", type=str, help="Path to detector description.")
    argparser.add_argument("station_id", type=int, help="Station id of station to use for reconstruction.")
    argparser.add_argument("lookup_table_path", type=str, help="Path to lookup tables.")
    argparser.add_argument("output_path", type=str, help="Path to output folder.")
    argparser.add_argument("--vertex", "-v", action='store_true', help="Reconstruct vertex position.")
    argparser.add_argument("--direction", "-d", action='store_true', help="Reconstruct neutrino direction.")
    argparser.add_argument("--save_nur", action='store_true', help="Store nur output file")
    argparser.add_argument("--debug", action='store_true', help="Create debug plots")
    argparser.add_argument(
        "--run", "-r", type=int, nargs="+", help="Run numbers to reconstruct. If not specified, reconstruction is performed for all runs in the input .nur file")

    args = argparser.parse_args()

    run_vertex_reco = args.vertex
    run_direction_reco = args.direction

    # load settings to use in reconstruction
    with open('./input/config_reconstruction.yaml') as f:
        reco_config = yaml.load(f, yaml.FullLoader)
    # propagation config (used in direction reconstruction)
    with open("./input/config_RNOG.yaml", 'r') as f:
        prop_config = yaml.load(f, Loader=yaml.FullLoader)

    vertex_reco_settings = reco_config['vertex']
    direction_reco_settings = reco_config['direction']

    eventreader = eventReader.eventReader()
    eventreader.begin(args.filename)
    if args.save_nur:
        eventwriter = eventWriter.eventWriter()
        nur_output_path = os.path.join(args.output_path, args.filename.split('/')[-1])
        logger.info(f"writing output nur to {nur_output_path}")
        eventwriter.begin(nur_output_path)
    file_id = os.path.basename(args.filename)

    run_ids = args.run
    if run_ids == None:
        run_ids = []
    det = detector.Detector(json_filename=args.detector, antenna_by_depth=False)
    station_id = args.station_id

    # make templates
    # for the vertex reconstruction, we need an electric field template
    n_samples = 1024
    viewing_angle = 1.5 * units.deg
    sampling_rate = reco_config['sampling_rate']
    ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
    ior = ice.get_index_of_refraction([0, 0, -1. * units.km])
    cherenkov_angle = np.arccos(1. / ior)
    efield_spec = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
        energy=1.e19 * units.eV,
        theta=viewing_angle + cherenkov_angle,
        N=n_samples,
        dt=1. / sampling_rate,
        shower_type='HAD',
        n_index=ior,
        R=5. * units.km,
        model='ARZ2020',
        seed=0
    )
    efield_template = NuRadioReco.framework.base_trace.BaseTrace()
    efield_template.set_frequency_spectrum(efield_spec, sampling_rate)

    # the template for the direction reconstruction should be convolved with the detector response
    # TODO: make the raytypeselecter do this?
    vpol_antenna = antenna_provider.load_antenna_pattern(det.get_antenna_model(station_id, 0))
    antenna_response = vpol_antenna.get_antenna_response_vectorized(
        efield_template.get_frequencies(),
        70 * units.deg,
        0*units.deg,
        *det.get_antenna_orientation(station_id, 0)
    )['theta']
    amp_response = det.get_amplifier_response(station_id, 0, efield_template.get_frequencies())

    direction_pulse_template = fft.freq2time(efield_spec * antenna_response * amp_response, sampling_rate)

    # initialize vertex reconstructor
    vertex3d = neutrino3DVertexReconstructor.neutrino3DVertexReconstructor(args.lookup_table_path)

    # run reco
    for evt in eventreader.run():
        run_number = evt.get_run_number()
        if len(run_ids):
            if run_number not in run_ids:
                continue
        results = dict()
        particle = evt.get_primary()
        E_nu = particle.get_parameter(pap.energy)
        output_path = args.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            subprocess.run(["mkdir", output_path])
        event_id = evt.get_id()
        weight = particle.get_parameter(pap.weight)
        zenith_sim = particle.get_parameter(pap.zenith)
        azimuth_sim = particle.get_parameter(pap.azimuth)
        x_sim = particle.get_parameter(pap.vertex)
        E_sh = evt.get_first_sim_shower().get_parameter(shp.energy)

        data = [E_nu, E_sh, file_id, run_number, event_id, weight, zenith_sim, azimuth_sim, x_sim]
        for key, value in zip(['E_nu', 'E_sh', 'file_id', 'run_number', 'event_id', 'weight',
                'zenith_sim', 'azimuth_sim', 'x_sim',], data):
            results[key] = value 

        logger.info("Starting reconstruction for run {}".format(run_number))

        station = evt.get_station(station_id)
        sim_station = station.get_sim_station()
        station.set_is_neutrino()

        resampler.begin()
        resampler.run(evt, station, det, sampling_rate=sampling_rate)
        if sim_station is not None:
            resampler.run(evt, sim_station, det, sampling_rate=sampling_rate)
        cabledelayadder.run(evt, station, det, mode='subtract')
        # same treatment for sim_station - useful for debugging
        if station.has_sim_station():
            resampler.run(evt, sim_station, det, sampling_rate=sampling_rate)
            cabledelayadder.run(evt, sim_station, det, mode='subtract')


        t0 = time.time()
        ### vertex reconstruction

        # we only use the vpol channels in the vertex reconstruction
        vertex_channels = [
            channel_id for channel_id in channel_dict['all']
            if channel_id not in channel_dict['hpol_channels']
            ]

        if run_vertex_reco:
            vertex3d.begin(
                station_id, vertex_channels, det, template=efield_template,
                debug_folder=output_path,
                **reco_config['vertex']
            )

            vertex3d.run(evt, station, det, debug=args.debug)

            sim_shower = evt.get_first_sim_shower()

            reco_vertex = station.get_parameter(stnp.nu_vertex)
            sim_vertex = sim_shower.get_parameter(shp.vertex)

            logger.debug("sim_vertex: {}".format(sim_shower.get_parameter(shp.vertex)))
            logger.debug("reco vertex: {}".format(station.get_parameter(stnp.nu_vertex)))
        else: # we are not running the vertex reconstruction, and instead use the simulated vertex - useful if only testing the direction reconstruction
            sim_shower = evt.get_first_sim_shower()
            logger.warning("!!! Setting reconstructed vertex to simulated vertex!!!")

            # shift to xmax
            def get_xmax(energy):
                return 0.25 * np.log(energy) - 2.78

            shower_axis = -hp.spherical_to_cartesian(zenith_sim, azimuth_sim)
            vertex_sim = sim_shower[shp.vertex]
            xmax_sim = vertex_sim + get_xmax(E_sh)*shower_axis

            station.set_parameter(stnp.nu_vertex, vertex_sim) ### FOR DEBUGGING ONLY!
            # sim_shower.set_parameter(shp.vertex, xmax_sim)
        results['x'] = station.get_parameter(stnp.nu_vertex)
        t1 = time.time()


        ### Direction reconstruction
        if run_direction_reco:
            passband = reco_config['direction']['passband']
            Hpol = channel_dict['hpol_channels']

            raytypeselecter.begin(debug=args.debug, debugplots_path=output_path)

            # TODO This needs some tidying up still...
            # The reference channel used to determine the pulse position by the ray type selecter
            # should be the phased array channel closest to the hpol channel, because this will
            # be used to infer the pulse arrival time in the hpol. However, in some rare geometries
            # only some of the phased-array channels actually get a signal from the neutrino vertex,
            # which raises an UnboundLocalError. In this case, we start with the lowest phased-array channel
            # instead
            try:
                raytypeselecter.run(
                    evt, station, det, vrms=None, use_channels=channel_dict['phased_array'][::-1], # using the top phased-array channel as reference channel
                    sim = False, template = direction_pulse_template,
                    ice_model=prop_config['propagation']['ice_model'], attenuation_model=prop_config['propagation']['attenuation_model'],
                    )
                reference_vpol = channel_dict['phased_array'][-1]
            except UnboundLocalError: # no RT solutions for given channel:
                raytypeselecter.run(
                    evt, station, det, vrms=None, use_channels=channel_dict['phased_array'], # using the bottom phased-array channel as reference channel
                    sim = False, template = direction_pulse_template,
                    ice_model=prop_config['propagation']['ice_model'], attenuation_model=prop_config['propagation']['attenuation_model'],
                    )
                reference_vpol = channel_dict['phased_array'][0]
            
            # to keep some debug plots happy, one can either re-run the raytypeselecter using sim=True, or manually set the missing parameters
            station.set_parameter(stnp.raytype_sim, station[stnp.raytype])
            station.set_parameter(stnp.pulse_position_sim, station[stnp.pulse_position])
            # try:
            #     raytypeselecter.run(
            #         evt, station, det, vrms=None, use_channels=channel_dict['phased_array'][::-1],
            #         sim = True, template = direction_pulse_template,
            #         ice_model=prop_config['propagation']['ice_model'], attenuation_model=prop_config['propagation']['attenuation_model'],
            #         )
            # except UnboundLocalError:
            #     raytypeselecter.run(
            #         evt, station, det, vrms=None, use_channels=channel_dict['phased_array'],
            #         sim = True, template = direction_pulse_template,
            #         ice_model=prop_config['propagation']['ice_model'], attenuation_model=prop_config['propagation']['attenuation_model'],
            #         )

            results['ray_type'] = station.get_parameter(stnp.raytype)
            results['ray_type_sim'] = station.get_parameter(stnp.raytype_sim)
            results['ch_Vpol'] = reference_vpol


            neutrinodirectionreconstructor.begin(
                evt, station, det, use_channels=channel_dict['all'],
                reference_vpol=reference_vpol,
                pa_cluster_channels=channel_dict['phased_array_cluster'],
                hpol_channels=channel_dict['hpol_channels'],
                propagation_config=prop_config,
                vrms=None,
                **reco_config['direction'],
                debug_folder = output_path,
                debug_formats=['.pdf']
            )
            try:
                neutrinodirectionreconstructor.run(debug=args.debug)
            except ValueError as e:
                logger.warning("Direction reconstruction failed. Error message:")
                logger.exception(e)
            try:
                results['zenith'] = station.get_parameter(stnp.nu_zenith)
                results['azimuth'] = station.get_parameter(stnp.nu_azimuth)
                results['E'] = station.get_parameter(stnp.nu_energy)
                results['viewing_angle'] = station.get_parameter(stnp.viewing_angle)
                pol_rec = station.get_parameter(stnp.polarization)
                results['polarization_angle'] = np.arctan2(pol_rec[2], pol_rec[1])
                results['chi2_fit'] = station[stnp.chi2]['chi2']
                results['shower_type'] = station.get_parameter(stnp.shower_type)

                additional_output = station.get_parameter(stnp.nu_reco_additional_output)
                for key in additional_output:
                    results[key] = additional_output[key]


            except KeyError as e:
                logger.exception(e)

        if args.save_nur:
            eventwriter.run(
                evt, mode=dict(
                    Channels=False,
                    ElectricFields=False,
                    SimChannels=False,
                    SimElectricFields=False
                )
            )
        t2 = time.time()

        print("Took {:.0f} s ({:.0f} s vertex fit / {:.0f} s direction reco)".format(t2-t0, t1-t0, t2-t1))

        import pickle
        with open('./reconstruction_results.p', 'wb') as f:
            pickle.dump(results, f)

    if args.save_nur:
        logger.info(f"Finished run {run_number}. Writing output nur to {nur_output_path}...")
        eventwriter.end()
