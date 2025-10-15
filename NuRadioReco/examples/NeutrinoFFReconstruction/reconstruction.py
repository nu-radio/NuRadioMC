"""
Example forward-folding neutrino reconstruction script.

"""

import os
import time
import argparse
import numpy as np
import logging

import NuRadioReco.modules.io.eventReader as eventReader
import NuRadioReco.modules.io.eventWriter as eventWriter
import NuRadioReco.framework.parameters as parameters
from NuRadioReco.utilities import units, fft
import NuRadioReco.detector.detector as detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.modules import channelAddCableDelay
from NuRadioReco.modules.neutrinoVertexReconstructor import neutrino3DVertexReconstructor
from NuRadioReco.modules.neutrinoDirectionReconstructor import rayTypeSelecter, neutrinoDirectionReconstructor
from NuRadioReco.modules import channelBandPassFilter, channelResampler, channelGenericNoiseAdder
import NuRadioMC.utilities.medium
import NuRadioMC.SignalGen.askaryan
import NuRadioReco.framework.base_trace
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

# Uncomment the following to show additional debug outputs
# directionrecologger = logging.getLogger('NuRadioReco.neutrinoDirectionReconstructor')
# directionrecologger.setLevel(logging.DEBUG)

# vertexlogger = logging.getLogger('NuRadioReco.neutrinoVertexReconstructor')
# vertexlogger.setLevel(logging.DEBUG)

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

    # We also need to specify:
    # - The signal windows: the size of the window around the pulse position to use in the fit
    #   This depends in general on the group delays of the antenna + signal chain and may be different for vpol and hpol antennas
    # - The sampling rate: we upsample the data before fitting in the time domain
    # - Any other non-default settings to pass to the NeutrinoVertexReconstructor or NeutrinoDirectionReconstructor

    window_vpol = [-20*units.ns, 20*units.ns]
    window_hpol = [0*units.ns, 25*units.ns]
    sampling_rate = 6.4 * units.GHz

    # see NuRadioReco.modules.neutrinoVertexReconstructor.neutrino3DVertexReconstructor for a description of all parameters
    vertex_settings = dict(
        passband=[80*units.MHz, 300*units.MHz], # using a bandpass filter with a relatively low highpass (increases sensitivity for antennas which are more off-cone)
        distances_2d=[100*units.m, 6000*units.m] # the default search distance is too small to include some of the expected vertices at higher energies.
    )

    # see NuRadioReco.modules.neutrinoDirectionReconstructor.neutrinoDirectionReconstructor for a description of all parameters
    direction_settings = dict(
        passband=[70*units.MHz, 700*units.MHz], # bandpass-filtering may improve SNR
        brute_force=False, # quicker
        use_fallback_timing=True, # include more Hpol channels by using approximate timing from Vpols
        fit_shower_type=True, # run another fit iteration with an EM shower as the fit hypothesis - may improve fit for EM events
        )

    ### some inputs can be passed interactively
    argparser = argparse.ArgumentParser(
        "Reconstruction",
        description="Script to run neutrino reconstruction, suitable for parallelization."
    )
    argparser.add_argument("filename", type=str, help="path to .nur file to reconstruct")
    argparser.add_argument("detector", type=str, help="Path to detector description.")
    argparser.add_argument("prop_config", type=str, help="Path to propagation settings (ice model, raytracer, attenuation model) to use in reconstruction.")
    argparser.add_argument("station_id", type=int, help="Station id of station to use for reconstruction.")
    argparser.add_argument("lookup_table_path", type=str, help="Path to lookup tables.")
    argparser.add_argument("--output_path", '-o', type=str, help="Where to store outputs (results, debug plots, nur files).", default='./output')
    argparser.add_argument("--vertex", "-v", action='store_true', help="Reconstruct vertex position.")
    argparser.add_argument("--direction", "-d", action='store_true', help="Reconstruct neutrino direction.")
    argparser.add_argument("--save_nur", action='store_true', help="Store nur output file")
    argparser.add_argument("--debug", action='store_true', help="Create debug plots")
    argparser.add_argument(
        "--run", "-r", type=int, nargs="+", help="Run numbers to reconstruct. If not specified, reconstruction is performed for all runs in the input .nur file")

    args = argparser.parse_args()

    run_vertex_reco = args.vertex
    run_direction_reco = args.direction

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

    # For the vertex reconstruction, we need an electric field template
    # We use the SignalGen.askaryan module to generate a template of an askaryan pulse
    n_samples = 1024
    viewing_angle = 1.5 * units.deg
    sampling_rate = det.get_sampling_frequency(station_id, channel_dict['all'][0])

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
    vpol_antenna = antenna_provider.load_antenna_pattern(det.get_antenna_model(station_id, channel_dict['phased_array'][0]))
    antenna_response = vpol_antenna.get_antenna_response_vectorized(
        efield_template.get_frequencies(),
        70 * units.deg,
        0*units.deg,
        *det.get_antenna_orientation(station_id, channel_dict['phased_array'][0])
    )['theta']
    amp_response = det.get_amplifier_response(station_id, channel_dict['phased_array'][0], efield_template.get_frequencies())

    direction_pulse_template = fft.freq2time(efield_spec * antenna_response * amp_response, sampling_rate)

    # initialize vertex reconstructor
    vertex3d = neutrino3DVertexReconstructor.neutrino3DVertexReconstructor(args.lookup_table_path)

    # run reco
    for evt in eventreader.run():
        run_number = evt.get_run_number()
        if len(run_ids):
            if run_number not in run_ids:
                continue

        output_path = args.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        results = dict()
        particle = evt.get_primary()
        results['file_id'] = file_id
        results['run_number'] = run_number
        results['event_id'] = evt.get_id()

        results['E_nu_sim'] = particle.get_parameter(pap.energy)
        results['weight'] = particle.get_parameter(pap.weight)
        results['zenith_sim'] = particle.get_parameter(pap.zenith)
        results['azimuth_sim'] = particle.get_parameter(pap.azimuth)
        results['vertex_sim'] = particle.get_parameter(pap.vertex)
        results['E_sh_sim'] = evt.get_first_sim_shower().get_parameter(shp.energy)

        logger.info("Starting reconstruction for run {}".format(run_number))

        station = evt.get_station(station_id)
        sim_station = station.get_sim_station()
        station.set_is_neutrino()

        resampler.begin()
        resampler.run(evt, station, det, sampling_rate=sampling_rate)
        cabledelayadder.run(evt, station, det, mode='subtract')

        # same treatment for sim_station - useful for debugging
        if station.has_sim_station():
            resampler.run(evt, sim_station, det, sampling_rate=sampling_rate)
            cabledelayadder.run(evt, sim_station, det, mode='subtract')

        t0 = time.time()
        ### Step 1: Vertex reconstruction

        # we only use the vpol channels in the vertex reconstruction
        vertex_channels = [
            channel_id for channel_id in channel_dict['all']
            if channel_id not in channel_dict['hpol_channels']
            ]

        if run_vertex_reco:
            vertex3d.begin(
                station_id, vertex_channels, det, template=efield_template,
                **vertex_settings,
                debug_folder=output_path,
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

            vertex_sim = sim_shower[shp.vertex]
            station.set_parameter(stnp.nu_vertex, vertex_sim) ### FOR DEBUGGING ONLY!

        results['vertex'] = station.get_parameter(stnp.nu_vertex)
        t1 = time.time()


        ### Step 2: Direction reconstruction
        if run_direction_reco:
            raytypeselecter.begin(
               propagation_config=args.prop_config,
               debug=args.debug, debugplots_path=output_path)

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
                    )
                reference_vpol = channel_dict['phased_array'][-1]
            except UnboundLocalError: # no RT solutions for given channel:
                raytypeselecter.run(
                    evt, station, det, vrms=None, use_channels=channel_dict['phased_array'], # using the bottom phased-array channel as reference channel
                    sim = False, template = direction_pulse_template,
                    )
                reference_vpol = channel_dict['phased_array'][0]

            # to keep some debug plots happy, one can either re-run the raytypeselecter using sim=True, or manually set the missing parameters
            station.set_parameter(stnp.raytype_sim, station[stnp.raytype])
            station.set_parameter(stnp.pulse_position_sim, station[stnp.pulse_position])
            # try:
            #     raytypeselecter.run(
            #         evt, station, det, vrms=None, use_channels=channel_dict['phased_array'][::-1],
            #         sim = True, template = direction_pulse_template,
            #         )
            # except UnboundLocalError:
            #     raytypeselecter.run(
            #         evt, station, det, vrms=None, use_channels=channel_dict['phased_array'],
            #         sim = True, template = direction_pulse_template,
            #         )

            results['ray_type'] = station.get_parameter(stnp.raytype)
            results['ray_type_sim'] = station.get_parameter(stnp.raytype_sim)
            results['reference_antenna'] = reference_vpol



            neutrinodirectionreconstructor.begin(
                evt, station, det, use_channels=channel_dict['all'],
                reference_vpol=reference_vpol,
                pa_cluster_channels=channel_dict['phased_array_cluster'],
                hpol_channels=channel_dict['hpol_channels'],
                propagation_config=args.prop_config,
                vrms=None,
                **direction_settings,
                debug_folder = output_path,
                debug_formats=['.pdf']
            )
            try:
                neutrinodirectionreconstructor.run(evt, station, det, debug=args.debug)
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
                evt,
                # mode=dict( # uncomment to only save reconstructed parameters, no voltage traces (resulting in much smaller file size)
                #     Channels=False,
                #     ElectricFields=False,
                #     SimChannels=False,
                #     SimElectricFields=False)
            )
        t2 = time.time()

        print("Took {:.0f} s ({:.0f} s vertex fit / {:.0f} s direction reco)".format(t2-t0, t1-t0, t2-t1))


        ## Some code to export the results
        ## If you are reconstructing multiple events something like a pandas DataFrame might be more convenient
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.float32):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

    if args.save_nur:
        logger.info(f"Finished run {run_number}. Writing output nur to {nur_output_path}...")
        eventwriter.end()
