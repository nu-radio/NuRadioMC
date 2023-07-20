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
from NuRadioReco.modules.neutrinoVertexReconstructor import neutrino3DVertexReconstructor_v2 as neutrino3DVertexReconstructor
from NuRadioReco.modules.neutrinoDirectionReconstruction import rayTypeSelecter, neutrinoDirectionReconstructor
from NuRadioReco.modules import channelBandPassFilter, channelResampler, channelGenericNoiseAdder
import NuRadioMC.utilities.medium
import NuRadioMC.SignalGen.askaryan
import NuRadioReco.framework.base_trace
import os
import sys
import time
import yaml
import argparse
from scipy import constants
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
    channel_dict = {}
    channel_dict['PA_ids_baseline'] = np.array([0,1,2,3])
    channel_dict['vpol_ids_baseline'] = np.array([0,1,2,3,4,5,6,9,10,12,13])
    channel_dict['hpol_ids_baseline'] = np.array([7,8,11,14])

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
    argparser.add_argument(
        "--run", "-r", type=int, nargs="+", help="Run numbers to reconstruct. If not specified, reconstruction is performed for all runs in the input .nur file")
    argparser.add_argument("--overwrite", "-o", action='store_true', help="Rerun existing files if direction reco is missing")
    argparser.add_argument("--hard-overwrite", "-O", action='store_true', help="Overwrite existing results files")
    args = argparser.parse_args()

    overwrite = int(args.overwrite) + 2 * int(args.hard_overwrite)


    # reconstruction settings
    sys.path.append(args.output_path)
    from reco_cfg import config as cfg

    debug_vertex = cfg["debug_vertex"]
    debug_direction = cfg["debug_direction"]
    noise_level_before_amp = cfg["noise_level_before_amp"]
    save_nur_output = cfg["save_nur_output"]
    restricted_input = cfg["restricted_input"]
    add_noise = cfg["add_noise"]
    add_noise_only = cfg["add_noise_only"]
    no_sim_station = cfg["no_sim_station"]
    apply_bandpass_filters = cfg["apply_bandpass_filters"]

    eventreader = eventReader.eventReader()
    eventreader.begin(args.filename)
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

    # make template
    n_samples = 1024
    viewing_angle = 1.5 * units.deg
    sampling_rate = cfg['sampling_rate'] #
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
    vpol_antenna = antenna_provider.load_antenna_pattern(det.get_antenna_model(station_id, 0))
    antenna_response = vpol_antenna.get_antenna_response_vectorized(
        efield_template.get_frequencies(),
        70 * units.deg,
        0*units.deg,
        *det.get_antenna_orientation(station_id, 0)
    )['theta']
    amp_response = det.get_amplifier_response(station_id, 0, efield_template.get_frequencies())
    filt = bandpassfilter.get_filter(
        efield_template.get_frequencies(), None, None, None,
        passband=cfg['direction']['passband'], filter_type='butterabs', order=10)

    direction_pulse_template = fft.freq2time(efield_spec * antenna_response * amp_response * filt, sampling_rate)

    # import matplotlib.pyplot as plt
    # plt.plot(efield_template.get_frequencies(), abs(efield_template.get_frequency_spectrum()))
    # plt.show()
    # plt.plot(efield_template.get_frequencies(), abs(fft.time2freq(direction_pulse_template, sampling_rate)))
    # plt.show()
    # raise ValueError

    # initialize vertex reconstructor
    vertex3d = neutrino3DVertexReconstructor.neutrino3DVertexReconstructor(args.lookup_table_path)

    # run reco
    for evt in eventreader.run():
        run_number = evt.get_run_number()
        if len(run_ids):
            if run_number not in run_ids:
                continue
        particle = evt.get_primary()
        E_nu = particle.get_parameter(pap.energy)
        csv_filename = "{}_{:06d}_{:02d}.csv".format(file_id, run_number, evt.get_id())
        output_csv_dir = args.output_path
        if not os.path.exists(output_csv_dir):
            subprocess.run(["mkdir", output_csv_dir])
        df = pd.DataFrame(
            columns=[
                'E_nu', 'E_sh', 'file_id', 'run_number', 'event_id', 'weight',
                'zenith_sim', 'azimuth_sim', 'x_sim', 'y_sim', 'z_sim',
                'zenith', 'azimuth', 'x', 'y', 'z', 'E', 'vw_sim','pol_sim', 'vw', 'pol',
                "corr_fit", "corr_sim", "corr_max", "corr_dnr_max", "ray_type","ray_type_sim", 
                "chi2_sim", "chi2_fit", "dof", "fit_channels", 'ch_Vpol', 'ch_Vpol_sim', 'shower_type'], dtype=object)
        event_id = evt.get_id()
        weight = particle.get_parameter(pap.weight)
        zenith_sim = particle.get_parameter(pap.zenith)
        azimuth_sim = particle.get_parameter(pap.azimuth)
        x_sim, y_sim, z_sim = particle.get_parameter(pap.vertex)
        E_sh = evt.get_first_sim_shower().get_parameter(shp.energy)

        data = [E_nu, E_sh, file_id, run_number, event_id, weight, zenith_sim, azimuth_sim, x_sim, y_sim, z_sim]
        df.loc[0,'E_nu':'z_sim'] = data

        csv_path = os.path.join(output_csv_dir, csv_filename)
        if os.path.exists(csv_path) and (overwrite < 1):
            logger.warning("File {} exists, skipping...".format(csv_path))
            continue
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            df.reset_index(drop=True, inplace=True)
        if (not np.isnan(df.loc[0,'zenith'])) and (overwrite < 2):
            logger.info(f"Reconstruction already completed for run {run_number}, skipping...")
            continue

        run_vertex_reco = args.vertex & (np.isnan(df.loc[0, 'x']) or (overwrite > 1))
        run_direction_reco = args.direction & (np.isnan(df.loc[0, 'zenith']) or (overwrite > 1))

        logger.info("Starting reconstruction for run {}".format(run_number))

        station = evt.get_station(station_id)
        if no_sim_station:
            station.set_sim_station(None)
        else:
            sim_station = station.get_sim_station()


        if add_noise: # if the simulations were run without noise, we may want to add noise here
            #TODO - make sure min_freq, max_freq are  correct!
            for channel in station.iter_channels():
                noise_fft = noiseadder.bandlimited_noise(
                    min_freq=0.001, max_freq=1.2*units.GHz, n_samples=channel.get_number_of_samples(),
                    sampling_rate=channel.get_sampling_rate(), amplitude=noise_level_before_amp,
                    type='rayleigh', time_domain=False,
                    )
                amp_response = det.get_amplifier_response(station_id, channel.get_id(), channel.get_frequencies())
                channel.set_frequency_spectrum(
                    channel.get_frequency_spectrum() + noise_fft * amp_response, channel.get_sampling_rate()
                )

        if add_noise_only and save_nur_output:
            eventwriter.run(evt)
            continue

        station.set_is_neutrino()

        resampler.begin()
        resampler.run(evt, station, det, sampling_rate=sampling_rate)
        cabledelayadder.run(evt, station, det, mode='subtract')
        # same treatment for sim_station - useful for debugging
        if station.has_sim_station():
            resampler.run(evt, sim_station, det, sampling_rate=sampling_rate)
            cabledelayadder.run(evt, sim_station, det, mode='subtract')


        t0 = time.time()
        ### vertex reconstruction
        vertex_channels = channel_dict['vpol_ids_baseline']

        ### Reduce the number of vertex channels
        # min_channel_distance = 3 * units.m
        # use_channels = []
        # channel_pos = []
        # channel_max_amp = []
        # for channel_id in vertex_channels:
        #     channel = station.get_channel(channel_id)
        #     channel_pos_i = det.get_relative_position(station_id, channel_id)
        #     channel_max_amp_i = channel.get_parameter(chnp.maximum_amplitude)
        #     mask = np.array([np.linalg.norm(channel_pos_i - j) for j in channel_pos]) < min_channel_distance
        #     if np.sum(mask):
        #         continue
        #         ind = np.where(mask)[0][0]
        #         if channel_max_amp[ind] > channel_max_amp_i:
        #             continue
        #         else:
        #             use_channels.pop(ind)
        #             channel_pos.pop(ind)
        #             channel_max_amp.pop(ind)
        #     use_channels.append(channel_id)
        #     channel_pos.append(channel_pos_i)
        #     channel_max_amp.append(channel_max_amp_i)

        # vertex_channels = use_channels

        if run_vertex_reco:
            vertex3d.begin(
                station_id, vertex_channels, det, template=efield_template,
                debug_folder=output_csv_dir,
                **cfg['vertex']
            )

            vertex3d.run(evt, station, det, debug=debug_vertex)

            sim_shower = evt.get_first_sim_shower()

            reco_vertex = station.get_parameter(stnp.nu_vertex)
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            try:
                corr = station.get_parameter(stnp.vertex_correlation_sums)
                df.loc[0, "corr_fit":"corr_dnr_max"] = corr
            except KeyError:
                pass
            logger.debug("sim_vertex: {}".format(sim_shower.get_parameter(shp.vertex)))
            logger.debug("reco vertex: {}".format(station.get_parameter(stnp.nu_vertex)))
        else:
            sim_shower = evt.get_first_sim_shower()
            if not np.isnan(df.loc[0, 'x']):
                station.set_parameter(stnp.nu_vertex, df.loc[0, 'x':'z'].values)
            else:
                logger.warning("Setting reconstructed vertex to simulated vertex!")
                ### shift to xmax
                def get_xmax(energy):
                    return 0.25 * np.log(energy) - 2.78

                shower_axis = -hp.spherical_to_cartesian(zenith_sim, azimuth_sim)
                vertex_sim = sim_shower[shp.vertex]
                xmax_sim = vertex_sim + get_xmax(E_sh)*shower_axis

                station.set_parameter(stnp.nu_vertex, vertex_sim) ### FOR DEBUGGING ONLY!
                # sim_shower.set_parameter(shp.vertex, xmax_sim)

        df.loc[0, 'x':'z'] = station.get_parameter(stnp.nu_vertex)
        # export to csv
        df.to_csv(csv_path)
        t1 = time.time()


        ### Direction reconstruction
        if run_direction_reco:
            with open("./input/config_RNOG.yaml", 'r') as f: #TODO: fix hardcoding
                prop_config = yaml.load(f, Loader=yaml.FullLoader)
            passband = cfg['direction']['passband']

            # compute Vrms for direction reconstructor:
            # noise_level = noise_level_before_amp # T = 300, bandwidth = 973.92 MHz # should be 14.2 for 2 GHz / 15.8 for 2.4
            ff = np.linspace(0, 2*units.GHz, 1000)
            amp_response = det.get_amplifier_response(station_id, channel_id=4, frequencies=ff)
            # filt = bandpassfilter.get_filter(
            #     ff, station_id, channel_id=4, det=det,
            #     passband=passband, filter_type='butterabs', order=10)
            ### currently use two separate bandpass filters in analytic_pulse
            filt1 = bandpassfilter.get_filter(
                ff, station_id, None, None,
                passband=[passband[0], 1150*units.MHz], filter_type='butter', order=8
            )
            filt2 = bandpassfilter.get_filter(
                ff, station_id, None, None,
                passband=[0, passband[1]], filter_type='butter', order=10
            )
            filt = filt1 * filt2
            bandwidth_filt = np.trapz(np.abs(amp_response * filt)**2, ff)
            Vrms_lowBW = (300 * 50 * constants.k * bandwidth_filt / units.Hz) ** 0.5
            print("Vrms: {:.3f} uV".format(Vrms_lowBW / units.microvolt))
            noise_level = Vrms_lowBW

            Hpol = channel_dict['hpol_ids_baseline']

            if apply_bandpass_filters: # if the data has already been bandpass filtered, don't rerun this
                bandpassfilter.run(evt, station, det, passband = [passband[0], 1150*units.MHz], filter_type = 'butter', order = 8)
                bandpassfilter.run(evt, station, det, passband = [0, passband[1]], filter_type = 'butter', order = 10)
                if not no_sim_station:
                    bandpassfilter.run(evt, sim_station, det, passband = [passband[0], 1150*units.MHz], filter_type = 'butter', order = 8)
                    bandpassfilter.run(evt, sim_station, det, passband = [0, passband[1]], filter_type = 'butter', order = 10)

            shower_ids = [sh.get_id() for sh in evt.get_sim_showers()]
            raytypeselecter.begin(debug=debug_direction, debugplots_path=output_csv_dir)
            try:
                raytypeselecter.run(
                    evt, station, det, noise_rms=noise_level, use_channels=channel_dict['PA_ids_baseline'][::-1],
                    sim = False, template = direction_pulse_template, icemodel='greenland_simple', att_model='GL1',
                    )
                ch_Vpol_rec = 3
            except UnboundLocalError: # no RT solutions for given channel:
                raytypeselecter.run(
                    evt, station, det, noise_rms=noise_level, use_channels=channel_dict['PA_ids_baseline'],
                    sim = False, template = direction_pulse_template, icemodel='greenland_simple', att_model='GL1',
                    )
                ch_Vpol_rec = 0
            
            ### this helps keep debug plots happy
            try:
                raytypeselecter.run(
                    evt, station, det, noise_rms=noise_level, use_channels=channel_dict['PA_ids_baseline'][::-1],
                    sim = True, template = direction_pulse_template, icemodel='greenland_simple', att_model='GL1',
                    )
                ch_Vpol_sim = 3
            except UnboundLocalError:
                raytypeselecter.run(
                    evt, station, det, noise_rms=noise_level, use_channels=channel_dict['PA_ids_baseline'],
                    sim = True, template = direction_pulse_template, icemodel='greenland_simple', att_model='GL1',
                    )
                ch_Vpol_sim = 0
            df.loc[0, 'ray_type'] = station.get_parameter(stnp.raytype)
            df.loc[0, 'ray_type_sim'] = station.get_parameter(stnp.raytype_sim)
            df.loc[0, 'ch_Vpol'] = ch_Vpol_rec
            df.loc[0, 'ch_Vpol_sim'] = ch_Vpol_sim
            if ch_Vpol_sim < ch_Vpol_rec: # we'll get errors from the sim station ray tracing
                station.set_sim_station(None)

            use_channels = np.arange(15)

            # use_channels = reco_channels
            print("Direction reconstruction using {} channels:".format(len(use_channels)), use_channels)
            PA_cluster_channels = np.concatenate([channel_dict['PA_ids_baseline'], [7,8]]) # PA + adjacent Hpols

            neutrinodirectionreconstructor.begin(
                evt, station, det, use_channels=use_channels,
                reference_Vpol=ch_Vpol_rec,
                reference_Hpol=np.min(Hpol),
                PA_cluster_channels=PA_cluster_channels,
                sim=False, template=False, Hpol_channels=Hpol,
                propagation_config=prop_config,
                restricted_input = restricted_input,
                Vrms_Vpol=noise_level, Vrms_Hpol=noise_level,
                **cfg['direction'], debug_formats=['.pdf']
            )
            try:
                neutrinodirectionreconstructor.run(debug_path = output_csv_dir, debug=debug_direction)
            except ValueError as e:
                logger.warning("Direction reconstruction failed. Error message:")
                logger.exception(e)
            try:
                df.loc[0, 'zenith'] = station.get_parameter(stnp.nu_zenith)
                df.loc[0, 'azimuth'] = station.get_parameter(stnp.nu_azimuth)
                df.loc[0, 'E'] = station.get_parameter(stnp.nu_energy)
                df.loc[0, ['vw_sim', 'vw']] = station.get_parameter(stnp.viewing_angle)
                pol_sim, pol_rec = station.get_parameter(stnp.polarization)
                df.loc[0, 'pol_sim'] = np.arctan2(pol_sim[2], pol_sim[1])
                df.loc[0, 'pol'] = np.arctan2(pol_rec[2], pol_rec[1])
                chi2_list = station[stnp.chi2]
                chi2_sim = chi2_list[0]
                chi2_fit = chi2_list[1]
                df.loc[0, 'chi2_sim'] = chi2_sim
                df.loc[0, 'chi2_fit'] = chi2_fit
                df.loc[0, 'dof'] = station[stnp.extra_channels]
                df.loc[0, 'fit_channels'] = str(station.get_parameter(stnp.direction_fit_pulses)) # need to convert back to dict!
                df.loc[0, 'shower_type'] = station.get_parameter(stnp.ccnc)
            except KeyError as e:
                logger.exception(e)

            df.to_csv(csv_path)
        if save_nur_output:
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
    logger.info(f"Finished run {run_number}. Writing output nur to {nur_output_path}...")
    eventwriter.end()
