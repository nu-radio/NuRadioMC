from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrans
from NuRadioMC.SignalGen import parametrizations as signalgen
from NuRadioMC.utilities import units
from NuRadioMC.SignalProp import analyticraytraycing as ray
from NuRadioMC.utilities import medium
from NuRadioMC.utilities import fft
from NuRadioMC.EvtGen.weight import get_weight
from matplotlib import pyplot as plt
import h5py
import argparse
import json
import time
import os
from scipy import constants
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.detector.detector as detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sim")

evt_time = datetime.datetime(2018, 1, 1)

debug = False

PLOT_FOLDER = 'plots'
if not os.path.isdir(PLOT_FOLDER):
    PLOT_FOLDER = './'

VERSION = 0.1
Tnoise = 350.  # define noise temperature, the noise Vrms will be calculated depending on the bandwidth of the detector

# # initialize detector simulation modules
# det = detector.Detector(json_filename)
# evt_time = datetime.datetime(2018, 1, 1)  # the time, relevant to determine the station configuration from the station id
# det.update(evt_time)


def add_empty_channel(sim_station, channel_id):
    channel = NuRadioReco.framework.channel.Channel(channel_id)
    channel.set_frequency_spectrum(np.zeros((3, len(ff)), dtype=np.complex), 1. / dt)
    channel['azimuth'] = 0
    channel['zenith'] = 180 * units.deg
    channel['raypath'] = 'none'
    sim_station.add_channel(channel)


def get_em_had_fraction(inelasticity, ccnc, flavor):
    """
    calculates the fraction of the neutrino energy that goes into the
    electromagnetic cascade (em) and the hadronic cascade (had)

    Parameters
    ----------
    inelasticity: float
        the inelasticity (fraction of energy that goes into had. cascade)
    ccnc: string ['nc', 'cc']
        neutral current (nc) or carged currend (cc) interaction
    flavor: int
        flavor id

    returns
    --------
    fem: float
        electrogmatnetic fraction
    fhad: float
        hadroninc fraction
    """
    fem = 0  # electrogmatnetic fraction
    fhad = 0  # hadroninc fraction
    if(ccnc == 'nc'):
        fhad = inelasticity
    else:
        if(np.abs(flavor) == 12):
            fem = (1 - inelasticity)
            fhad = inelasticity
        elif(np.abs(flavor) == 14):
            fhad = inelasticity
        elif(np.abs(flavor) == 16):
            fhad = inelasticity
    return fem, fhad


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

# read in detector positions
det = detector.Detector(json_filename=args.detectordescription)
station_id = 101

# read time and frequency resolution from detector (assuming all channels have the same sampling)
dt = 1. / det.get_sampling_frequency(station_id, 0)
bandwidth = 0.5 / dt
n_samples = det.get_number_of_samples(station_id, 0)
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)

Vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverterPerChannel.begin(debug=False)
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
if(args.outputfilenameNuRadioReco is not None):
    eventWriter.begin(args.outputfilenameNuRadioReco)

fin = h5py.File(args.inputfilename, 'r')
n_events = len(fin['event_ids'])
n_antennas = det.get_number_of_channels(station_id)

# define arrays that will be saved at the end
weights = np.zeros(n_events)
triggered = np.zeros(n_events, dtype=np.bool)
launch_vectors = np.zeros((n_events, n_antennas, 2, 3))
receive_vectors = np.zeros((n_events, n_antennas, 2, 3))
ray_tracing_C0 = np.zeros((n_events, n_antennas, 2))
ray_tracing_C1 = np.zeros((n_events, n_antennas, 2))
ray_tracing_solution_type = np.zeros((n_events, n_antennas, 2), dtype=np.int)
polarization = np.zeros((n_events, n_antennas, 2))
travel_times = np.zeros((n_events, n_antennas, 2))
travel_distances = np.zeros((n_events, n_antennas, 2))
SNRs = np.zeros(n_events)

t_start = time.time()
for iE in range(n_events):
    if(iE > 0 and iE % 1000 == 0):
        eta = datetime.timedelta(seconds=(time.time() - t_start) * (n_events - iE) / iE)
        logger.info("processing event {}/{} = {}%, ETA {}".format(iE, n_events, 100. * iE / n_events, eta))
    # read all quantities from hdf5 file and store them in local variables
    event_id = fin['event_ids'][iE]
    flavor = fin['flavors'][iE]
    energy = fin['energies'][iE]
    ccnc = fin['ccncs'][iE]
    x = fin['xx'][iE]
    y = fin['yy'][iE]
    z = fin['zz'][iE]
    zenith_nu = fin['zeniths'][iE]
    azimuth_nu = fin['azimuths'][iE]
    inelasticity = fin['inelasticity'][iE]

    # be careful, zenith/azimuth angle always refer to where the neutrino came from,
    # i.e., opposite to the direction of propagation. We need the propagation directio nhere,
    # so we multiply the shower axis with '-1'
    shower_axis = -1 * hp.spherical_to_cartesian(zenith_nu, azimuth_nu)
    x1 = np.array([x, y, z])

    # calculate correct chereknov angle for ice density at vertex position
    ice = medium.southpole_simple()
    n_index = ice.get_index_of_refraction(x1)
    rho = np.arccos(1. / n_index)

    # create NuRadioReco event structure
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    # save relevant neutrino properties
    sim_station['zenith_nu'] = zenith_nu
    sim_station['azimuth_nu'] = azimuth_nu
    sim_station['energy'] = energy
    sim_station['flavor'] = flavor
    sim_station['ccnc'] = ccnc
    sim_station['vertex'] = np.array([x, y, z])
    sim_station['inelasticity'] = inelasticity

    candidate_event = False

    # first step: peorform raytracing to see if solution exists
    for channel_id in range(det.get_number_of_channels(station_id)):
        x2 = det.get_relative_position(station_id, channel_id)
        r = ray.ray_tracing(x1, x2, ice, log_level=logging.WARNING)
        if(not r.has_solution()):
            logger.debug("event {} and station {}, channel {} does not have any ray tracing solution".format(event_id, station_id, channel_id))
            add_empty_channel(sim_station, channel_id)
            continue
        dRhos = []
        viewing_angles = []
        for iS in range(r.get_number_of_solutions()):  # loop through all ray tracing solution
            ray_tracing_C0[iE, channel_id, iS] = r.get_results()[iS]['C0']
            ray_tracing_C1[iE, channel_id, iS] = r.get_results()[iS]['C1']
            ray_tracing_solution_type[iE, channel_id, iS] = r.get_solution_type(iS)
            launch_vector = r.get_launch_vector(iS)
            launch_vectors[iE, channel_id, iS] = launch_vector
            viewing_angle = hp.get_angle(shower_axis, launch_vector)  # calculates angle between shower axis and launch vector
            viewing_angles.append(viewing_angle)
            dRho = (viewing_angle - rho)
            logger.debug('solution {} {}: viewing angle {:.1f} = dRho = {:.1f}'.format(iS, ray.solution_types[r.get_solution_type(iS)], viewing_angle / units.deg, (viewing_angle - rho) / units.deg))
            dRhos.append(dRho)
        # discard event if dRho (angle off cherenkov cone) is too large
        if(min(np.abs(dRhos)) > 30 * units.deg):
            logger.debug('dRho too large, event unlikely to be observed, skipping event')
            add_empty_channel(sim_station, channel_id)
            continue

        n = r.get_number_of_solutions()
        Rs = np.zeros(n)
        Ts = np.zeros(n)
        tts = np.zeros((n, n_samples))
        for iS in range(n):  # loop through all ray tracing solution
            R = r.get_path_length(iS)  # calculate path length
            Rs[iS] = R
            T = r.get_travel_time(iS)  # calculate travel time
            travel_distances[iE, channel_id, iS] = R
            travel_times[iE, channel_id, iS] = T
            Ts[iS] = T
            receive_vector = r.get_receive_vector(iS)
            receive_vectors[iE, channel_id, iS] = receive_vector  # save receive vector
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            logger.debug("R = {:.1f} m, t = {:.1f}ns, receive angles {:.0f} {:.0f}".format(R / units.m, T / units.ns, zenith / units.deg, azimuth / units.deg))
#             eR, eTheta, ePhi = get_frequency_spectrum(energy, viewing_angles[iS], ff, 0, n, R)

            fem, fhad = get_em_had_fraction(inelasticity, ccnc, flavor)
            # get neutrino pulse from Askaryan module
#             eR, eTheta, ePhi = signalgen.get_frequency_spectrum(energy * fhad, viewing_angles[iS], n_samples, dt, 0, n_index, R, 'Alvarez2000')
            eTheta = signalgen.get_frequency_spectrum(energy * fhad, viewing_angles[iS], n_samples, dt, 0, n_index, R, 'Alvarez2000')

            # apply frequency dependent attenuation
            attn = r.get_attenuation(iS, ff)
#             eR *= attn
            eTheta *= attn
#             ePhi *= attn

            if(fem > 0):
                eTheta2 = signalgen.get_frequency_spectrum(energy * fem, viewing_angles[iS], n_samples, dt, 1, n_index, R, 'Alvarez2000')
#                 eR2 *= attn
                eTheta2 *= attn
#                 ePhi2 *= attn
                # add EM signal to had signal in the time domain
#                 eR = fft.time2freq(fft.freq2time(eR) + fft.freq2time(eR2))
                eTheta = fft.time2freq(fft.freq2time(eTheta) + fft.freq2time(eTheta2))
#                 ePhi = fft.time2freq(fft.freq2time(ePhi) + fft.freq2time(ePhi2))

            # TODO verify that calculation of polarization vector is correct!
            polarization_direction = np.cross(receive_vector, np.cross(shower_axis, receive_vector))
            polarization_direction /= np.linalg.norm(polarization_direction)
            cs = cstrans.cstrafo(zenith, azimuth)
            polarization_direction_onsky = cs.transform_from_ground_to_onsky(polarization_direction)
            polarization[iE, channel_id, iS] = np.arctan2(polarization_direction_onsky[1], polarization_direction_onsky[2])
            eR, eTheta, ePhi = np.outer(polarization_direction_onsky, eTheta)
#             print("{} {:.2f} {:.0f}".format(polarization_direction_onsky, np.linalg.norm(polarization_direction_onsky), np.arctan2(np.abs(polarization_direction_onsky[1]), np.abs(polarization_direction_onsky[2])) / units.deg))

            # in case of a reflected ray we need to account for fresnel reflection at the surface
            if(r.get_solution_type(iS) == 'reflected'):
                from NuRadioReco.utilities import geometryUtilities as geo_utl
                r_parallel = geo_utl.get_fresnel_r_parallel(zenith, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))
                r_perpendicular = geo_utl.get_fresnel_r_perpendicular(zenith, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))

                eTheta *= r_parallel
                ePhi *= r_perpendicular
                logger.debug("reflection coefficient is r_parallel = {:.2f}, r_perpendicular = {:.2f}".format(r_parallel, r_perpendicular))

            if(debug):
                fig, (ax, ax2) = plt.subplots(1, 2)
                ax.plot(ff, np.abs(eTheta) / units.micro / units.V * units.m)
                ax2.plot(tt, fft.freq2time(eTheta) / units.micro / units.V * units.m)
                ax2.set_ylabel("amplitude [$\mu$V/m]")
                fig.tight_layout()
                fig.suptitle("$E_C$ = {:.1g}eV $\Delta \Omega$ = {:.1f}deg, R = {:.0f}m".format(energy * fhad, viewing_angles[iS], R))
                fig.subplots_adjust(top=0.9)
                plt.show()

            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / dt)
            channel.set_trace_start_time(T)
            channel['azimuth'] = azimuth
            channel['zenith'] = zenith
            channel['raypath'] = ray.solution_types[r.get_solution_type(iS)]
            sim_station.add_channel(channel)

            if(np.max(np.abs(channel.get_trace())) > 3 * Vrms):  # apply a simple threshold cut to speed up the simulation, application of antenna response will just decrease the signal amplitude
                candidate_event = True

    # perform only a detector simulation if event had at least one candidate channel
    if(not candidate_event):
        continue

    logger.debug("performing detector simulation")
    # finalize NuRadioReco event structure
    station = NuRadioReco.framework.station.Station(station_id)
    station.set_sim_station(sim_station)
    evt = NuRadioReco.framework.event.Event(0, event_id)
    evt.set_station(station)
    station.set_station_time(evt_time)

    # start detector simulation
    efieldToVoltageConverterPerChannel.run(evt, station, det)  # convolve efield with antenna pattern
    # downsample trace back to detector sampling rate
    channelResampler.run(evt, station, det, sampling_rate=1. / dt)
    # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
    channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.MHz],
                              filter_type='butter10')
    logger.debug("Vrms= {:.2f}muV".format(Vrms / units.V / units.micro))
#     one_sigma = 11 * units.micro * units.V
#     one_sigma = 16 * units.nano * units.V
    triggerSimulator.run(evt, station, det,
                        threshold=3 * Vrms,
                        triggered_channels=None,
#                         triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
#                          triggered_channels=[0, 1, 2, 3],
                         number_concidences=1)
    if(station.has_triggered()):  # calculate more time consuming ARIANNA trigger only if station passes simple trigger
        triggerSimulatorARIANNA.run(evt, station, det,
                             threshold_high=3 * Vrms,
                             threshold_low=-3 * Vrms,
                             triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                             number_concidences=3)

    SNRs[iE] = station.get_parameter("channels_max_amplitude") / Vrms

    # calculate weight
    weights[iE] = get_weight(zenith_nu, energy, mode='simple')

    # save events that trigger the detector and have weight > 0
    if(station.has_triggered() and (weights[iE] > 1e-5)):
        if(args.outputfilenameNuRadioReco is not None):
            eventWriter.run(evt)
        logger.info("event triggered")

    triggered[iE] = station.has_triggered()

if(args.outputfilenameNuRadioReco is not None):
    eventWriter.end()  # close output file

# save simulation run in hdf5 format (only triggered events)
fout = h5py.File(args.outputfilename, 'w')
for key in fin.keys():
    fout[key] = fin[key][triggered]
for key in fin.attrs.keys():
    fout.attrs[key] = fin.attrs[key]
fout.attrs['n_events'] = n_events
# fin.copy(fin['/'], fout['/'], name='/events')
fout['launch_vectors'] = launch_vectors[triggered]
fout['receive_vectors'] = receive_vectors[triggered]
fout['travel_times'] = travel_times[triggered]
fout['travel_distances'] = travel_distances[triggered]
fout['ray_tracing_C0'] = ray_tracing_C0[triggered]
fout['ray_tracing_C1'] = ray_tracing_C1[triggered]
fout['ray_tracing_solution_type'] = ray_tracing_solution_type[triggered]
fout['triggered'] = triggered[triggered]
fout['weights'] = weights[triggered]
fout['polarization'] = polarization[triggered]
fout['SNRs'] = SNRs[triggered]

t_total = time.time() - t_start
logger.warning("{:d} events processed in {:.0f} seconds = {:.2f}ms/event".format(n_events, t_total, 1.e3 * t_total / n_events))

# calculate effective
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights[triggered])
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

dX = fin.attrs['xmax'] - fin.attrs['xmin']
dY = fin.attrs['ymax'] - fin.attrs['ymin']
dZ = fin.attrs['zmax'] - fin.attrs['zmin']
V = dX * dY * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events

print("Veff = {:.2g} km^3 sr".format(Veff / units.km ** 3))
#     a = 1 / 0
fin.close()
fout.close()

