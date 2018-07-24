from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from NuRadioMC.EvtGen import readEventList
from NuRadioMC.SignalGen.RalstonBuniy.askaryan_module import get_frequency_spectrum
from NuRadioMC.utilities import units
from NuRadioMC.SignalProp import analyticraytraycing as ray
from NuRadioMC.utilities import medium
from NuRadioMC.utilities import fft
from NuRadioMC.EvtGen.weight import get_weight
from matplotlib import pyplot as plt
import argparse
import json
import time
import os
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.detector.detector as detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sim")

# define time and frequency resolution of simulation
dt = 1 * units.ns
n_samples = 256
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
evt_time = datetime.datetime(2018, 1, 1)

debug = False

PLOT_FOLDER = 'plots'
if not os.path.isdir(PLOT_FOLDER):
    PLOT_FOLDER = './'

VERSION = 0.1

# # initialize detector simulation modules
# det = detector.Detector(json_filename)
# evt_time = datetime.datetime(2018, 1, 1)  # the time, relevant to determine the station configuration from the station id
# det.update(evt_time)


def add_empty_channel(sim_station, channel_id):
    channel = NuRadioReco.framework.channel.Channel(channel_id)
    channel.set_frequency_spectrum(np.zeros((3, len(ff)), dtype=np.complex), 1. / dt)
    channel['azimuth'] = 0
    channel['zenith'] = 180 * units.deg
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


parser = argparse.ArgumentParser(description='Parse ARA event list.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

# read in detector positions
det = detector.Detector(json_filename=args.detectordescription)
station_id = 101

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
triggerSimulator = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin("test_01.ari")

eventlist = readEventList.read_eventlist(args.inputfilename)
weights = np.zeros(len(eventlist))
triggered = np.zeros(len(eventlist), dtype=np.bool)
n_events = 1000

t_start = time.time()
for iE, event in enumerate(eventlist[:n_events]):
    event_id, flavor, energy, ccnc, x, y, z, zenith, azimuth, inelasticity = event

    shower_axis = hp.spherical_to_cartesian(zenith, azimuth)
    x1 = np.array([x, y, z])

    # calculate correct chereknov angle for ice density at vertex position
    ice = medium.southpole_simple()
    n_index = ice.get_index_of_refraction(x1)
    rho = np.arccos(1. / n_index)

    # create NuRadioReco event structure
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)

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
            launch_vector = r.get_launch_vector(iS)
            viewing_angle = hp.get_angle(shower_axis, launch_vector)  # calculates angle between shower axis and launch vector
            viewing_angles.append(viewing_angle)
            dRho = (viewing_angle - rho)
            logger.debug('solution {}: viewing angle {:.1f} = dRho = {:.1f}'.format(iS, viewing_angle / units.deg, (viewing_angle - rho) / units.deg))
            dRhos.append(dRho)
        # discard event if dRho (angle off cherenkov cone) is too large
        if(min(np.abs(dRhos)) > 15 * units.deg):
            logger.debug('dRho too large, event unlikely to be observed, skipping event')
            add_empty_channel(sim_station, channel_id)
            continue

        candidate_event = True

        n = r.get_number_of_solutions()
        Rs = np.zeros(n)
        Ts = np.zeros(n)
        tts = np.zeros((n, n_samples))
        for iS in range(n):  # loop through all ray tracing solution
            R = r.get_path_length(iS)  # calculate path length
            Rs[iS] = R
            T = r.get_travel_time(iS)  # calculate travel time
            Ts[iS] = T
            logger.debug("R = {:.1f} m, t = {:.1f}ns".format(R / units.m, T / units.ns))
#             eR, eTheta, ePhi = get_frequency_spectrum(energy, viewing_angles[iS], ff, 0, n, R)

            fem, fhad = get_em_had_fraction(inelasticity, ccnc, flavor)
            # get neutrino pulse from Askaryan module
            eR, eTheta, ePhi = get_frequency_spectrum(energy * fhad, viewing_angles[iS], ff, 0, n_index, R, a=1.5 * units.m)
            # apply frequency dependent attenuation
            attn = r.get_attenuation(iS, ff)
            eR *= attn
            eTheta *= attn
            ePhi *= attn

            if(fem > 0 and 0):
                eR2, eTheta2, ePhi2 = get_frequency_spectrum(energy * fem, viewing_angles[iS], ff, 1, n_index, R, a=1.5 * units.m)
                eR2 *= attn
                eTheta2 *= attn
                ePhi2 *= attn
                # add EM signal to had signal in the time domain
                eR = fft.time2freq(fft.freq2time(eR) + fft.freq2time(eR2))
                eTheta = fft.time2freq(fft.freq2time(eTheta) + fft.freq2time(eTheta2))
                ePhi = fft.time2freq(fft.freq2time(ePhi) + fft.freq2time(ePhi2))

            receive_vector = r.get_receive_vector(iS)
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)

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
            channel['azimuth'] = azimuth
            channel['zenith'] = zenith
            sim_station.add_channel(channel)

    # perform only a detector simulation if event had at least one candidate channel
    if(not candidate_event):
        continue

    # finalize NuRadioReco event structure
    station = NuRadioReco.framework.station.Station(station_id)
    station.set_sim_station(sim_station)
    evt = NuRadioReco.framework.event.Event(0, event_id)
    evt.set_station(station)
    station.set_station_time(evt_time)

    # start detector simulation
    # convolve efield with antenna pattern, adjust timing accordingly, account for transmission loss at boundary
    efieldToVoltageConverterPerChannel.run(evt, station, det)
    one_sigma = 11 * units.micro * units.V
#     one_sigma = 16 * units.nano * units.V
    triggerSimulator.run(evt, station, det,
                         threshold_high=3 * one_sigma,
                         threshold_low=-3 * one_sigma,
                         triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                         number_concidences=2)
    # save events that trigger the detector
#     if(station.has_triggered()):
    eventWriter.run(evt)

    triggered[iE] = station.has_triggered()

    # calculate weight
    weights[iE] = get_weight(zenith, energy, mode='simple')

eventWriter.end()  # close output file

t_total = time.time() - t_start
logger.warning("{:d} events processed in {:.0f} seconds = {:.2f}ms/event".format(n_events, t_total, 1.e3 * t_total / n_events))

# calculate effective
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights[triggered])
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

dX = 2 * units.km
dY = 2 * units.km
dZ = 1 * units.km
V = dX * dY * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events

print("Veff = {:.2g} km^3 sr".format(Veff / units.km ** 3))
#     a = 1 / 0

