from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from NuRadioMC.EvtGen import readARAEventList
from NuRadioMC.SignalGen.RalstonBuniy.create_askaryan import get_time_trace
from NuRadioMC.utilities import units
import argparse
import json
import os
from ROOT import gROOT
gROOT.LoadMacro("../SignalProp/UzairProp/RayTraceRK4.C+")
from ROOT import RayTraceRK4
from matplotlib import pyplot as plt

# ARIANNA Reco import import numpy as np
import ARIANNAreco.modules.channelSignalReconstructor
import ARIANNAreco.modules.io.eventWriter
import ARIANNAreco.modules.efieldToVoltageConverter
import ARIANNAreco.modules.hardwareResponseIncorporator
import ARIANNAreco.modules.channelResampler
import ARIANNAreco.modules.triggerSimulator
import ARIANNAreco.modules.simEfieldResampler
import ARIANNAreco.framework.event
import ARIANNAreco.framework.station
import ARIANNAreco.framework.sim_station
import ARIANNAreco.framework.channel
from ARIANNAreco.detector import detector
import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("readARAEventList")

PLOT_FOLDER = 'plots'

VERSION = 0.1

parser = argparse.ArgumentParser(description='Parse ARA event list.')
parser.add_argument('inputfilename', type=str,
                    help='path to ARAsim input event list')
parser.add_argument('observerlist', type=str,
                    help='path to file containing the detector positions')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

# read in detector positions
with open(args.observerlist) as fobs:
    station_list = json.load(fobs)

eventlist = readARAEventList.read_ARA_eventlist(args.inputfilename)

# # initialize detector simulation modules

station_id = 2  # specify station id for which MC data set should be generated
triggered_channels = [0]

det = detector.Detector()
evt_time = datetime.datetime(2018, 1, 1)  # the time, relevant to determine the station configuration from the station id
det.update(evt_time)

# initialize all modules
simEFieldResampler = ARIANNAreco.modules.simEfieldResampler.simEFieldResampler()
efieldToVoltageConverter = ARIANNAreco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = ARIANNAreco.modules.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelResampler = ARIANNAreco.modules.channelResampler.channelResampler()
channelResampler.begin()
triggerSimulator = ARIANNAreco.modules.triggerSimulator.triggerSimulator()
triggerSimulator.begin()
eventWriter = ARIANNAreco.modules.io.eventWriter.eventWriter()
output_filename = "example_{}.ari".format(station_id)
eventWriter.begin(output_filename)

for event in eventlist:
    evid, nuflavorint, nu_nubar, pnu, currentint, posnu_r, posnu_theta, posnu_phi, nnu_theta, nnu_phi, elast_y = event
    if(-999 in event):
        logger.warning("random input generation not supported, skipping event {}".format(evid))
        continue
    position = posnu_r * hp.spherical_to_cartesian(posnu_theta, posnu_phi)
    shower_axis = hp.spherical_to_cartesian(nnu_theta, 0)  # ignoring phi angle, assuming that everything happens in the x-z plane

    dt = 1 * units.ns
    ff = np.fft.rfftfreq(256, dt)

    # first step: peorform raytracing to see if solution exists
    for iStation, station in enumerate(station_list):

        x_antenna = station["position"]
        getres = RayTraceRK4(position[0], position[1], position[2],
                             x_antenna[0], x_antenna[1], x_antenna[2], 0,
                             ff, len(ff))

        if(getres[0] == -1):
            logger.info("no raytracing solution found, skipping to next station")
            continue
        launch_angle = getres[0]
        receive_angle = getres[1]
#         dt = getres[2]

        attenuation = np.array([getres[i + 9] for i in range(len(ff))])

        # calculate cherenkov angle
        dPhi = np.arctan2(x_antenna[1] - position[1], x_antenna[0] - position[0])
        incoming_signal = hp.spherical_to_cartesian(launch_angle, 0)  # ignoring phi angle, assuming that everything happens in the x-z plane
        cherenkov_angle = hp.get_angle(shower_axis, incoming_signal)
        tt, ex, ey, ez = get_time_trace(pnu, cherenkov_angle, ff.min(), ff.max(), ff[1] - ff[0], 0)

        fig, ax = plt.subplots(1, 1)
        ax.plot(position[0] / units.m, position[2] / units.m, 'ko', label='start')
        ax.plot([position[0] / units.m, position[0] / units.m + 50 * incoming_signal[0]],
                [position[2] / units.m, position[2] / units.m + 50 * incoming_signal[2]], 'k-')

        # plot shower direction
        ax.plot([position[0] / units.m, position[0] / units.m + 50 * shower_axis[0]],
                [position[2] / units.m, position[2] / units.m + 50 * shower_axis[2]], 'C0--', label='shower axis')

        ax.plot(x_antenna[0] / units.m, x_antenna[2] / units.m, 'kD', label='antenna')
        receive_vector = hp.spherical_to_cartesian(receive_angle, 0)
        receive_vector *= -1
        ax.plot([x_antenna[0] / units.m, x_antenna[0] / units.m + 50 * receive_vector[0]],
                [x_antenna[2] / units.m, x_antenna[2] / units.m + 50 * receive_vector[2]], 'k-')
        ax.legend(numpoints=1)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOT_FOLDER, "event_{:04d}_station_{:02d}_geometry.png".format(evid, iStation)))

        fig, ax = plt.subplots(1, 1)
        ax.plot(ff / units.MHz, attenuation)
        ax.set_xlabel("frequency [MHz]")
        ax.set_ylabel("attenuation")
        fig.savefig(os.path.join(PLOT_FOLDER, "event_{:04d}_station_{:02d}_attenuatio.png".format(evid, iStation)))

        spectrum_x = np.fft.rfft(ex, norm='ortho')
        spectrum_y = np.fft.rfft(ey, norm='ortho')
        spectrum_z = np.fft.rfft(ez, norm='ortho')
        spectrum_x *= attenuation
        spectrum_y *= attenuation
        spectrum_z *= attenuation
        ex_attenuated = np.fft.irfft(spectrum_x, norm='ortho')
        ey_attenuated = np.fft.irfft(spectrum_y, norm='ortho')
        ez_attenuated = np.fft.irfft(spectrum_z, norm='ortho')

        fig, (ax, axf) = plt.subplots(1, 2)
        ax.plot(tt / units.ns, ex / units.mV * units.m, label='eR')
        ax.plot(tt / units.ns, ey / units.mV * units.m, label='eTheta')
        ax.plot(tt / units.ns, ez / units.mV * units.m, label='ePhi')
        ax.plot(tt / units.ns, ex_attenuated / units.mV * units.m, '--C0')
        ax.plot(tt / units.ns, ey_attenuated / units.mV * units.m, '--C1')
        ax.plot(tt / units.ns, ez_attenuated / units.mV * units.m, '--C2')
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("electric-field [mV/m]")
        ax.legend()
        ax.set_title("electic field at antenna")
        axf.plot(ff / units.MHz, np.abs(spectrum_x))
        axf.plot(ff / units.MHz, np.abs(spectrum_y))
        axf.plot(ff / units.MHz, np.abs(spectrum_z))
        axf.set_xlabel("frequency [MHz]")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_FOLDER, "event_{:04d}_station_{:02d}_trace.png".format(evid, iStation)))

        trace = np.array([ex_attenuated, ey_attenuated, ez_attenuated])
        sim_station = ARIANNAreco.framework.sim_station.SimStation(station_id, 1. / dt, trace, None)
        zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
        sim_station['azimuth'] = azimuth
        sim_station['zenith'] = zenith
        station = ARIANNAreco.framework.station.Station(station_id)
        station.set_sim_station(sim_station)
        evt = ARIANNAreco.framework.event.Event(0, evid)

        # upsample simulated efield to achive a good time resolution in shifting channel traces to the correct time
        simEFieldResampler.run(evt, station, det, sampling_rate=100 * units.GHz)

        # manually set station time
        station.set_station_time(evt_time)

        # convolve efield with antenna pattern, adjust timing accordingly, account for transmission loss at boundary
        efieldToVoltageConverter.run(evt, station, det)
        simEFieldResampler.run(evt, station, det, sampling_rate=1 * units.GHz)  # resample back to ARIANNA time resolution to reduce file size
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
        triggerSimulator.run(evt, station, det,
                             threshold_high=40 * units.mV,
                             threshold_low=-40 * units.mV,
                             triggered_channels=triggered_channels,
                             number_concidences=1)
        
        fig, (ax, axf) = plt.subplots(1, 2)
        trace = station.get_channel(0).get_trace()
        tt = station.get_channel(0).get_times()
        ax.plot(tt / units.ns, trace / units.mV)
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("voltage [mV]")
        ax.legend()
        ax.set_title("after detector response")
        axf.plot(station.get_channel(0).get_frequencies() / units.MHz, np.abs(station.get_channel(0).get_frequency_spectrum()))
        axf.set_xlabel("frequency [MHz]")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_FOLDER, "event_{:04d}_station_{:02d}_afterdetector.png".format(evid, iStation)))

        a = 1 / 0
        if(not station['triggered']):
            logger.info("removing event because it will not be triggered by ARIANNA")
            continue
        eventWriter.run(evt)
        eventWriter.end()

