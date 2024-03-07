from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
from scipy import signal
import argparse
from datetime import datetime
import pickle
import os

from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import emitterParameters as ep
from NuRadioReco.framework.parameters import electricFieldParameters as efp

import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.efieldTimeDirectionFitter
import NuRadioReco.modules.channelTimeWindow
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.eventTypeIdentifier

from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from radiotools import helper as hp
# plt.switch_backend('agg')

if __name__ == "__main__":
    plot = 1

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
    voltageToEfieldConverterPerChannel = NuRadioReco.modules.voltageToEfieldConverterPerChannel.voltageToEfieldConverterPerChannel()
    voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
    electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
    efieldTimeDirectionFitter = NuRadioReco.modules.efieldTimeDirectionFitter.efieldTimeDirectionFitter()
    channelTimeWindow = NuRadioReco.modules.channelTimeWindow.channelTimeWindow()
    channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
    eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

    efieldTimeDirectionFitter.begin(debug=plot)
    channelTimeWindow.begin(debug=False)
    correlationDirectionFitter.begin(debug=False, log_level=logging.DEBUG)

    electricFieldSignalReconstructor.begin(log_level=logging.WARNING)

    # Parse eventfile as argument
    parser = argparse.ArgumentParser(description='NuRadioSim file')
    parser.add_argument('inputfilename', type=str,
                        help='path to NuRadioMC simulation result')
    # parser.add_argument('detectordescription', type=str,
    #                     help='path to detectordescription')
    args = parser.parse_args()

    det = detector.Detector("detector_db.json", source='json',antenna_by_depth=False)
    det.update(datetime(2018, 12, 30, 22, 30, 22))
    pos_SP1 = det.get_absolute_position(51)

    pos_SP1_2 = np.array([41153.2175 * units.feet, 50381.75 * units.feet, -1.5 * units.m])
    pos_spice = np.array([42600, 48800, 0]) * units.feet

    print('distance = {:.2f}'.format(np.linalg.norm(pos_SP1 - pos_spice)))

    t1 = datetime(2018, 12, 30, 22, 30, 22)
    t2 = datetime(2018, 12, 31, 2, 3, 11)
    # initialize modules
    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    eventReader.begin(args.inputfilename)

    results = {'corr_LPDA': [],
            'corr_dipole': [],
            'depth': [],
            'exp': [],
            'time_LPDA': [],
            'time_dipole': [],
            'chi2_time_dipole': [],
            'chi2_time_LPDA': [],
            'pol_angle': []}

    for evt in eventReader.run():
        for station in evt.get_stations():
            station_id = station.get_id()
            t = station.get_station_time()
            print("event {}, station {}, time {}".format(evt.get_id(), station_id, t))
            det.update(t)
            d = evt.get_first_sim_emitter()[ep.position][2]

            # calcualte expected angles
            r = ray.ray_tracing(medium.southpole_simple(), log_level=logging.WARNING)
            r.set_start_and_end_point(pos_spice + np.array([0, 0, d]), pos_SP1)
            r.find_solutions()
            if(not r.has_solution()):
                continue

            results['depth'].append(d)
            rvec = r.get_receive_vector(0)
            zen, az = hp.cartesian_to_spherical(*rvec)
            az = hp.get_normalized_angle(az)
            results['exp'].append((zen, az))
            station.get_sim_station().set_parameter(stnp.zenith, zen)
            station.get_sim_station().set_parameter(stnp.azimuth, az)
            print("{} depth = {:.1f}m -> {:.2f} {:.2f} (solution type {})".format(t, d, zen / units.deg, az / units.deg, r.get_solution_type(0)))

            eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='neutrino')
            channelResampler.run(evt, station, det, 50 * units.GHz)
            channelBandPassFilter.run(evt, station, det, passband=[120 * units.MHz, 300 * units.MHz], filter_type='butterabs', order=10)
            channelBandPassFilter.run(evt, station, det, passband=[10 * units.MHz, 1000 * units.MHz], filter_type='rectangular')
            hardwareResponseIncorporator.run(evt, station, det)
            channelSignalReconstructor.run(evt, station, det)
    #         channelTimeWindow.run(evt, station, det, window_function='hanning', around_pulse=True, window_width=20*units.ns,
    #                             window_rise_time=20*units.ns)

            correlationDirectionFitter.run(evt, station, det, n_index=1.353, ZenLim=[90 * units.deg, 180 * units.deg],
                                        AziLim=[300 * units.deg, 330 * units.deg],
                                        channel_pairs=((0, 2), (1, 3)))

            print("reco correlation LPDAs = {:.1f} ({:.1f}) {:.1f} ({:.1f})".format(station[stnp.zenith] / units.deg,
                                                                            (station[stnp.zenith] - zen) / units.deg,
                                                                            station[stnp.azimuth] / units.deg,
                                                                            (station[stnp.azimuth] - az) / units.deg))
            results['corr_LPDA'].append((station[stnp.zenith], station[stnp.azimuth]))
    #
            correlationDirectionFitter.run(evt, station, det, n_index=1.353, ZenLim=[90 * units.deg, 180 * units.deg],
                                        AziLim=[300 * units.deg, 330 * units.deg],
                                        channel_pairs=((6, 4), (5, 7)))

            print("reco correlation dipoles = {:.1f} ({:.1f}) {:.1f} ({:.1f})".format(station[stnp.zenith] / units.deg,
                                                                            (station[stnp.zenith] - zen) / units.deg,
                                                                            station[stnp.azimuth] / units.deg,
                                                                            (station[stnp.azimuth] - az) / units.deg))
            results['corr_dipole'].append((station[stnp.zenith], station[stnp.azimuth]))

            voltageToEfieldConverterPerChannel.run(evt, station, det, pol=0)
            electricFieldSignalReconstructor.run(evt, station, det)
            efieldTimeDirectionFitter.run(evt, station, det, channels_to_use=[0, 1, 2, 3])
            print("reco time LPDAs = {:.1f} ({:.1f}) {:.1f} ({:.1f})".format(station[stnp.zenith] / units.deg,
                                                                            (station[stnp.zenith] - zen) / units.deg,
                                                                            station[stnp.azimuth] / units.deg,
                                                                            (station[stnp.azimuth] - az) / units.deg))
            results['time_LPDA'].append((station[stnp.zenith], station[stnp.azimuth]))
            results['chi2_time_LPDA'].append(station[stnp.chi2_efield_time_direction_fit])
            efieldTimeDirectionFitter.run(evt, station, det, channels_to_use=range(4, 8))
            print("reco time 4 dipoles = {:.1f} ({:.1f}) {:.1f} ({:.1f})".format(station[stnp.zenith] / units.deg,
                                                                            (station[stnp.zenith] - zen) / units.deg,
                                                                            station[stnp.azimuth] / units.deg,
                                                                            (station[stnp.azimuth] - az) / units.deg))
            results['time_dipole'].append((station[stnp.zenith], station[stnp.azimuth]))
            results['chi2_time_dipole'].append(station[stnp.chi2_efield_time_direction_fit])

            if plot:
                fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
                ax = ax.flatten(order='F')
                for channel in station.iter_channels():
                    cid = channel.get_id()
                    if not cid in range(8):
                        continue
                    tt = channel.get_times()  # + channel.get_trace_start_time()
                    tt -= tt[0]
                    ax[cid].plot(tt / units.ns, channel.get_trace() / units.mV, lw=1)
    #                 ax[cid].plot(tt/units.ns, channel.get_trace()[2]/units.mV)
    #                 ax[cid].axvline(channel[efp.signal_time] + channel.get_trace_start_time())
                    ax[cid].set_xlim(10, 150)
                    ax[cid].set_title("channel {}".format(cid), fontsize='xx-small')
                ax[3].set_xlabel("time [ns]")
                ax[7].set_xlabel("time [ns]")
                fig.suptitle("voltage trace (signal chain deconvolved) d = {:.0f}m".format(d))
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)

                fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
                ax = ax.flatten(order='F')
                for efield in station.get_electric_fields():
                    eid = efield.get_channel_ids()[0]
                    if not eid in range(8):
                        continue
                    tt = efield.get_times()  # + efield.get_trace_start_time()
                    t0 = tt[0]
                    tt -= t0
                    ax[eid].plot(tt / units.ns, efield.get_trace()[1] / units.mV, lw=1, label='rec')
                    ax[eid].plot(tt / units.ns, np.abs(signal.hilbert(efield.get_trace()[1])) / units.mV, '--', lw=1)

    #                 ax[eid].plot(tt/units.ns, efield.get_trace()[2]/units.mV)
                    ax[eid].axvline(efield[efp.signal_time] - t0, linestyle='--', lw=1)
                    ax[eid].set_xlim(10, 150)
                # add simulated efields
                for efield in station.get_sim_station().get_electric_fields():
                    eid = efield.get_channel_ids()[0]
                    if not eid in range(8):
                        continue
                    if(efield[efp.ray_path_type] == 'reflected'):
    #                 print("{} {}".format(efield[efp.reflection_coefficient_theta], efield[efp.reflection_coefficient_phi]))
                        ax[eid].set_title("channel {}, r = {:.2f} {:.2f}".format(eid, np.abs(efield[efp.reflection_coefficient_theta]), np.abs(efield[efp.reflection_coefficient_phi])), fontsize='xx-small')
                for i in range(8):
                    ax[i].legend(fontsize='xx-small')
                ax[3].set_xlabel("time [ns]")
                ax[7].set_xlabel("time [ns]")
                fig.suptitle("reconstructed electric field d = {:.0f}m".format(d))
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                if(not os.path.exists("plots/efields")):
                    os.makedirs("plots/efields")
                fig.savefig("plots/efields/d_{:04.0f}m.png".format(d))
                plt.close("all")
    #             a = 1/0

            voltageToEfieldConverter.run(evt, station, det, use_channels=[0,1,2,3], use_MC_direction=True)
            for efield in station.get_electric_fields_for_channels([0,1,2,3]):
                etrace = efield.get_trace()
                ehilbert_theta = np.abs(signal.hilbert(etrace[1]))
                imax = np.argmax(ehilbert_theta)
                nbins_20ns = int(20 * units.ns / (efield.get_times()[1] - efield.get_times()[0]))
                pol_angle = np.arctan2(np.sum(etrace[2][imax - nbins_20ns:imax + nbins_20ns]**2),np.sum(etrace[1][imax - nbins_20ns:imax + nbins_20ns]**2))
                print("pol angle = {:.2f}deg".format(pol_angle / units.deg))
                results['pol_angle'].append(pol_angle)
                if plot:
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(efield.get_times() / units.ns, efield.get_trace()[1] / units.mV, "-C0", lw=1, label='theta')
                    ax.plot(efield.get_times() / units.ns, np.abs(signal.hilbert(efield.get_trace()[1])) / units.mV, 'C0--', lw=1)
                    ax.plot(efield.get_times() / units.ns, efield.get_trace()[2] / units.mV, "-C1", lw=1, label='phi')
                    ax.plot(efield.get_times() / units.ns, np.abs(signal.hilbert(efield.get_trace()[2])) / units.mV, 'C1--', lw=1)

                    # plotting the simulated efield does not make much sense as it has a much higher frequency content than what can be reconstructed
                    # for sefield in station.get_sim_station().get_electric_fields_for_channels([0]):
                    #     if(sefield[efp.ray_path_type] == 'direct'):
                    #         ax.plot(sefield.get_times() / units.ns, sefield.get_trace()[1] / units.mV, "--C0", lw=1, label='sim direct')
                    #         ax.plot(sefield.get_times() / units.ns, sefield.get_trace()[2] / units.mV, "--C1", lw=1)
                    #     elif(sefield[efp.ray_path_type] == 'reflected'):
                    #         ax.plot(sefield.get_times() / units.ns, sefield.get_trace()[1] / units.mV, ":C0", lw=1, label='sim reflected')
                    #         ax.plot(sefield.get_times() / units.ns, sefield.get_trace()[2] / units.mV, ":C1", lw=1)
                    #         ax.set_title("r = {:.2f} {:.2f}".format(np.abs(sefield[efp.reflection_coefficient_theta]), np.abs(sefield[efp.reflection_coefficient_phi])), fontsize='xx-small')
                    ax.legend()
                    fig.tight_layout()
                    # plt.show()
                    # a = 1/0
                    fig.savefig("plots/efields/3D_d_{:04.0f}m.png".format(d))
                    plt.close(fig)
                break # there should be only one efield for the 4 channels, but better be safe than sorry


    with open("sim_results_03.pkl", 'wb') as fout:
        pickle.dump(results, fout, protocol=2)
