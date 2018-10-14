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
import time
import six
from scipy import constants
# import detector simulation modules
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.detector.detector as detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import datetime
import logging
from six import iteritems
import yaml
import os
# import confuse
# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sim")

VERSION = 0.1

def merge_config(user, default):
    if isinstance(user,dict) and isinstance(default,dict):
        for k,v in iteritems(default):
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k],v)
    return user







class simulation():

    def __init__(self, eventlist,
                 outputfilename,
                 detectorfile,
                 station_id,
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1),
                 config_file=None):
        """
        initialize the NuRadioMC end-to-end simulation

        Parameters
        ----------
        eventlist: string
            the path to the hdf5 file containing the list of neutrino events
        outputfilename: string
            specify hdf5 output filename.
        detectorfile: string
            path to the json file containing the detector description
        station_id: int
            the station id for which the simulation is performed. Must match a station
            deself._fined in the detector description
        outputfilenameNuRadioReco: string or None
            outputfilename of NuRadioReco detector sim file, this file contains all
            waveforms of the triggered events
            default: None, i.e., no output file will be written which is useful for
            effective volume calculations
        debug: bool
            True activates debug mode, default False
        evt_time: datetime object
            the time of the events, default 1/1/2018
        """
        
        config_file_default = os.path.join(os.path.dirname(__file__), 'config_default.yaml')
        logger.warning('reading default config from {}'.format(config_file_default))
        with open(config_file_default, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile)
        if(config_file is not None):
            logger.warning('reading local config overrides from {}'.format(config_file))
            with open(config_file, 'r') as ymlfile:
                local_config=yaml.load(ymlfile)
                new_cfg = merge_config(local_config, self._cfg)
                self._cfg = new_cfg
        
        self._eventlist = eventlist
        self._outputfilename = outputfilename
        self._detectorfile = detectorfile
        self._station_id = station_id
        self._Tnoise = float(self._cfg['trigger']['noise_temperature'])
        self._outputfilenameNuRadioReco = outputfilenameNuRadioReco
        self._debug = debug
        self._evt_time = evt_time
        
        self._ice = medium.get_ice_model(self._cfg['propagation']['ice_model'])
        
        self._mout = {}
        self._mout_attrs = {}

        # read in detector positions
        logger.debug("Detectorfile {}".format(self._detectorfile))
        self._det = detector.Detector(json_filename=self._detectorfile)
        
        # print noise information
        logger.warning("running with noise {}".format(bool(self._cfg['noise'])))
        logger.warning("setting signal to zero {}".format(bool(self._cfg['signal']['zerosignal'])))

        # read sampling rate from config (this sampling rate will be used internally)
        self._dt = 1. / (self._cfg['sampling_rate'] * units.GHz)
        
        self._sampling_rate_detector = self._det.get_sampling_frequency(station_id, 0)
        logger.warning('internal sampling rate is {:.3g}GHz, final detector sampling rate is {:.3g}GHz'.format(self.get_sampling_rate(), self._sampling_rate_detector))
        
        bandwidth = self._cfg['trigger']['bandwidth']
        if(bandwidth is None):
            self._bandwidth = 0.5 / self._dt
        else:
            self._bandwidth = bandwidth
        self._n_samples = self._det.get_number_of_samples(station_id, 0) / self._sampling_rate_detector / self._dt
        self._n_samples = int(np.ceil(self._n_samples / 2.) * 2)  # round to nearest even integer
        self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
        self._tt = np.arange(0, self._n_samples * self._dt, self._dt)
        self._Vrms = (self._Tnoise * 50 * constants.k *
                       self._bandwidth / units.Hz) ** 0.5
        logger.warning('noise temperature = {}, bandwidth = {:.0f} MHz, Vrms = {:.2f} muV'.format(self._Tnoise, self._bandwidth/units.MHz, self._Vrms/units.V/units.micro))


    def run(self):
        """
        run the NuRadioMC simulation
        """

        self._channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        self._eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
        if(self._outputfilenameNuRadioReco is not None):
            self._eventWriter.begin(self._outputfilenameNuRadioReco)

        self._read_input_hdf5() # we read in the full input file into memory at the beginning to limit io to the beginning and end of the run
        self._n_events = len(self._fin['event_ids'])
        self._n_antennas = self._det.get_number_of_channels(self._station_id)
        
        self._create_meta_output_datastructures()
        
        # check if the same detector was simulated before (then we can save the ray tracing part)
        self._check_if_was_pre_simulated()
        
    
        inputTime = 0.0
        rayTracingTime = 0.0
        detSimTime = 0.0
        outputTime = 0.0
        time_attenuation_length = 0.
        t_start = time.time()
        
        for self._iE in range(self._n_events):
            t1 = time.time()
            if(self._iE > 0 and self._iE % max(1, int(self._n_events / 100.)) == 0):
                eta = datetime.timedelta(seconds=(time.time() - t_start) * (self._n_events - self._iE) / self._iE)
                total_time = inputTime + rayTracingTime + detSimTime + outputTime
                logger.warning("processing event {}/{} = {:.1f}%, ETA {}, time consumption: ray tracing = {:.0f}% (att. length {:.0f}%), detector simulation = {:.0f}% reading input = {:.0f}%".format(
                    self._iE, self._n_events, 100. * self._iE / self._n_events, eta, 100. * rayTracingTime / total_time, 100. * time_attenuation_length / rayTracingTime, 100. * detSimTime / total_time, 100.*inputTime/total_time))
#             if(self._iE > 0 and self._iE % max(1, int(self._n_events / 10000.)) == 0):
#                 print("*", end='')

            # read all quantities from hdf5 file and store them in local variables
            self._read_input_neutrino_properties()

            # calculate weight
            self._mout['weights'][self._iE] = get_weight(self._zenith_nu, self._energy, mode=self._cfg['weights']['weight_mode'])
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if(self._mout['weights'][self._iE] < self._cfg['speedup']['minimum_weight_cut']):
                logger.debug("neutrino weight is smaller than {}, skipping event".format(self._cfg['speedup']['minimum_weight_cut']))
                continue

            # be careful, zenith/azimuth angle always refer to where the neutrino came from,
            # i.e., opposite to the direction of propagation. We need the propagation directio nhere,
            # so we multiply the shower axis with '-1'
            self._shower_axis = -1 * hp.spherical_to_cartesian(self._zenith_nu, self._azimuth_nu)
            x1 = np.array([self._x, self._y, self._z])

            # calculate correct chereknov angle for ice density at vertex position
            n_index = self._ice.get_index_of_refraction(x1)
            cherenkov_angle = np.arccos(1. / n_index)
                
            self._create_sim_station()
            candidate_event = False

            # first step: peorform raytracing to see if solution exists
            #print("start raytracing. time: " + str(time.time()))
            t2 = time.time()
            inputTime += (t2 - t1)
            ray_tracing_performed = ('ray_tracing_C0' in self._fin) and (self._was_pre_simulated)
            for channel_id in range(self._det.get_number_of_channels(self._station_id)):
                x2 = self._det.get_relative_position(self._station_id, channel_id)
                r = ray.ray_tracing(x1, x2, self._ice, log_level=logging.WARNING)

                if(ray_tracing_performed):  # check if raytracing was already performed
                    r.set_solution(self._fin['ray_tracing_C0'][self._iE, channel_id], self._fin['ray_tracing_C1'][self._iE, channel_id],
                                   self._fin['ray_tracing_solution_type'][self._iE, channel_id])
                else:
                    r.find_solutions()
                if(not r.has_solution()):
                    logger.debug("event {} and station {}, channel {} does not have any ray tracing solution".format(
                        self._event_id, self._station_id, channel_id))
                    self._add_empty_channel(channel_id)
                    continue
                delta_Cs = []
                viewing_angles = []
                # loop through all ray tracing solution
                for iS in range(r.get_number_of_solutions()):
                    self._mout['ray_tracing_C0'][self._iE, channel_id, iS] = r.get_results()[iS]['C0']
                    self._mout['ray_tracing_C1'][self._iE, channel_id, iS] = r.get_results()[iS]['C1']
                    self._mout['ray_tracing_solution_type'][self._iE, channel_id, iS] = r.get_solution_type(iS)
                    self._launch_vector = r.get_launch_vector(iS)
                    self._mout['launch_vectors'][self._iE, channel_id, iS] = self._launch_vector
                    # calculates angle between shower axis and launch vector
                    viewing_angle = hp.get_angle(self._shower_axis, self._launch_vector)
                    viewing_angles.append(viewing_angle)
                    delta_C = (viewing_angle - cherenkov_angle)
                    logger.debug('solution {} {}: viewing angle {:.1f} = delta_C = {:.1f}'.format(
                        iS, ray.solution_types[r.get_solution_type(iS)], viewing_angle / units.deg, (viewing_angle - cherenkov_angle) / units.deg))
                    delta_Cs.append(delta_C)

                # discard event if delta_C (angle off cherenkov cone) is too large
                if(min(np.abs(delta_Cs)) > self._cfg['speedup']['delta_C_cut']):
                    logger.debug('delta_C too large, event unlikely to be observed, skipping event')
                    self._add_empty_channel(channel_id)
                    continue

                n = r.get_number_of_solutions()
                Rs = np.zeros(n)
                Ts = np.zeros(n)
                tts = np.zeros((n, self._n_samples))
                for iS in range(n):  # loop through all ray tracing solution
                    if(ray_tracing_performed):
                        R = self._fin['travel_distances'][self._iE, channel_id, iS]
                        T = self._fin['travel_times'][self._iE, channel_id, iS]
                    else:
                        R = r.get_path_length(iS)  # calculate path length
                        T = r.get_travel_time(iS)  # calculate travel time
                    self._mout['travel_distances'][self._iE, channel_id, iS] = R
                    self._mout['travel_times'][self._iE, channel_id, iS] = T
                    Rs[iS] = R
                    Ts[iS] = T
                    self._launch_vector = r.get_launch_vector(iS)
                    receive_vector = r.get_receive_vector(iS)
                    # save receive vector
                    self._mout['receive_vectors'][self._iE, channel_id, iS] = receive_vector
                    zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
                    logger.debug("ch {}, s {} R = {:.1f} m, t = {:.1f}ns, receive angles {:.0f} {:.0f}".format(
                        channel_id, iS, R / units.m, T / units.ns, zenith / units.deg, azimuth / units.deg))

                    fem, fhad = self._get_em_had_fraction(self._inelasticity, self._ccnc, self._flavor)
                    # get neutrino pulse from Askaryan module
                    spectrum = signalgen.get_frequency_spectrum(
                        self._energy * fhad, viewing_angles[iS], self._n_samples, self._dt, 0, n_index, R, self._cfg['signal']['model'])

                    # apply frequency dependent attenuation
                    t_att = time.time()
                    attn = r.get_attenuation(iS, self._ff)
                    time_attenuation_length += (time.time() - t_att)
                    spectrum *= attn

                    if(fem > 0):
                        spectrum_em = signalgen.get_frequency_spectrum(
                            self._energy * fem, viewing_angles[iS], self._n_samples, self._dt, 1, n_index, R, self._cfg['signal']['model'])
                        spectrum_em *= attn
                        # add EM signal to had signal in the time domain
                        spectrum = fft.time2freq(fft.freq2time(spectrum) + fft.freq2time(spectrum_em))

                    
                    polarization_direction_onsky = self._calculate_polarization_vector()
                    cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*receive_vector))
                    polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
                    logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
                        zenith / units.deg, azimuth / units.deg, polarization_direction_onsky[0],
                        polarization_direction_onsky[1], polarization_direction_onsky[2],
                        *polarization_direction_at_antenna))
                    self._mout['polarization'][self._iE, channel_id, iS] = polarization_direction_at_antenna
                    eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)
        #             print("{} {:.2f} {:.0f}".format(polarization_direction_onsky, np.linalg.norm(polarization_direction_onsky), np.arctan2(np.abs(polarization_direction_onsky[1]), np.abs(polarization_direction_onsky[2])) / units.deg))

                    # in case of a reflected ray we need to account for fresnel
                    # reflection at the surface
                    if(ray.solution_types[r.get_solution_type(iS)] == 'reflected'):
                        from NuRadioReco.utilities import geometryUtilities as geo_utl
                        r_parallel = geo_utl.get_fresnel_r_parallel(
                            zenith, n_2=1., n_1=self._ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))
                        r_perpendicular = geo_utl.get_fresnel_r_perpendicular(
                            zenith, n_2=1., n_1=self._ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))

                        eTheta *= r_parallel
                        ePhi *= r_perpendicular
                        logger.debug("reflection coefficient is r_parallel = {:.2f}, r_perpendicular = {:.2f}".format(
                            r_parallel, r_perpendicular))

                    if(self._debug):
                        fig, (ax, ax2) = plt.subplots(1, 2)
                        ax.plot(self._ff, np.abs(eTheta) / units.micro / units.V * units.m)
                        ax2.plot(self._tt, fft.freq2time(eTheta) / units.micro / units.V * units.m)
                        ax2.set_ylabel("amplitude [$\mu$V/m]")
                        fig.tight_layout()
                        fig.suptitle("$E_C$ = {:.1g}eV $\Delta \Omega$ = {:.1f}deg, R = {:.0f}m".format(
                            self._energy * fhad, viewing_angles[iS], R))
                        fig.subplots_adjust(top=0.9)
                        plt.show()

                    channel = NuRadioReco.framework.channel.Channel(channel_id)
                    channel.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / self._dt)
                    channel.set_trace_start_time(T)
                    channel[chp.azimuth] = azimuth
                    channel[chp.zenith] = zenith
                    channel[chp.ray_path_type] = ray.solution_types[r.get_solution_type(iS)]
                    self._sim_station.add_channel(channel)

                    # apply a simple threshold cut to speed up the simulation,
                    # application of antenna response will just decrease the
                    # signal amplitude
                    if(np.max(np.abs(channel.get_trace())) > 2 * self._Vrms):
                        candidate_event = True

            #print("start detector simulation. time: " + str(time.time()))
            t3 = time.time()
            rayTracingTime += (t3 - t2)
            # perform only a detector simulation if event had at least one
            # candidate channel
            if(not candidate_event):
                continue
            logger.debug("performing detector simulation")
            # self._finalize NuRadioReco event structure
            self._station = NuRadioReco.framework.station.Station(self._station_id)
            self._station.set_sim_station(self._sim_station)
            self._evt = NuRadioReco.framework.event.Event(0, self._event_id)
            self._evt.set_station(self._station)
            self._station.set_station_time(self._evt_time)

            self._detector_simulation()
            self._calculate_signal_properties()
            self._save_triggers_to_hdf5()
            if(self._outputfilenameNuRadioReco is not None):
                self._eventWriter.run(self._evt)
            t4 = time.time()
            detSimTime += (t4 - t3)
            

        # save simulation run in hdf5 format (only triggered events)
        t5 = time.time()
        self._write_ouput_file()
        
        t_total = time.time() - t_start
        logger.warning("{:d} events processed in {:.0f} seconds = {:.2f}ms/event".format(self._n_events,
                                                                                         t_total, 1.e3 * t_total / self._n_events))

        self.calculate_Veff()

        outputTime = time.time() - t5
        print("inputTime = " + str(inputTime) + "\nrayTracingTime = " + str(rayTracingTime) +
              "\ndetSimTime = " + str(detSimTime) + "\noutputTime = " + str(outputTime))
        
    
    def _increase_signal(self, channel_id, factor):
        """
        increase the signal of a simulated station by a factor of x
        this is e.g. used to approximate a phased array concept with a single antenna
        
        Parameters
        ----------
        channel_id: int or None
            if None, all available channels will be modified
        """
        if(channel_id is None):
            for sim_channels in self._station.get_sim_station().iter_channels():
                for sim_channel in sim_channels:
                    sim_channel.set_trace(sim_channel.get_trace() * factor, sampling_rate=sim_channel.get_sampling_rate())
                
        else:
            sim_channels = self._station.get_sim_station().get_channel(channel_id)
            for sim_channel in sim_channels:
                sim_channel.set_trace(sim_channel.get_trace() * factor, sampling_rate=sim_channel.get_sampling_rate())
        
    def _read_input_hdf5(self):
        """
        reads input file into memory
        """
        fin = h5py.File(self._eventlist, 'r')
        self._fin = {}
        self._fin_attrs = {}
        for key, value in iteritems(fin):
            self._fin[key] = np.array(value)
        for key, value in iteritems(fin.attrs):
            self._fin_attrs[key] = value
        fin.close()
    
    def _calculate_signal_properties(self):
        if(self._station.has_triggered()):
            self._channelSignalReconstructor.run(self._evt, self._station, self._det)
            for channel in self._station.get_channels():
                self._mout['maximum_amplitudes'][self._iE, channel.get_id()] = channel.get_parameter(chp.maximum_amplitude)
                self._mout['maximum_amplitudes_envelope'][self._iE, channel.get_id()] = channel.get_parameter(chp.maximum_amplitude_envelope)

            self._mout['SNRs'][self._iE] = self._station.get_parameter(stnp.channels_max_amplitude) / self._Vrms

    def _save_triggers_to_hdf5(self):

        if('trigger_names' not in self._mout_attrs):
            self._mout_attrs['trigger_names'] = []
            for trigger in six.itervalues(self._station.get_triggers()):
                self._mout_attrs['trigger_names'].append(trigger.get_name())
        # the 'multiple_triggers' output array is not initialized in the constructor because the number of 
        # simulated triggers is unknown at the beginning. So we check if the key already exists and if not, 
        # we first create this data structure
        if('multiple_triggers' not in self._mout):
            self._mout['multiple_triggers'] = np.zeros((self._n_events, len(self._mout_attrs['trigger_names'])))
        for iT, trigger_name in enumerate(self._mout_attrs['trigger_names']):
            self._mout['multiple_triggers'][self._iE, iT] = self._station.get_trigger(trigger_name).has_triggered()

        self._mout['triggered'][self._iE] = np.any(self._mout['multiple_triggers'][self._iE])
        if(self._mout['triggered'][self._iE]):
            logger.info("event triggered")

    
    def get_Vrms(self):
        return self._Vrms

    def get_sampling_rate(self):
        return 1. / self._dt

    def get_bandwidth(self):
        return self._bandwidth
    
    def _check_if_was_pre_simulated(self):
        """
        checks if the same detector was simulated before (then we can save the ray tracing part)
        """
        self._was_pre_simulated = False
        if('detector' in self._fin_attrs):
            with open(self._detectorfile) as fdet:
                if(fdet.read() == self._fin_attrs['detector']):
                    self._was_pre_simulated = True
                    print("the simulation was already performed with the same detector")
        return self._was_pre_simulated

    
    def _create_meta_output_datastructures(self):
        """
        creates the data structures of the parameters that will be saved into the hdf5 output file
        """
        self._mout = {}
        self._mout_attributes = {}
        self._mout['weights'] = np.zeros(self._n_events)
        self._mout['triggered'] = np.zeros(self._n_events, dtype=np.bool)
#         self._mout['multiple_triggers'] = np.zeros((self._n_events, self._number_of_triggers), dtype=np.bool)
        self._mout_attributes['trigger_names'] = None
        self._mout['launch_vectors'] = np.zeros((self._n_events, self._n_antennas, 2, 3)) * np.nan
        self._mout['receive_vectors'] = np.zeros((self._n_events, self._n_antennas, 2, 3)) * np.nan
        self._mout['ray_tracing_C0'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
        self._mout['ray_tracing_C1'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
        self._mout['ray_tracing_solution_type'] = np.zeros((self._n_events, self._n_antennas, 2), dtype=np.int) * np.nan
        self._mout['polarization'] = np.zeros((self._n_events, self._n_antennas, 2, 3)) * np.nan
        self._mout['travel_times'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
        self._mout['travel_distances'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
        self._mout['SNRs'] = np.zeros(self._n_events) * np.nan
        self._mout['maximum_amplitudes'] = np.zeros((self._n_events, self._n_antennas)) * np.nan
        self._mout['maximum_amplitudes_envelope'] = np.zeros((self._n_events, self._n_antennas)) * np.nan
        
    def _read_input_neutrino_properties(self):
        self._event_id = self._fin['event_ids'][self._iE]
        self._flavor = self._fin['flavors'][self._iE]
        self._energy = self._fin['energies'][self._iE]
        self._ccnc = self._fin['ccncs'][self._iE]
        self._x = self._fin['xx'][self._iE]
        self._y = self._fin['yy'][self._iE]
        self._z = self._fin['zz'][self._iE]
        self._zenith_nu = self._fin['zeniths'][self._iE]
        self._azimuth_nu = self._fin['azimuths'][self._iE]
        self._inelasticity = self._fin['inelasticity'][self._iE]
        
    def _create_sim_station(self):
        """
        created an empyt sim_station object and saves the meta arguments such as neutrino direction, self._energy and self._flavor
        """
        # create NuRadioReco event structure
        self._sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
        # save relevant neutrino properties
        self._sim_station[stnp.nu_zenith] = self._zenith_nu
        self._sim_station[stnp.nu_azimuth] = self._azimuth_nu
        self._sim_station[stnp.nu_energy] = self._energy
        self._sim_station[stnp.nu_flavor] = self._flavor
        self._sim_station[stnp.ccnc] = self._ccnc
        self._sim_station[stnp.nu_vertex] = np.array([self._x, self._y, self._z])
        self._sim_station[stnp.inelasticity] = self._inelasticity


    def _add_empty_channel(self, channel_id):
        channel = NuRadioReco.framework.channel.Channel(channel_id)
        channel.set_frequency_spectrum(np.zeros((3, len(self._ff)), dtype=np.complex), 1. / self._dt)
        channel[chp.azimuth] = 0
        channel[chp.zenith] = 180 * units.deg
        channel[chp.ray_path_type] = 'none'
        channel.set_trace_start_time(np.nan)
        self._sim_station.add_channel(channel)

    def _write_ouput_file(self):
        fout = h5py.File(self._outputfilename, 'w')
        for (key, value) in iteritems(self._mout):
            fout[key] = value[self._mout['triggered']]
            
        for (key, value) in iteritems(self._mout_attrs):
            fout.attrs[key] = value

        with open(self._detectorfile) as fdet:
            fout.attrs['detector'] = fdet.read()
        # save antenna position separately to hdf5 output
        n_channels = self._det.get_number_of_channels(self._station_id)
        positions = np.zeros((n_channels, 3))
        for channel_id in range(n_channels):
            positions[channel_id] = self._det.get_relative_position(self._station_id, channel_id)
        fout.attrs['antenna_positions'] = positions    
        
        fout.attrs['Tnoise'] = self._Tnoise
        fout.attrs['Vrms'] = self._Vrms
        fout.attrs['dt'] = self._dt
        fout.attrs['bandwidth'] = self._bandwidth
        fout.attrs['n_samples'] = self._n_samples
        fout.attrs['config'] = yaml.dump(self._cfg)

        # now we also save all input parameters back into the out file
        for key in self._fin.keys():
            if(not key in fout.keys()):  # only save data sets that havn't been recomputed and saved already
                fout[key] = np.array(self._fin[key])[self._mout['triggered']]
        for key in self._fin_attrs.keys():
            if(not key in fout.attrs.keys()):  # only save atrributes sets that havn't been recomputed and saved already
                fout.attrs[key] = self._fin_attrs[key]
        fout.close()
        
    def calculate_Veff(self):
        # calculate effective
        density_ice = 0.9167 * units.g / units.cm ** 3
        density_water = 1000 * units.kg / units.m ** 3

        n_triggered = np.sum(self._mout['triggered'])
        n_triggered_weighted = np.sum(self._mout['weights'][self._mout['triggered']])
        logger.warning('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(
            n_triggered, self._n_events, n_triggered / self._n_events))

        V = None
        if('xmax' in self._fin_attrs):
            dX = self._fin_attrs['xmax'] - self._fin_attrs['xmin']
            dY = self._fin_attrs['ymax'] - self._fin_attrs['ymin']
            dZ = self._fin_attrs['zmax'] - self._fin_attrs['zmin']
            V = dX * dY * dZ
        elif('rmin' in self._fin_attrs):
            rmin = self._fin_attrs['rmin']
            rmax = self._fin_attrs['rmax']
            dZ = self._fin_attrs['zmax'] - self._fin_attrs['zmin']
            V = np.pi * (rmax**2 - rmin**2) * dZ
        Veff = V * density_ice / density_water * 4 * np.pi * n_triggered_weighted / self._n_events
        logger.warning("Veff = {:.2g} km^3 sr".format(Veff / units.km ** 3))
        
    def _get_em_had_fraction(self, inelasticity, ccnc, flavor):
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
    
    # TODO verify that calculation of polarization vector is correct!
    def _calculate_polarization_vector(self):
        """ calculates the polarization vector in spherical coordinates (eR, eTheta, ePhi)
        """ 
        polarization_direction = np.cross(self._launch_vector, np.cross(self._shower_axis, self._launch_vector))
        polarization_direction /= np.linalg.norm(polarization_direction)
        cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*self._launch_vector))
        return cs.transform_from_ground_to_onsky(polarization_direction)
