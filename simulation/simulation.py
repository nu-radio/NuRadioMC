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
# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sim")

VERSION = 0.1
delta_C_cut = 40 * units.deg
minimum_weight_cut = 1e-5


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


class simulation():

    def __init__(self, eventlist,
                 outputfilename,
                 detectorfile,
                 station_id,
                 Tnoise,
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1)):
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
            defined in the detector description
        Tnoise: float
            noise temperature in Kelvin (assuming white noise)
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
        self.__eventlist = eventlist
        self.__outputfilename = outputfilename
        self.__detectorfile = detectorfile
        self.__station_id = station_id
        self.__Tnoise = Tnoise
        self.__outputfilenameNuRadioReco = outputfilenameNuRadioReco
        self.__debug = debug
        self.__evt_time = evt_time

        # read in detector positions
        logger.info("Detectorfile {}".format(self.__detectorfile))
        self.__det = detector.Detector(json_filename=self.__detectorfile)

        # read time and frequency resolution from detector (assuming all
        # channels have the same sampling)
        self.__dt = 1. / self.__det.get_sampling_frequency(station_id, 0)
        self.__bandwidth = 0.5 / self.__dt
        self.__n_samples = self.__det.get_number_of_samples(station_id, 0)
        self.__ff = np.fft.rfftfreq(self.__n_samples, self.__dt)
        self.__tt = np.arange(0, self.__n_samples * self.__dt, self.__dt)
        self.__Vrms = (Tnoise * 50 * constants.k *
                       self.__bandwidth / units.Hz) ** 0.5

    def get_Vrms(self):
        return self.__Vrms

    def get_sampling_rate(self):
        return 1. / self.__dt

    def get_bandwidth(self):
        return self.__bandwidth

    def run(self, detector_simulation, number_of_triggers=1):
        """
        run the NuRadioMC simulation

        Parameters
        ----------
        detector_simulation: function
            a function that containes the detector simulation
        number_of_triggers: int
            the number of different triggers
        """

        def add_empty_channel(sim_station, channel_id):
            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel.set_frequency_spectrum(np.zeros((3, len(self.__ff)), dtype=np.complex), 1. / self.__dt)
            channel[chp.azimuth] = 0
            channel[chp.zenith] = 180 * units.deg
            channel[chp.ray_path_type] = 'none'
            channel.set_trace_start_time(np.nan)
            sim_station.add_channel(channel)

        channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
        if(self.__outputfilenameNuRadioReco is not None):
            eventWriter.begin(self.__outputfilenameNuRadioReco)

        fin = h5py.File(self.__eventlist, 'r')
        fout = h5py.File(self.__outputfilename, 'w')
        n_events = len(fin['event_ids'])
#        n_events = 1000
        n_antennas = self.__det.get_number_of_channels(self.__station_id)
        logger.info("processing {} events".format(n_events))

        # check if the same detector was simulated before (then we can save the ray tracing part)
        same_detector = False
        if('detector' in fin.attrs):
            with open(self.__detectorfile) as fdet:
                if(fdet.read() == fin.attrs['detector']):
                    same_detector = True
                    logger.info("the simulation was already performed with the same detector")

        # define arrays that will be saved at the end
        weights = np.zeros(n_events)
        triggered = np.zeros(n_events, dtype=np.bool)
        multiple_triggers = np.zeros((n_events, number_of_triggers), dtype=np.bool)
        trigger_names = None
        launch_vectors = np.zeros((n_events, n_antennas, 2, 3)) * np.nan
        receive_vectors = np.zeros((n_events, n_antennas, 2, 3)) * np.nan
        ray_tracing_C0 = np.zeros((n_events, n_antennas, 2)) * np.nan
        ray_tracing_C1 = np.zeros((n_events, n_antennas, 2)) * np.nan
        ray_tracing_solution_type = np.zeros((n_events, n_antennas, 2), dtype=np.int) * np.nan
        polarization = np.zeros((n_events, n_antennas, 2)) * np.nan
        travel_times = np.zeros((n_events, n_antennas, 2)) * np.nan
        travel_distances = np.zeros((n_events, n_antennas, 2)) * np.nan
        SNRs = np.zeros(n_events) * np.nan
        maximum_amplitudes = np.zeros((n_events, n_antennas)) * np.nan
        maximum_amplitudes_envelope = np.zeros((n_events, n_antennas)) * np.nan

        inputTime = 0.0
        rayTracingTime = 0.0
        detSimTime = 0.0
        outputTime = 0.0
        time_attenuation_length = 0.
        t_start = time.time()
        for iE in range(n_events):
            #print("start event. time: " + str(time.time()))
            t1 = time.time()
            if(iE > 0 and iE % max(1, int(n_events / 100.)) == 0):
                eta = datetime.timedelta(seconds=(time.time() - t_start) * (n_events - iE) / iE)
                total_time = inputTime + rayTracingTime + detSimTime + outputTime
                logger.warning("processing event {}/{} = {:.1f}%, ETA {}, time consumption: ray tracing = {:.0f}% (att. length {:.0f}%), detector simulation = {:.0f}% ".format(
                    iE, n_events, 100. * iE / n_events, eta, 100. * rayTracingTime / total_time, 100. * time_attenuation_length / rayTracingTime, 100. * detSimTime / total_time))
#             if(iE > 0 and iE % max(1, int(n_events / 10000.)) == 0):
#                 print("*", end='')
            # read all quantities from hdf5 file and store them in local
            # variables
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

            # calculate weight
            weights[iE] = get_weight(zenith_nu, energy, mode='simple')
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if(weights[iE] < minimum_weight_cut):
                logger.debug("neutrino weight is smaller than {}, skipping event".format(minimum_weight_cut))
                continue

            # be careful, zenith/azimuth angle always refer to where the neutrino came from,
            # i.e., opposite to the direction of propagation. We need the propagation directio nhere,
            # so we multiply the shower axis with '-1'
            shower_axis = -1 * hp.spherical_to_cartesian(zenith_nu, azimuth_nu)
            x1 = np.array([x, y, z])

            # calculate correct chereknov angle for ice density at vertex
            # position
            ice = medium.ARAsim_southpole()
            n_index = ice.get_index_of_refraction(x1)
            cherenkov_angle = np.arccos(1. / n_index)

            # create NuRadioReco event structure
            sim_station = NuRadioReco.framework.sim_station.SimStation(self.__station_id)
            # save relevant neutrino properties
            sim_station[stnp.nu_zenith] = zenith_nu
            sim_station[stnp.nu_azimuth] = azimuth_nu
            sim_station[stnp.nu_energy] = energy
            sim_station[stnp.nu_flavor] = flavor
            sim_station[stnp.ccnc] = ccnc
            sim_station[stnp.nu_vertex] = np.array([x, y, z])
            sim_station[stnp.inelasticity] = inelasticity

            candidate_event = False

            # first step: peorform raytracing to see if solution exists
            #print("start raytracing. time: " + str(time.time()))
            t2 = time.time()
            inputTime += (t2 - t1)
            ray_tracing_performed = ('ray_tracing_C0' in fin) and (same_detector)
            for channel_id in range(self.__det.get_number_of_channels(self.__station_id)):
                x2 = self.__det.get_relative_position(self.__station_id, channel_id)
                r = ray.ray_tracing(x1, x2, ice, log_level=logging.WARNING)

                if(ray_tracing_performed):  # check if raytracing was already performed
                    r.set_solution(fin['ray_tracing_C0'][iE, channel_id], fin['ray_tracing_C1'][iE, channel_id],
                                   fin['ray_tracing_solution_type'][iE, channel_id])
                else:
                    r.find_solutions()
                if(not r.has_solution()):
                    logger.debug("event {} and station {}, channel {} does not have any ray tracing solution".format(
                        event_id, self.__station_id, channel_id))
                    add_empty_channel(sim_station, channel_id)
                    continue
                delta_Cs = []
                viewing_angles = []
                # loop through all ray tracing solution
                for iS in range(r.get_number_of_solutions()):
                    ray_tracing_C0[iE, channel_id, iS] = r.get_results()[iS]['C0']
                    ray_tracing_C1[iE, channel_id, iS] = r.get_results()[iS]['C1']
                    ray_tracing_solution_type[iE, channel_id, iS] = r.get_solution_type(iS)
                    launch_vector = r.get_launch_vector(iS)
                    launch_vectors[iE, channel_id, iS] = launch_vector
                    # calculates angle between shower axis and launch vector
                    viewing_angle = hp.get_angle(shower_axis, launch_vector)
                    viewing_angles.append(viewing_angle)
                    delta_C = (viewing_angle - cherenkov_angle)
                    logger.debug('solution {} {}: viewing angle {:.1f} = delta_C = {:.1f}'.format(
                        iS, ray.solution_types[r.get_solution_type(iS)], viewing_angle / units.deg, (viewing_angle - cherenkov_angle) / units.deg))
                    delta_Cs.append(delta_C)

                # discard event if delta_C (angle off cherenkov cone) is too large
                if(min(np.abs(delta_Cs)) > delta_C_cut):
                    logger.debug('delta_C too large, event unlikely to be observed, skipping event')
                    add_empty_channel(sim_station, channel_id)
                    continue

                n = r.get_number_of_solutions()
                Rs = np.zeros(n)
                Ts = np.zeros(n)
                tts = np.zeros((n, self.__n_samples))
                for iS in range(n):  # loop through all ray tracing solution
                    if(ray_tracing_performed):
                        R = fin['travel_distances'][iE, channel_id, iS]
                        T = fin['travel_times'][iE, channel_id, iS]
                    else:
                        R = r.get_path_length(iS)  # calculate path length
                        T = r.get_travel_time(iS)  # calculate travel time
                    travel_distances[iE, channel_id, iS] = R
                    travel_times[iE, channel_id, iS] = T
                    Rs[iS] = R
                    Ts[iS] = T
                    receive_vector = r.get_receive_vector(iS)
                    # save receive vector
                    receive_vectors[iE, channel_id, iS] = receive_vector
                    zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
                    logger.debug("ch {}, s {} R = {:.1f} m, t = {:.1f}ns, receive angles {:.0f} {:.0f}".format(
                        channel_id, iS, R / units.m, T / units.ns, zenith / units.deg, azimuth / units.deg))
        #             eR, eTheta, ePhi = get_frequency_spectrum(energy, viewing_angles[iS], ff, 0, n, R)

                    fem, fhad = get_em_had_fraction(inelasticity, ccnc, flavor)
                    # get neutrino pulse from Askaryan module
        #             eR, eTheta, ePhi = signalgen.get_frequency_spectrum(energy * fhad, viewing_angles[iS], n_samples, dt, 0, n_index, R, 'Alvarez2000')
                    eTheta = signalgen.get_frequency_spectrum(
                        energy * fhad, viewing_angles[iS], self.__n_samples, self.__dt, 0, n_index, R, 'Alvarez2000')

                    # apply frequency dependent attenuation
                    t_att = time.time()
                    attn = r.get_attenuation(iS, self.__ff)
                    time_attenuation_length += (time.time() - t_att)
        #             eR *= attn
                    eTheta *= attn
        #             ePhi *= attn

                    if(fem > 0):
                        eTheta2 = signalgen.get_frequency_spectrum(
                            energy * fem, viewing_angles[iS], self.__n_samples, self.__dt, 1, n_index, R, 'Alvarez2000')
        #                 eR2 *= attn
                        eTheta2 *= attn
        #                 ePhi2 *= attn
                        # add EM signal to had signal in the time domain
        #                 eR = fft.time2freq(fft.freq2time(eR) + fft.freq2time(eR2))
                        eTheta = fft.time2freq(fft.freq2time(eTheta) + fft.freq2time(eTheta2))
        #                 ePhi = fft.time2freq(fft.freq2time(ePhi) + fft.freq2time(ePhi2))

                    # TODO verify that calculation of polarization vector is correct!
                    polarization_direction = np.cross(launch_vector, np.cross(shower_axis, launch_vector))
                    polarization_direction /= np.linalg.norm(polarization_direction)
                    cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*launch_vector))
                    polarization_direction_onsky = cs.transform_from_ground_to_onsky(polarization_direction)
                    logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}'.format(
                        zenith / units.deg, azimuth / units.deg, polarization_direction_onsky[0], polarization_direction_onsky[1], polarization_direction_onsky[2]))
                    polarization[iE, channel_id, iS] = np.arctan2(
                        polarization_direction_onsky[1], polarization_direction_onsky[2])
                    eR, eTheta, ePhi = np.outer(polarization_direction_onsky, eTheta)
        #             print("{} {:.2f} {:.0f}".format(polarization_direction_onsky, np.linalg.norm(polarization_direction_onsky), np.arctan2(np.abs(polarization_direction_onsky[1]), np.abs(polarization_direction_onsky[2])) / units.deg))

                    # in case of a reflected ray we need to account for fresnel
                    # reflection at the surface
                    if(ray.solution_types[r.get_solution_type(iS)] == 'reflected'):
                        from NuRadioReco.utilities import geometryUtilities as geo_utl
                        r_parallel = geo_utl.get_fresnel_r_parallel(
                            zenith, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))
                        r_perpendicular = geo_utl.get_fresnel_r_perpendicular(
                            zenith, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))

                        eTheta *= r_parallel
                        ePhi *= r_perpendicular
                        logger.debug("reflection coefficient is r_parallel = {:.2f}, r_perpendicular = {:.2f}".format(
                            r_parallel, r_perpendicular))

                    if(self.__debug):
                        fig, (ax, ax2) = plt.subplots(1, 2)
                        ax.plot(self.__ff, np.abs(eTheta) / units.micro / units.V * units.m)
                        ax2.plot(self.__tt, fft.freq2time(eTheta) / units.micro / units.V * units.m)
                        ax2.set_ylabel("amplitude [$\mu$V/m]")
                        fig.tight_layout()
                        fig.suptitle("$E_C$ = {:.1g}eV $\Delta \Omega$ = {:.1f}deg, R = {:.0f}m".format(
                            energy * fhad, viewing_angles[iS], R))
                        fig.subplots_adjust(top=0.9)
                        plt.show()

                    channel = NuRadioReco.framework.channel.Channel(channel_id)
                    channel.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / self.__dt)
                    channel.set_trace_start_time(T)
                    channel[chp.azimuth] = azimuth
                    channel[chp.zenith] = zenith
                    channel[chp.ray_path_type] = ray.solution_types[r.get_solution_type(iS)]
                    sim_station.add_channel(channel)

                    # apply a simple threshold cut to speed up the simulation,
                    # application of antenna response will just decrease the
                    # signal amplitude
                    if(np.max(np.abs(channel.get_trace())) > 2 * self.__Vrms):
                        candidate_event = True

            #print("start detector simulation. time: " + str(time.time()))
            t3 = time.time()
            rayTracingTime += (t3 - t2)
            # perform only a detector simulation if event had at least one
            # candidate channel
            if(not candidate_event):
                continue
            logger.debug("performing detector simulation")
            # finalize NuRadioReco event structure
            station = NuRadioReco.framework.station.Station(self.__station_id)
            station.set_sim_station(sim_station)
            evt = NuRadioReco.framework.event.Event(0, event_id)
            evt.set_station(station)
            station.set_station_time(self.__evt_time)

            detector_simulation(evt, station, self.__det, self.__dt, self.__Vrms)

            if(trigger_names is None):
                trigger_names = []
                for trigger in station.get_triggers():
                    trigger_names.append(trigger.get_name())
            for iT, trigger_name in enumerate(trigger_names):
                multiple_triggers[iE, iT] = station.get_trigger(trigger_name).has_triggered()

            triggered[iE] = np.any(multiple_triggers[iE])
            if(triggered[iE]):
                logger.info("event triggered")

            # save events that trigger the detector and have weight > 0
            if(triggered[iE] and (weights[iE] > minimum_weight_cut)):
                channelSignalReconstructor.run(evt, station, self.__det)
                for channel in station.get_channels():
                    maximum_amplitudes[iE, channel.get_id()] = channel.get_parameter(chp.maximum_amplitude)
                    maximum_amplitudes_envelope[iE, channel.get_id()] = channel.get_parameter(chp.maximum_amplitude_envelope)

                SNRs[iE] = station.get_parameter(stnp.channels_max_amplitude) / self.__Vrms
                if(self.__outputfilenameNuRadioReco is not None):
                    eventWriter.run(evt)
            t4 = time.time()
            detSimTime += (t4 - t3)

        # save simulation run in hdf5 format (only triggered events)
        t5 = time.time()
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
        fout['maximum_amplitudes'] = maximum_amplitudes[triggered]
        fout['maximum_amplitudes_envelope'] = maximum_amplitudes_envelope[triggered]
        fout['multiple_triggers'] = multiple_triggers[triggered]
        fout.attrs['trigger_names'] = trigger_names
        with open(self.__detectorfile) as fdet:
            fout.attrs['detector'] = fdet.read()
        fout.attrs['Tnoise'] = self.__Tnoise
        fout.attrs['Vrms'] = self.__Vrms
        fout.attrs['dt'] = self.__dt
        fout.attrs['bandwidth'] = self.__bandwidth
        fout.attrs['n_samples'] = self.__n_samples

        # now we also save all input parameters back into the out file
        for key in fin.keys():
            if(not key in fout.keys()):  # only save data sets that havn't been recomputed and saved already
                fout[key] = np.array(fin[key])[triggered]
        for key in fin.attrs.keys():
            if(not key in fout.attrs.keys()):  # only save atrributes sets that havn't been recomputed and saved already
                fout.attrs[key] = fin.attrs[key]

        t_total = time.time() - t_start
        logger.warning("{:d} events processed in {:.0f} seconds = {:.2f}ms/event".format(n_events,
                                                                                         t_total, 1.e3 * t_total / n_events))

        # calculate effective
        density_ice = 0.9167 * units.g / units.cm ** 3
        density_water = 1000 * units.kg / units.m ** 3

        n_triggered = np.sum(weights[triggered])
        logger.warning('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(
            n_triggered, n_events, n_triggered / n_events))

        V = None
        if('xmax' in fin.attrs):
            dX = fin.attrs['xmax'] - fin.attrs['xmin']
            dY = fin.attrs['ymax'] - fin.attrs['ymin']
            dZ = fin.attrs['zmax'] - fin.attrs['zmin']
            V = dX * dY * dZ
        elif('rmin' in fin.attrs):
            rmin = fin.attrs['rmin']
            rmax = fin.attrs['rmax']
            dZ = fin.attrs['zmax'] - fin.attrs['zmin']
            V = np.pi * (rmax**2 - rmin**2) * dZ
        Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events
        logger.warning("Veff = {:.2g} km^3 sr".format(Veff / units.km ** 3))
        fin.close()
        fout.close()

        outputTime = time.time() - t5
        logger.info("inputTime = " + str(inputTime) + "\nrayTracingTime = " + str(rayTracingTime) +
              "\ndetSimTime = " + str(detSimTime) + "\noutputTime = " + str(outputTime))
