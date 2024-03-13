import os
import collections
import datetime
import logging
import warnings
import time
import copy
import yaml
import numpy as np
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrans
from numpy.random import Generator, Philox
import h5py
from scipy import constants
# import detector simulation modules
from NuRadioMC.SignalGen import askaryan
from NuRadioMC.SignalGen import emitter as emitter_signalgen
import NuRadioMC.utilities.medium
from NuRadioMC.utilities.earth_attenuation import get_weight
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities.Veff import remove_duplicate_triggers
from NuRadioMC.simulation.output_writer_hdf5 import outputWriterHDF5
from NuRadioReco.utilities import units
import NuRadioReco.modules.io.eventWriter
from NuRadioReco.utilities.logging import LOGGING_STATUS, setup_logger
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
from NuRadioReco.detector import detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.particle
import NuRadioReco.framework.event
import NuRadioReco.framework.sim_emitter
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
# parameters describing simulated Monte Carlo particles
from NuRadioReco.framework.parameters import particleParameters as simp
from NuRadioReco.framework.parameters import emitterParameters as ep
# parameters set in the event generator
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
# import NuRadioMC.simulation.time_logger

logger = setup_logger("NuRadioMC.simulation")

# initialize a few NuRadioReco modules
# TODO: Is this the best way to do it? Better to initialize them on demand
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()

def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


def merge_config(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k], v)
    return user


def calculate_sim_efield(showers, sid, cid,
                         det, propagator, medium, config,
                         time_logger=None):
    """
    Calculate the simulated electric field for a given shower and channel.

    Parameters
    ----------
    showers : list of showers
        A list of the showers that should be simulated (as defined in the input HDF5 file)
    sid : int
        the station id, only needed to identify the correct channel in the detector description
    cid : int
        the channel id for which the electric field should be calculated as defined in the detector description
    det : Detector object
        the detector description defining all channels. 
    propagator : propagator object
        the propagator that should be used to calculate the electric field from emitter to receiver, typically a ray tracer
    medium: medium object
        the medium in which the electric field is propagating, typically ice
    config : dict
        the NuRadioMC configuration dictionary (from the yaml file)

    Returns
    -------
    list of SimEfield objects
        A list of SimEfield objects, one for each shower and propagation solution

    """
    logger.debug("Calculating electric field for station %d , channel %d from list of showers", sid, cid)
    p = propagator # shorthand for more compact coding

    sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
    sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)

    x2 = det.get_relative_position(sid, cid) + det.get_absolute_position(sid)
    dt = 1. / config['sampling_rate']
    # rescale the number of samples to the internal (higher) sampling rate used in the simulation
    n_samples = det.get_number_of_samples(sid, cid) / det.get_sampling_frequency(sid, cid) / dt
    n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer

    for shower in showers:
        logger.debug(f"Calculating electric field for shower {shower.get_id()} and station {sid}, channel {cid}")
        shower_axis = -1 * shower.get_axis() # We need the propagation direction here, so we multiply the shower axis with '-1'
        x1 = shower.get_parameter(shp.vertex)
        n_index = medium.get_index_of_refraction(x1)
        cherenkov_angle = np.arccos(1. / n_index)

        p.set_start_and_end_point(x1, x2)
        p.use_optional_function('set_shower_axis', shower_axis)
        if config['speedup']['redo_raytracing']:  # check if raytracing was already performed
            pass
            # TODO: initiatlize ray tracer with existing results if available
        p.find_solutions()
        if not p.has_solution():
            logger.debug(f"shower {shower.get_id()} and station {sid}, channel {cid} from {x1} to {x2} does not have any ray tracing solution")
            continue

        n = p.get_number_of_solutions()
        logger.status(f"found {n} solutions for shower {shower.get_id()} and station {sid}, channel {cid} from {x1} to {x2}")
        delta_Cs = np.zeros(n)
        viewing_angles = np.zeros(n)
        # loop through all ray tracing solution
        for iS in range(p.get_number_of_solutions()):
            viewing_angles[iS] = hp.get_angle(shower_axis, p.get_launch_vector(iS))
            delta_Cs[iS] = viewing_angles[iS] - cherenkov_angle
        # discard event if delta_C (angle off cherenkov cone) is too large
        if min(np.abs(delta_Cs)) > config['speedup']['delta_C_cut']:
            logger.debug(f'delta_C too large, event unlikely to be observed, (min(Delta_C) = {min(np.abs(delta_Cs))/units.deg:.1f}deg), skipping event')
            continue
        for iS in range(n): # loop through all ray tracing solution
            # skip individual channels where the viewing angle difference is too large
            # discard event if delta_C (angle off cherenkov cone) is too large
            if np.abs(delta_Cs[iS]) > config['speedup']['delta_C_cut']:
                logger.debug('delta_C too large, ray tracing solution unlikely to be observed, skipping ray tracing solution')
                continue
            # TODO: Fill with previous values if RT was already performed
            R = p.get_path_length(iS)  # calculate path length
            T = p.get_travel_time(iS)  # calculate travel time
            if R is None or T is None:
                logger.warning(f'travel distance or travel time could not be calculated, skipping ray tracing solution. Shower ID: {shower.get_id()} Station ID: {sid} Channel ID: {cid}')
                continue
            kwargs = {}
            # if the input file specifies a specific shower realization, or
            # if the shower was already simulated (e.g. for a different channel or ray tracing solution)
            # use that realization
            if config['signal']['model'] in ["ARZ2019", "ARZ2020"] and shower.has_parameter(shp.charge_excess_profile_id):
                kwargs['iN'] = shower[shp.charge_excess_profile_id]
                logger.debug(f"reusing shower {kwargs['iN']} ARZ shower library")
            elif config['signal']['model'] == "Alvarez2009" and shower.has_parameter(shp.k_L):
                kwargs['k_L'] = shower[shp.k_L]
                logger.debug(f"reusing k_L parameter of Alvarez2009 model of k_L = {kwargs['k_L']:.4g}")

            spectrum, additional_output = askaryan.get_frequency_spectrum(shower[shp.energy], viewing_angles[iS],
                            n_samples, dt, shower[shp.type], n_index, R,
                            config['signal']['model'], seed=config['seed'], full_output=True, **kwargs)
            # save shower realization to SimShower and hdf5 file
            if config['signal']['model'] in ["ARZ2019", "ARZ2020"]:
                if not shower.has_parameter(shp.charge_excess_profile_id):
                    shower.set_parameter(shp.charge_excess_profile_id, additional_output['iN'])
                    logger.debug(f"setting shower profile for ARZ shower library to i = {additional_output['iN']}")
            if config['signal']['model'] == "Alvarez2009":
                if not shower.has_parameter(shp.k_L):
                    shower.set_parameter(shp.k_L, additional_output['k_L'])

            polarization_direction_onsky = calculate_polarization_vector(shower_axis, p.get_launch_vector(iS), config)
            receive_vector = p.get_receive_vector(iS)
            eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)


            # this is common stuff which is the same between emitters and showers
            electric_field = NuRadioReco.framework.electric_field.ElectricField([cid],
                                    position=det.get_relative_position(sid, cid),
                                    shower_id=shower.get_id(), ray_tracing_id=iS)
            electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / dt)
            electric_field = p.apply_propagation_effects(electric_field, iS)
            # Trace start time is equal to the interaction time relative to the first
            # interaction plus the wave travel time.
            if shower.has_parameter(shp.vertex_time):
                trace_start_time = shower[shp.vertex_time] + T
            else:
                trace_start_time = T

            # We shift the trace start time so that the trace time matches the propagation time.
            # The centre of the trace corresponds to the instant when the signal from the shower
            # vertex arrives at the observer. The next line makes sure that the centre time
            # of the trace is equal to vertex_time + T (wave propagation time)
            trace_start_time -= 0.5 * electric_field.get_number_of_samples() / electric_field.get_sampling_rate()

            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            electric_field.set_trace_start_time(trace_start_time)
            electric_field[efp.azimuth] = azimuth
            electric_field[efp.zenith] = zenith
            electric_field[efp.ray_path_type] = propagation.solution_types[p.get_solution_type(iS)]
            electric_field[efp.nu_vertex_distance] = R
            electric_field[efp.nu_vertex_travel_time] = T
            electric_field[efp.nu_viewing_angle] = viewing_angles[iS]
            electric_field[efp.polarization_angle] = np.arctan2(*polarization_direction_onsky[1:][::-1]) #: electric field polarization in onsky-coordinates. 0 corresponds to polarization in e_theta, 90deg is polarization in e_phi
            electric_field[efp.raytracing_solution] = p.get_raytracing_output(iS)
            electric_field[efp.launch_vector] = p.get_launch_vector(iS)

            sim_station.add_electric_field(electric_field)
            logger.debug(f"Added electric field to SimStation for shower {shower.get_id()} and station {sid}, channel {cid} with ray tracing solution {iS} and viewing angle {viewing_angles[iS]/units.deg:.1f}deg")

            # TODO: Implement this speedup cut
            # apply a simple threshold cut to speed up the simulation,
            # application of antenna response will just decrease the
            # signal amplitude
            # if np.max(np.abs(electric_field.get_trace())) > float(config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[self._station_id][channel_id]:
            #     candidate_station = True
    return sim_station

def calculate_sim_efield_for_emitter(emitters, sid, cid,
                         det, propagator, medium, config,
                         rnd, antenna_pattern_provider,
                         time_logger=None):
    """
    Calculate the simulated electric field for a given shower and channel.

    Parameters
    ----------
    emitters : list of emitters
        A list of the emitters that should be simulated (as defined in the input HDF5 file)
    sid : int
        the station id, only needed to identify the correct channel in the detector description
    cid : int
        the channel id for which the electric field should be calculated as defined in the detector description
    det : Detector object
        the detector description defining all channels. 
    propagator : propagator object
        the propagator that should be used to calculate the electric field from emitter to receiver, typically a ray tracer
    medium: medium object
        the medium in which the electric field is propagating, typically ice
    config : dict
        the NuRadioMC configuration dictionary (from the yaml file)
    rnd: numpy random number generator
        the random number generator to be used for the simulation
    antenna_pattern_provider: antenna pattern provider object
        the antenna pattern provider object to be used for the simulation

    Returns
    -------
    list of SimEfield objects
        A list of SimEfield objects, one for each shower and propagation solution

    """
    logger.debug(f"Calculating electric field for station {sid}, channel {cid} from list of emitters")
    p = propagator # shorthand for more compact coding

    sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
    sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)

    x2 = det.get_relative_position(sid, cid) + det.get_absolute_position(sid)
    n_samples = det.get_number_of_samples(sid, cid)
    dt = 1. / config['sampling_rate']

    for emitter in emitters:
        x1 = emitter.get_parameter(shp.position)
        n_index = medium.get_index_of_refraction(x1)

        p.set_start_and_end_point(x1, x2)
        if config['speedup']['redo_raytracing']:  # check if raytracing was already performed
            pass
            # TODO: initiatlize ray tracer with existing results if available
        p.find_solutions()
        if not p.has_solution():
            logger.debug(f"emitter {emitter.get_id()} and station {sid}, channel {cid} from {x1} to {x2} does not have any ray tracing solution")
            continue

        n = p.get_number_of_solutions()
        for iS in range(n): # loop through all ray tracing solution
            # TODO: Fill with previous values if RT was already performed
            R = p.get_path_length(iS)  # calculate path length
            T = p.get_travel_time(iS)  # calculate travel time
            if R is None or T is None:
                logger.warning(f'travel distance or travel time could not be calculated, skipping ray tracing solution. Emitter ID: {emitter.get_id()} Station ID: {sid} Channel ID: {cid}')
                continue
            # if the input file specifies a specific shower realization, or
            # if the shower was already simulated (e.g. for a different channel or ray tracing solution)
            # use that realization
            amplitude = emitter[ep.amplitude]
            emitter_model = emitter[ep.model]
            emitter_kwargs = {}
            emitter_kwargs["launch_vector"] = p.get_launch_vector(iS)
            for key in ep.keys():
                if key.name not in ['amplitude', 'model', 'position']:
                    if emitter.has_parameter(key):
                        emitter_kwargs[key.name] = emitter[key]

            if emitter_model.startswith("efield_"):
                if emitter_model == "efield_idl1_spice":
                    if emitter.has_parameter(ep.realization_id):
                        emitter_kwargs['iN'] = emitter[ep.realization_id]
                    else:
                        emitter_kwargs['rnd'] = rnd

                (eR, eTheta, ePhi), additional_output = emitter_signalgen.get_frequency_spectrum(amplitude, n_samples, dt, emitter_model, **emitter_kwargs, full_output=True)
                if emitter_model == "efield_idl1_spice":
                    if not emitter.has_parameter(ep.realization_id):
                        emitter.set_parameter(ep.realization_id, additional_output['iN'])
                        logger.debug(f"setting emitter realization to i = {additional_output['iN']}")
            else:
                # the emitter fuction returns the voltage output of the pulser. We need to convole with the antenna response of the emitting antenna
                # to obtain the emitted electric field.
                # get emitting antenna properties
                antenna_model = emitter[ep.antenna_type]
                antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
                ori = [emitter[ep.orientation_theta], emitter[ep.orientation_phi],
                       emitter[ep.emitter_rotation_theta], emitter[ep.emitter_rotation_phi]]
                # source voltage given to the emitter
                voltage_spectrum_emitter = emitter.get_frequency_spectrum(amplitude, n_samples, dt,
                                                                            emitter_model, **emitter_kwargs)
                # convolve voltage output with antenna response to obtain emitted electric field
                frequencies = np.fft.rfftfreq(n_samples, d=dt)
                zenith_emitter, azimuth_emitter = hp.cartesian_to_spherical(*p.get_launch_vector(iS))
                VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_emitter, azimuth_emitter, *ori)
                c = constants.c * units.m / units.s
                eTheta = VEL['theta'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
                ePhi = VEL['phi'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
                eR = np.zeros_like(eTheta)
            # rescale amplitudes by 1/R, for emitters this is not part of the "SignalGen" class
            eTheta *= 1 / R
            ePhi *= 1 / R

            # this is common stuff which is the same between emitters and showers. Make sure to do any changes to this code in both places
            electric_field = NuRadioReco.framework.electric_field.ElectricField([cid],
                                    position=det.get_relative_position(sid, cid),
                                    shower_id=emitter.get_id(), ray_tracing_id=iS)
            electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / dt)
            electric_field = p.apply_propagation_effects(electric_field, iS)
            # Trace start time is equal to the emitter time in case one was defined
            # (relevant for multiple emitters per event group)
            if emitter.has_parameter(ep.time):
                trace_start_time = emitter[ep.time] + T
            else:
                trace_start_time = T

            # We shift the trace start time so that the trace time matches the propagation time.
            # The centre of the trace corresponds to the instant when the signal from the shower
            # vertex arrives at the observer. The next line makes sure that the centre time
            # of the trace is equal to vertex_time + T (wave propagation time)
            trace_start_time -= 0.5 * electric_field.get_number_of_samples() / electric_field.get_sampling_rate()

            zenith, azimuth = hp.cartesian_to_spherical(*p.get_receive_vector(iS))
            electric_field.set_trace_start_time(trace_start_time)
            electric_field[efp.azimuth] = azimuth
            electric_field[efp.zenith] = zenith
            electric_field[efp.ray_path_type] = propagation.solution_types[p.get_solution_type(iS)]
            electric_field[efp.nu_vertex_distance] = R
            electric_field[efp.raytracing_solution] = p.get_raytracing_output(iS)

            sim_station.add_electric_field(electric_field)

            # TODO: Implement this speedup cut
            # apply a simple threshold cut to speed up the simulation,
            # application of antenna response will just decrease the
            # signal amplitude
            # if np.max(np.abs(electric_field.get_trace())) > float(config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[self._station_id][channel_id]:
            #     candidate_station = True
    return sim_station

def apply_det_response_sim(sim_station, det, config,
                        detector_simulation_filter_amp=None,
                        evt=None,
                        event_time=None,
                        detector_simulation_part1=None,
                        time_logger=None):
    """
    Apply the detector response to the simulated electric field, i.e., the voltage traces
    seen by the readout system, per shower, raytracing solution and channel.
    This includes the effect of the antenna response, the
    analog signal chain. The result is a list of SimChannel objects which are added to the
    SimStation object.


    Parameters
    ----------
    sim_station : sim_station object that contains the electric fields at the observer positions
        A list of SimEfield objects, one for each shower and propagation solution
    det : Detector object
        the detector description defining all channels.
    config : dict
        the NuRadioMC configuration dictionary (from the yaml file)
    detector_simulation_filter_amp: function (optional)
        a function that applies the filter and amplifier response to the electric field
        the arguments to the function are (event, station, detector)
        if not provided, the function `detector_simulation_part1` needs to be provided.
    evt : NuRadioReco event object (optional)
        all NuRadioReco modules that get executed will be registered to the event.
        If no event is provided, a dummy event is created so that the function runs, but
        then this information is not available to the user. 
    event_time: time object (optional)
        the time of the event to be simulated
    detector_simulation_part1: function (optional)
        this function gives the user the full flexibility to implement all processing
        arguments to the function are (event, station, detector)

    Returns nothing. The SimChannels are added to the SimStation object.
    """
    if evt is None:
        evt = NuRadioReco.framework.event.Event(0, 0)
    if event_time is not None:
        sim_station.set_station_time(event_time)

    if detector_simulation_filter_amp is None and detector_simulation_part1 is None:
        logger.error("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part1")
        raise ValueError("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part1")

    # convert efields to voltages at digitizer
    if detector_simulation_part1 is not None:
        detector_simulation_part1(sim_station, det)
    else:
        efieldToVoltageConverterPerEfield.run(evt, sim_station, det)  # convolve efield with antenna pattern
        detector_simulation_filter_amp(evt, sim_station, det)
        channelAddCableDelay.run(evt, sim_station, det)

    if config['speedup']['amp_per_ray_solution']:
        channelSignalReconstructor.run(evt, sim_station, det)

def apply_det_response(evt, det, config,
                        detector_simulation_filter_amp=None,
                        add_noise=None,
                        Vrms_per_channel=None,
                        integrated_channel_response=None,
                        noiseless_channels=None,
                        detector_simulation_part2=None,
                        time_logger=None):
    """
    Apply the detector response to the simulated electric field, i.e., the voltage traces
    seen by the readout system. This function combines all electric fields (from different showers and
    ray tracing solutions) of one detector channel/antenna. This includes the effect of the antenna response, the
    analog signal chain. The result is a list of Channel objects which are added to the
    Station object.


    Parameters
    ----------
    evt : NuRadioReco.framework.event.Event
        Event object containing all the showers/emitters and electric fields
    det : Detector object
        the detector description defining all channels.
    config : dict
        the NuRadioMC configuration dictionary (from the yaml file)
    detector_simulation_filter_amp: function (optional)
        a function that applies the filter and amplifier response to the electric field
        the arguments to the function are (event, station, detector)
        if not provided, the function `detector_simulation_part1` needs to be provided.
    add_noise : bool
        if True, noise is added to the channels
    Vrms_per_channel : dict
        the noise RMS per channel
    integrated_channel_response : dict
        the integrated channels response int(S21^2, dt) per channel. Corresponds to bandwidth for signal 
        chains without amplification. This is used to normalize the noise level
        to the bandwidth it is generated for.
    noiseless_channels : dict (keys station_id) of list of ints
        the channels that should not have noise added
    detector_simulation_part2: function (optional)
        this function gives the user the full flexibility to implement all processing
        arguments to the function are (event, station, detector)

    Returns nothing. The Channels are added to the Station object.
    """
    if detector_simulation_filter_amp is None and detector_simulation_part2 is None:
        logger.error("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part1")
        raise ValueError("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part1")

    station = evt.get_station()  # will raise an error if there are more than one station, but this should never happen
    # convert efields to voltages at digitizer
    if detector_simulation_part2 is not None:
        detector_simulation_part2(evt, station, det)
    else:
        dt = 1. / (config['sampling_rate'] * units.GHz)
        # start detector simulation
        efieldToVoltageConverter.run(evt, station, det)  # convolve efield with antenna pattern
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        channelResampler.run(evt, station, det, sampling_rate=1. / dt)

        if add_noise:
            max_freq = 0.5 / dt
            channel_ids = det.get_channel_ids(station.get_id())
            Vrms = {}
            for channel_id in channel_ids:
                norm = integrated_channel_response[station.get_id()][channel_id]
                Vrms[channel_id] = Vrms_per_channel[station.get_id()][channel_id] / (norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(evt, station, det, amplitude=Vrms, min_freq=0 * units.MHz,
                                            max_freq=max_freq, type='rayleigh',
                                            excluded_channels=noiseless_channels[station.get_id()])

        detector_simulation_filter_amp(evt, station, det)

def build_dummy_event(sid, det, config):
    """
    Builds a dummy event for simulation.

    Parameters
    ----------
        sid : int
            The station ID.
        det : Detector object
            The detector description defining all channels.
        config : dict
            The NuRadioMC configuration dictionary (from the yaml file)

    Returns:
        object: The built event object.
    """

    evt = NuRadioReco.framework.event.Event(0, 0)
    sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
    sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)

    dt = 1. / config['sampling_rate']
    # rescale the number of samples to the internal (higher) sampling rate used in the simulation
    n_samples = det.get_number_of_samples(sid, 0) / det.get_sampling_frequency(sid, 0) / dt
    n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer

    for channel_id in det.get_channel_ids(sid):
        electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_id],
                                    det.get_relative_position(sim_station.get_id(), channel_id))
        trace = np.zeros(n_samples)
        trace[n_samples // 2] = 100 * units.V  # set a signal that should satisfy any trigger and speedup cuts
        trace[n_samples // 2 + 1] = -100 * units.V
        electric_field.set_trace(np.array([np.zeros(n_samples), trace, trace]), 1. / dt)
        electric_field.set_trace_start_time(0)
        electric_field[efp.azimuth] = 0
        electric_field[efp.zenith] = 100 * units.deg
        electric_field[efp.ray_path_type] = 0
        sim_station.add_electric_field(electric_field)

    station = NuRadioReco.framework.station.Station(sid)
    station.set_sim_station(sim_station)
    evt.set_station(station)
    return evt


def build_NuRadioEvents_from_hdf5(fin, fin_attrs, idxs):
    """
    Build NuRadioReco event structures from the input file

    The function automatically determines if a particle or an emitter is simulated and builds the
    corresponding event structure.

    Parameters
    ----------
    fin : dict
        the input file data dictionary
    fin_attrs : dict
        the input file attributes dictionary
    idxs : list of ints
        the indices of the events that should be built

    Returns
    -------
        an event group object containing the showers and particles or emitters
        the output should contain all relevant information from the hdf5 file (except the attributes)
        to perform a NuRadioMC simulation
    """
    parent_id = idxs[0]
    event_group_id = fin['event_group_ids'][parent_id]
    event_group = NuRadioReco.framework.event.Event(event_group_id, parent_id)
    # add event generator info event
    for enum_entry in genattrs:
        if enum_entry.name in fin_attrs:
            event_group.set_generator_info(enum_entry, fin_attrs[enum_entry.name])

    particle_mode = "simulation_mode" not in fin_attrs or fin_attrs['simulation_mode'] != "emitter"
    if particle_mode:  # first case: simulation of a particle interaction which produces showers
        # there is only one primary particle per event group
        input_particle = NuRadioReco.framework.particle.Particle(event_group_id)
        input_particle[simp.flavor] = fin['flavors'][parent_id]
        input_particle[simp.energy] = fin['energies'][parent_id]
        input_particle[simp.interaction_type] = fin['interaction_type'][parent_id]
        input_particle[simp.inelasticity] = fin['inelasticity'][parent_id]
        input_particle[simp.vertex] = np.array([fin['xx'][parent_id],
                                                fin['yy'][parent_id],
                                                fin['zz'][parent_id]])
        input_particle[simp.zenith] = fin['zeniths'][parent_id]
        input_particle[simp.azimuth] = fin['azimuths'][parent_id]
        input_particle[simp.inelasticity] = fin['inelasticity'][parent_id]
        input_particle[simp.n_interaction] = fin['n_interaction'][parent_id]
        input_particle[simp.shower_id] = fin['shower_ids'][parent_id]
        if fin['n_interaction'][parent_id] <= 1:
            # parents before the neutrino and outgoing daughters without shower are currently not
            # simulated. The parent_id is therefore at the moment only rudimentarily populated.
            input_particle[simp.parent_id] = None  # primary does not have a parent
        input_particle[simp.vertex_time] = 0
        if 'vertex_times' in fin:
            input_particle[simp.vertex_time] = fin['vertex_times'][parent_id]
        event_group.add_particle(input_particle)

        # now loop over all showers and add them to the event group
        for idx in idxs:
            vertex_time = 0
            if 'vertex_times' in fin:
                vertex_time = fin['vertex_times'][idx]

            # create NuRadioReco event structure
            sim_shower = NuRadioReco.framework.radio_shower.RadioShower(fin['shower_ids'][idx])
            # save relevant neutrino properties
            sim_shower[shp.zenith] = fin['zeniths'][idx]
            sim_shower[shp.azimuth] = fin['azimuths'][idx]
            sim_shower[shp.energy] = fin['shower_energies'][idx]
            sim_shower[shp.flavor] = fin['flavors'][idx]
            sim_shower[shp.interaction_type] = fin['interaction_type'][idx]
            sim_shower[shp.n_interaction] = fin['n_interaction'][idx]
            sim_shower[shp.vertex] = np.array([fin['xx'][idx], fin['yy'][idx], fin['zz'][idx]])
            sim_shower[shp.vertex_time] = vertex_time
            sim_shower[shp.type] = fin['shower_type'][idx]
            # TODO direct parent does not necessarily need to be the primary in general, but full
            # interaction chain is currently not populated in the input files.
            sim_shower[shp.parent_id] = event_group_id
            logger.debug(f"Adding shower {sim_shower.get_id()} to event group {event_group.get_id()}")
            event_group.add_sim_shower(sim_shower)

    else:  # emitter mode: simulation of one or several artificial emitters
        for idx in idxs:
            emitter_obj = NuRadioReco.framework.sim_emitter.SimEmitter(fin['shower_ids'][idx])  # shower_id is equivalent to emitter_id in this case
            emitter_obj[ep.position] = np.array([fin['xx'][idx], fin['yy'][idx], fin['zz'][idx]])
            emitter_obj[ep.model] = fin['emitter_model'][idx]
            emitter_obj[ep.amplitude] = fin['emitter_amplitudes'][idx]
            for key in ep:
                if not emitter_obj.has_parameter(key):
                    if 'emitter_' + key.name in fin:
                        emitter_obj[key] = fin['emitter_' + key.name][idx]
            event_group.add_sim_emitter(emitter_obj)
    return event_group

def get_config(config_file):
    """
    Read the configuration file and return the configuration dictionary.

    The configuration dictionary is a combination of the default configuration
    and the local configuration file. The local configuration file can override
    the default configuration.

    Parameters
    ----------
    config_file : string
        the path to the configuration file

    Returns
    -------
    dict
        the configuration dictionary
    """
    config_file_default = os.path.join(os.path.dirname(__file__), 'config_default.yaml')
    logger.status('reading default config from %s', config_file_default)
    with open(config_file_default, 'r', encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    if config_file is not None:
        logger.status('reading local config overrides from %s', config_file)
        with open(config_file, 'r', encoding="utf-8") as ymlfile:
            local_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            new_cfg = merge_config(local_config, cfg)
            cfg = new_cfg
    return cfg

def calculate_polarization_vector(shower_axis, launch_vector, config):
    """ calculates the polarization vector in spherical coordinates (eR, eTheta, ePhi)

    Parameters
    ----------
    shower_axis: array-like
        shower axis in cartesian coordinates
    launch_vector: array-like
        launch vector in cartesian coordinates
    config: dict
        configuration dictionary

    Returns
    -------
    array-like
        polarization vector in spherical coordinates (eR, eTheta, ePhi)
    """
    if config['signal']['polarization'] == 'auto':
        polarization_direction = np.cross(launch_vector, np.cross(shower_axis, launch_vector))
        polarization_direction /= np.linalg.norm(polarization_direction)
        cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*launch_vector))
        return cs.transform_from_ground_to_onsky(polarization_direction)

    if config['signal']['polarization'] == 'custom':
        ePhi = float(config['signal']['ePhi'])
        eTheta = (1 - ePhi ** 2) ** 0.5
        v = np.array([0, eTheta, ePhi])
        return v / np.linalg.norm(v)

    msg = f"{config['signal']['polarization']} for config.signal.polarization is not a valid option"
    logger.error(msg)
    raise ValueError(msg)

def increase_signal(station, channel_id, factor):
    """
    increase the signal of a simulated station by a factor of x
    this is e.g. used to approximate a phased array concept with a single antenna

    Parameters
    ----------
    channel_id: int or None
        if None, all available channels will be modified
    """
    if channel_id is None:
        for electric_field in station.get_sim_station().get_electric_fields():
            electric_field.set_trace(electric_field.get_trace() * factor, sampling_rate=electric_field.get_sampling_rate())

    else:
        sim_channels = station.get_sim_station().get_electric_fields_for_channels([channel_id])
        for sim_channel in sim_channels:
            sim_channel.set_trace(sim_channel.get_trace() * factor, sampling_rate=sim_channel.get_sampling_rate())

def calculate_particle_weight(event_group, idx, cfg, fin):
    """
    Calculate the (survival) propability of a neutrino reaching the simulation volume.

    Depending on the config settings, the weights is used from the input file, or
    calulated based on the Earth attenuation and cross section model defined in the config

    Parameters
    ----------
    event_group : EventGroup
        The event group containing the particle.
    idx : int
        The index of the particle in the event group.
    cfg : dict
        The configuration parameters.
    fin : dictionary
        the data part of the hdf5 input file

    Returns
    -------
    float
        The weight of the particle.

    Notes
    -----
    This function calculates the weight of a particle based on the given event group, particle index, and configuration parameters.
    The weight is determined based on the weight mode specified in the configuration parameters.
    If the weight mode is set to "existing", the function checks if the input file contains weights and assigns the weight to the particle.
    If the weight mode is set to None, the weight is set to 1.
    Otherwise, the weight is calculated based on the particle's zenith, energy, flavor, weight mode, cross section type, vertex position, and azimuth.

    """
    primary = event_group.get_primary()
    if cfg['weights']['weight_mode'] == "existing":
        if "weights" in fin:
            primary[simp.weight] = fin["weights"][idx]
        else:
            logger.error(
                "config file specifies to use weights from the input hdf5 file but the input file does not contain this information.")
    elif cfg['weights']['weight_mode'] is None:
        primary[simp.weight][simp.weight] = 1.
    else:
        primary[simp.weight] = get_weight(primary[simp.zenith],
                                          primary[simp.energy],
                                          primary[simp.flavor],
                                          mode=cfg['weights']['weight_mode'],
                                          cross_section_type=cfg['weights']['cross_section_type'],
                                          vertex_position=primary[simp.vertex],
                                          phi_nu=primary[simp.azimuth])
    # all entries for the event for this primary get the calculated primary's weight
    return primary[simp.weight]


def group_into_events(station, event_group, particle_mode, split_event_time_diff,
                      zerosignal=False):
    """
    Group the signals from a station into multiple events based on signal arrival times.

    Parameters:
    -----------
    station : NuRadioReco.framework.station.Station
        The station object containing the signals.
    event_group : NuRadioMC.framework.event.Event
        The event group object containing all showers or emitters and other meta attributes
    particle_mode : bool
        Flag indicating whether the events are a particle simulations or not (i.e. an emitter simulation)
    split_event_time_diff : float
        The time difference threshold for splitting events.
    zerosignal : bool, optional
        Flag indicating whether to zero out the signals. Default is False.

    Returns:
    --------
    events : list of NuRadioReco.framework.event.Event
        The list of events created from the grouped signals.
    """
    event_group_id = event_group.get_run_number()
    start_times = []
    channel_identifiers = []
    for channel in station.get_sim_station().iter_channels():
        channel_identifiers.append(channel.get_unique_identifier())
        start_times.append(channel.get_trace_start_time())
    start_times = np.array(start_times)
    start_times_sort = np.argsort(start_times)
    delta_start_times = start_times[start_times_sort][1:] - start_times[start_times_sort][:-1]  # this array is sorted in time
    split_event_time_diff = float(split_event_time_diff)
    iSplit = np.atleast_1d(np.squeeze(np.argwhere(delta_start_times > split_event_time_diff)))
    n_sub_events = len(iSplit) + 1
    if n_sub_events > 1:
        logger.info(f"splitting event group id {event_group_id} into {n_sub_events} sub events because time separation larger than {split_event_time_diff/units.ns}ns")

    tmp_station = copy.deepcopy(station)
    event_group_has_triggered = False
    events = []
    for iEvent in range(n_sub_events):
        iStart = 0
        iStop = len(channel_identifiers)
        if n_sub_events > 1:
            if iEvent > 0:
                iStart = iSplit[iEvent - 1] + 1
        if iEvent < n_sub_events - 1:
            iStop = iSplit[iEvent] + 1
        indices = start_times_sort[iStart: iStop]
        if n_sub_events > 1:
            tmp = ""
            for start_time in start_times[indices]:
                tmp += f"{start_time/units.ns:.0f}, "
            tmp = tmp[:-2] + " ns"
            logger.info(f"creating event {iEvent} of event group {event_group_id} ranging rom {iStart} to {iStop} with indices {indices} corresponding to signal times of {tmp}")
        evt = NuRadioReco.framework.event.Event(event_group_id, iEvent)  # create new event
        if particle_mode:
            # add MC particles that belong to this (sub) event to event structure
            # add only primary for now, since full interaction chain is not typically in the input hdf5s
            evt.add_particle(event_group.get_primary())  # add primary particle to event
        # copy over generator information from temporary event to event
        evt._generator_info = event_group._generator_info

        station = NuRadioReco.framework.station.Station(tmp_station.get_id())
        sim_station = NuRadioReco.framework.sim_station.SimStation(tmp_station.get_id())
        sim_station.set_is_neutrino()
        tmp_sim_station = tmp_station.get_sim_station()
        shower_ids_of_sub_event = []
        for iCh in indices:
            ch_uid = channel_identifiers[iCh]
            shower_id = ch_uid[1]
            if shower_id not in shower_ids_of_sub_event:
                shower_ids_of_sub_event.append(shower_id)
            sim_station.add_channel(tmp_sim_station.get_channel(ch_uid))
            efield_uid = ([ch_uid[0]], ch_uid[1], ch_uid[2])  # the efield unique identifier has as first parameter an array of the channels it is valid for
            for efield in tmp_sim_station.get_electric_fields():
                if efield.get_unique_identifier() == efield_uid:
                    sim_station.add_electric_field(efield)
                    logger.debug(f"adding sim efield {efield_uid} to sim station")

        if particle_mode:
            # add showers that contribute to this (sub) event to event structure
            for shower_id in shower_ids_of_sub_event:
                evt.add_sim_shower(event_group.get_sim_shower(shower_id))
        else:
            for shower_id in shower_ids_of_sub_event:
                evt.add_sim_emitter(event_group.get_sim_emitter(shower_id))
        station.set_sim_station(sim_station)
        station.set_station_time(event_group.get_event_time()) 
        evt.set_station(station)
        if bool(zerosignal):
            increase_signal(station, None, 0)
        events.append(evt)
    logger.status(f"created {len(events)} events from event group {event_group_id}")
    return events

def read_input_hdf5(filename):
    """
    Reads input file into memory.

    Parameters
    ----------
    filename : str
        The path to the input file.

    Returns
    -------
    fin : dict
        A dictionary containing the data from the input file.
    fin_stations : dict
        A nested dictionary containing the station data from the input file.
    fin_attrs : dict
        A dictionary containing the attributes from the input file.

    """
    fin_hdf5 = h5py.File(filename, 'r')
    fin = {}
    fin_stations = {}
    fin_attrs = {}
    for key, value in fin_hdf5.items():
        if isinstance(value, h5py._hl.group.Group):
            fin_stations[key] = {}
            for key2, value2 in value.items():
                fin_stations[key][key2] = np.array(value2)
        else:
            if len(value) and type(value[0]) == bytes:
                fin[key] = np.array(value).astype('U')
            else:
                fin[key] = np.array(value)
    for key, value in fin_hdf5.attrs.items():
        fin_attrs[key] = value

    fin_hdf5.close()
    return fin, fin_stations, fin_attrs

def remove_all_traces(evt):
    """
    Remove all traces from the event.

    Parameters
    ----------
    evt : NuRadioReco.framework.event.Event
    """
    for station in evt.get_stations():
        sim_station = station.get_sim_station()
        for electric_field in sim_station.get_electric_fields():
            electric_field._time_trace = None
            electric_field._frequency_spectrum = None
        for sim_channel in sim_station.iter_channels():
            sim_channel._time_trace = None
            sim_channel._frequency_spectrum = None
        for electric_field in station.get_electric_fields():
            electric_field._time_trace = None
            electric_field._frequency_spectrum = None
        for channel in station.iter_channels():
            channel._time_trace = None
            channel._frequency_spectrum = None


class simulation:

    def __init__(self, inputfilename,
                 outputfilename,
                 detectorfile=None,
                 det=None,
                 det_kwargs={},
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1),
                 config_file=None,
                 log_level=LOGGING_STATUS,
                 default_detector_station=None,
                 default_detector_channel=None,
                 file_overwrite=False,
                 write_detector=True,
                 event_list=None,
                 log_level_propagation=logging.WARNING,
                 ice_model=None,
                 **kwargs):
        """
        initialize the NuRadioMC end-to-end simulation

        Parameters
        ----------
        inputfilename: string, or pair
            the path to the hdf5 file containing the list of neutrino events
            alternatively, the data and attributes dictionary can be passed directly to the method
        outputfilename: string
            specify hdf5 output filename.
        detectorfile: string
            path to the json file containing the detector description
        det: detector object
            Pass a detector class object
        det_kwargs: dict
            Pass arguments to the detector (only used when det == None)
        station_id: int
            the station id for which the simulation is performed. Must match a station
            defined in the detector description
        outputfilenameNuRadioReco: string or None
            outputfilename of NuRadioReco detector sim file, this file contains all
            waveforms of the triggered events
            default: None, i.e., no output file will be written which is useful for
            effective volume calculations
        debug: bool
            True activates debug mode, default False
        evt_time: datetime object
            the time of the events, default 1/1/2018
        config_file: string
            path to config file
        log_level: logging.LEVEL
            the log level
        default_detector_station: int or None
            DEPRECATED: Define reference stations in the detector JSON file instead
        default_detector_channel: int or None
            DEPRECATED: Define reference channels in the detector JSON file instead
        file_overwrite: bool
            True allows overwriting of existing files, default False
        write_detector: bool
            If true, the detector description is written into the .nur files along with the events
            default True
        event_list: None or list of ints
            if provided, only the event listed in this list are being simulated
        log_level_propagation: logging.LEVEL
            the log level of the propagation module
        ice_model: medium object (default None)
            allows to specify a custom ice model. This model is used if the config file specifies the ice model as "custom".
        """
        logger.setLevel(log_level)
        if 'write_mode' in kwargs:
            logger.warning('Parameter write_mode is deprecated. Define the output format in the config file instead.')

        self._log_level = log_level
        self._log_level_ray_propagation = log_level_propagation
        self._config = get_config(config_file)
        if self._config['seed'] is None:
            # the config seeting None means a random seed. To have the simulation be reproducable, we generate a new
            # random seed once and save this seed to the config setting. If the simulation is rerun, we can get
            # the same random sequence.
            self._config['seed'] = np.random.randint(0, 2 ** 32 - 1)

        self._rnd = Generator(Philox(self._config['seed']))

        self._outputfilename = outputfilename
        if os.path.exists(self._outputfilename):
            msg = f"hdf5 output file {self._outputfilename} already exists"
            if file_overwrite is False:
                logger.error(msg)
                raise FileExistsError(msg)
            logger.warning(msg)

        self._outputfilenameNuRadioReco = outputfilenameNuRadioReco
        self._debug = debug  # TODO remove
        self._evt_time = evt_time  # TODO: fill event time properly to station and event objects
        self.__write_detector = write_detector
        logger.status("setting event time to {}".format(evt_time))
        self._event_group_list = event_list

        self._antenna_pattern_provider = antennapattern.AntennaPatternProvider()


        if self._config['propagation']['ice_model'] == "custom":
            if ice_model is None:
                logger.error("ice model is set to 'custom' in config file but no custom ice model is provided.")
                raise AttributeError("ice model is set to 'custom' in config file but no custom ice model is provided.")
            self._ice = ice_model
        else:
            self._ice = NuRadioMC.utilities.medium.get_ice_model(self._config['propagation']['ice_model'])

        # Initialize detector
        if det is None:
            logger.status("Detectorfile {}".format(os.path.abspath(detectorfile)))
            kwargs = dict(json_filename=detectorfile, default_station=default_detector_station,
                              default_channel=default_detector_channel, antenna_by_depth=False)
            kwargs.update(det_kwargs)
            self._det = detector.Detector(**kwargs)
        else:
            self._det = det

        self._det.update(self._evt_time)

        # initialize propagation module
        prop = propagation.get_propagation_module(self._config['propagation']['module'])
        self._propagator = prop(
            self._ice, self._config['propagation']['attenuation_model'],
            log_level=self._log_level_ray_propagation,
            n_frequencies_integration=int(self._config['propagation']['n_freq']),
            n_reflections=int(self._config['propagation']['n_reflections']),
            config=self._config,
            detector=self._det
        )

        self._station_ids = self._det.get_station_ids()
        self._event_ids_counter = {}
        for station_id in self._station_ids:
            self._event_ids_counter[station_id] = -1  # we initialize with -1 becaue we increment the counter before we use it the first time

        # print noise information
        logger.status("running with noise {}".format(bool(self._config['noise'])))
        logger.status("setting signal to zero {}".format(bool(self._config['signal']['zerosignal'])))
        if bool(self._config['propagation']['focusing']):
            logger.status("simulating signal amplification due to focusing of ray paths in the firn.")

        # read sampling rate from config (this sampling rate will be used internally)
        self._dt = 1. / (self._config['sampling_rate'] * units.GHz)  # TODO: Maybe remove


        #### Input HDF5 block   TODO: Add option to start from nur files
        if isinstance(inputfilename, str):
            logger.status(f"reading input from {inputfilename}")
            # we read in the full input file into memory at the beginning to limit io to the beginning and end of the run
            self._fin, self._fin_stations, self._fin_attrs = read_input_hdf5(inputfilename)
        else:
            logger.status("getting input on-the-fly")
            self._inputfilename = "on-the-fly"
            self._fin = inputfilename[0]
            self._fin_attrs = inputfilename[1]
            self._fin_stations = {}

        # check if the input file contains events, if not save empty output file (for book keeping) and terminate simulation
        if len(self._fin['xx']) == 0:
            logger.status(f"input file {self._inputfilename} is empty")
            return
        ### END input HDF5 block

        self._output_writer_hdf5 = outputWriterHDF5(self._outputfilename, self._config, self._det, self._station_ids,
                                                    self._propagator.get_number_of_raytracing_solutions())

        # Perfom a dummy detector simulation to determine how the signals are filtered.
        # This variable stores the integrated channel response for each channel, i.e.
        # the integral of the squared signal chain response over all frequencies, int S21^2 df.
        # For a system without amplification, it is equivalent to the bandwidth of the system.
        self._integrated_channel_response = {}

        self._integrated_channel_response_normalization = {}
        self._max_amplification_per_channel = {}

        # first create dummy event and station with channels, run the signal chain, 
        # and determine the integrated channel response (to be able to normalize the noise level)
        for sid in self._station_ids:

            evt = build_dummy_event(sid, self._det, self._config)
            # TODO: It seems to be sufficient to just apply the `_detector_simulation_filter_amp` function to a dummy event
            apply_det_response(evt, self._det, self._config, self._detector_simulation_filter_amp,
                               add_noise=False)

            self._integrated_channel_response[sid] = {}
            self._integrated_channel_response_normalization[sid] = {}
            self._max_amplification_per_channel[sid] = {}

            for channel_id in range(self._det.get_number_of_channels(sid)):
                ff = np.linspace(0, 0.5 / self._dt, 10000)
                filt = np.ones_like(ff, dtype=complex)
                for i, (name, instance, kwargs) in enumerate(evt.iter_modules(sid)):
                    if hasattr(instance, "get_filter"):
                        filt *= instance.get_filter(ff, sid, channel_id, self._det, **kwargs)

                self._max_amplification_per_channel[sid][channel_id] = np.abs(filt).max()

                mean_integrated_response = np.mean(
                    np.abs(filt)[np.abs(filt) > np.abs(filt).max() / 100] ** 2)  # a factor of 100 corresponds to -40 dB in amplitude
                self._integrated_channel_response_normalization[sid][channel_id] = mean_integrated_response

                integrated_channel_response = np.trapz(np.abs(filt) ** 2, ff)
                self._integrated_channel_response[sid][channel_id] = integrated_channel_response

                logger.debug(f"Station.channel {sid}.{channel_id} estimated bandwidth is "
                             f"{integrated_channel_response / mean_integrated_response / units.MHz:.1f} MHz")
        ################################

        self._bandwidth = next(iter(next(iter(self._integrated_channel_response.values())).values()))  # get value of first station/channel key pair

        noise_temp = self._config['trigger']['noise_temperature']
        Vrms = self._config['trigger']['Vrms']

        if noise_temp is not None and Vrms is not None:
            raise AttributeError(f"Specifying noise temperature (set to {noise_temp}) and Vrms (set to {Vrms}) is not allowed).")

        self._Vrms_per_channel = collections.defaultdict(dict)
        self._Vrms_efield_per_channel = collections.defaultdict(dict)

        if noise_temp is not None:
            if noise_temp == "detector":
                logger.status("Use noise temperature from detector description to determine noise Vrms in each channel.")
                self._noise_temp = None  # the noise temperature is defined in the detector description
            else:
                self._noise_temp = float(noise_temp)
                logger.status(f"Use a noise temperature of {noise_temp / units.kelvin:.1f} K for each channel to determine noise Vrms.")

            self._noiseless_channels = collections.defaultdict(list)
            for station_id in self._integrated_channel_response:
                for channel_id in self._integrated_channel_response[station_id]:
                    if self._noise_temp is None:
                        noise_temp_channel = self._det.get_noise_temperature(station_id, channel_id)
                    else:
                        noise_temp_channel = self._noise_temp

                    if self._det.is_channel_noiseless(station_id, channel_id):
                        self._noiseless_channels[station_id].append(channel_id)

                    # Calculation of Vrms. For details see from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
                    # (last two Eqs. in "noise voltage and power" section) or our wiki https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

                    # Bandwidth, i.e., \Delta f in equation
                    integrated_channel_response = self._integrated_channel_response[station_id][channel_id]
                    max_amplification = self._max_amplification_per_channel[station_id][channel_id]

                    self._Vrms_per_channel[station_id][channel_id] = (noise_temp_channel * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
                    self._Vrms_efield_per_channel[station_id][channel_id] = self._Vrms_per_channel[station_id][channel_id] / max_amplification / units.m  # VEL = 1m

                    # for logging
                    mean_integrated_response = self._integrated_channel_response_normalization[station_id][channel_id]

                    logger.status(f'Station.channel {station_id}.{channel_id:02d}: noise temperature = {noise_temp_channel} K, '
                                  f'est. bandwidth = {integrated_channel_response / mean_integrated_response / units.MHz:.2f} MHz, '
                                  f'max. filter amplification = {max_amplification:.2e} -> Vrms = '
                                  f'integrated response = {integrated_channel_response / units.MHz:.2e}MHz -> Vrms = '
                                  f'{self._Vrms_per_channel[station_id][channel_id] / units.mV:.2f} mV -> efield Vrms = {self._Vrms_efield_per_channel[station_id][channel_id] / units.V / units.m / units.micro:.2f}muV/m (assuming VEL = 1m) ')

            self._Vrms = next(iter(next(iter(self._Vrms_per_channel.values())).values()))

        elif Vrms is not None:
            self._Vrms = float(Vrms) * units.V
            self._noise_temp = None
            logger.status(f"Use a fix noise Vrms of {self._Vrms / units.mV:.2f} mV in each channel.")

            for station_id in self._integrated_channel_response:
                for channel_id in self._integrated_channel_response[station_id]:
                    max_amplification = self._max_amplification_per_channel[station_id][channel_id]
                    self._Vrms_per_channel[station_id][channel_id] = self._Vrms  # to be stored in the hdf5 file
                    self._Vrms_efield_per_channel[station_id][channel_id] = self._Vrms / max_amplification / units.m  # VEL = 1m

                    # for logging
                    integrated_channel_response = self._integrated_channel_response[station_id][channel_id]
                    mean_integrated_response = self._integrated_channel_response_normalization[station_id][channel_id]

                    logger.status(f'Station.channel {station_id}.{channel_id:02d}: '
                                  f'est. bandwidth = {integrated_channel_response / mean_integrated_response / units.MHz:.2f} MHz, '
                                  f'max. filter amplification = {max_amplification:.2e} '
                                  f'integrated response = {integrated_channel_response / units.MHz:.2e}MHz ->'
                                  f'efield Vrms = {self._Vrms_efield_per_channel[station_id][channel_id] / units.V / units.m / units.micro:.2f}muV/m (assuming VEL = 1m) ')

        else:
            raise AttributeError("noise temperature and Vrms are both set to None")

        self._Vrms_efield = next(iter(next(iter(self._Vrms_efield_per_channel.values())).values()))
        speed_cut = float(self._config['speedup']['min_efield_amplitude'])
        logger.status(f"All signals with less then {speed_cut:.1f} x Vrms_efield will be skipped.")

        # define function for distance speedup cut
        self._distance_cut_polynomial = None
        if self._config['speedup']['distance_cut']:
            coef = self._config['speedup']['distance_cut_coefficients']
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(coef)

            def get_distance_cut(shower_energy):
                if shower_energy <= 0:
                    return 100 * units.m
                return max(100 * units.m, 10 ** self.__distance_cut_polynomial(np.log10(shower_energy)))

            self._get_distance_cut = get_distance_cut

    def run(self):
        """
        run the NuRadioMC simulation
        """
        if len(self._fin['xx']) == 0:
            logger.status(f"writing empty hdf5 output file")
            self._write_output_file(empty=True)
            logger.status(f"terminating simulation")
            return 0
        logger.status(f"Starting NuRadioMC simulation")
        t_start = time.time()
        t_last_update = t_start

        efieldToVoltageConverter.begin(time_resolution=self._config['speedup']['time_res_efieldconverter'])
        channelGenericNoiseAdder.begin(seed=self._config['seed'])
        if self._outputfilenameNuRadioReco is not None:
            eventWriter.begin(self._outputfilenameNuRadioReco, log_level=self._log_level)
        unique_event_group_ids = np.unique(self._fin['event_group_ids'])
        self._n_showers = len(self._fin['event_group_ids'])
        self._shower_ids = np.array(self._fin['shower_ids'])
        self._shower_index_array = {}  # this array allows to convert the shower id to an index that starts from 0 to be used to access the arrays in the hdf5 file.

        input_time = 0.0
        askaryan_time = 0.0
        rayTracingTime = 0.0
        detSimTime = 0.0
        outputTime = 0.0
        weightTime = 0.0
        distance_cut_time = 0.0

        n_shower_station = len(self._station_ids) * self._n_showers
        iCounter = 0

        # calculate bary centers of station
        self._station_barycenter = np.zeros((len(self._station_ids), 3))
        for iSt, station_id in enumerate(self._station_ids):
            pos = []
            for channel_id in range(self._det.get_number_of_channels(station_id)):
                pos.append(self._det.get_relative_position(station_id, channel_id))
            self._station_barycenter[iSt] = np.mean(np.array(pos), axis=0) + self._det.get_absolute_position(station_id)

        # loop over event groups
        for i_event_group_id, event_group_id in enumerate(unique_event_group_ids):
            logger.debug(f"simulating event group id {event_group_id}")
            if self._event_group_list is not None and event_group_id not in self._event_group_list:
                logger.debug(f"skipping event group {event_group_id} because it is not in the event group list provided to the __init__ function")
                continue
            event_indices = np.atleast_1d(np.squeeze(np.argwhere(self._fin['event_group_ids'] == event_group_id)))

            # the weight calculation is independent of the station, so we do this calculation only once
            # the weight also depends just on the "mother" particle, i.e. the incident neutrino which determines
            # the propability of arriving at our simulation volume. All subsequent showers have the same weight. So
            # we calculate it just once and save it to all subshowers.
            t1 = time.time()

            event_group = build_NuRadioEvents_from_hdf5(self._fin, self._fin_attrs, event_indices)
            event_group.set_event_time(self._evt_time)

            # determine if a particle (neutrinos, or a secondary interaction of a neutrino, or surfaec muons) is simulated
            particle_mode = "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] != "emitter"
            weight = 1
            if particle_mode:
                weight = calculate_particle_weight(event_group, event_indices[0], self._config, self._fin)
            weightTime += time.time() - t1
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if weight < self._config['speedup']['minimum_weight_cut']:
                logger.debug("neutrino weight is smaller than {}, skipping event".format(self._config['speedup']['minimum_weight_cut']))
                continue

            output_buffer = {}
            # loop over all stations (each station is treated independently)
            for iSt, sid in enumerate(self._station_ids):
                output_buffer[sid] = {}
                station = NuRadioReco.framework.station.Station(sid)
                sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
                sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)
                station.set_sim_station(sim_station)
                event_group.set_station(station)

                # loop over all trigger channels
                for iCh, channel_id in enumerate(self._det.get_channel_ids(sid)):
                    if particle_mode:
                        sim_station = calculate_sim_efield(showers=event_group.get_sim_showers(),
                                                        sid=sid, cid=channel_id,
                                                        det=self._det, propagator=self._propagator, medium=self._ice,
                                                        config=self._config,
                                                        time_logger=None)
                    else:
                        sim_station = calculate_sim_efield_for_emitter(emitters=event_group.get_sim_emitters(),
                                            sid=sid, cid=channel_id,
                                            det=self._det, propagator=self._propagator, medium=self._ice, config=self._config,
                                            rnd=self._rnd, antenna_pattern_provider=self._antenna_pattern_provider,
                                            time_logger=None)
                    # skip to next channel if the efield is below the speed cut
                    if len(sim_station.get_electric_fields()) == 0:
                        logger.status(f"Eventgroup {event_group.get_run_number()} Station {sid} channel {channel_id:02d} has {len(sim_station.get_electric_fields())} efields, skipping to next channel")
                        continue

                    # applies the detector response to the electric fields (the antennas are defined
                    # in the json detector description file)
                    apply_det_response_sim(sim_station, self._det, self._config, self._detector_simulation_filter_amp,
                                           event_time=self._evt_time)
                    logger.debug(f"adding sim_station to station {sid} for event group {event_group.get_run_number()}, channel {channel_id}")
                    station.add_sim_station(sim_station)  # this will add the channels and efields to the existing sim_station object
                # end channel loop, skip to next event group if all signals are empty (due to speedup cuts)
                sim_station = station.get_sim_station()  # needed to get sim_station object containing all channels and not just the last one.
                if len(sim_station.get_electric_fields()) == 0:
                    logger.status(f"Eventgroup {event_group.get_run_number()} Station {sid} has {len(sim_station.get_electric_fields())} efields, skipping to next station")
                    continue

                # group events into events based on signal arrival times
                events = group_into_events(station, event_group, particle_mode, self._config['split_event_time_diff'])

                evt_group_triggered = False
                for evt in events:
                    station = evt.get_station()
                    apply_det_response(evt, self._det, self._config, self._detector_simulation_filter_amp,
                                       bool(self._config['noise']),
                                       self._Vrms_efield_per_channel, self._integrated_channel_response,
                                       self._noiseless_channels)  # TODO: Add option to pass fully custon detector simulation

                    # calculate trigger
                    self._detector_simulation_trigger(evt, station, self._det)
                    logger.debug(f"event {evt.get_run_number()},{evt.get_id()} tiggered: {evt.get_station().has_triggered()}")

                    if not evt.get_station().has_triggered():
                        continue
                    evt_group_triggered = True
                    channelSignalReconstructor.run(evt, station, self._det)

                    if self._outputfilenameNuRadioReco is not None:
                        # downsample traces to detector sampling rate to save file size
                        sampling_rate_detector = self._det.get_sampling_frequency(sid, self._det.get_channel_ids(sid)[0])
                        channelResampler.run(evt, station, self._det, sampling_rate=sampling_rate_detector)
                        channelResampler.run(evt, station.get_sim_station(), self._det, sampling_rate=sampling_rate_detector)
                        electricFieldResampler.run(evt, station.get_sim_station(), self._det, sampling_rate=sampling_rate_detector)

                        output_mode = {'Channels': self._config['output']['channel_traces'],
                                       'ElectricFields': self._config['output']['electric_field_traces'],
                                       'SimChannels': self._config['output']['sim_channel_traces'],
                                       'SimElectricFields': self._config['output']['sim_electric_field_traces']}
                        if self.__write_detector:
                            eventWriter.run(evt, self._det, mode=output_mode)
                        else:
                            eventWriter.run(evt, mode=output_mode)
                    remove_all_traces(evt)  # remove all traces to save memory
                    output_buffer[sid][evt.get_id()] = evt
                if(evt_group_triggered):
                    # TODO: Write hdf5 output per event group
                    self._output_writer_hdf5.add_event_group(output_buffer)

        if self._outputfilenameNuRadioReco is not None:
            eventWriter.end()
            logger.debug("closing nur file")

        self._output_writer_hdf5.end()



    def _is_in_fiducial_volume(self, pos):
        """ Checks if pos is in fiducial volume """

        for check_attr in ['fiducial_zmin', 'fiducial_zmax']:
            if not check_attr in self._fin_attrs:
                logger.warning("Fiducial volume not defined. Return True")
                return True

        pos = copy.deepcopy(pos) - np.array([self._fin_attrs.get("x0", 0), self._fin_attrs.get("y0", 0), 0])

        if not (self._fin_attrs["fiducial_zmin"] < pos[2] < self._fin_attrs["fiducial_zmax"]):
            return False

        if "fiducial_rmax" in self._fin_attrs:
            radius = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
            return self._fin_attrs["fiducial_rmin"] < radius < self._fin_attrs["fiducial_rmax"]
        elif "fiducial_xmax" in self._fin_attrs:
            return (self._fin_attrs["fiducial_xmin"] < pos[0] < self._fin_attrs["fiducial_xmax"] and
                    self._fin_attrs["fiducial_ymin"] < pos[1] < self._fin_attrs["fiducial_ymax"])
        else:
            raise ValueError("Could not contruct fiducial volume from input file.")


    def _check_vertex_times(self):

        if 'vertex_times' in self._fin:
            return True
        else:
            warn_msg = 'The input file does not include vertex times. '
            warn_msg += 'Vertices from the same event will not be time-ordered.'
            logger.warning(warn_msg)
            return False

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

        if 'detector' in self._fin_attrs:
            if isinstance(self._det, detector.rnog_detector.Detector):
                if self._det.export_as_string() == self._fin_attrs['detector']:
                    self._was_pre_simulated = True
            else:
                with open(self._detectorfile, 'r') as fdet:
                    if fdet.read() == self._fin_attrs['detector']:
                        self._was_pre_simulated = True

        if self._was_pre_simulated:
            logger.status("the simulation was already performed with the same detector")

        return self._was_pre_simulated

 

    def calculate_Veff(self):
        # calculate effective
        triggered = remove_duplicate_triggers(self._mout['triggered'], self._fin['event_group_ids'])
        n_triggered = np.sum(triggered)
        n_triggered_weighted = np.sum(self._mout['weights'][triggered])
        n_events = self._fin_attrs['n_events']
        logger.status(f'fraction of triggered events = {n_triggered:.0f}/{n_events:.0f} = {n_triggered / self._n_showers:.3f} (sum of weights = {n_triggered_weighted:.2f})')

        V = self._fin_attrs['volume']
        Veff = V * n_triggered_weighted / n_events
        logger.status(f"Veff = {Veff / units.km ** 3:.4g} km^3, Veffsr = {Veff * 4 * np.pi/units.km**3:.4g} km^3 sr")

    

    @property
    def _bandwidth_per_channel(self):
        warnings.warn("This variable `_bandwidth_per_channel` is deprecated. "
                      "Please use `integrated_channel_response` instead.", DeprecationWarning)
        return self._integrated_channel_response

    @_bandwidth_per_channel.setter
    def _bandwidth_per_channel(self, value):
        warnings.warn("This variable `_bandwidth_per_channel` is deprecated. "
                        "Please use `integrated_channel_response` instead.", DeprecationWarning)
        self._integrated_channel_response = value

    @property
    def integrated_channel_response(self):
        return self._integrated_channel_response

    @integrated_channel_response.setter
    def integrated_channel_response(self, value):
        self._integrated_channel_response = value
