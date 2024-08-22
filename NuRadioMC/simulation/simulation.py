import os
import collections
import datetime
import logging
import warnings
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
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.detector import detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.particle
import NuRadioReco.framework.event
import NuRadioReco.framework.sim_emitter
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import channelParameters as chp
# parameters describing simulated Monte Carlo particles
from NuRadioReco.framework.parameters import particleParameters as simp
from NuRadioReco.framework.parameters import emitterParameters as ep
# parameters set in the event generator
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
import NuRadioMC.simulation.time_logger

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
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()

def merge_config(user, default):
    """
    Merge the user configuration dictionary with the default configuration dictionary recursively.

    Parameters
    ----------
    user : dict
        The user configuration dictionary.
    default : dict
        The default configuration dictionary.

    Returns
    -------
    dict
        The merged configuration dictionary.

    """
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k], v)
    return user


def calculate_sim_efield(showers, sid, cid,
                         det, propagator, medium, config,
                         time_logger=None,
                         min_efield_amplitude=None,
                         distance_cut=None):
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
    time_logger: time_logger object
        the time logger to be used for the simulation
    min_efield_amplitude: float
        speedup cut: the minimum electric field amplitude, if all efields from all showers are belwo this threshold value
        the station will not be set as candidate station.

    Returns
    -------
    list of SimEfield objects
        A list of SimEfield objects, one for each shower and propagation solution

    """
    if distance_cut is not None:
        time_logger.start_time('distance cut')
        vertex_positions = []
        shower_energies = []
        for i, shower in enumerate(showers):
            vertex_positions.append(shower.get_parameter(shp.vertex))
            shower_energies.append(shower.get_parameter(shp.energy))
        vertex_positions = np.array(vertex_positions)
        shower_energies = np.array(shower_energies)
        vertex_distances = np.linalg.norm(vertex_positions - vertex_positions[0], axis=1)
        time_logger.stop_time('distance cut')
    logger.debug("Calculating electric field for station %d , channel %d from list of showers", sid, cid)
    p = propagator # shorthand for more compact coding

    sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
    sim_station.set_candidate(False)
    if min_efield_amplitude is None:
        sim_station.set_candidate(True)
    sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)

    x2 = det.get_relative_position(sid, cid) + det.get_absolute_position(sid)
    dt = 1. / config['sampling_rate']
    # rescale the number of samples to the internal (higher) sampling rate used in the simulation
    n_samples = det.get_number_of_samples(sid, cid) / det.get_sampling_frequency(sid, cid) / dt
    n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer

    for iSh, shower in enumerate(showers):
        x1 = shower.get_parameter(shp.vertex)
        if distance_cut is not None:
            time_logger.start_time('distance cut')
            mask_shower_sum = np.abs(vertex_distances - vertex_distances[iSh]) < config['speedup']['distance_cut_sum_length']
            shower_energy_sum = np.sum(shower_energies[mask_shower_sum])
            if np.linalg.norm(x1 - x2) > distance_cut(shower_energy_sum):
                # logger.warning(f"shower {shower.get_id()} is too far away ({np.linalg.norm(x1 - x2)/units.km:.2f} > {distance_cut(shower_energy_sum)/units.km:.2f}) from station {sid} channel {cid}, skipping shower")
                time_logger.stop_time('distance cut')
                continue
            # logger.warning(f"shower {shower.get_id()} is close enough ({np.linalg.norm(x1 - x2)/units.km:.2f} < {distance_cut(shower_energy_sum)/units.km:.2f}) to station {sid} channel {cid}, continuing with shower")
            time_logger.stop_time('distance cut')
        time_logger.start_time('ray tracing')
        logger.debug(f"Calculating electric field for shower {shower.get_id()} and station {sid}, channel {cid}")
        shower_axis = -1 * shower.get_axis() # We need the propagation direction here, so we multiply the shower axis with '-1'
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
        logger.debug(f"found {n} solutions for shower {shower.get_id()} and station {sid}, channel {cid} from {x1} to {x2}")
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
        time_logger.stop_time('ray tracing')
        for iS in range(n): # loop through all ray tracing solution
            time_logger.start_time('ray tracing (time)')
            # skip individual channels where the viewing angle difference is too large
            # discard event if delta_C (angle off cherenkov cone) is too large
            if np.abs(delta_Cs[iS]) > config['speedup']['delta_C_cut']:
                logger.debug('delta_C too large, ray tracing solution unlikely to be observed, skipping ray tracing solution')
                continue
            # TODO: Fill with previous values if RT was already performed
            R = p.get_path_length(iS)  # calculate path length
            T = p.get_travel_time(iS)  # calculate travel time
            time_logger.start_time('ray tracing (time)')
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

            time_logger.start_time('signal generation')
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
            time_logger.stop_time('signal generation')

            # this is common stuff which is the same between emitters and showers
            electric_field = NuRadioReco.framework.electric_field.ElectricField([cid],
                                    position=det.get_relative_position(sid, cid),
                                    shower_id=shower.get_id(), ray_tracing_id=iS)
            electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / dt)
            time_logger.start_time('propagation effects')
            electric_field = p.apply_propagation_effects(electric_field, iS)
            time_logger.stop_time('propagation effects')
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

            if min_efield_amplitude is not None:
                if np.max(np.abs(electric_field.get_trace())) > min_efield_amplitude:
                    sim_station.set_candidate(True)
                # if np.max(np.abs(electric_field.get_trace())) < min_efield_amplitude:
                #     logger.debug(f"Amplitude to low: electric field NOT added to SimStation for shower {shower.get_id()} and station {sid}, channel {cid} with ray tracing solution {iS} and viewing angle {viewing_angles[iS]/units.deg:.1f}deg")
                #     continue
            sim_station.add_electric_field(electric_field)
            logger.debug(f"Added electric field to SimStation for shower {shower.get_id()} and station {sid}, channel {cid} with ray tracing solution {iS} and viewing angle {viewing_angles[iS]/units.deg:.1f}deg")
    return sim_station

def calculate_sim_efield_for_emitter(emitters, sid, cid,
                         det, propagator, medium, config,
                         rnd, antenna_pattern_provider,
                         time_logger=None,
                         min_efield_amplitude=None):
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
    time_logger: time_logger object
        the time logger to be used for the simulation
    min_efield_amplitude: float
        speedup cut: the minimum electric field amplitude, if all efields from all showers are belwo this threshold value
        the station will not be set as candidate station.

    Returns
    -------
    list of SimEfield objects
        A list of SimEfield objects, one for each shower and propagation solution

    """
    logger.debug(f"Calculating electric field for station {sid}, channel {cid} from list of emitters")
    p = propagator # shorthand for more compact coding

    sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
    sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)
    sim_station.set_candidate(False)
    if min_efield_amplitude is None:
        sim_station.set_candidate(True)

    x2 = det.get_relative_position(sid, cid) + det.get_absolute_position(sid)
    dt = 1. / config['sampling_rate']
    # rescale the number of samples to the internal (higher) sampling rate used in the simulation
    n_samples = det.get_number_of_samples(sid, cid) / det.get_sampling_frequency(sid, cid) / dt
    n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer

    for emitter in emitters:
        time_logger.start_time('ray tracing')
        x1 = emitter.get_parameter(ep.position)
        n_index = medium.get_index_of_refraction(x1)

        p.set_start_and_end_point(x1, x2)
        if config['speedup']['redo_raytracing']:  # check if raytracing was already performed
            pass
            # TODO: initiatlize ray tracer with existing results if available
        p.find_solutions()
        time_logger.stop_time('ray tracing')
        if not p.has_solution():
            logger.debug(f"emitter {emitter.get_id()} and station {sid}, channel {cid} from {x1} to {x2} does not have any ray tracing solution")
            continue

        n = p.get_number_of_solutions()
        for iS in range(n): # loop through all ray tracing solution
            time_logger.start_time('ray tracing (time)')
            # TODO: Fill with previous values if RT was already performed
            R = p.get_path_length(iS)  # calculate path length
            T = p.get_travel_time(iS)  # calculate travel time
            time_logger.stop_time('ray tracing (time)')
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
            for key in ep:
                if key.name not in ['amplitude', 'model', 'position']:
                    if emitter.has_parameter(key):
                        emitter_kwargs[key.name] = emitter[key]

            time_logger.start_time('signal generation')
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
                       emitter[ep.rotation_theta], emitter[ep.rotation_phi]]
                # source voltage given to the emitter
                voltage_spectrum_emitter = emitter_signalgen.get_frequency_spectrum(amplitude, n_samples, dt,
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
            time_logger.stop_time('signal generation')

            # this is common stuff which is the same between emitters and showers. Make sure to do any changes to this code in both places
            electric_field = NuRadioReco.framework.electric_field.ElectricField([cid],
                                    position=det.get_relative_position(sid, cid),
                                    shower_id=emitter.get_id(), ray_tracing_id=iS)
            electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / dt)
            time_logger.start_time('propagation effects')
            electric_field = p.apply_propagation_effects(electric_field, iS)
            time_logger.stop_time('propagation effects')
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
            electric_field[efp.nu_vertex_travel_time] = T
            electric_field[efp.raytracing_solution] = p.get_raytracing_output(iS)
            electric_field[efp.launch_vector] = p.get_launch_vector(iS)

            if min_efield_amplitude is not None:
                if np.max(np.abs(electric_field.get_trace())) > min_efield_amplitude:
                    sim_station.set_candidate(True)

            sim_station.add_electric_field(electric_field)
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
    if time_logger is not None: time_logger.start_time('detector response (sim)')
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
    if time_logger is not None: time_logger.stop_time('detector response (sim)')

def apply_det_response(evt, det, config,
                        detector_simulation_filter_amp=None,
                        add_noise=None,
                        Vrms_per_channel=None,
                        integrated_channel_response=None,
                        noiseless_channels=None,
                        detector_simulation_part2=None,
                        time_logger=None,
                        channel_ids=None):
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
        if not provided, the function `detector_simulation_part2` needs to be provided.
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
    time_logger: time_logger object
        the time logger to be used for the simulation
    channel_ids: list of ints
        the channel ids for which the detector response should be calculated. If None, all channels are used.

    Returns nothing. The Channels are added to the Station object.
    """
    if time_logger is not None: time_logger.start_time('detector response')
    if detector_simulation_filter_amp is None and detector_simulation_part2 is None:
        logger.error("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part2")
        raise ValueError("No detector response function provided. Please provide either detector_simulation_filter_amp or detector_simulation_part2")

    station = evt.get_station()  # will raise an error if there are more than one station, but this should never happen
    # convert efields to voltages at digitizer
    if detector_simulation_part2 is not None:
        detector_simulation_part2(evt, station, det)
    else:
        dt = 1. / (config['sampling_rate'])
        # start detector simulation
        efieldToVoltageConverter.run(evt, station, det, channel_ids=channel_ids)  # convolve efield with antenna pattern
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        channelResampler.run(evt, station, det, sampling_rate=1. / dt)

        if add_noise:
            max_freq = 0.5 / dt
            Vrms = {}
            for channel_id in det.get_channel_ids(station.get_id()):
                norm = integrated_channel_response[station.get_id()][channel_id]
                Vrms[channel_id] = Vrms_per_channel[station.get_id()][channel_id] / (norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(evt, station, det, amplitude=Vrms, min_freq=0 * units.MHz,
                                            max_freq=max_freq, type='rayleigh',
                                            excluded_channels=noiseless_channels[station.get_id()])

        detector_simulation_filter_amp(evt, station, det)

    if time_logger is not None: time_logger.stop_time('detector response')

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
    cid = det.get_channel_ids(sid)[0]
    n_samples = det.get_number_of_samples(sid, cid) / det.get_sampling_frequency(sid, cid) / dt
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


def build_NuRadioEvents_from_hdf5(fin, fin_attrs, idxs, time_logger=None):
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
    event_group : `NuRadioReco.framework.event.Event`
        an event group object containing the showers and particles or emitters
        the output should contain all relevant information from the hdf5 file (except the attributes)
        to perform a NuRadioMC simulation
    """
    if time_logger is not None: time_logger.start_time('event builder (hdf5 -> nur)')
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
        else:
            logger.warning("The input file does not include vertex times, setting vertex time to zero. Vertices from the same event will not be time-ordered.")
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
            if('shower_realization_ARZ' in fin):
                sim_shower[shp.charge_excess_profile_id] = fin['shower_realization_ARZ'][idx]
            if('shower_realization_Alvarez2009' in fin):
                sim_shower[shp.k_L] = fin['shower_realization_Alvarez2009'][idx]
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
    if time_logger is not None: time_logger.stop_time('event builder (hdf5 -> nur)')
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
    cfg : dict
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

def calculate_particle_weight(event_group, idx, cfg, fin=None):
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
        if fin is not None and "weights" in fin:
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
                      zerosignal=False, time_logger=None):
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
    if time_logger is not None: time_logger.start_time('group into events')
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
    logger.info(f"created {len(events)} events from event group {event_group_id}")
    if time_logger is not None: time_logger.stop_time('group into events')
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
            if len(value) and isinstance(value[0], bytes):
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
                 trigger_channels = None,
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
        trigger_channels: list of ints or dict of list of ints
            list of channel ids that are used for the trigger (per station_id). If None, all channels are used.
        """
        logger.setLevel(log_level)
        if 'write_mode' in kwargs:
            logger.warning('Parameter write_mode is deprecated. Define the output format in the config file instead.')

        self.__trigger_channel_ids = trigger_channels
        if(self.__trigger_channel_ids is None):
            logger.warning("No trigger channels specified. All channels will be simulated even if they don't contribute to any trigger. This can be inefficient. \
                           Processing time can be saved by specifying the trigger channels.")
        self._log_level = log_level
        self._log_level_ray_propagation = log_level_propagation
        self.__time_logger = NuRadioMC.simulation.time_logger.timeLogger(logger)

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
        self._evt_time = evt_time
        self.__write_detector = write_detector
        logger.status("setting event time to {}".format(evt_time))
        self._event_group_list = event_list

        # initialize detector simulation modules
        # due to mostly historical reasons, the detector simulation is provided as protected member functions,
        # the user inherits from the simulation class and provides the detector simulation functions as protected
        # member functions. This has the advantage that the user has potential access to all internal member variables
        # of the simulation class and thereby full control. However, this is discouraged as it makes it difficult to assure
        # backwards compatibility.
        # The default behaviour is that the user provides a function that defines the signal chain (i.e. filter and amplifiers etc.)
        # and another function that defines the trigger.
        # The user can also provide a function that defines the full detector simulation (i.e. the signal chain and the trigger) including
        # how noise is added etc.
        # The user needs to either provide the functions `_detector_simulation_filter_amp` and `_detector_simulation_trigger` or
        # the function `_detector_simulation_part1` and `_detector_simulation_part2`.
        self.detector_simulation_part1 = None
        self.detector_simulation_part2 = None
        self.detector_simulation_filter_amp = None
        self.detector_simulation_trigger = None
        if hasattr(self, '_detector_simulation_filter_amp'):
            self.detector_simulation_filter_amp = self._detector_simulation_filter_amp
        if hasattr(self, '_detector_simulation_trigger'):
            self.detector_simulation_trigger = self._detector_simulation_trigger
        if hasattr(self, '_detector_simulation_part1'):
            self.detector_simulation_part1 = self._detector_simulation_part1
        if hasattr(self, '_detector_simulation_part2'):
            self.detector_simulation_part2 = self._detector_simulation_part2
        if((self.detector_simulation_filter_amp is None) ^ (self.detector_simulation_trigger is None)):
            logger.error("Please provide both detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")
            raise ValueError("Please provide both detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")
        if((self.detector_simulation_part1 is None) ^ (self.detector_simulation_part2 is None)):
            logger.error("Please provide both detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")
            raise ValueError("Please provide both detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")
        if((self.detector_simulation_part1 is not None) and (self.detector_simulation_filter_amp is not None)):
            logger.error("Please provide either detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")
            raise ValueError("Please provide either detector_simulation_filter_amp and detector_simulation_trigger or detector_simulation_part1 and detector_simulation_part2")

        # Initialize detector
        self._antenna_pattern_provider = antennapattern.AntennaPatternProvider()
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
        if self._config['propagation']['ice_model'] == "custom":
            if ice_model is None:
                logger.error("ice model is set to 'custom' in config file but no custom ice model is provided.")
                raise AttributeError("ice model is set to 'custom' in config file but no custom ice model is provided.")
            self._ice = ice_model
        else:
            self._ice = NuRadioMC.utilities.medium.get_ice_model(self._config['propagation']['ice_model'])
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
            # TODO: It seems to be sufficient to just apply the `detector_simulation_filter_amp` function to a dummy event
            apply_det_response(evt, self._det, self._config, self.detector_simulation_filter_amp,
                               add_noise=False, detector_simulation_part2=self.detector_simulation_part2)

            self._integrated_channel_response[sid] = {}
            self._integrated_channel_response_normalization[sid] = {}
            self._max_amplification_per_channel[sid] = {}

            for channel_id in self._det.get_channel_ids(sid):
                ff = np.linspace(0, 0.5 * self._config['sampling_rate'], 10000)
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
                                  f'max. filter amplification = {max_amplification:.2e} '
                                  f'integrated response = {integrated_channel_response / units.MHz:.2e}MHz -> Vrms = '
                                  f'{self._Vrms_per_channel[station_id][channel_id] / units.mV:.4f} mV -> efield Vrms = {self._Vrms_efield_per_channel[station_id][channel_id] / units.V / units.m / units.micro:.2f}muV/m (assuming VEL = 1m) ')

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
        logger.status(f"All stations where all efields from all showers have amplitudes of less then {speed_cut:.1f} x Vrms_efield will be skipped.")

        # define function for distance speedup cut
        self._get_distance_cut = None
        if self._config['speedup']['distance_cut']:
            coef = self._config['speedup']['distance_cut_coefficients']
            distance_cut_polynomial = np.polynomial.polynomial.Polynomial(coef)

            def get_distance_cut(shower_energy):
                if shower_energy <= 0:
                    return 100 * units.m
                return max(100 * units.m, 10 ** distance_cut_polynomial(np.log10(shower_energy)))

            self._get_distance_cut = get_distance_cut

        particle_mode = "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] != "emitter"
        self._output_writer_hdf5 = outputWriterHDF5(self._outputfilename, self._config, self._det, self._station_ids,
                                                    self._propagator.get_number_of_raytracing_solutions(),
                                                    particle_mode=particle_mode)


    def run(self):
        """
        run the NuRadioMC simulation
        """
        if len(self._fin['xx']) == 0:
            logger.status("The input file does not contain any showers or emitters. Writing empty hdf5 output file.")
            self._output_writer_hdf5.write_empty_output_file(self._fin_attrs)
            logger.status("terminating simulation")
            return 0
        logger.status("Starting NuRadioMC simulation")
        self.__time_logger.reset_times()

        efieldToVoltageConverter.begin(time_resolution=self._config['speedup']['time_res_efieldconverter'])
        channelGenericNoiseAdder.begin(seed=self._config['seed'])
        if self._outputfilenameNuRadioReco is not None:
            eventWriter.begin(self._outputfilenameNuRadioReco, log_level=self._log_level)

        particle_mode = "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] != "emitter"
        event_group_ids = np.array(self._fin['event_group_ids'])
        unique_event_group_ids = np.unique(event_group_ids)

        # calculate bary centers of station
        station_barycenter = np.zeros((len(self._station_ids), 3))
        for iSt, station_id in enumerate(self._station_ids):
            pos = []
            for channel_id in self._det.get_channel_ids(station_id):
                pos.append(self._det.get_relative_position(station_id, channel_id))
            station_barycenter[iSt] = np.mean(np.array(pos), axis=0) + self._det.get_absolute_position(station_id)

        # loop over event groups
        for i_event_group_id, event_group_id in enumerate(unique_event_group_ids):
            logger.debug(f"simulating event group id {event_group_id}")
            if self._event_group_list is not None and event_group_id not in self._event_group_list:
                logger.debug(f"skipping event group {event_group_id} because it is not in the event group list provided to the __init__ function")
                continue
            event_indices = np.atleast_1d(np.squeeze(np.argwhere(event_group_ids == event_group_id)))

            self.__time_logger.show_time(len(unique_event_group_ids), i_event_group_id)

            event_group = build_NuRadioEvents_from_hdf5(self._fin, self._fin_attrs, event_indices, time_logger=self.__time_logger)
            event_group.set_event_time(self._evt_time)

            # determine if a particle (neutrinos, or a secondary interaction of a neutrino, or surfaec muons) is simulated
            self.__time_logger.start_time("weight calc.")
            weight = 1
            if particle_mode:
                weight = calculate_particle_weight(event_group, event_indices[0], self._config, self._fin)
            self.__time_logger.stop_time("weight calc.")
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if weight < self._config['speedup']['minimum_weight_cut']:
                logger.debug("neutrino weight is smaller than %f, skipping event", self._config['speedup']['minimum_weight_cut'])
                continue

            # these quantities get computed to apply the distance cut as a function of shower energies
            # the shower energies of closeby showers will be added as they can constructively interfere
            if self._config['speedup']['distance_cut']:
                self.__time_logger.start_time("distance cut")
                shower_energies = []
                vertex_positions = []
                for shower in event_group.get_sim_showers():
                    shower_energies.append([shower[shp.energy]])
                    vertex_positions.append([shower[shp.vertex]])
                shower_energies = np.array(shower_energies)
                vertex_positions = np.array(vertex_positions)
                self.__time_logger.stop_time("distance cut")

            output_buffer = {}
            # loop over all stations (each station is treated independently)
            for iSt, sid in enumerate(self._station_ids):
                if self._config['speedup']['distance_cut']:
                    # perform a quick cut to reject event group completely if no shower is close enough to the station
                    vertex_distances_to_station = np.linalg.norm(vertex_positions - station_barycenter[iSt], axis=1)
                    distance_cut = self._get_distance_cut(np.sum(shower_energies)) + 100 * units.m  # 100m safety margin is added to account for extent of station around bary center.
                    if vertex_distances_to_station.min() > distance_cut:
                        logger.debug(f"event group {event_group.get_run_number()} is too far away from station {sid}, skipping to next station")
                        # continue
                output_buffer[sid] = {}
                station = NuRadioReco.framework.station.Station(sid)
                sim_station = NuRadioReco.framework.sim_station.SimStation(sid)
                sim_station.set_is_neutrino()  # naming not ideal, but this function defines in-ice emission (compared to in-air emission from air showers)
                station.set_sim_station(sim_station)
                event_group.set_station(station)

                # we allow to first only simualte trigger channels. As the trigger channels might be different per station,
                # we need to determine the channels to simulate first per station
                channel_ids = self._det.get_channel_ids(sid)
                if self.__trigger_channel_ids is not None:
                    if isinstance(self.__trigger_channel_ids, dict):
                        channel_ids = self.__trigger_channel_ids[sid]
                    else:
                        channel_ids = self.__trigger_channel_ids

                # loop over all trigger channels
                candidate_station = False
                for iCh, channel_id in enumerate(channel_ids):
                    if particle_mode:
                        sim_station = calculate_sim_efield(showers=event_group.get_sim_showers(),
                                                        sid=sid, cid=channel_id,
                                                        det=self._det, propagator=self._propagator, medium=self._ice,
                                                        config=self._config,
                                                        time_logger=self.__time_logger,
                                                        min_efield_amplitude=float(self._config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[sid][channel_id],
                                                        distance_cut=self._get_distance_cut)
                    else:
                        sim_station = calculate_sim_efield_for_emitter(emitters=event_group.get_sim_emitters(),
                                            sid=sid, cid=channel_id,
                                            det=self._det, propagator=self._propagator, medium=self._ice, config=self._config,
                                            rnd=self._rnd, antenna_pattern_provider=self._antenna_pattern_provider,
                                            min_efield_amplitude=float(self._config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[sid][channel_id],
                                            time_logger=self.__time_logger)
                    if sim_station.is_candidate():
                        candidate_station = True
                    # skip to next channel if the efield is below the speed cut
                    if len(sim_station.get_electric_fields()) == 0:
                        logger.info(f"Eventgroup {event_group.get_run_number()} Station {sid} channel {channel_id:02d} has {len(sim_station.get_electric_fields())} efields, skipping to next channel")
                        continue

                    # applies the detector response to the electric fields (the antennas are defined
                    # in the json detector description file)
                    apply_det_response_sim(sim_station, self._det, self._config, self.detector_simulation_filter_amp,
                                           event_time=self._evt_time, time_logger=self.__time_logger,
                                           detector_simulation_part1=self.detector_simulation_part1)
                    logger.debug(f"adding sim_station to station {sid} for event group {event_group.get_run_number()}, channel {channel_id}")
                    station.add_sim_station(sim_station)  # this will add the channels and efields to the existing sim_station object
                # end channel loop, skip to next event group if all signals are empty (due to speedup cuts)
                sim_station = station.get_sim_station()  # needed to get sim_station object containing all channels and not just the last one.
                if len(sim_station.get_electric_fields()) == 0:
                    logger.info(f"Eventgroup {event_group.get_run_number()} Station {sid} has {len(sim_station.get_electric_fields())} efields, skipping to next station")
                    continue
                if candidate_station is False:
                    logger.info(f"skipping station {sid} because all electric fields are below threshold value")
                    continue

                # group events into events based on signal arrival times
                events = group_into_events(station, event_group, particle_mode, self._config['split_event_time_diff'], time_logger=self.__time_logger)

                evt_group_triggered = False
                for evt in events:
                    station = evt.get_station()
                    apply_det_response(evt, self._det, self._config, self.detector_simulation_filter_amp,
                                       bool(self._config['noise']),
                                       self._Vrms_per_channel, self._integrated_channel_response,
                                       self._noiseless_channels,
                                       time_logger=self.__time_logger,
                                       channel_ids=channel_ids,
                                       detector_simulation_part2=self.detector_simulation_part2)

                    # calculate trigger
                    self.__time_logger.start_time("trigger")
                    self.detector_simulation_trigger(evt, station, self._det)
                    logger.debug(f"event {evt.get_run_number()},{evt.get_id()} tiggered: {evt.get_station().has_triggered()}")
                    self.__time_logger.stop_time("trigger")
                    if not evt.get_station().has_triggered():
                        continue

                    triggerTimeAdjuster.run(evt, station, self._det)
                    evt_group_triggered = True
                    output_buffer[sid][evt.get_id()] = evt
                # end event loop

                # Only simulate the remaining channels & store the event when the event triggered.
                if not evt_group_triggered:
                    continue

                # now simulate non-trigger channels
                # we loop through all non-trigger channels and simulate the electric fields for all showers.
                # then we apply the detector response to the electric fields and find the event in which they will be visible in the readout window
                non_trigger_channels = list(set(self._det.get_channel_ids(sid)) - set(channel_ids))
                if (len(non_trigger_channels) > 0):
                    logger.status(f"Simulating non-trigger channels for station {sid}")
                    for iCh, channel_id in enumerate(non_trigger_channels):
                        if particle_mode:
                            sim_station = calculate_sim_efield(showers=event_group.get_sim_showers(),
                                                            sid=sid, cid=channel_id,
                                                            det=self._det, propagator=self._propagator, medium=self._ice,
                                                            config=self._config,
                                                            time_logger=self.__time_logger,
                                                            min_efield_amplitude=float(self._config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[sid][channel_id],
                                                            distance_cut=self._get_distance_cut)
                        else:
                            sim_station = calculate_sim_efield_for_emitter(emitters=event_group.get_sim_emitters(),
                                                sid=sid, cid=channel_id,
                                                det=self._det, propagator=self._propagator, medium=self._ice, config=self._config,
                                                rnd=self._rnd, antenna_pattern_provider=self._antenna_pattern_provider,
                                                min_efield_amplitude=float(self._config['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[sid][channel_id],
                                                time_logger=self.__time_logger)

                        # skip to next channel if the efield is below the speed cut
                        if not sim_station.get_electric_fields():
                            logger.info(f"Eventgroup {event_group.get_run_number()} Station {sid} channel {channel_id:02d} has "
                                        f"{len(sim_station.get_electric_fields())} efields, skipping to next channel")
                            continue

                        # applies the detector response to the electric fields (the antennas are defined
                        # in the json detector description file)
                        apply_det_response_sim(sim_station, self._det, self._config, self.detector_simulation_filter_amp,
                                            event_time=self._evt_time, time_logger=self.__time_logger,
                                            detector_simulation_part1=self.detector_simulation_part1)
                        logger.debug(f"adding sim_station to station {sid} for event group {event_group.get_run_number()}, channel {channel_id}")
                        station.add_sim_station(sim_station)  # this will add the channels and efields to the existing sim_station object
                        for evt in output_buffer[sid].values():
                            # we need to use any existing channel to get the correct trace_start_time. All channels have the same start time at the end
                            # of the simulation.
                            trace_start_time = evt.get_station().get_channel(channel_ids[0]).get_trace_start_time()
                            # # determine the trigger that was used to determine the readout window
                            # trigger_name = None
                            # trigger_time = None
                            # pre_trigger_time = None
                            # for trigger in evt.get_station().get_triggers().values():
                            #     # we need to select the trigger that was used to determine the readout window, which is the one that has the
                            #     # `pre_trigger_times` set. All other triggers will return None. We can double check that by testing that the
                            #     #  trigger_time is the same as start time as the trace
                            #     if trigger.get_pre_trigger_times() is not None:
                            #         pre_trigger_time = trigger.get_pre_trigger_times()
                            #         trigger_time = trigger.get_trigger_time()
                            #         trigger_name = trigger.get_name()
                            #         break

                            # assert(trigger_time is not None)
                            # assert(trigger_name is not None)
                            # assert(trace_start_time == trigger_time)

                            for sim_channel in sim_station.get_channels_by_channel_id(channel_id):
                                tt = sim_channel.get_times()
                                t0 = tt[0]
                                t1 = tt[-1]
                                t0_readout = trace_start_time
                                t1_readout = t0_readout + self._det.get_number_of_samples(sid, sim_channel.get_id()) * self._det.get_sampling_frequency(sid, sim_channel.get_id())

                                if not station.has_channel(sim_channel.get_id()):
                                    # add empty channel with the correct length and time if it doesn't exist yet.
                                    channel = NuRadioReco.framework.channel.Channel(channel_id)
                                    n_samples = int(round(self._det.get_number_of_samples(sid, channel_id)) * self._config['sampling_rate'] / self._det.get_sampling_frequency(sid, sim_channel.get_id()))
                                    channel.set_trace(np.zeros(n_samples), self._config['sampling_rate'])
                                    channel.set_trace_start_time(trace_start_time)
                                    station.add_channel(channel)

                                # determine if the two intervals have any overlap, if yes, add the signal to the existing empty trace
                                if max(t0, t0_readout) < min(t1, t1_readout):
                                    # we need to create a new Channel object in which we copy the SimChannel trace. This is necessary because
                                    # Channel and SimChannel are different objects and a SimChannel can't be added to a Channel.
                                    tmp_channel = NuRadioReco.framework.channel.Channel(sim_channel.get_id())
                                    tmp_channel.set_trace(sim_channel.get_trace(), sim_channel.get_sampling_rate())
                                    tmp_channel.set_trace_start_time(sim_channel.get_trace_start_time() - 55 * units.ns)

                                    channel = station.get_channel(sim_channel.get_id())  # important to get the channel (again) to have a reference to the correct object

                                    assert(tmp_channel.get_sampling_rate() == channel.get_sampling_rate())

                                    # we need to identify the overlapping time interval and only add this to channel
                                    # to achieve time accuracy, we need to shift `tmp_channel` using the Fourier shift theorem.
                                    time_offset = np.abs(tmp_channel.get_trace_start_time() - channel.get_trace_start_time())
                                    i_start = int(round(time_offset * channel.get_sampling_rate()))
                                    residual_time_offset = time_offset - i_start / channel.get_sampling_rate()
                                    print(f"residual_time_offset = {residual_time_offset}")
                                    tmp_channel.apply_time_shift(-residual_time_offset)  # we only shift the residual time offset that is left after the integer shift
                                    # determine the indices for both traces
                                    i_min = max(0, i_start)
                                    i_min_tmp = max(0, -i_start)
                                    i_max = min(len(channel.get_trace()), len(tmp_channel.get_trace()) + i_min - i_min_tmp)
                                    i_max_tmp = i_min_tmp + i_max - i_min
                                    channel_trace = channel.get_trace()
                                    tmp_channel_trace = tmp_channel.get_trace()
                                    channel_trace[i_min:i_max] += tmp_channel_trace[i_min_tmp:i_max_tmp]
                                    # logger.status(f"channel type {type(channel)}, sim_channel type {type(sim_channel)}")
                                    channel.set_trace(channel_trace, channel.get_sampling_rate())

                for evt in output_buffer[sid].values():
                    # the only thing left is to add noise to the non-trigger traces
                    # we need to do it a bit differently than for the trigger traces, because we need to add noise to traces where the amplifier response
                    # was already applied to.
                    station = evt.get_station()
                    if bool(self._config['noise']):
                        for channel_id in non_trigger_channels:
                            if channel_id in self._noiseless_channels[sid]:
                                continue
                            # we might not have a channel object in case there was no ray tracing solution to this channel, or if the timing did not match
                            # the readout window. In this case we need to create a channel object and add it to the station
                            if station.has_channel(channel_id):
                                channel = station.get_channel(channel_id)
                            else:
                                channel = NuRadioReco.framework.channel.Channel(channel_id)
                                n_samples = int(round(self._det.get_number_of_samples(sid, channel_id)) * self._config['sampling_rate'] / self._det.get_sampling_frequency(sid, channel_id))
                                channel.set_trace(np.zeros(n_samples), self._config['sampling_rate'])
                                # we need to use any other channel to get the correct trace_start_time. All channels have the same start time at the end
                                # of the simulation.
                                channel.set_trace_start_time(station.get_channel(station.get_channel_ids()[0]).get_trace_start_time())
                                station.add_channel(channel)
                            # logger.status(f"norm  = {norm}, Vrms = {Vrms[channel_id]}, max_freq = {max_freq}")
                            ff = channel.get_frequencies()
                            filt = np.ones_like(ff, dtype=complex)
                            for i, (name, instance, kwargs) in enumerate(evt.iter_modules(sid)):
                                if hasattr(instance, "get_filter"):
                                    filt *= instance.get_filter(ff, sid, channel_id, self._det, **kwargs)
                            noise = channelGenericNoiseAdder.bandlimited_noise_from_spectrum(len(channel.get_trace()), channel.get_sampling_rate(),
                                                                                            spectrum=filt, amplitude=self._Vrms_per_channel[station.get_id()][channel_id],
                                                                                            type='rayleigh', time_domain=False)
                            # from NuRadioReco.utilities import fft
                            # logger.warning(f"adding noise to channel {channel.get_id()} with Vrms = {Vrms[channel_id]/units.mV:.4f}mV, realized noise Vrms = {np.std(fft.freq2time(noise, 1/dt))/units.mV:.4f}mV")
                            channel.set_frequency_spectrum(channel.get_frequency_spectrum() + noise, channel.get_sampling_rate())

                    channelSignalReconstructor.run(evt, station, self._det)
                    # save RMS and bandwidth to channel object
                    evt.set_generator_info(genattrs.Vrms, self._Vrms)
                    evt.set_generator_info(genattrs.dt, 1. / self._config['sampling_rate'])
                    evt.set_generator_info(genattrs.Tnoise, self._noise_temp)
                    evt.set_generator_info(genattrs.bandwidth, next(iter(next(iter(self._integrated_channel_response.values())).values())))
                    for channel in station.iter_channels():
                        channel[chp.Vrms_NuRadioMC_simulation] = self._Vrms_per_channel[sid][channel.get_id()]
                        channel[chp.bandwidth_NuRadioMC_simulation] = self._integrated_channel_response[sid][channel.get_id()]

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

                self._output_writer_hdf5.add_event_group(output_buffer)

        if self._outputfilenameNuRadioReco is not None:
            eventWriter.end()
            logger.debug("closing nur file")

        self._output_writer_hdf5.calculate_Veff()
        if not self._output_writer_hdf5.write_output_file():
            logger.warning("No events were triggered. Writing empty HDF5 output file.")
            self._output_writer_hdf5.write_empty_output_file(self._fin_attrs)


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
        return 1. / self._config['sampling_rate']

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
