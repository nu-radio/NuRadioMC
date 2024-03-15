import os
import collections
import numpy as np
import h5py
import yaml
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrans
import NuRadioMC
from NuRadioMC.utilities.Veff import remove_duplicate_triggers
from NuRadioReco.utilities import version
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import emitterParameters as ep
from NuRadioReco.framework.parameters import particleParameters as pap
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger("NuRadioMC.HDF5OutputWriter")

class outputWriterHDF5:
    """
    Class to write output HDF5 files
    """
    def __init__(
            self,
            output_filename,
            config,
            detector,
            station_ids,
            number_of_ray_tracing_solutions,
            particle_mode=True
    ):
        self._station_ids = station_ids
        self._nS = number_of_ray_tracing_solutions
        self._particle_mode = particle_mode
        # self._fin = fin
        # self._fin_attrs = fin_attrs
        # self._n_showers = len(self._fin['shower_ids'])

        # self._hdf5_keys = ['event_group_ids', 'xx', 'yy', 'zz', 'vertex_times',
        #                    'azimuths', 'zemiths', 'energies', 
        #                    'shower_energies', ''n_interaction',
        #                    'shower_ids', 'event_ids', 'triggered', 'n_samples',
        #                    'dt', 'Tnoise', 'Vrms', 'bandwidth', 'trigger_names']

        """
        creates the data structures of the parameters that will be saved into the hdf5 output file
        """
        self._mout = collections.OrderedDict()
        self._mout_groups = collections.OrderedDict()
        self._mout_attributes = collections.OrderedDict()
        self._mout_groups_attributes = collections.OrderedDict()



        self.__first_event = True
        self._output_filename = output_filename
        self._det = detector

        #### write attributes
        self._mout_attributes['config'] = config
        # self._mout_attributes['detector'] = detector.export_as_string()  # TODO: cherrypick this function from Christophs refactor


        # self._mout['weights'] = []
        # self._mout['triggered'] = []
        # self._mout_attributes['trigger_names'] = None

        for station_id in self._station_ids:
            self._mout_groups[station_id] = collections.OrderedDict()
            self._mout_groups_attributes[station_id] = collections.OrderedDict()



    def __add_parameter(self, dict_to_fill, key, value, first_event=None):
        """
        Add a parameter to the output file
        """
        if first_event is None:
            if key not in dict_to_fill:
                dict_to_fill[key] = [value]
            else:
                dict_to_fill[key].append(value)
        else:
            if first_event:
                dict_to_fill[key] = [value]
            else:
                if key not in dict_to_fill:
                    logger.error(f"key {key} not in dict but not first event")
                    raise KeyError(f"key {key} not in dict but not first event")
                dict_to_fill[key].append(value)


    def add_event_group(self, event_buffer):
        """
        Add an event group to the output file
        """
        logger.debug("adding event group to output file")



        # first add atributes to the output file. Attributes of all events should be the same, 
        # raising an error if not. 
        trigger_names = []
        extent_array_by = 0
        for sid in event_buffer:
            for eid in event_buffer[sid]:
                evt = event_buffer[sid][eid]
                for enum_entry in genattrs:
                    if(evt.has_generator_info(enum_entry)):
                        if enum_entry.name not in self._mout_attributes:
                            self._mout_attributes[enum_entry.name] = evt.get_generator_info(enum_entry)
                        else:  # if the attribute is already present, we need to check if it is the same for all events
                            assert all(np.atleast_1d(self._mout_attributes[enum_entry.name] == evt.get_generator_info(enum_entry)))
                for stn in evt.get_stations():
                    # save station attributes
                    tmp_keys = [[chp.Vrms_NuRadioMC_simulation, "Vrms"], [chp.bandwidth_NuRadioMC_simulation, "bandwidth"]]
                    for (key_cp, key_hdf5) in tmp_keys:
                        channel_values = []
                        for channel in stn.iter_channels():
                            channel_values.append(channel[key_cp])
                        if key_hdf5 not in self._mout_groups_attributes[sid]:
                            self._mout_groups_attributes[sid][key_hdf5] = np.array(channel_values)
                        else:
                            assert all(np.atleast_1d(self._mout_groups_attributes[sid][key_hdf5] == np.array(channel_values))), f"station {sid} key {key_hdf5} is {self._mout_groups_attributes[sid][key_hdf5]}, but current channel is {np.array(channel_values)}"
                    for trigger in stn.get_triggers().values():
                        if trigger.get_name() not in trigger_names:
                            trigger_names.append(trigger.get_name())
                            logger.debug(f"extending data structure by trigger {trigger.get_name()} to output file")
                            extent_array_by += 1
        # the available triggers are not available from the start because a certain trigger
        # might only trigger a later event. Therefore we need to extend the array
        # if we find a new trigger
        if extent_array_by:
            self._mout_attributes['trigger_names'] = trigger_names
            keys = ["multiple_triggers", "trigger_times"]
            if keys[0] in self._mout:
                for key in keys:
                    for i in range(len(self._mout[key])):
                        logger.debug(f"extending data structure by {extent_array_by} to output file for key {key}")
                        self._mout[key][i] = self._mout[key][i] + [False] * extent_array_by
            for station_id in self._station_ids:
                sg = self._mout_groups[station_id]
                if keys[0] in sg:
                    for key in keys:
                        for i in range(len(sg[key])):
                            sg[key][i] = sg[key][i] + [False] * extent_array_by

        shower_ids = []
        for sid in event_buffer:  # loop over all stations (every station is treated independently)
            shower_ids_stn = []
            logger.debug(f"adding station {sid} to output file")
            for eid in event_buffer[sid]:
                logger.debug(f"adding event {eid} to output file")
                evt = event_buffer[sid][eid]
                if self._particle_mode:
                    for shower in evt.get_sim_showers():
                        if not shower.get_id() in shower_ids:
                            logger.debug(f"adding shower {shower.get_id()} to output file")
                            # shower ids might not be in increasing order. We need to sort the hdf5 output later
                            shower_ids.append(shower.get_id())
                            particle = evt.get_parent(shower)
                            self.__add_parameter(self._mout, 'shower_ids', shower.get_id(), self.__first_event)
                            self.__add_parameter(self._mout, 'event_group_ids', evt.get_run_number(), self.__first_event)
                            self.__add_parameter(self._mout, 'xx', shower[shp.vertex][0], self.__first_event)
                            self.__add_parameter(self._mout, 'yy', shower[shp.vertex][1], self.__first_event)
                            self.__add_parameter(self._mout, 'zz', shower[shp.vertex][2], self.__first_event)
                            self.__add_parameter(self._mout, 'vertex_times', shower[shp.vertex_time], self.__first_event)
                            self.__add_parameter(self._mout, 'azimuths', shower[shp.azimuth], self.__first_event)
                            self.__add_parameter(self._mout, 'zeniths', shower[shp.zenith], self.__first_event)
                            self.__add_parameter(self._mout, 'shower_energies', shower[shp.energy], self.__first_event)
                            self.__add_parameter(self._mout, 'shower_type', shower[shp.type], self.__first_event)
                            if(shower.has_parameter(shp.charge_excess_profile_id)):
                                self.__add_parameter(self._mout, 'shower_realization_ARZ', shower[shp.charge_excess_profile_id], self.__first_event)
                            if(shower.has_parameter(shp.k_L)):
                                self.__add_parameter(self._mout, 'shower_realization_Alvarez2009', shower[shp.k_L], self.__first_event)

                            self.__add_parameter(self._mout, 'energies', particle[pap.energy], self.__first_event)
                            self.__add_parameter(self._mout, 'flavors', shower[shp.flavor], self.__first_event)
                            self.__add_parameter(self._mout, 'n_interaction', shower[shp.n_interaction], self.__first_event)
                            self.__add_parameter(self._mout, 'interaction_type', shower[shp.interaction_type], self.__first_event)
                            self.__add_parameter(self._mout, 'inelasticity', particle[pap.inelasticity], self.__first_event)
                            self.__add_parameter(self._mout, 'weights', particle[pap.weight], self.__first_event)
                            self.__first_event = False
                else:  # emitters have different properties, so we need to treat them differently than showers
                    for emitter in evt.get_sim_emitters():
                        if not emitter.get_id() in shower_ids:  # the key "shower_ids" is also used for emitters and identifies the emitter id. This is done because it is the only way to have the same input files for both shower/particle and emitter simulations. 
                            logger.debug(f"adding shower {emitter.get_id()} to output file")
                            # shower ids might not be in increasing order. We need to sort the hdf5 output later
                            shower_ids.append(emitter.get_id())
                            self.__add_parameter(self._mout, 'shower_ids', emitter.get_id(), self.__first_event)
                            self.__add_parameter(self._mout, 'event_group_ids', evt.get_run_number(), self.__first_event)
                            self.__add_parameter(self._mout, 'xx', emitter[ep.position][0], self.__first_event)
                            self.__add_parameter(self._mout, 'yy', emitter[ep.position][1], self.__first_event)
                            self.__add_parameter(self._mout, 'zz', emitter[ep.position][2], self.__first_event)
                            self.__add_parameter(self._mout, 'emitter_amplitudes', emitter[ep.amplitude], self.__first_event)

                            for key in ep:
                                if key.name not in ['position', 'amplitude']:
                                    if emitter.has_parameter(key):
                                        if key not in self._mout:
                                            keyname = 'emitter_' + key.name
                                            self.__add_parameter(self._mout, keyname, emitter[key], self.__first_event)

                            self.__first_event = False
                # now save station data
                stn = evt.get_station()  # there can only ever be one station per event! If there are more than one station, this line will crash. 
                sg = self._mout_groups[sid]
                self.__add_parameter(sg, 'event_group_ids', evt.get_run_number())
                self.__add_parameter(sg, 'event_ids', evt.get_id())
                maximum_amplitudes = []
                maximum_amplitudes_envelope = []
                for channel in stn.iter_channels():
                    maximum_amplitudes.append(channel[chp.maximum_amplitude])
                    maximum_amplitudes_envelope.append(channel[chp.maximum_amplitude_envelope])
                self.__add_parameter(sg, 'maximum_amplitudes', maximum_amplitudes)
                self.__add_parameter(sg, 'maximum_amplitudes_envelope', maximum_amplitudes_envelope)

                multiple_triggers = []
                trigger_times = []
                for iT, tname in enumerate(self._mout_attributes['trigger_names']):
                    if stn.has_triggered(tname):
                        multiple_triggers.append(True)
                        trigger_times.append(stn.get_trigger(tname).get_trigger_time())
                    else:
                        multiple_triggers.append(False)
                        trigger_times.append(np.nan)
                self.__add_parameter(sg, 'multiple_triggers_per_event', multiple_triggers)
                self.__add_parameter(sg, 'trigger_times_per_event', np.array(trigger_times, dtype=float))
                self.__add_parameter(sg, 'triggered_per_event', np.any(multiple_triggers))

                self.__add_parameter(sg, 'triggered', stn.has_triggered())

                # depending on the simulation mode we have either showers or emitters but we can 
                # treat them the same way as long as we only call common member functions such as 
                # `get_id()`
                iterable = None
                if self._particle_mode:
                    iterable = evt.get_sim_showers()
                else:
                    iterable = evt.get_sim_emitters()
                for shower in iterable:
                    if not shower.get_id() in shower_ids_stn:
                        shower_ids_stn.append(shower.get_id())
                        self.__add_parameter(sg, 'shower_id', shower.get_id())
                        self.__add_parameter(sg, 'event_group_id_per_shower', evt.get_run_number())
                        self.__add_parameter(sg, 'event_id_per_shower', shower.get_id())

                        # we need to save data per shower, channel and ray tracing solution. Due to the simple table structure
                        # of the hdf5 files we need to preserve the ordering of the showers and channels. As the order in the
                        # NuRadio data structure is different, we need to go through some effort to get the right order.
                        # The shower ids will be sorted at the very end. 
                        # The channel ids already have the correct ordering. 
                        # The ray tracing solutions are also ordered, because the efield object contains the correct ray tracing solution id. 
                        channel_rt_data = {}
                        keys_channel_rt_data = ['travel_times', 'travel_distances']
                        if self._mout_attributes['config']['speedup']['amp_per_ray_solution']:
                            keys_channel_rt_data.extend(['time_shower_and_ray', 'max_amp_shower_and_ray'])
                        nCh = stn.get_number_of_channels()
                        for key in keys_channel_rt_data:
                            channel_rt_data[key] = np.zeros((nCh, self._nS)) * np.nan
                        keys_channel_rt_data_3D = ['launch_vectors', 'receive_vectors', 'polarization']
                        for key in keys_channel_rt_data_3D:
                            channel_rt_data[key] = np.zeros((nCh, self._nS, 3)) * np.nan
                        # important: we need to loop over the channels of the station object, not
                        # the channels present in the sim_station object. This is because the sim
                        # channel object only contains the channels that have a signal, i.e., a ray
                        # tracing solution and a strong enough Askaryan signal. But we want to loop over all 
                        # channels of the station, because we want to save the data for all channels, not only
                        # the ones that have a signal. This also preserves the order of the channels.
                        for iCh, channel in enumerate(stn.iter_channels()):
                            for efield in stn.get_sim_station().get_electric_fields_for_channels([channel.get_id()]):
                                if efield.get_shower_id() == shower.get_id():
                                    iS = efield.get_ray_tracing_solution_id()
                                    for key, value in efield[efp.raytracing_solution].items():
                                        if key not in channel_rt_data:
                                            channel_rt_data[key] = np.zeros((nCh, self._nS)) * np.nan
                                        channel_rt_data[key][iCh, iS] = value
                                    channel_rt_data['launch_vectors'][iCh, iS] = efield[efp.launch_vector]
                                    receive_vector = hp.spherical_to_cartesian(efield[efp.zenith], efield[efp.azimuth])
                                    channel_rt_data['receive_vectors'][iCh, iS] = receive_vector
                                    channel_rt_data['travel_times'][iCh, iS] = efield[efp.nu_vertex_travel_time]
                                    channel_rt_data['travel_distances'][iCh, iS] = efield[efp.nu_vertex_distance]

                                    if self._particle_mode:
                                        # only the polarization angle is saved in the electric field object, so we need to
                                        # calculate the vector it to the ground frame in cartesian coordinates.
                                        cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*receive_vector))
                                        polarization_angle = efield[efp.polarization_angle]
                                        polarization_direction_onsky = np.array([0, np.cos(polarization_angle), np.sin(polarization_angle)])
                                        polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
                                        channel_rt_data['polarization'][iCh, iS] = polarization_direction_at_antenna

                                    if self._mout_attributes['config']['speedup']['amp_per_ray_solution']:
                                        sim_station = stn.get_sim_station()
                                        sim_channel = sim_station.get_channel((channel.get_id(), shower.get_id(), iS))
                                        channel_rt_data['max_amp_shower_and_ray'][iCh, iS] = sim_channel[chp.maximum_amplitude_envelope]
                                        channel_rt_data['time_shower_and_ray'][iCh, iS] = sim_channel[chp.signal_time]

                        for key, value in channel_rt_data.items():
                            self.__add_parameter(sg, key, value)
            # end event loop
            # now determine triggers per shower. This is a bit tricky, we need to consider all events
            # and count a shower if it contributed to any of the events. The trigger_times field contains
            # the earliest trigger time of all stations and triggers
            shower_id_to_index = {shower_id: i for i, shower_id in enumerate(shower_ids_stn)}
            triggered = np.zeros(len(shower_ids_stn), dtype=bool)
            multiple_triggers = np.zeros((len(shower_ids_stn), len(self._mout_attributes['trigger_names'])), dtype=bool)
            trigger_times = np.ones((len(shower_ids_stn), len(self._mout_attributes['trigger_names'])), dtype=float) * np.nan
            for eid in event_buffer[sid]:
                evt = event_buffer[sid][eid]
                stn = evt.get_station()  # there can only ever be one station per event! If there are more than one station, this line will crash. 
                # depending on the simulation mode we have either showers or emitters but we can 
                # treat them the same way as long as we only call common member functions such as 
                # `get_id()`
                iterable = None
                if self._particle_mode:
                    iterable = evt.get_sim_showers()
                else:
                    iterable = evt.get_sim_emitters()
                for shower in iterable:
                    if stn.has_triggered():
                        triggered[shower_id_to_index[shower.get_id()]] = True
                        for iT, tname in enumerate(self._mout_attributes['trigger_names']):
                            if stn.has_triggered(tname):
                                multiple_triggers[shower_id_to_index[shower.get_id()], iT] = True
                                if np.isnan(trigger_times[shower_id_to_index[shower.get_id()], iT]):
                                    trigger_times[shower_id_to_index[shower.get_id()], iT] = stn.get_trigger(tname).get_trigger_time()
                                else:
                                    trigger_times[shower_id_to_index[shower.get_id()], iT] = min(trigger_times[shower_id_to_index[shower.get_id()], iT], stn.get_trigger(tname).get_trigger_time())
            # fill to output structure
            for shower_id in shower_ids_stn:
                i = shower_id_to_index[shower_id]
                self.__add_parameter(sg, 'triggered', triggered[i])
                self.__add_parameter(sg, 'multiple_triggers', multiple_triggers[i])
                self.__add_parameter(sg, 'trigger_times', trigger_times[i])
        # end station loop

        # save trigger information on first level
        shower_id_to_index = {shower_id: i for i, shower_id in enumerate(shower_ids)}
        triggered = np.zeros(len(shower_ids_stn), dtype=bool)
        multiple_triggers = np.zeros((len(shower_ids), len(self._mout_attributes['trigger_names'])), dtype=bool)
        trigger_times = np.ones((len(shower_ids), len(self._mout_attributes['trigger_names'])), dtype=float) * np.nan
        for shower_id in shower_ids:
            iSh = shower_id_to_index[shower_id]
            for stn_id in self._station_ids:
                sg = self._mout_groups[stn_id]
                if 'shower_id' not in sg:
                    continue
                shower_ids_stn = sg['shower_id']
                iSh_stn = np.where(shower_ids_stn == shower_id)[0]
                if len(iSh_stn) == 0:
                    continue
                iSh_stn = iSh_stn[0]
                triggered[iSh] = triggered[iSh] or sg['triggered'][iSh_stn]
                if 'multiple_triggers' in sg:
                    multiple_triggers[iSh] = multiple_triggers[iSh] | sg['multiple_triggers'][iSh_stn]
                if 'trigger_times' in sg:
                    for iT, tname in enumerate(self._mout_attributes['trigger_names']):
                        if not np.isnan(sg['trigger_times'][iSh_stn][iT]):
                            if np.isnan(trigger_times[iSh, iT]):
                                trigger_times[iSh, iT] = sg['trigger_times'][iSh_stn][iT]
                            else:
                                trigger_times[iSh, iT] = min(trigger_times[iSh, iT], sg['trigger_times'][iSh_stn][iT])
        # fill to output structure
        for shower_id in shower_ids:
            i = shower_id_to_index[shower_id]
            self.__add_parameter(self._mout, 'triggered', triggered[i])
            self.__add_parameter(self._mout, 'multiple_triggers', multiple_triggers[i])
            self.__add_parameter(self._mout, 'trigger_times', trigger_times[i])

        if self._particle_mode:
            # we also want to save the first interaction even if it didn't contribute to any trigger
            # this is important to know the initial neutrino properties (only relevant for the simulation of 
            # secondary interactions)
            stn_buffer = event_buffer[self._station_ids[0]]
            evt = stn_buffer[list(stn_buffer.keys())[0]]
            particle = evt.get_primary()
            if(particle[pap.shower_id] not in shower_ids):
                keys_to_populate = list(self._mout.keys())
                logger.info(f"adding primary shower {particle[pap.shower_id]} to output file")
                self.__add_parameter(self._mout, 'shower_ids', particle[pap.shower_id])
                self.__add_parameter(self._mout, 'event_group_ids', evt.get_run_number())
                self.__add_parameter(self._mout, 'xx', particle[pap.vertex][0])
                self.__add_parameter(self._mout, 'yy', particle[pap.vertex][1])
                self.__add_parameter(self._mout, 'zz', particle[pap.vertex][2])
                self.__add_parameter(self._mout, 'vertex_times', particle[pap.vertex_time])
                self.__add_parameter(self._mout, 'azimuths', particle[pap.azimuth])
                self.__add_parameter(self._mout, 'zeniths', particle[pap.zenith])
                self.__add_parameter(self._mout, 'shower_energies', np.nan)
                self.__add_parameter(self._mout, 'shower_type', "")
                self.__add_parameter(self._mout, 'energies', particle[pap.energy])
                self.__add_parameter(self._mout, 'flavors', particle[pap.flavor])
                self.__add_parameter(self._mout, 'n_interaction', particle[pap.n_interaction])
                self.__add_parameter(self._mout, 'interaction_type', particle[pap.interaction_type])
                self.__add_parameter(self._mout, 'inelasticity', particle[pap.inelasticity])
                self.__add_parameter(self._mout, 'weights', particle[pap.weight])
                self.__add_parameter(self._mout, 'triggered', False)
                multiple_triggers = np.zeros(len(self._mout_attributes['trigger_names']), dtype=bool)
                self.__add_parameter(self._mout, 'multiple_triggers', multiple_triggers)
                self.__add_parameter(self._mout, 'trigger_times', np.ones(len(self._mout_attributes['trigger_names']), dtype=float) * np.nan)

                keys_populated = ['shower_ids', 'event_group_ids', 'xx', 'yy', 'zz', 'vertex_times', 'azimuths', 'zeniths',
                                'shower_energies', 'shower_type', 'energies', 'flavors', 'n_interaction', 'interaction_type',
                                'inelasticity', 'weights', 'triggered', 'multiple_triggers', 'trigger_times']
                for key in keys_to_populate:
                    if key not in keys_populated:
                        logger.debug(f"key {key} not populated for primary shower, adding nan")
                        self.__add_parameter(self._mout, key, np.nan)


    def write_empty_output_file(self, fin_attrs):
        """
        Write an empty output file with the given file attributes.

        Parameters:
            fin_attrs (callable): A function that returns a dictionary of file attributes.

        Returns:
            None
        """
        folder = os.path.dirname(self._output_filename)
        if not os.path.exists(folder) and folder != '':
            logger.warning(f"output folder {folder} does not exist, creating folder...")
            os.makedirs(folder)
        fout = h5py.File(self._output_filename, 'w')
        # save meta arguments
        for (key, value) in fin_attrs():
            fout.attrs[key] = value
        # save NuRadioMC and NuRadioReco versions
        fout.attrs['NuRadioMC_version'] = NuRadioMC.__version__
        fout.attrs['NuRadioMC_version_hash'] = version.get_NuRadioMC_commit_hash()
        fout.close()


    def write_output_file(self):
        """
        Write the output file in HDF5 format.

        Returns:
            bool: False if there are no events to save, True otherwise.
        """
        if len(self._mout['shower_ids']) == 0:
            logger.warning("no events to save, not writing output file")
            return False
        folder = os.path.dirname(self._output_filename)
        if not os.path.exists(folder) and folder != '':
            logger.warning(f"output folder {folder} does not exist, creating folder...")
            os.makedirs(folder)
        fout = h5py.File(self._output_filename, 'w')

        logger.status("start saving events")
        # save data sets
        # all arrays need to be sorted by shower id
        sort = np.argsort(np.array(self._mout['shower_ids']))
        for (key, value) in self._mout.items():
            logger.debug(f"saving {key} {value} {type(value)}")
            if np.array(value).dtype.char == 'U':
                fout[key] = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))[sort]
            else:
                fout[key] = np.array(value)[sort]

        # save all data sets of the station groups
        keys_per_event = ['event_group_ids', 'event_ids', 'multiple_triggers_per_event', 'trigger_times_per_event',
                            'maximum_amplitudes', 'maximum_amplitudes_envelope', 'triggered_per_event']
        for (key, value) in self._mout_groups.items():
            sg = fout.create_group("station_{:d}".format(key))
            if 'shower_id' not in value:
                continue
            sort = np.argsort(np.array(value['shower_id']))
            for (key2, value2) in value.items():
                # a few arrays are counting values for different events, so we need to sort them
                if(key2 not in keys_per_event):
                    sg[key2] = np.array(value2)[sort]
                else:
                    sg[key2] = np.array(value2)

        # TODO save detector
        # if isinstance(self._det, detector.rnog_detector.Detector):
        #     fout.attrs['detector'] = self._det.export_as_string()
        # else:
        #     with open(self._detectorfile, 'r') as fdet:
        #         fout.attrs['detector'] = fdet.read()

        # save antenna position separately to hdf5 output
        for station_id in self._mout_groups:
            n_channels = self._det.get_number_of_channels(station_id)
            positions = np.zeros((n_channels, 3))
            for channel_id in range(n_channels):
                positions[channel_id] = self._det.get_relative_position(station_id, channel_id) + self._det.get_absolute_position(station_id)
            fout["station_{:d}".format(station_id)].attrs['antenna_positions'] = positions
        fout.attrs['config'] = yaml.dump(self._mout_attributes['config'])

        # save NuRadioMC and NuRadioReco versions
        fout.attrs['NuRadioMC_version'] = NuRadioMC.__version__
        fout.attrs['NuRadioMC_version_hash'] = version.get_NuRadioMC_commit_hash()

        for key in self._mout_attributes:
            if key not in fout.attrs:
                if self._mout_attributes[key] is not None:
                    fout.attrs[key] = self._mout_attributes[key]
                else:
                    logger.warning(f"attribute {key} is None, not saving it")
        fout.close()
        return True

    def calculate_Veff(self):
        """
        Calculate the effective volume (Veff)

        Returns:
            float: The calculated effective volume (Veff)
        """
        # calculate effective
        try: # sometimes not all relevant attributes are set, e.g. for emitter simulations. 
            triggered = remove_duplicate_triggers(self._mout['triggered'], self._mout['event_group_ids'])
            n_triggered = np.sum(triggered)
            n_triggered_weighted = np.sum(np.array(self._mout['weights'])[triggered])
            n_events = self._mout_attributes['n_events']
            logger.status(f'fraction of triggered events = {n_triggered:.0f}/{n_events:.0f} (sum of weights = {n_triggered_weighted:.2f})')

            V = self._mout_attributes['volume']
            Veff = V * n_triggered_weighted / n_events
            logger.status(f"Veff = {Veff / units.km ** 3:.4g} km^3, Veffsr = {Veff * 4 * np.pi/units.km**3:.4g} km^3 sr")
            return Veff
        except:
            return None
