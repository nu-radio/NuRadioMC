import numpy as np
import h5py
import yaml
import os
import collections
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as ep
from NuRadioReco.framework.parameters import particleParameters as pap
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrans
from NuRadioReco.utilities import units
from NuRadioReco.utilities.logging import setup_logger

logger = setup_logger("NuRadioMC.HDF5OutputWriter")

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
            number_of_ray_tracing_solutions
    ):
        self._station_ids = station_ids
        self._nS = number_of_ray_tracing_solutions
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
        self._mout = {}
        self._mout_groups = {}
        self._mout_attributes = {}

        self.__first_event = True

        #### write attributes
        self._mout_attributes['config'] = config
        # self._mout_attributes['detector'] = detector.export_as_string()  # TODO: cherrypick this function from Christophs refactor


        self._mout['weights'] = []
        self._mout['triggered'] = []
        # self._mout['weights'] = np.zeros(self._n_showers)
        # self._mout['triggered'] = np.zeros(self._n_showers, dtype=bool)
#         self._mout['multiple_triggers'] = np.zeros((self._n_showers, self._number_of_triggers), dtype=bool)
        self._mout_attributes['trigger_names'] = None

        for station_id in self._station_ids:
            self._mout_groups[station_id] = {}


    def __add_parameter(self, dict, key, value, first_event=None):
        """
        Add a parameter to the output file
        """
        if first_event is None:
            if key not in dict:
                dict[key] = [value]
            else:
                dict[key].append(value)
        else:
            if first_event:
                dict[key] = [value]
            else:
                if key not in dict:
                    logger.error(f"key {key} not in dict but not first event")
                    raise KeyError(f"key {key} not in dict but not first event")
                dict[key].append(value)


    def add_event_group(self, event_buffer):
        """
        Add an event group to the output file
        """
        logger.status("adding event group to output file")



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
                        else:
                            assert all(np.atleast_1d(self._mout_attributes[enum_entry.name] == evt.get_generator_info(enum_entry)))
                for stn in evt.get_stations():
                    for trigger in stn.get_triggers().values():
                        if trigger.get_name() not in trigger_names:
                            trigger_names.append(trigger.get_name())
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
                        self._mout[key][i] = self._mout[key][i] + [False] * extent_array_by
            for station_id in self._station_ids:
                sg = self._mout_groups[station_id]
                if keys[0] in sg:
                    for key in keys:
                        for i in range(len(sg[key])):
                            sg[key][i] = sg[key][i] + [False] * extent_array_by
        # TODO: missing attributes: n_samples, dt, Tnoise, triggers

        shower_ids = []
        for sid in event_buffer:  # loop over all stations (every station is treated independently)
            shower_ids_stn = []
            logger.status(f"adding station {sid} to output file")
            for eid in event_buffer[sid]:
                logger.status(f"adding event {eid} to output file")
                evt = event_buffer[sid][eid]
                for shower in evt.get_sim_showers():
                    logger.status(f"adding shower {shower.get_id()} to output file")
                    if not shower.get_id() in shower_ids:
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

                        self.__add_parameter(self._mout, 'energies', particle[pap.energy], self.__first_event)
                        self.__add_parameter(self._mout, 'flavors', particle[pap.flavor], self.__first_event)
                        self.__add_parameter(self._mout, 'n_interaction', particle[pap.n_interaction], self.__first_event)
                        self.__add_parameter(self._mout, 'interaction_type', particle[pap.interaction_type], self.__first_event)
                        assert(particle[pap.interaction_type]==shower[shp.interaction_type])
                        self.__add_parameter(self._mout, 'inelasticity', particle[pap.inelasticity], self.__first_event)
                        self.__add_parameter(self._mout, 'weights', particle[pap.weight], self.__first_event)

                        self.__add_parameter(self._mout, 'triggered', evt.has_triggered(), self.__first_event)
                        multiple_triggers = []
                        trigger_times = []
                        # set the trigger time to the earliest trigger time of all stations
                        for iT, tname in enumerate(self._mout_attributes['trigger_names']):
                            if evt.has_triggered(tname):
                                multiple_triggers.append(True)
                                tmp_time = None
                                for stn in evt.get_stations():
                                    if stn.has_triggered(tname):
                                        if(tmp_time is None):
                                            tmp_time = stn.get_trigger(tname).get_trigger_time()
                                        else:
                                            tmp_time = min(tmp_time, stn.get_trigger(tname).get_trigger_time())
                                trigger_times.append(tmp_time)
                            else:
                                multiple_triggers.append(False)
                                trigger_times.append(np.nan)
                        self.__add_parameter(self._mout, 'multiple_triggers', multiple_triggers, self.__first_event)
                        self.__add_parameter(self._mout, 'trigger_times', trigger_times, self.__first_event)
                        self.__first_event = False
                # now save station data
                stn = evt.get_station()  # there can only ever be one station per event! If there are more than one station, this line will crash. 
                sg = self._mout_groups[sid]
                self.__add_parameter(sg, 'event_group_ids', evt.get_run_number())
                self.__add_parameter(sg, 'event_ids', evt.get_id())
                # self.__add_parameter(sg, 'maximum_amplitudes', )
                # self.__add_parameter(sg, 'maximum_amplitudes_envelope', )

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
                self.__add_parameter(sg, 'trigger_times_per_event', trigger_times)
                self.__add_parameter(sg, 'triggered_per_event', np.any(multiple_triggers))

                self.__add_parameter(sg, 'triggered', stn.has_triggered())

                for shower in evt.get_sim_showers():
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
                                    for key, value in efield[ep.raytracing_solution].items():
                                        if key not in channel_rt_data:
                                            channel_rt_data[key] = np.zeros((nCh, self._nS)) * np.nan
                                        channel_rt_data[key][iCh, iS] = value
                                    channel_rt_data['launch_vectors'][iCh, iS] = efield[ep.launch_vector]
                                    receive_vector = hp.spherical_to_cartesian(efield[ep.zenith], efield[ep.azimuth])
                                    channel_rt_data['receive_vectors'][iCh, iS] = receive_vector
                                    channel_rt_data['travel_times'][iCh, iS] = efield[ep.nu_vertex_travel_time]
                                    channel_rt_data['travel_distances'][iCh, iS] = efield[ep.nu_vertex_distance]

                                    # only the polarization angle is saved in the electric field object, so we need to
                                    # calculate the vector it to the ground frame in cartesian coordinates.
                                    cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*receive_vector))
                                    polarization_angle = efield[ep.polarization_angle]
                                    if(np.abs(polarization_angle - 90 * units.deg) < 0.0001):
                                        polarization_direction_onsky = np.array([0, 0, 1])
                                    else:
                                        p_phi = np.tan(polarization_angle)**2/(1+np.tan(polarization_angle)**2)
                                        p_theta =  (1 - p_phi**2)**0.5
                                        polarization_direction_onsky = np.array([0, p_theta, p_phi])
                                    polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
                                    channel_rt_data['polarization'][iCh, iS] = polarization_direction_at_antenna

                                    if self._mout_attributes['config']['speedup']['amp_per_ray_solution']:
                                        sim_station = stn.get_sim_station()
                                        uid = (shower.get_id(), channel.get_id(), iS)
                                        if uid in sim_station.get_channel_ids():
                                            sim_channel = sim_station.get_channel((shower.get_id(), channel.get_id(), iS))
                                            channel_rt_data['max_amp_shower_and_ray'][iCh, iS] = sim_channel[chp.maximum_amplitude_envelope]
                                            channel_rt_data['time_shower_and_ray'][iCh, iS] = sim_channel[chp.signal_time]

                        for key, value in channel_rt_data.items():
                            self.__add_parameter(sg, key, value)



                            



                            

        pass

    def end(self):
        import json
        print(self._mout_attributes)
        for key in self._mout_attributes:
            print(key, self._mout_attributes[key])
        # print(json.dumps(self._mout_attributes, sort_keys=True, indent=4))
            
        print(self._mout['shower_ids'])
    def _create_station_output_structure(self, n_showers, n_antennas):
        nS = self._raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions
        sg = {}
        sg['triggered'] = np.zeros(n_showers, dtype=bool)
        # we need the reference to the shower id to be able to find the correct shower in the upper level hdf5 file
        sg['shower_id'] = np.zeros(n_showers, dtype=int) * -1
        sg['event_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        sg['event_group_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        sg['launch_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['receive_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['polarization'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['travel_times'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        sg['travel_distances'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        if config['speedup']['amp_per_ray_solution']:
            sg['max_amp_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
            sg['time_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
        for parameter_entry in self._raytracer.get_output_parameters():
            if parameter_entry['ndim'] == 1:
                sg[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS)) * np.nan
            else:
                sg[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS, parameter_entry['ndim'])) * np.nan
        return sg

    def write_output_file(self, empty=False):
        folder = os.path.dirname(self._outputfilename)
        if not os.path.exists(folder) and folder != '':
            logger.warning(f"output folder {folder} does not exist, creating folder...")
            os.makedirs(folder)
        fout = h5py.File(self._outputfilename, 'w')

        if not empty:
            # here we add the first interaction to the saved events
            # if any of its children triggered

            # Careful! saved should be a copy of the triggered array, and not
            # a reference! saved indicates the interactions to be saved, while
            # triggered should indicate if an interaction has produced a trigger
            saved = np.copy(self._mout['triggered'])
            if 'n_interaction' in self._fin:  # if n_interactions is not specified, there are not parents
                parent_mask = self._fin['n_interaction'] == 1
                for event_id in np.unique(self._fin['event_group_ids']):
                    event_mask = self._fin['event_group_ids'] == event_id
                    if True in self._mout['triggered'][event_mask]:
                        saved[parent_mask & event_mask] = True

            logger.status("start saving events")
            # save data sets
            for (key, value) in self._mout.items():
                fout[key] = value[saved]

            # save all data sets of the station groups
            for (key, value) in self._mout_groups.items():
                sg = fout.create_group("station_{:d}".format(key))
                for (key2, value2) in value.items():
                    sg[key2] = np.array(value2)[np.array(value['triggered'])]

            # save "per event" quantities
            if 'trigger_names' in self._mout_attrs:
                n_triggers = len(self._mout_attrs['trigger_names'])
                for station_id in self._mout_groups:
                    n_events_for_station = len(self._output_triggered_station[station_id])
                    if n_events_for_station > 0:
                        n_channels = self._det.get_number_of_channels(station_id)
                        sg = fout["station_{:d}".format(station_id)]
                        sg['event_group_ids'] = np.array(self._output_event_group_ids[station_id])
                        sg['event_ids'] = np.array(self._output_sub_event_ids[station_id])
                        sg['maximum_amplitudes'] = np.array(self._output_maximum_amplitudes[station_id])
                        sg['maximum_amplitudes_envelope'] = np.array(self._output_maximum_amplitudes_envelope[station_id])
                        sg['triggered_per_event'] = np.array(self._output_triggered_station[station_id])

                        # the multiple triggeres 2d array might have different number of entries per event
                        # because the number of different triggers can increase dynamically
                        # therefore we first create an array with the right size and then fill it
                        tmp = np.zeros((n_events_for_station, n_triggers), dtype=bool)
                        for iE, values in enumerate(self._output_multiple_triggers_station[station_id]):
                            tmp[iE] = values
                        sg['multiple_triggers_per_event'] = tmp
                        tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
                        for iE, values in enumerate(self._output_trigger_times_station[station_id]):
                            tmp_t[iE] = values
                        sg['trigger_times_per_event'] = tmp_t


        # save meta arguments
        for (key, value) in self._mout_attrs.items():
            fout.attrs[key] = value

        if isinstance(self._det, detector.rnog_detector.Detector):
            fout.attrs['detector'] = self._det.export_as_string()
        else:
            with open(self._detectorfile, 'r') as fdet:
                fout.attrs['detector'] = fdet.read()

        if not empty:
            # save antenna position separately to hdf5 output
            for station_id in self._mout_groups:
                n_channels = self._det.get_number_of_channels(station_id)
                positions = np.zeros((n_channels, 3))
                for channel_id in range(n_channels):
                    positions[channel_id] = self._det.get_relative_position(station_id, channel_id) + self._det.get_absolute_position(station_id)
                fout["station_{:d}".format(station_id)].attrs['antenna_positions'] = positions
                fout["station_{:d}".format(station_id)].attrs['Vrms'] = list(self._Vrms_per_channel[station_id].values())
                fout["station_{:d}".format(station_id)].attrs['bandwidth'] = list(self._integrated_channel_response[station_id].values())

            fout.attrs.create("Tnoise", self._noise_temp, dtype=float)
            fout.attrs.create("Vrms", self._Vrms, dtype=float)
            fout.attrs.create("dt", self._dt, dtype=float)
            fout.attrs.create("bandwidth", self._bandwidth, dtype=float)
            fout.attrs['n_samples'] = self._n_samples
        fout.attrs['config'] = yaml.dump(config)

        # save NuRadioMC and NuRadioReco versions
        from NuRadioReco.utilities import version
        import NuRadioMC
        fout.attrs['NuRadioMC_version'] = NuRadioMC.__version__
        fout.attrs['NuRadioMC_version_hash'] = version.get_NuRadioMC_commit_hash()

        if not empty:
            # now we also save all input parameters back into the out file
            for key in self._fin.keys():
                if key.startswith("station_"):
                    continue
                if not key in fout.keys():  # only save data sets that havn't been recomputed and saved already
                    if np.array(self._fin[key]).dtype.char == 'U':
                        fout[key] = np.array(self._fin[key], dtype=h5py.string_dtype(encoding='utf-8'))[saved]

                    else:
                        fout[key] = np.array(self._fin[key])[saved]

        for key in self._fin_attrs.keys():
            if not key in fout.attrs.keys():  # only save atrributes sets that havn't been recomputed and saved already
                if key not in ["trigger_names", "Tnoise", "Vrms", "bandwidth", "n_samples", "dt", "detector", "config"]:  # don't write trigger names from input to output file, this will lead to problems with incompatible trigger names when merging output files
                    fout.attrs[key] = self._fin_attrs[key]
        fout.close()