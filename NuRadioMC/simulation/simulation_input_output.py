import h5py
import numpy as np
import os.path
from six import iteritems
import six
import logging
import yaml
import time
import NuRadioReco.framework.particle
from NuRadioReco.framework.parameters import particleParameters as simp
from NuRadioReco.framework.parameters import showerParameters as shp


class simulation_input_output():

    def _read_input_hdf5(self):
        """
        reads input file into memory
        """
        fin = h5py.File(self._inputfilename, 'r')
        self._fin = {}
        self._fin_stations = {}
        self._fin_attrs = {}
        for key, value in iteritems(fin):
            if isinstance(value, h5py._hl.group.Group):
                self._fin_stations[key] = {}
                for key2, value2 in iteritems(value):
                    self._fin_stations[key][key2] = np.array(value2)
            else:
                if len(value) and type(value[0]) == bytes:
                    self._fin[key] = np.array(value).astype('U')
                else:
                    self._fin[key] = np.array(value)
        for key, value in iteritems(fin.attrs):
            self._fin_attrs[key] = value

        fin.close()

    def _read_input_particle_properties(self, idx=None):
        if idx is None:
            idx = self._primary_index
        self._event_group_id = self._fin['event_group_ids'][idx]

        self.input_particle = NuRadioReco.framework.particle.Particle(0)
        self.input_particle[simp.flavor] = self._fin['flavors'][idx]
        self.input_particle[simp.energy] = self._fin['energies'][idx]
        self.input_particle[simp.interaction_type] = self._fin['interaction_type'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.vertex] = np.array([self._fin['xx'][idx],
                                                     self._fin['yy'][idx],
                                                     self._fin['zz'][idx]])
        self.input_particle[simp.zenith] = self._fin['zeniths'][idx]
        self.input_particle[simp.azimuth] = self._fin['azimuths'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.n_interaction] = self._fin['n_interaction'][idx]
        if self._fin['n_interaction'][idx] <= 1:
            # parents before the neutrino and outgoing daughters without shower are currently not
            # simulated. The parent_id is therefore at the moment only rudimentarily populated.
            self.input_particle[simp.parent_id] = None  # primary does not have a parent

        self.input_particle[simp.vertex_time] = 0
        if 'vertex_times' in self._fin:
            self.input_particle[simp.vertex_time] = self._fin['vertex_times'][idx]

    def _read_input_shower_properties(self):
        """ read in the properties of the shower with index _shower_index from input """
        self._event_group_id = self._fin['event_group_ids'][self._shower_index]

        self._shower_vertex = np.array([self._fin['xx'][self._shower_index],
                                        self._fin['yy'][self._shower_index],
                                        self._fin['zz'][self._shower_index]])

        self._vertex_time = 0
        if 'vertex_times' in self._fin:
            self._vertex_time = self._fin['vertex_times'][self._shower_index]

    def _save_triggers_to_hdf5(self, event, station, output_data, local_shower_index, global_shower_index):
        extend_array = self._create_trigger_structures(station)
        # now we also need to create the trigger structure also in the sg (station group) dictionary that contains
        # the information fo the current station and event group
        n_showers = output_data['launch_vectors'].shape[0]
        if 'multiple_triggers' not in output_data:
            output_data['multiple_triggers'] = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            output_data['trigger_times'] = np.nan * np.zeros_like(output_data['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            nx, ny = output_data['multiple_triggers'].shape
            tmp[:, 0:ny] = output_data['multiple_triggers']
            output_data['multiple_triggers'] = tmp
            # repeat for trigger times
            tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
            tmp_t[:, :ny] = output_data['trigger_times']
            output_data['trigger_times'] = tmp_t
        self._output_event_group_ids[self._station_id].append(event.get_run_number())
        self._output_sub_event_ids[self._station_id].append(event.get_id())
        multiple_triggers = np.zeros(len(self._mout_attrs['trigger_names']), dtype=np.bool)
        trigger_times = np.nan * np.zeros_like(multiple_triggers)
        for iT, trigger_name in enumerate(self._mout_attrs['trigger_names']):
            if station.has_trigger(trigger_name):
                multiple_triggers[iT] = station.get_trigger(trigger_name).has_triggered()
                trigger_times[iT] = station.get_trigger(trigger_name).get_trigger_time()
                for iSh in local_shower_index:  # now save trigger information per shower of the current station
                    output_data['multiple_triggers'][iSh][iT] = station.get_trigger(trigger_name).has_triggered()
                    output_data['trigger_times'][iSh][iT] = trigger_times[iT]
        for iSh, iSh2 in zip(local_shower_index, global_shower_index):  # now save trigger information per shower of the current station
            output_data['triggered'][iSh] = np.any(output_data['multiple_triggers'][iSh])
            self._mout['triggered'][iSh2] |= output_data['triggered'][iSh]
            self._mout['multiple_triggers'][iSh2] |= output_data['multiple_triggers'][iSh]
            self._mout['trigger_times'][iSh2] = np.fmin(self._mout['trigger_times'][iSh2], output_data['trigger_times'][iSh])
        output_data['event_id_per_shower'][local_shower_index] = event.get_id()
        output_data['event_group_id_per_shower'][local_shower_index] = event.get_run_number()
        self._output_multiple_triggers_station[self._station_id].append(multiple_triggers)
        self._output_trigger_times_station[self._station_id].append(trigger_times)
        self._output_triggered_station[self._station_id].append(np.any(multiple_triggers))

    def _create_empty_multiple_triggers(self):
        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = np.array([])
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, 1), dtype=np.bool)
            for station_id in self._station_ids:
                n_showers = self._mout_groups[station_id]['launch_vectors'].shape[0]
                self._mout_groups[station_id]['multiple_triggers'] = np.zeros((n_showers, 1), dtype=np.bool)
                self._mout_groups[station_id]['triggered'] = np.zeros(n_showers, dtype=np.bool)

    def _create_trigger_structures(
            self,
            station
    ):

        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = []
        extend_array = False
        for trigger in six.itervalues(station.get_triggers()):
            if trigger.get_name() not in self._mout_attrs['trigger_names']:
                self._mout_attrs['trigger_names'].append((trigger.get_name()))
                extend_array = True
        # the 'multiple_triggers' output array is not initialized in the constructor because the number of
        # simulated triggers is unknown at the beginning. So we check if the key already exists and if not,
        # we first create this data structure
        if 'multiple_triggers' not in self._mout:
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            self._mout['trigger_times'] = np.nan * np.zeros_like(self._mout['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            nx, ny = self._mout['multiple_triggers'].shape
            tmp[:, 0:ny] = self._mout['multiple_triggers']
            self._mout['multiple_triggers'] = tmp
        return extend_array

    def _create_event_structure(
            self,
            iEvent,
            indices,
            channel_identifiers, sim_showers
    ):
        evt = NuRadioReco.framework.event.Event(self._event_group_id, iEvent)  # create new event

        if self._particle_mode:
            # add MC particles that belong to this (sub) event to event structure
            # add only primary for now, since full interaction chain is not typically in the input hdf5s
            evt.add_particle(self.primary)
        # copy over generator information from temporary event to event
        evt._generator_info = self._generator_info

        new_station = NuRadioReco.framework.station.Station(self._station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
        sim_station.set_is_neutrino()
        shower_ids_of_sub_event = []
        for iCh in indices:
            ch_uid = channel_identifiers[iCh]
            shower_id = ch_uid[1]
            if shower_id not in shower_ids_of_sub_event:
                shower_ids_of_sub_event.append(shower_id)
            sim_station.add_channel(self._station.get_sim_station().get_channel(ch_uid))
            efield_uid = ([ch_uid[0]], ch_uid[1], ch_uid[
                2])  # the efield unique identifier has as first parameter an array of the channels it is valid for
            for efield in self._station.get_sim_station().get_electric_fields():
                if efield.get_unique_identifier() == efield_uid:
                    sim_station.add_electric_field(efield)
        if self._particle_mode:
            # add showers that contribute to this (sub) event to event structure
            for shower_id in shower_ids_of_sub_event:
                evt.add_sim_shower(sim_showers[str(shower_id)])
        new_station.set_sim_station(sim_station)
        new_station.set_station_time(self._evt_time)
        evt.set_station(new_station)
        return evt, new_station, shower_ids_of_sub_event


    def _write_nur_file(
            self,
            event,
            station
    ):
        # downsample traces to detector sampling rate to save file size
        self._channelResampler.run(event, station, self._det, sampling_rate=self._sampling_rate_detector)
        self._channelResampler.run(event, station.get_sim_station(), self._det,
                                   sampling_rate=self._sampling_rate_detector)
        self._electricFieldResampler.run(event, station.get_sim_station(), self._det,
                                         sampling_rate=self._sampling_rate_detector)

        output_mode = {'Channels': self._cfg['output']['channel_traces'],
                       'ElectricFields': self._cfg['output']['electric_field_traces'],
                       'SimChannels': self._cfg['output']['sim_channel_traces'],
                       'SimElectricFields': self._cfg['output']['sim_electric_field_traces']}
        if self._write_detector:
            self._eventWriter.run(event, self._det, mode=output_mode)
        else:
            self._eventWriter.run(event, mode=output_mode)

    def _write_progress_output(
            self,
            iCounter,
            i_event_group_id,
            unique_event_group_ids
    ):
        n_shower_station = len(self._station_ids) * self._n_showers
        eta = NuRadioMC.simulation.simulation_base.pretty_time_delta((time.time() - self._t_start) * (n_shower_station - iCounter) / iCounter)
        total_time_sum = self._input_time + self._rayTracingTime + self._detSimTime + self._outputTime + self._weightTime + self._distance_cut_time  # askaryan time is part of the ray tracing time, so it is not counted here.
        total_time = time.time() - self._t_start
        if total_time > 0:
            logger.status(
                "processing event group {}/{} and shower {}/{} ({} showers triggered) = {:.1f}%, ETA {}, time consumption: ray tracing = {:.0f}%, askaryan = {:.0f}%, detector simulation = {:.0f}% reading input = {:.0f}%, calculating weights = {:.0f}%, distance cut {:.0f}%, unaccounted = {:.0f}% ".format(
                    i_event_group_id,
                    len(unique_event_group_ids),
                    iCounter,
                    n_shower_station,
                    np.sum(self._mout['triggered']),
                    100. * iCounter / n_shower_station,
                    eta,
                    100. * (self._rayTracingTime - self._askaryan_time) / total_time,
                    100. * self._askaryan_time / total_time,
                    100. * self._detSimTime / total_time,
                    100. * self._input_time / total_time,
                    100. * self._weightTime / total_time,
                    100 * self._distance_cut_time / total_time,
                    100 * (total_time - total_time_sum) / total_time))