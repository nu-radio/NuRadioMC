import numpy as np
import h5py
import yaml
import os
import collections
from NuRadioReco.framework.parameters import channelParameters as chp
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
            station_id,
            fin, 
            fin_attrs
    ):
        self._mout = {}
        self._mout_attrs = {}
        self._mout_groups = {}
        self._fin = fin
        self._fin_attrs = fin_attrs

    
    def add_event_group(event_buffer):
        """
        Add an event group to the output file
        """
        
        pass



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