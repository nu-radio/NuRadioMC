from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.utilities.earth_attenuation import get_weight
from NuRadioMC.SignalProp import propagation
import h5py
# import detector simulation modules
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
import NuRadioReco.framework.particle
from NuRadioReco.detector import antennapattern
# parameters describing simulated Monte Carlo particles
from NuRadioReco.framework.parameters import particleParameters as simp
# parameters set in the event generator
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
import datetime
import logging
from six import iteritems
import yaml
import os
from NuRadioMC.utilities.Veff import remove_duplicate_triggers
import NuRadioMC.simulation.channel_efield_simulator
import NuRadioMC.simulation.shower_simulator
import NuRadioMC.simulation.station_simulator
import NuRadioMC.simulation.hardware_response_simulator
import NuRadioMC.simulation.output_writer_hdf5
import NuRadioMC.simulation.output_writer_nur
import NuRadioMC.simulation.time_logger




STATUS = 31
class NuRadioMCLogger(logging.Logger):

    def status(self, msg, *args, **kwargs):
        if self.isEnabledFor(STATUS):
            self._log(STATUS, msg, args, **kwargs)

logging.setLoggerClass(NuRadioMCLogger)
logging.addLevelName(STATUS, 'STATUS')
logger = logging.getLogger("NuRadioMC")
assert isinstance(logger, NuRadioMCLogger)


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
        for k, v in iteritems(default):
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k], v)
    return user




class simulation():

    def __init__(self, inputfilename,
                 outputfilename,
                 detectorfile,
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1),
                 config_file=None,
                 log_level=logging.WARNING,
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
        if 'write_mode' in kwargs.keys():
            logger.warning('Parameter write_mode is deprecated. Define the output format in the config file instead.')
        self._log_level_ray_propagation = log_level_propagation
        config_file_default = os.path.join(os.path.dirname(__file__), 'config_default.yaml')
        logger.status('reading default config from {}'.format(config_file_default))
        with open(config_file_default, 'r') as ymlfile:
            self.__cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if config_file is not None:
            logger.status('reading local config overrides from {}'.format(config_file))
            with open(config_file, 'r') as ymlfile:
                local_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
                new_cfg = merge_config(local_config, self.__cfg)
                self.__cfg = new_cfg

        if self.__cfg['seed'] is None:
            # the config seeting None means a random seed. To have the simulation be reproducable, we generate a new
            # random seed once and save this seed to the config setting. If the simulation is rerun, we can get
            # the same random sequence.
            self.__cfg['seed'] = np.random.randint(0, 2 ** 32 - 1)

        self._outputfilename = outputfilename
        if os.path.exists(self._outputfilename):
            msg = f"hdf5 output file {self._outputfilename} already exists"
            if file_overwrite == False:
                logger.error(msg)
                raise FileExistsError(msg)
            else:
                logger.warning(msg)
        self._detectorfile = detectorfile
        self._outputfilenameNuRadioReco = outputfilenameNuRadioReco
        self._debug = debug
        self._evt_time = evt_time
        self._write_detector = write_detector
        logger.status("setting event time to {}".format(evt_time))
        self.__event_group_list = event_list
        
        # initialize propagation module
        self.__prop = NuRadioMC.SignalProp.propagation.get_propagation_module(self.__cfg['propagation']['module'])
        self.__time_logger = NuRadioMC.simulation.time_logger.timeLogger(logger)
        if self.__cfg['propagation']['ice_model'] == "custom":
            if ice_model is None:
                logger.error("ice model is set to 'custom' in config file but no custom ice model is provided.")
                raise AttributeError("ice model is set to 'custom' in config file but no custom ice model is provided.")
            self._ice = ice_model
        else:
            self._ice = NuRadioMC.utilities.medium.get_ice_model(self.__cfg['propagation']['ice_model'])

        # read in detector positions
        logger.status("Detectorfile {}".format(os.path.abspath(self._detectorfile)))
        self._det = None
        if default_detector_station is not None:
            logger.warning(
                'Deprecation warning: Passing the default detector station is deprecated. Default stations and default'
                'channel should be specified in the detector description directly.'
            )
            logger.status(f"Default detector station provided (station {default_detector_station}) -> Using generic detector")
            self._det = NuRadioReco.detector.generic_detector.GenericDetector(json_filename=self._detectorfile, default_station=default_detector_station,
                                                 default_channel=default_detector_channel, antenna_by_depth=False)
        else:
            self._det = NuRadioReco.detector.detector.Detector(json_filename=self._detectorfile, antenna_by_depth=False)

        self._det.update(evt_time)

        self._station_ids = self._det.get_station_ids()
        
        # print noise information
        logger.status("running with noise {}".format(bool(self.__cfg['noise'])))
        logger.status("setting signal to zero {}".format(bool(self.__cfg['signal']['zerosignal'])))
        if bool(self.__cfg['propagation']['focusing']):
            logger.status("simulating signal amplification due to focusing of ray paths in the firn.")

        # read sampling rate from config (this sampling rate will be used internally)

        if isinstance(inputfilename, str):
            logger.status(f"reading input from {inputfilename}")
            self._inputfilename = inputfilename
            self.__read_input_hdf5()  # we read in the full input file into memory at the beginning to limit io to the beginning and end of the run
        else:
            logger.status("getting input on-the-fly")
            self._inputfilename = "on-the-fly"
            self.__fin = inputfilename[0]
            self.__fin_attrs = inputfilename[1]
            self.__fin_stations = {}
        # store all relevant attributes of the input file in a dictionary
        self._generator_info = {}
        self._particle_mode = "simulation_mode" not in self.__fin_attrs or self.__fin_attrs['simulation_mode'] != "emitter"

        for enum_entry in genattrs:
            if enum_entry.name in self.__fin_attrs:
                self._generator_info[enum_entry] = self.__fin_attrs[enum_entry.name]

        # check if the input file contains events, if not save empty output file (for book keeping) and terminate simulation
        if len(self.__fin['xx']) == 0:
            logger.status(f"input file {self._inputfilename} is empty")
            return



        self._distance_cut_polynomial = None
        if self.__cfg['speedup']['distance_cut']:
            coef = self.__cfg['speedup']['distance_cut_coefficients']
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
        if len(self.__fin['xx']) == 0:
            logger.status(f"writing empty hdf5 output file")
            self._write_output_file(empty=True)
            logger.status(f"terminating simulation")
            return 0
        logger.status(f"Starting NuRadioMC simulation")
        """
        TODO: Allow running stations with different numbers of channels
        """
        self.__channel_ids = self._det.get_channel_ids(self._det.get_station_ids()[0])
        sampling_rate_detector = self._det.get_sampling_frequency(self._det.get_station_ids()[0], self.__channel_ids[0])
        n_samples = self._det.get_number_of_samples(self._det.get_station_ids()[0], self.__channel_ids[0]) / sampling_rate_detector * self.__cfg['sampling_rate'] * units.GHz
        n_samples = int(np.ceil(n_samples / 2.) * 2)
        unique_event_group_ids = np.unique(self.__fin['event_group_ids'])
        self._n_showers = len(self.__fin['event_group_ids'])
        self._shower_ids = np.array(self.__fin['shower_ids'])
        self.__particle_mode = "simulation_mode" not in self.__fin_attrs or self.__fin_attrs['simulation_mode'] != "emitter"
        self.__raytracer = self.__prop(
            self._ice, self.__cfg['propagation']['attenuation_model'],
            log_level=self._log_level_ray_propagation,
            n_frequencies_integration=int(self.__cfg['propagation']['n_freq']),
            n_reflections=int(self.__cfg['propagation']['n_reflections']),
            config=self.__cfg,
            detector=self._det
        )
        self.__channel_simulator = NuRadioMC.simulation.channel_efield_simulator.channelEfieldSimulator(
            self._det,
            self.__raytracer,
            self.__channel_ids,
            self.__cfg,
            self.__fin,
            self.__fin_attrs,
            self._ice,
            n_samples,
            self.__time_logger
        )
        self.__shower_simulator = NuRadioMC.simulation.shower_simulator.showerSimulator(
            self._det,
            self.__channel_ids,
            self.__cfg,
            self.__fin,
            self.__fin_attrs,
            self.__channel_simulator,
            self.__raytracer.get_number_of_raytracing_solutions()
        )
        self.__hardware_response_simulator = NuRadioMC.simulation.hardware_response_simulator.hardwareResponseSimulator(
            self._det,
            self.__cfg,
            self._station_ids,
            self.__fin,
            self.__fin_attrs,
            self._detector_simulation_trigger,
            self._detector_simulation_filter_amp,
            self.__raytracer,
            self._evt_time,
            self.__time_logger
        )
        self._Vrms = self.__hardware_response_simulator.get_noise_vrms()
        self._Vrms_per_channel = self.__hardware_response_simulator.get_noise_vrms_per_channel()
        efield_v_rms_per_channel = self.__hardware_response_simulator.get_efield_v_rms_per_channel()
        # check if the same detector was simulated before (then we can save the ray tracing part)
        pre_simulated = self.__check_if_was_pre_simulated()

        self.__station_simulator = NuRadioMC.simulation.station_simulator.stationSimulator(
            self._det,
            self.__channel_ids,
            self.__cfg,
            self.__fin,
            self.__fin_attrs,
            self._fin_stations,
            self.__shower_simulator,
            self.__raytracer,
            pre_simulated,
            efield_v_rms_per_channel
        )
        self.__time_logger.reset_times(['ray tracing', 'askaryan', 'detector simulation', ])


        # calculate bary centers of station
        self._station_barycenter = self.__calculate_station_barycenter()
        self.__output_writer_hdf5 = NuRadioMC.simulation.output_writer_hdf5.outputWriterHDF5(
            self._outputfilename,
            self.__cfg,
            self._det,
            self._station_ids,
            self.__raytracer,
            self.__hardware_response_simulator,
            self._inputfilename,
            self._particle_mode
        )
        if self._outputfilenameNuRadioReco is None:
            self.__output_writer_nur = None
        else:
            self.__output_writer_nur = NuRadioMC.simulation.output_writer_nur.outputWriterNur(
                self._outputfilenameNuRadioReco,
                self._write_detector,
                self._det
            )
        # loop over event groups
        for i_event_group_id, event_group_id in enumerate(unique_event_group_ids):
            if i_event_group_id > 10e99:
                print('breaking')
                break
            logger.debug(f"simulating event group id {event_group_id}")
            if self.__event_group_list is not None and event_group_id not in self.__event_group_list:
                logger.debug(f"skipping event group {event_group_id} because it is not in the event group list provided to the __init__ function")
                continue
            event_indices = np.atleast_1d(np.squeeze(np.argwhere(self.__fin['event_group_ids'] == event_group_id)))

            # the weight calculation is independent of the station, so we do this calculation only once
            # the weight also depends just on the "mother" particle, i.e. the incident neutrino which determines
            # the propability of arriving at our simulation volume. All subsequent showers have the same weight. So
            # we calculate it just once and save it to all subshowers.
            self.__primary_index = event_indices[0]
            # determine if a particle (neutrinos, or a secondary interaction of a neutrino, or surfaec muons) is simulated
            if self._particle_mode:
                event_group_weight = self.__calculate_particle_weights(event_indices)
            else:
                event_group_weight = 1
            self.__output_writer_hdf5.store_event_group_weight(
                event_group_weight,
                event_indices
            )
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if event_group_weight < self.__cfg['speedup']['minimum_weight_cut']:
                logger.debug("neutrino weight is smaller than {}, skipping event".format(self.__cfg['speedup']['minimum_weight_cut']))
                continue



            # these quantities get computed to apply the distance cut as a function of shower energies
            # the shower energies of closeby showers will be added as they can constructively interfere
            if 'shower_energies' in self.__fin.keys():
                shower_energies = np.array(self.__fin['shower_energies'])[event_indices]
            else:
                shower_energies = np.zeros(event_indices.shape)
            vertex_positions = np.array([np.array(self.__fin['xx'])[event_indices],
                                         np.array(self.__fin['yy'])[event_indices],
                                         np.array(self.__fin['zz'])[event_indices]])
            self.__station_simulator.set_event_group(
                i_event_group_id,
                event_group_id,
                event_indices,
                self.__particle_mode
            )
            self.__hardware_response_simulator.set_event_group(
                event_group_id
            )
            # loop over all stations (each station is treated independently)
            for iSt, self._station_id in enumerate(self._station_ids):
                logger.debug(f"simulating station {self._station_id}")

                # perform a quick cut to reject event group completely if no shower is close enough to the station
                if not self.__distance_cut_station(
                        vertex_positions,
                        shower_energies,
                        self._station_barycenter[iSt]
                ):
                    continue

                ray_tracing_performed = False
                if 'station_{:d}'.format(self._station_id) in self._fin_stations:
                    ray_tracing_performed = (self.__raytracer.get_output_parameters()[0]['name'] in self._fin_stations['station_{:d}'.format(self._station_id)]) and self.__was_pre_simulated
                self._dummy_event = NuRadioReco.framework.event.Event(0, 0) # a dummy event object, which does nothing but is needed because some modules require an event to be passed
                sim_showers = {}
                station_output, efield_array, is_candidate_station = self.__station_simulator.simulate_station(self._station_id)
                if not is_candidate_station:
                    continue
                event_objects, station_objects, sub_event_shower_ids, station_has_triggered, hardware_response_output = self.__hardware_response_simulator.simulate_detector_response(
                    self._station_id,
                    efield_array,
                    event_indices
                )
                if np.any(station_has_triggered):
                    trigger_indices = np.where(station_has_triggered)[0]
                    # if several sub-events have triggered, most data is only saved for the
                    # last event in the group
                    self.__output_writer_hdf5.add_station(
                        self._station_id,
                        event_objects,
                        station_objects,
                        station_output,
                        hardware_response_output,
                        event_group_id,
                        sub_event_shower_ids,
                        station_has_triggered
                    )
                    self.__output_writer_hdf5.add_station_per_shower(
                        self._station_id,
                        event_objects,
                        station_objects,
                        station_output,
                        hardware_response_output,
                        event_group_id,
                        sub_event_shower_ids
                    )
                    if self.__output_writer_nur is not None:
                        self.__output_writer_nur.save_event(
                            event_objects
                        )
                    self.__time_logger.show_time(len(unique_event_group_ids), i_event_group_id)
        # Create trigger structures if there are no triggering events.
        # This is done to ensure that files with no triggering n_events
        # merge properly.
#         self._create_empty_multiple_triggers()

        # save simulation run in hdf5 format (only triggered events)
        # self._write_output_file()
        self.__output_writer_hdf5.save_output()
        if self.__output_writer_nur is not None:
            self.__output_writer_nur.end()

        try:
            self.calculate_Veff()
        except:
            logger.error("error in calculating effective volume")

        
    def _calculate_emitter_output(self):
        pass


    def _get_channel_index(self, channel_id):
        index = self.__channel_ids.index(channel_id)
        if index < 0:
            raise ValueError('Channel with ID {} not found in station {} of detector description!'.format(channel_id, self._station_id))
        return index
    def _is_simulate_noise(self):
        """
        returns True if noise should be added
        """
        return bool(self.__cfg['noise'])

    def _is_in_fiducial_volume(self):
        """
        checks wether a vertex is in the fiducial volume

        if the fiducial volume is not specified in the input file, True is returned (this is required for the simulation
        of pulser calibration measuremens)
        """
        tt = ['fiducial_rmin', 'fiducial_rmax', 'fiducial_zmin', 'fiducial_zmax']
        has_fiducial = True
        for t in tt:
            if not t in self.__fin_attrs:
                has_fiducial = False
        if not has_fiducial:
            return True

        r = (self._shower_vertex[0] ** 2 + self._shower_vertex[1] ** 2) ** 0.5
        if r >= self.__fin_attrs['fiducial_rmin'] and r <= self.__fin_attrs['fiducial_rmax']:
            if self._shower_vertex[2] >= self.__fin_attrs['fiducial_zmin'] and self._shower_vertex[2] <= self.__fin_attrs['fiducial_zmax']:
                return True
        return False



    def get_Vrms(self):
        return self._Vrms

    def get_sampling_rate(self):
        return self.__cfg['sampling_rate'] * units.GHz


    def __check_if_was_pre_simulated(self):
        """
        checks if the same detector was simulated before (then we can save the ray tracing part)
        """
        self.__was_pre_simulated = False
        if 'detector' in self.__fin_attrs:
            with open(self._detectorfile, 'r') as fdet:
                if fdet.read() == self.__fin_attrs['detector']:
                    self.__was_pre_simulated = True
                    logger.debug("the simulation was already performed with the same detector")
        return self.__was_pre_simulated



    def calculate_Veff(self):
        # calculate effective
        trigger_status = self.__output_writer_hdf5.get_trigger_status()
        triggered = remove_duplicate_triggers(trigger_status, self.__fin['event_group_ids'])
        n_triggered = np.sum(triggered)
        weights = self.__output_writer_hdf5.get_weights()
        n_triggered_weighted = np.sum(weights[triggered])
        n_events = self.__fin_attrs['n_events']
        logger.status(f'fraction of triggered events = {n_triggered:.0f}/{n_events:.0f} = {n_triggered / self._n_showers:.3f} (sum of weights = {n_triggered_weighted:.2f})')
        V = self.__fin_attrs['volume']
        Veff = V * n_triggered_weighted / n_events
        logger.status(f"Veff = {Veff / units.km ** 3:.4g} km^3, Veffsr = {Veff * 4 * np.pi/units.km**3:.4g} km^3 sr")


    def __calculate_station_barycenter(self):
        station_barycenter = np.zeros((len(self._station_ids), 3))
        for iSt, station_id in enumerate(self._station_ids):
            pos = []
            for channel_id in self._det.get_channel_ids(station_id):
                pos.append(self._det.get_relative_position(station_id, channel_id))
            station_barycenter[iSt] = np.mean(np.array(pos), axis=0) + self._det.get_absolute_position(station_id)
        return station_barycenter

    def __calculate_particle_weights(
            self,
            evt_indices
    ):
        primary = self.__read_input_particle_properties(self.__primary_index)  # this sets the self.input_particle for self.__primary_index
        # calculate the weight for the primary particle
        if self.__cfg['weights']['weight_mode'] == "existing":
            if "weights" in self.__fin:
                self._mout['weights'] = self.__fin["weights"]
            else:
                logger.error(
                    "config file specifies to use weights from the input hdf5 file but the input file does not contain this information.")
        elif self.__cfg['weights']['weight_mode'] is None:
            primary[simp.weight] = 1.
        else:
            primary[simp.weight] = get_weight(primary[simp.zenith],
                                                   primary[simp.energy],
                                                   primary[simp.flavor],
                                                   mode=self.__cfg['weights']['weight_mode'],
                                                   cross_section_type=self.__cfg['weights']['cross_section_type'],
                                                   vertex_position=primary[simp.vertex],
                                                   phi_nu=primary[simp.azimuth])
        # all entries for the event for this primary get the calculated primary's weight
        return primary[simp.weight]


    def __distance_cut_station(
            self,
            vertex_positions,
            shower_energies,
            station_barycenter
    ):
        """
        Checks if the station fulfills the distance cut criterium.
        Returns True if the station barycenter is within the
        maximum distance (and should therefore be simulated)
        and False otherwise.
        
        Parameters
        ----------
        vertex_positions: array of float
            Positions of all sub-showers of the event
        shower_energies: array of float
            energies of all sub-showers of the event
        Returns
        -------

        """
        if not self.__cfg['speedup']['distance_cut']:
            return True
        vertex_distances_to_station = np.linalg.norm(vertex_positions.T - station_barycenter, axis=1)
        distance_cut = self._get_distance_cut(np.sum(
            shower_energies)) + 100 * units.m  # 100m safety margin is added to account for extent of station around bary center.
        if vertex_distances_to_station.min() > distance_cut:
            logger.debug(
                f"skipping station {self._station_id} because minimal distance {vertex_distances_to_station.min() / units.km:.1f}km > {distance_cut / units.km:.1f}km (shower energy = {shower_energies.max():.2g}eV) bary center of station {station_barycenter}")
        return vertex_distances_to_station.min() <= distance_cut

    def __read_input_hdf5(self):
        """
        reads input file into memory
        """
        fin = h5py.File(self._inputfilename, 'r')
        self.__fin = {}
        self._fin_stations = {}
        self.__fin_attrs = {}
        for key, value in iteritems(fin):
            if isinstance(value, h5py._hl.group.Group):
                self._fin_stations[key] = {}
                for key2, value2 in iteritems(value):
                    self._fin_stations[key][key2] = np.array(value2)
            else:
                if len(value) and type(value[0]) == bytes:
                    self.__fin[key] = np.array(value).astype('U')
                else:
                    self.__fin[key] = np.array(value)
        for key, value in iteritems(fin.attrs):
            self.__fin_attrs[key] = value

        fin.close()

    def __read_input_particle_properties(self, idx=None):
        if idx is None:
            idx = self.__primary_index
        
        input_particle = NuRadioReco.framework.particle.Particle(0)
        input_particle[simp.flavor] = self.__fin['flavors'][idx]
        input_particle[simp.energy] = self.__fin['energies'][idx]
        input_particle[simp.interaction_type] = self.__fin['interaction_type'][idx]
        input_particle[simp.inelasticity] = self.__fin['inelasticity'][idx]
        input_particle[simp.vertex] = np.array([self.__fin['xx'][idx],
                                                     self.__fin['yy'][idx],
                                                     self.__fin['zz'][idx]])
        input_particle[simp.zenith] = self.__fin['zeniths'][idx]
        input_particle[simp.azimuth] = self.__fin['azimuths'][idx]
        input_particle[simp.inelasticity] = self.__fin['inelasticity'][idx]
        input_particle[simp.n_interaction] = self.__fin['n_interaction'][idx]
        if self.__fin['n_interaction'][idx] <= 1:
            # parents before the neutrino and outgoing daughters without shower are currently not
            # simulated. The parent_id is therefore at the moment only rudimentarily populated.
            input_particle[simp.parent_id] = None  # primary does not have a parent

        input_particle[simp.vertex_time] = 0
        if 'vertex_times' in self.__fin:
            input_particle[simp.vertex_time] = self.__fin['vertex_times'][idx]
        return input_particle