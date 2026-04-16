import numpy as np
import datetime
from radiotools import helper as hp

from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation as sim
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.channelAddCableDelay

channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

class ShowerSimulator():
    """
    Class to simulate voltage traces for a specific shower(s) with user-defined vertex,
    energy, direction, etc. The Askaryan emission, propagation effects and detector response
    are simulated consistently with simulation.py, but no noise is added and triggers aren't
    run. This is, e.g., useful for studies where only the MC true neutrino signal is of
    interrrest, and it can act as a pure neutrino signal model, which is needed in forward-folding
    reconstruction.

    Parameters
    ----------
    config_file: string
        path to config file
    detectorfile: string
        path to the json file containing the detector description
    det: detector object
        Pass a detector class object
    station_id: int
        Station in the detector description to run the simulation for
    use_channels: list(int)
        Channels in the station to run the simulation for. If None, all 
        channels in the station will be used.
    reference_channel: int
        Channel to clalculate reference time for if no trace_start_times are provided. The pulse should
        appear pre_pulse_time into the trace for this channel.
    evt_time: datetime object
        The time of the simulated event, default 1/1/2018
    detector_simulation_filter_amp: function
        Function to use as the _detector_simulation_filter_amp in the simulation class, e.g,
        hardware response and/or bandpass filter of the detector. Currently has to be provided.
    add_cable_delay: bool
        Whether to add the cable delay of the detector to the simulated traces, default True.
    pre_pulse_time: float
        Time of the readout pulse in the reference channel relative to the start of the trace,
        if vertex_time is 0, default 200 ns. Ignored if trace_start_times are provided.

    Methods
    -------
    simulate_showers(showers, trace_start_times=None)
        Simulate the traces for a list of shower objects.

    simulate_single_shower(energy, zenith, azimuth, vertex, vertex_time, type, charge_excess_profile_id=1, trace_start_times=None)
        Simulate the traces for a single shower with the given parameters.

    simulate_single_shower_spherical_vertex_coordinates(energy, zenith, azimuth, vertex_r, vertex_theta, vertex_phi, vertex_time, type, charge_excess_profile_id=1, trace_start_times=None)
        Simulate the traces for a single shower with the given parameters, where the vertex position
        is given in spherical coordinates instead of Cartesian coordinates.
    """

    def __init__(
            self,
            config_file,
            detectorfile=None,
            det=None,
            station_id=11,
            use_channels=None,
            reference_channel=0,
            evt_time=datetime.datetime(2018, 1, 1),
            detector_simulation_filter_amp=None,
            add_cable_delay=True,
            pre_pulse_time=200*units.ns,
        ):

        if detectorfile is None and det is None:
            raise ValueError("Either a detector file or a detector object needs to be provided.")

        self.station_id = station_id
        self.reference_channel = reference_channel

        # Initialize the relavant modules using the simulation class:
        dummy_input_file = [{"xx": [None, None]}, [None, None]]
        class mySimulation(sim.simulation):
            def _detector_simulation_filter_amp(self, evt, station, det):
                detector_simulation_filter_amp(evt, station, det)
        sim_class = mySimulation(
            inputfilename = dummy_input_file,
            outputfilename = "dummy_output_file.hdf5",
            detectorfile=detectorfile,
            det=det,
            config_file=config_file,
            evt_time=evt_time
        )
        self.config = sim_class._config
        self.det = sim_class._det
        self.propagator = sim_class._propagator
        self.ice = sim_class._ice

        if use_channels is not None:
            self.channel_ids = use_channels
        else:
            self.channel_ids = self.det.get_channel_ids(station_id)

        self.detector_simulation_filter_amp = detector_simulation_filter_amp

        self.add_cable_delay = add_cable_delay

        self.pre_pulse_time = pre_pulse_time

        # Get readout sampling rates and number of samples for the channels to be simulated:
        self.readout_sampling_rates = np.zeros(len(self.channel_ids))
        self.readout_n_samples = np.zeros(len(self.channel_ids), dtype=int)
        for i_ch, channel_id in enumerate(self.channel_ids):
            channel_info = self.det.get_channel(self.station_id, channel_id)
            self.readout_sampling_rates[i_ch] = channel_info["adc_sampling_frequency"]
            self.readout_n_samples[i_ch] = channel_info["adc_n_samples"]

        # Determine wether the channels have equal number of samples and sampling rate:
        if np.all(self.readout_sampling_rates == self.readout_sampling_rates[0]) and np.all(self.readout_n_samples == self.readout_n_samples[0]):
            self.channels_are_similar = True
        else:
            self.channels_are_similar = False


    def simulate_showers(self, showers, trace_start_times=None):
        """
        Simulate the traces for a list of user-defined showers (e.g. from a neutrino interaction).

        Parameters
        ----------
        showers: list of NuRadioReco.framework.radio_shower.RadioShower objects
            List of shower objects to simulate.
        trace_start_times: array-like
            Start times of the traces for each readout channel. If None, the start times will be set such
            that the pulse appears self.pre_pulse_time + shower[shp.vertex_time] into the trace of the reference channel.

        Returns
        -------
        station: NuRadioReco.framework.station.Station
            Station object containing the simulated channels.
        traces: numpy.ndarray
            Numpy array of the simulated traces
        trace_start_times:
            Start times of the traces for each readout channel calculated dynamically if not trace_start_times is provided.
        """

        evt = NuRadioReco.framework.event.Event(0, 0)
        station = NuRadioReco.framework.station.Station(self.station_id)
        if self.channels_are_similar:
            traces = np.zeros([len(self.channel_ids), self.readout_n_samples[0]], dtype=float)
        else:
            traces = np.zeros(len(self.channel_ids), dtype=object)

        # Calculate trace start times if not provided:
        if trace_start_times is None:
            start_point = showers[0][shp.vertex]
            end_point = self.det.get_relative_position(self.station_id, self.reference_channel)
            self.propagator.set_start_and_end_point(start_point, end_point)
            self.propagator.find_solutions()
            reference_travel_time = self.propagator.get_travel_time(0)

            reference_cable_delay = self.det.get_cable_delay(self.station_id, self.reference_channel)

            trace_start_times = np.repeat(reference_travel_time + reference_cable_delay - self.pre_pulse_time, len(self.channel_ids))
        elif len(np.atleast_1d(trace_start_times)) == 1:
            trace_start_times = np.repeat(trace_start_times, len(self.channel_ids))

        for i_ch, channel_id in enumerate(self.channel_ids):

            # Electric field simulation:
            sim_station = sim.calculate_sim_efield(showers, self.station_id, channel_id, self.det, self.propagator, self.ice, self.config)

            # Detector simulation:
            if len(sim_station.get_electric_fields()) != 0:
                sim.apply_det_response_sim(sim_station, self.det, self.config, self.detector_simulation_filter_amp)

            # Add cable delays:
            if self.add_cable_delay:
                channelAddCableDelay.run(evt, sim_station, self.det, mode="add")

            # Make empty channel and add sim channels to it:
            readout_channel = NuRadioReco.framework.channel.Channel(channel_id)
            readout_channel.set_trace(
                trace = np.zeros(self.readout_n_samples[i_ch]),
                sampling_rate = self.readout_sampling_rates[i_ch],
                trace_start_time = trace_start_times[i_ch]
            )
            for sim_channel in sim_station.get_channels_by_channel_id(channel_id):
                sim_channel.resample(self.readout_sampling_rates[i_ch])
                readout_channel.add_to_trace(sim_channel, raise_error=False,  min_residual_time_offset=0 * units.ns)

            # Add readout channel to station and save trace:
            station.add_channel(readout_channel)
            traces[i_ch] = readout_channel.get_trace()

        return station, traces, trace_start_times


    def simulate_single_shower(self, energy, zenith, azimuth, vertex, vertex_time, type, charge_excess_profile_id=1, trace_start_times=None):
        """
        Simulate a the traces for a single shower with the given parameters, which can be used to simulate the
        neutrino signal with a given energy, direction, vertex position, and vertex time.

        Parameters
        ----------
        energy: float
            Energy of the shower to simulate in eV
        zenith: float
            Zenith angle of the shower to simulate in radians
        azimuth: float
            Azimuth angle of the shower to simulate in radians
        vertex: array-like
            Vertex position of the shower in Cartesian coordinates (x,y,z) in meters
        vertex_time: float
            Time of the shower in ns
        type: string
            Type of the shower, either "HAD" or "EM"
        charge_excess_profile_id: int
            Id of the charge excess profile to use for the simulation, default 1.
        trace_start_times: array-like
            Start times of the traces for each readout channel. If None, the start times will be set such
            that the pulse appears self.pre_pulse_time + vertex_time into the trace of the reference channel.
            If a single trace_start_time is provided, it will be used for all channels.

        Returns
        -------
        station: NuRadioReco.framework.station.Station
            Station object containing the simulated channels.
        traces: numpy.ndarray
            Numpy array of the simulated traces
        trace_start_times:
            Start times of the traces for each readout channel which
            are automatically calcualted if trace_start_times is None.
        """

        shower = NuRadioReco.framework.radio_shower.RadioShower(0)
        shower[shp.zenith] = zenith
        shower[shp.azimuth] = azimuth
        shower[shp.energy] = energy
        shower[shp.vertex] = vertex
        shower[shp.type] = type
        shower[shp.vertex_time] = vertex_time
        shower[shp.charge_excess_profile_id] = charge_excess_profile_id

        station, traces, trace_start_times = self.simulate_showers([shower], trace_start_times)

        return station, traces, trace_start_times


    def simulate_single_shower_spherical_vertex_coordinates(self, energy, zenith, azimuth, vertex_r, vertex_theta, vertex_phi, vertex_time, type, charge_excess_profile_id=1, trace_start_times=None):
        """
        Reparameterization of simulate_single_shower where the vertex position is given in spherical
        coordinates instead of Cartesian coordinates. This often gives less correlated parameters for
        forward-folding reconstruction.

        Parameters
        ----------
        energy: float
            Energy of the shower to simulate in eV
        zenith: float
            Zenith angle of the shower to simulate in radians
        azimuth: float
            Azimuth angle of the shower to simulate in radians
        vertex_r: float
            Radial distance of the shower vertex from the origin in meters
        vertex_theta: float
            Polar angle of the shower vertex in radians
        vertex_phi: float
            Azimuthal angle of the shower vertex in radians
        vertex_time: float
            Time of the shower in ns
        type: string
            Type of the shower, either "HAD" or "EM"
        charge_excess_profile_id: int
            Id of the charge excess profile to use for the simulation, default 1.
        trace_start_times: array-like
            Start times of the traces for each readout channel. If None, the start times will be set such
            that the pulse appears self.pre_pulse_time + vertex_time into the trace of the reference channel.
            If a single trace_start_time is provided, it will be used for all channels.

        Returns
        -------
        station: NuRadioReco.framework.station.Station
            Station object containing the simulated channels.
        traces: numpy.ndarray
            Numpy array of the simulated traces
        trace_start_times:
            Start times of the traces for each readout channel. If None, the start times will be set such
            that the pulse appears self.pre_pulse_time + vertex_time into the trace of the reference channel.
            If a single trace_start_time is provided, it will be used for all channels.
        """

        shower = NuRadioReco.framework.radio_shower.RadioShower(0)
        shower[shp.zenith] = zenith
        shower[shp.azimuth] = azimuth
        shower[shp.energy] = energy
        shower[shp.vertex] = hp.spherical_to_cartesian(vertex_theta, vertex_phi) * vertex_r
        shower[shp.type] = type
        shower[shp.vertex_time] = vertex_time
        shower[shp.charge_excess_profile_id] = charge_excess_profile_id

        station, traces, trace_start_times = self.simulate_showers([shower], trace_start_times)

        return station, traces, trace_start_times
