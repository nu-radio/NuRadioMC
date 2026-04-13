import os
from matplotlib.pylab import det
import numpy as np
from NuRadioMC.simulation import simulation as sim
import NuRadioReco.framework.radio_shower
from NuRadioReco.utilities import units
import datetime
from radiotools import helper as hp
import NuRadioReco.framework.event

from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.channelAddCableDelay

channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

class NeutrinoSimulator():
    """
    Class to simulate voltage traces for a given neutrino interaction vertex, energy 
    and direction with Askaryan emission, propagation effects and detector response
    taken into account. The simulation does not include noise and trigger simulation,
    which is useful for studies where only the MC true neutrino signal is of interrrest.
    It hence acts as a pure neutrino signal model which is needed in, e.g., forward-folding
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
        Channel to clalculate reference time for (only works with reference_channel=0 for now)
    reference_channel: int
        Channel to clalculate reference time for (only works with reference_channel=0 for now)
        if no trace_start_times are provided. Pulse should appear pre_pulse_time into the trace
        for this channel.
    evt_time: datetime object
        The time of the simulated event, default 1/1/2018
    detector_simulation_filter_amp: function
        Function to use as the _detector_simulation_filter_amp in the simulation class, e.g,
        hardware response and/or bandpass filter of the detector. Currently has to be provided.
    add_cable_delay: bool
        Whether to add the cable delay of the detector to the simulated traces, default True.
    pre_pulse_time: float
        Time of the readout pulse in the reference channel relative to the start of the trace,
        if interaction_time is 0, default 200 ns. Ignored if trace_start_times are provided.
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
        dummy_imput_file = [{"xx": [None, None]}, [None, None]]
        class mySimulation(sim.simulation):
            def _detector_simulation_filter_amp(self, evt, station, det):
                detector_simulation_filter_amp(evt, station, det)
        sim_class = mySimulation(
            inputfilename = dummy_imput_file,
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


    def simulate(self, energy, zenith, azimuth, vertex, interaction_time, type, charge_excess_profile_id=1, trace_start_times=None):
        """
        Simulate a neutrino signal with a given energy, direction, and vertex position. 
        
        Parameters
        ----------
        energy: float
            Energy of the neutrino shower to simulate in eV
        zenith: float
            Zenith angle of the neutrino shower to simulate in radians
        azimuth: float
            Azimuth angle of the neutrino shower to simulate in radians
        vertex: array-like
            Vertex position of the neutrino interaction in Cartesian coordinates (x,y,z) in meters
        interaction_time: float
            Time of the neutrino interaction in seconds since the epoch, used to calculate the hardware response
        type: string
            Type of the neutrino interaction, either "HAD" or "EM"
        charge_excess_profile_id: int
            Id of the charge excess profile to use for the simulation, default 1.
        trace_start_times: array-like
            Start times of the traces for each readout channel. If None, the start times will be set such
            that the pulse appears pre_pulse_time + interaction_time into the trace of the reference channel.

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

        showers = []
        shower = NuRadioReco.framework.radio_shower.RadioShower(0)
        shower[shp.zenith] = zenith 
        shower[shp.azimuth] = azimuth
        shower[shp.energy] = energy
        shower[shp.vertex] = vertex
        shower[shp.type] = type
        shower[shp.vertex_time] = interaction_time
        shower[shp.charge_excess_profile_id] = charge_excess_profile_id
        showers.append(shower)

        evt = NuRadioReco.framework.event.Event(0, 0)
        station = NuRadioReco.framework.station.Station(self.station_id)
        traces = np.zeros(len(self.channel_ids), dtype=object)

        for i_ch, channel_id in enumerate(self.channel_ids):

            sim_station = sim.calculate_sim_efield(showers, self.station_id, channel_id, self.det, self.propagator, self.ice, self.config)

            if len(sim_station.get_electric_fields()) != 0:
                sim.apply_det_response_sim(sim_station, self.det, self.config, self.detector_simulation_filter_amp)

            if self.add_cable_delay:
                channelAddCableDelay.run(evt, sim_station, self.det, mode="add")

            if trace_start_times is None and channel_id == self.reference_channel:
                reference_travel_time = self.propagator.get_travel_time(0)
                reference_cable_delay = self.det.get_cable_delay(self.station_id, self.reference_channel)
                trace_start_times = np.repeat(reference_travel_time + reference_cable_delay - self.pre_pulse_time, len(self.channel_ids))

            # Make empty channel and add sim channels to it:
            channel_info = self.det.get_channel(self.station_id, channel_id)
            readout_sampling_rate = channel_info["adc_sampling_frequency"]
            readout_n_samples = channel_info["adc_n_samples"]
            readout_channel = NuRadioReco.framework.channel.Channel(channel_id)
            readout_channel.set_trace(
                np.zeros(readout_n_samples),
                readout_sampling_rate,
                trace_start_time=trace_start_times[i_ch]
            )
            
            for sim_channel in sim_station.get_channels_by_channel_id(channel_id):

                sim_channel.resample(readout_sampling_rate)

                readout_channel.add_to_trace(sim_channel, raise_error=False,  min_residual_time_offset=0 * units.ns)

            station.add_channel(readout_channel)
            traces[i_ch] = readout_channel.get_trace()

        return station, traces, trace_start_times
