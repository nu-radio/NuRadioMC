import NuRadioMC.simulation.simulation_base
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelResampler
from NuRadioReco.utilities import units
class simulation_detector(NuRadioMC.simulation.simulation_base.simulation_base):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(simulation_detector, self).__init__(*args, **kwargs)
        self._channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        self._eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
        self._efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()
        self._efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self._efieldToVoltageConverter.begin(time_resolution=self._cfg['speedup']['time_res_efieldconverter'])
        self._channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        self._channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        self._channelGenericNoiseAdder.begin(seed=self._cfg['seed'])
        self._channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        self._electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
        if self._outputfilenameNuRadioReco is not None:
            self._eventWriter.begin(self._outputfilenameNuRadioReco, log_level=self._log_level)

    def _simulate_sim_station_detector_response(
            self
    ):
        # convert efields to voltages at digitizer
        if hasattr(self, '_detector_simulation_part1'):
            # we give the user the opportunity to define a custom detector simulation
            self._detector_simulation_part1()
        else:
            self._efieldToVoltageConverterPerEfield.run(self._evt, self._station,
                                                        self._det)  # convolve efield with antenna pattern
            self._detector_simulation_filter_amp(self._evt, self._station.get_sim_station(), self._det)
            self._channelAddCableDelay.run(self._evt, self._sim_station, self._det)

    def _simulate_detector_response(
            self
    ):
        if hasattr(self, '_detector_simulation_part2'):
            # we give the user the opportunity to specify a custom detector simulation module sequence
            # which might be needed for certain analyses
            self._detector_simulation_part2()
        else:
            # start detector simulation
            self._efieldToVoltageConverter.run(self._evt, self._station,
                                               self._det)  # convolve efield with antenna pattern
            # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
            # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
            self._channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)

            if self._is_simulate_noise():
                max_freq = 0.5 / self._dt
                channel_ids = self._det.get_channel_ids(self._station.get_id())
                Vrms = {}
                for channel_id in channel_ids:
                    norm = self._bandwidth_per_channel[self._station.get_id()][channel_id]
                    Vrms[channel_id] = self._Vrms_per_channel[self._station.get_id()][channel_id] / (
                                norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
                self._channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms,
                                                   min_freq=0 * units.MHz,
                                                   max_freq=max_freq, type='rayleigh',
                                                   excluded_channels=self._noiseless_channels[station_id])

            self._detector_simulation_filter_amp(self._evt, self._station, self._det)

            self._detector_simulation_trigger(self._evt, self._station, self._det)