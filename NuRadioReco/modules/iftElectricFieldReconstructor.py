import numpy as np
from NuRadioReco.utilities import units, fft
import NuRadioReco.detector.antennapattern
import scipy
import nifty5 as ift

class IftElectricFieldReconstructor:

    def __init__(self):
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        return

    def begin(self, passband=None, debug = False):
        self.__passband = passband
        self.__debug = debug
        return

    def run(self, event, station, detector, channel_ids, use_sim=False):
        if use_sim:
            self.__vertex_position = station.get_sim_station().get_parameter(stnp.nu_vertex)
        else:
            self.__vertex_position = station.get_parameter(stnp.vertex_2D_fit)

        ref_channel = channel_ids[0]
        frequencies = ref_channel.get_frequencies()
        sampling_rate = ref_channel.get_sampling_rate()
        samples = len(ref_channel.get_trace())
        time_domain = ift.RGSpace(len(ref_channel.get_trace()))
        frequency_domain = time_domain.get_default_codomain()
        gain_operators, phase_operators, filter_operator = self.__get_detector_operators(
            station,
            detector,
            frequency_domain,
            sampling_rate,
            samples,
            channel_ids
        )

        for i_channel, channel_id in enumerate(channel_ids):
            if len(list(station.get_electric_fields_for_channels([channel_id]))) == 0:
                continue
    def __get_detector_operators(self,
        station,
        detector,
        frequency_domain,
        sampling_rate,
        samples,
        channel_ids
    ):
        gain_operators = []
        phase_operators = []
        frequencies = frequency_domain.get_k_length_array().val/samples*sampling_rate
        if self.__passband is not None:
            b, a = scipy.signal.butter(10, self.__passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies)
            filter_field = ift.Field(ift.DomainTuple.make(frequency_domain), np.abs(h))
            filter_operator = ift.DiagonalOperator(filter_field, frequency_domain)
        for channel_id in channel_ids:
            electric_field = list(station.get_electric_fields_for_channels([channel_id]))[0]
            receiving_zenith = electric_field.get_parameter(efp.zenith)
            amp_response_func = NuRadioReco.detector.RNO_G.analog_components.load_amp_response(detector.get_amplifier_type(station.get_id(), channel_id))
            amp_gain = amp_response_func['gain'](frequencies)
            amp_phase = amp_response_func['phase'](frequencies)
            antenna_type = detector.get_antenna_model(station.get_id(), channel_id)
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(antenna_type)
            antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel_id)
            antenna_response = antenna_pattern.get_antenna_response_vectorized(
                frequencies,
                receiving_zenith,
                0,
                antenna_orientation[0],
                antenna_orientation[1],
                antenna_orientation[2],
                antenna_orientation[3])['theta']
            total_gain = np.abs(amp_gain) * np.abs(antenna_response)
            gain_field = ift.Field(ift.DomainTuple.make(frequency_domain), total_gain)
            gain_operators.append(ift.DiagonalOperator(gain_field, frequency_domain))
            total_phase = np.unwrap(np.angle(antenna_response))+amp_phase
            phase_field = ift.Field(ift.DomainTuple.make(frequency_domain), total_phase)
            phase_operators.append(ift.Adder(phase_field))
        return gain_operators, phase_operators, filter_operator
