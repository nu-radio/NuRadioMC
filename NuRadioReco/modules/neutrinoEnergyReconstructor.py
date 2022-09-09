import numpy as np
import NuRadioMC.SignalProp.analyticraytracing
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.radio_shower
from NuRadioReco.utilities import trace_utilities, units


class neutrinoEnergyReconstructor():
    """
    Module to perform an energy reconstruction as described in https://arxiv.org/abs/2107.02604
    Look at NuRadioReco/examples/RNO_energy_reconstruction for an example how to use it.
    Requires vertex reconstruction and IFT electric field reconstruction to be run beforehand.
    """
    def __init__(self):
        self.__s_parametrization = np.array([4.70, -22.20, 29.99])
        self.__s_prime_parametrization = np.array([6.46, -16.00, 12.49])
        self.__s_passbands = np.array([
            [.13, .3],
            [.3, .5]
        ])
        self.__s_prime_passbands = np.array([
            [.13, .2],
            [.2, .3]
        ])
        self.__s_s_prime_cut_value = 10.
        self.__channel_groups = None
        self.__raytracer = None

    def begin(self, channel_groups, ice, attenuation_model='GL1'):
        """
        Set up module

        Parameters
        -------------------
        channel_groups: 2D array of integers
            List of groups of channels which were used together by the iftElectricFieldReconstructor
        ice: ice model object
            Class describing the index of refraction model of the ice
        attenuation model: string
            Name of the model describing the signal attenuation
        """
        self.__channel_groups = channel_groups
        self.__raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
            ice,
            attenuation_model
        )
        pass

    def run(self, event, station, detector):
        """
        Runs the reconstruction. Reconstructed energies are stored in a new shower object that is added to the event.

        Parameters
        ------------
        event
        station
        detector
        """
        if not station.has_parameter(stnp.nu_vertex):
            raise ValueError('Cannot run energy reconstruction without a reconstructed vertex!')
        rec_vertex = station.get_parameter(stnp.nu_vertex)
        for i_group, channel_group in enumerate(self.__channel_groups):
            max_snr_in_group = 0
            self.__raytracer.set_start_and_end_point(rec_vertex, detector.get_relative_position(station.get_id(), channel_group[0]))
            self.__raytracer.find_solutions()
            for i_efield, efield in enumerate(station.get_electric_fields()):
                ray_type = efield.get_parameter(efp.ray_path_type)
                channel = station.get_channel(channel_group[0])
                # Check if this is the ray type with the largest SNR
                if not channel.has_parameter(chp.signal_region_snrs) or not channel.has_parameter(chp.signal_ray_types):
                    continue
                current_snr = 0
                for i_region, region_snr in enumerate(channel.get_parameter(chp.signal_region_snrs)):
                    if ray_type == channel.get_parameter(chp.signal_ray_types)[i_region]:
                        current_snr = region_snr
                if current_snr <= max_snr_in_group:
                    continue
                max_snr_in_group = current_snr
                attenuation = None
                path_length = None
                for i_solution in range(self.__raytracer.get_number_of_raytracing_solutions()):
                    if self.__raytracer.get_solution_type(i_solution) == ray_type:
                        attenuation = self.__raytracer.get_attenuation(
                            i_solution,
                            efield.get_frequencies(),
                            10. * units.GHz
                        )
                        path_length = self.__raytracer.get_path_length(i_solution)
                if attenuation is not None:
                    corrected_efield = NuRadioReco.framework.base_trace.BaseTrace()
                    corrected_efield.set_frequency_spectrum(
                        efield.get_frequency_spectrum() / attenuation * (path_length / units.km),
                        efield.get_sampling_rate()
                    )
                    energy_fluence_1 = trace_utilities.get_electric_field_energy_fluence(
                        corrected_efield.get_filtered_trace(self.__s_passbands[0], 'butter', 10),
                        corrected_efield.get_times()
                    )
                    energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                        corrected_efield.get_filtered_trace(self.__s_passbands[1], 'butter', 10),
                        corrected_efield.get_times()
                    )
                    if energy_fluence_1[1] / energy_fluence_2[1] > self.__s_s_prime_cut_value:
                        energy_fluence_1 = trace_utilities.get_electric_field_energy_fluence(
                            corrected_efield.get_filtered_trace(self.__s_prime_passbands[0], 'butter', 10),
                            corrected_efield.get_times()
                        )
                        energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                            corrected_efield.get_filtered_trace(self.__s_prime_passbands[1], 'butter', 10),
                            corrected_efield.get_times()
                        )
                        log_s_parameter = np.log10((energy_fluence_1[1]) / (energy_fluence_2[1]))
                        rec_energy = np.sqrt(np.sum(energy_fluence_1)) / (
                                self.__s_prime_parametrization[0] * log_s_parameter ** 2 + self.__s_prime_parametrization[1] * log_s_parameter +
                                self.__s_prime_parametrization[2]
                        ) * units.EeV
                    else:
                        log_s_parameter = np.log10((energy_fluence_1[1]) / (energy_fluence_2[1]))
                        rec_energy = np.sqrt(np.sum(energy_fluence_1)) / (
                                self.__s_parametrization[0] * log_s_parameter ** 2 + self.__s_parametrization[1] * log_s_parameter +
                                self.__s_parametrization[2]
                        ) * units.EeV
                    new_shower = NuRadioReco.framework.radio_shower.RadioShower(shower_id=i_group, station_ids=[station.get_id()])
                    new_shower.set_parameter(shp.energy, rec_energy)
                    event.add_shower(new_shower)
