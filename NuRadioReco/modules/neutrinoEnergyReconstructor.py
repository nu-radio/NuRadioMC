import numpy as np
import NuRadioMC.SignalProp.analyticraytracing
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.radio_shower
from NuRadioReco.utilities import trace_utilities, units
from NuRadioReco.modules.base.module import register_run


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
        self.__efield_choice = 'first'

    def begin(self, channel_groups, ice, attenuation_model='GL1', efield_choice='first'):  
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
        efield_choice: 'first' or 'median'
            With efield_scaling=True the IFT stores one copy of the field per channel.
            'first' uses the first copy of the selected ray type (as before), 'median'
            the median of the energies of all copies.
        """
        if efield_choice not in ('first', 'median'):  
            raise ValueError("efield_choice must be 'first' or 'median'")
        self.__efield_choice = efield_choice
        self.__channel_groups = channel_groups
        self.__raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
            ice,
            attenuation_model
        )
        pass

    @register_run()
    def run(self, event, station, detector):
        """
        Runs the reconstruction. Reconstructed energies are stored in a new shower object that is added to the event.

        Parameters
        ------------
        event
        station
        detector
        """
        if station.has_parameter(stnp.nu_vertex):
            vertex = station.get_parameter(stnp.nu_vertex)
        else:
            raise ValueError('Cannot run energy reconstruction without a reconstructed vertex!')
        for i_group, channel_group in enumerate(self.__channel_groups):
            max_snr_in_group = 0
            antenna_position = detector.get_relative_position(station.get_id(), channel_group[0]) + detector.get_absolute_position(station.get_id())  
            self.__raytracer.set_start_and_end_point(vertex, antenna_position)
            self.__raytracer.find_solutions()
            for i_efield, efield in enumerate(station.get_electric_fields()):
                ray_type = efield.get_parameter(efp.ray_path_type)
                current_snr = self.__ray_type_snr(station, channel_group, ray_type) 
                if current_snr <= max_snr_in_group:
                    continue
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
                if attenuation is None: 
                    continue
                max_snr_in_group = current_snr 
                rec_energy, log_s_parameter, branch = self.__energy_from_efield(efield, attenuation, path_length) 
                if self.__efield_choice == 'median': 
                    copies = [self.__energy_from_efield(other, attenuation, path_length) for other in station.get_electric_fields()
                              if other.get_parameter(efp.ray_path_type) == ray_type]
                    copies = sorted([copy for copy in copies if np.isfinite(copy[0])], key=lambda copy: copy[0])
                    if len(copies) > 0:
                        rec_energy = np.median([copy[0] for copy in copies])
                        log_s_parameter, branch = copies[len(copies) // 2][1], copies[len(copies) // 2][2]

                showers = list(event.get_showers())
                if len(showers) == 0:
                    shower = NuRadioReco.framework.radio_shower.RadioShower(
                        shower_id=0,
                        station_ids=[station.get_id()]
                    )
                    event.add_shower(shower)
                else:
                    shower = showers[0]

                shower.set_parameter(shp.energy, rec_energy)

    def __ray_type_snr(self, station, channel_group, ray_type): 
        """
        Largest SNR any channel of the group has for this ray type. SNR i and ray type i
        are paired by index, as before; this is only guaranteed with
        channelSignalPropertiesFromNeighbors (2026-10-20 version, one entry per region).
        An interpolated region has SNR NaN and is never counted (NaN > best is False).
        """
        best = 0.
        for channel_id in channel_group:
            channel = station.get_channel(channel_id)
            if not channel.has_parameter(chp.signal_region_snrs) or not channel.has_parameter(chp.signal_ray_type):
                continue
            snrs = np.atleast_1d(channel.get_parameter(chp.signal_region_snrs))
            ray_types = np.atleast_1d(channel.get_parameter(chp.signal_ray_type))
            for i_region in range(min(len(snrs), len(ray_types))):
                if ray_types[i_region] == ray_type and snrs[i_region] > best:
                    best = float(snrs[i_region])
        return best

    def __energy_from_efield(self, efield, attenuation, path_length): 
        """Energy, log10(s) and branch ('s' or 's_prime') for one electric field."""
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
            return rec_energy, log_s_parameter, 's_prime'
        log_s_parameter = np.log10((energy_fluence_1[1]) / (energy_fluence_2[1]))
        rec_energy = np.sqrt(np.sum(energy_fluence_1)) / (
                self.__s_parametrization[0] * log_s_parameter ** 2 + self.__s_parametrization[1] * log_s_parameter +
                self.__s_parametrization[2]
        ) * units.EeV
        return rec_energy, log_s_parameter, 's'
