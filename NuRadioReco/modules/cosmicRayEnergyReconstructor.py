import numpy as np
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import units
import NuRadioReco.utilities.trace_utilities
import logging
import radiotools.helper as hp
import radiotools.coordinatesystems
import radiotools.atmosphere.models


class cosmicRayEnergyReconstructor:
    """
    Reconstructs the energy of an air shower from its radio signal

    Requires the following modules to be run beforehand:
    
    * a 10th order Butterworth bandpass filter with passband 80-300 MHz
    * a direction reconstruction
    * the voltageToAnalyticEfieldConverter
    
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.cosmicRayEnergyReconstructor')
        self.__atmosphere = radiotools.atmosphere.models.Atmosphere()
        self.__parametrizations = {
            'mooresbay': {
                'scale': np.array([(442.46, -281.75, 324.96), (394.08, -308.36, 436.30)]),
                'falloff': np.array([(-.1584, -.07943), (.8070, -1.4098)])
            },
            'southpole': {
                'scale': np.array([(976.30, -1213.43, 626.98), (643.39, -667.08, 478.06)]),
                'falloff': np.array([(-.2273, .05627), (1.3372, -2.1653)])
            },
            'auger': {
                'scale': np.array([(229.96, -123.75, 110.51), (214.46, -111.01, 119.18)]),
                'falloff': np.array([(-.1445, -.09820), (.5936, -1.1763)])
            },
            'summit': {
                'scale': np.array([[ 281.34, -551.65,  610.25],[ 411.01, -590.02,  570.2 ]]),
                'falloff': np.array([[-0.2285,  0.4058], [ 2.0967, -1.2992]])
            }
        }
        self.__elevations = {  # TODO: This should be changed once we have implemented a proper coordinate system
            'mooresbay': 30.,
            'southpole': 2800.,
            'auger': 1560.,
            'summit': 3216.
        }
        self.__site = None

    def begin(self, site=None):
        """
        Initialize the cosmicRayEnergyReconstructor (optional)

        Parameters
        ----------
        site : string | None (default: None)
            Specifies the site of the station. The parameterization of
            the cosmic ray energy depends on the site of the detector.

            If None, the site will be determined from the detector
            passed to the `run` function.

        """
        self.__site = site
        if site not in self.__parametrizations.keys():
            self.logger.error('Unsupported site. Please select one of the following: {}'.format(self.__parametrizations.keys()))
            raise ValueError

    @register_run()
    def run(self, event, station, detector, electric_field=None):
        """
        Determine the cosmic ray energy from the electric field fluence.

        The reconstructed cosmic ray energy will be stored in the
        station in the :obj:`cr_energy_em <NuRadioReco.framework.parameters.stationParameters.cr_energy_em>` parameter.

        Parameters
        ----------
        event : Event
        station : Station
            The station containing the reconstructed electric field.
            If it contains multiple electric fields, only the last electric field
            will be used, unless another electric field is passed as a keyword-argument
        detector : Detector
        electric_field : ElectricField | None (default: None)
            If not None, reconstruct the energy for this electric field.
            Otherwise, reconstruct the last electric field in the station.
            Useful if a station contains multiple reconstructed electric fields.

        Returns
        -------
        rec_energy : float
            The reconstructed cosmic ray energy

        """

        if not station.is_cosmic_ray():
            self.logger.warning('Event is not a cosmic ray!')
        if not station.has_parameter(stnp.zenith) or not station.has_parameter(stnp.azimuth):
            self.logger.error('No incoming direction available. Energy can not be reconstructed!')
            return
        zenith = station.get_parameter(stnp.zenith)
        azimuth = station.get_parameter(stnp.azimuth)
        site = self.__site
        if site is None:
            site = detector.get_site(station.get_id())
            if site not in self.__parametrizations.keys():
                self.logger.error('Unsupported site. Please select one of the following: {}'.format(self.__parametrizations.keys()))
                raise ValueError
        parametrization_for_site = self.__parametrizations[site]
        elevation = self.__elevations[site]

        if zenith < 30. * units.deg:
            self.logger.warning('Zenith angle is smaller than 30deg. Energy reconstruction is likely to be inaccurate!')
        if electric_field is None:
            n_efields = len(station.get_electric_fields())
            if n_efields == 0:
                self.logger.error('No E-field found. Please run the voltageToAnalyticEfieldConverter beforehand!')
                return
            if n_efields > 1:
                self.logger.warning('Multiple E-fields were found. Only the last E-field will be used.')
            electric_field = station.get_electric_fields()[-1]

        spectrum_slope = electric_field.get_parameter(efp.cr_spectrum_slope)
        alpha = hp.get_angle_to_magnetic_field_vector(zenith, azimuth, site)
        cs = radiotools.coordinatesystems.cstrafo(zenith, azimuth, site=site)
        efield_trace_vxB_vxvxB = cs.transform_to_vxB_vxvxB(cs.transform_from_onsky_to_ground(electric_field.get_trace()))
        efield_trace_vxB_vxvxB[0] /= np.sin(alpha)  # correct energy fluence for effect of angle to magnetic field
        energy_fluence = NuRadioReco.utilities.trace_utilities.get_electric_field_energy_fluence(efield_trace_vxB_vxvxB, electric_field.get_times())
        energy_fluence = np.abs(energy_fluence[0]) + np.abs(energy_fluence[1])
        xmax_distance = self.__atmosphere.get_distance_xmax_geometric(zenith, 750., elevation)  # parametrization is for Xmax of 750g/cm^2

        # find out if we are inside or outside of the Cherenkov ring
        second_order_spectrum_parameter = electric_field.get_parameter(efp.cr_spectrum_quadratic_term)
        if second_order_spectrum_parameter <= spectrum_slope * .1:
            scale_parameter = parametrization_for_site['scale'][0][0] * zenith ** 2 + parametrization_for_site['scale'][0][1] * zenith + parametrization_for_site['scale'][0][0]
            falloff_parameter = parametrization_for_site['falloff'][0][0] * zenith + parametrization_for_site['falloff'][0][1]
        else:
            scale_parameter = parametrization_for_site['scale'][1][0] * zenith ** 2 + parametrization_for_site['scale'][1][1] * zenith + parametrization_for_site['scale'][1][0]
            falloff_parameter = parametrization_for_site['falloff'][1][0] * zenith + parametrization_for_site['falloff'][1][1]
        rec_energy = 1.e18 * np.sqrt(energy_fluence) * (xmax_distance / units.km) / (scale_parameter * np.exp(falloff_parameter * np.abs(spectrum_slope) ** 0.8))
        station.set_parameter(stnp.cr_energy_em, rec_energy)

        return rec_energy
