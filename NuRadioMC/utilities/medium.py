"""
Module providing the ice models in NuRadioMC.

For more details on the implementation, and the available modules,
see the documentation :doc:`here </NuRadioMC/pages/Manuals/icemodels>`.

"""

from NuRadioMC.utilities import medium_base
import numpy as np
import os
from NuRadioReco.utilities import units

import functools

logger = medium_base.logger

try:
    import radiopropa as RP
    radiopropa_is_imported = True
except ImportError:
    radiopropa_is_imported = False

"""
1) When implementing a new model it should at least inherit from
'IceModel' from the module 'medium_base'. Overwrite all the function.
Inheritance from daughter classes like 'IceModelSimple' is also
possible and overwriting functions may not be needed in this case.

2) When implementing a new model and using the radiopropa numerical
tracer, do not forget to implement scalar field of the refractive index
also in the c++ code of radiopropa for a fast simulation. Implement the
model in IceModel.cpp and IceModel.h. Then edit the function to get the
radiopropa ice model, so it can be used in NuRadioMC. For example

        def get_ice_model_radiopropa(self):
            #args is a placeholder for the arguments needed by the specific module
            scalar field = radiopropa.New_IceModel(*args)
            return RadioPropaIceWrapper(self,scalar_field)

3) You can also choose to only implement the new ice model in radiopropa if
radiopropa is always necessary and make the new model in this script access
the c++ implemented model (e.g. green_firn model)


4) If you want to adjust (add, replace, remove) predefined modules
in the a RadioPropaIceWrapper object, you can do this by redefining the
'get_ice_model_radiopropa()' in your IceModel object. For exemple

        def get_ice_model_radiopropa(self):
            scalar field = radiopropa.IceModelSimple(*args)
            ice = RadioPropaIceWrapper(self,scalar_field)
            extra_dicontinuity = radiopropa.Discontinuity(*args)
            ice.add_module(extra_discontinuity)
            return ice
"""


class southpole_simple(medium_base.IceModelSimple):
    """RICE2014/SP model from https://doi.org/10.1088/1475-7516/2018/07/055"""
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 RICE2014/SP model
        # define model parameters (RICE 2014/southpole)
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 71.*units.meter,
            delta_n = 0.426,
            )


class southpole_2015(medium_base.IceModelSimple):
    """SPICE2015/SP model from https://doi.org/10.1088/1475-7516/2018/07/055"""
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 SPICE2015/SP model
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 77.*units.meter,
            delta_n = 0.423,
            )


class ARAsim_southpole(medium_base.IceModelSimple):
    """ARA SPICE 2015/southpole model"""
    def __init__(self):
        # define model parameters (SPICE 2015/southpole)
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 75.75757575757576*units.meter,
            delta_n = 0.43,
            )



class ARA_2022(medium_base.IceModelSimple):
    """ARA/southpole model from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.122006"""
    def __init__(self):
        # define model parameters (ARA/southpole) -> https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.122006
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 49.5049505*units.meter,
            delta_n = 0.454,
            )


class birefringence_medium(medium_base.IceModelBirefringence):

    def __init__(self, bir_model='southpole_A'):
        """
        Birefringent medium

        Parameters
        ----------
        bir_model : str
            Which birefringent medium to use. Options are:

            * southpole_A - assumes a constant index of refraction at shallow and deep depths
            * southpole_B - assumes a converging index of refraction at shallow depths
            * southpole_C - no birefringence as nx = ny = nz
            * southpole_D - assumes a constant average over all depths
            * southpole_E - assumes ny and nz to be the same value at the average of the two
            * greenland_A - most reasonable interpolation
            * greenland_B - assumes ny and nx to be the same value at the average of the two
            * greenland_C - assumes ny and nx to diverge more than the data indicates

        Notes
        -----
        The southpole models A, B, D, E were presented in the paper: N. Heyer and C. Glaser,
        First-principle calculation of birefringence effects for in-ice
        radio detection of neutrinos, eprint: 2205.06169.

        The Greenland ice models are based on the data in Aguilar, J. A. and others,
        Radiofrequency Ice Dielectric Measurements at Summit Station, Greenland, eprint: 2212.10285

        In order to use the data, spline interpolated functions were fitted to the data points.
        As the data only cover a certain depth range there is room for interpretation on
        how to extrapolate the data to more shallow and deeper depth regions.
        The file IceModel_interpolation.py can be used to adjust the interpolation method and come up with new ice models.
        As the data was taken at the South Pole, the models created with this data set should be called
        "birefringence_southpole_X.npy", where the X marks the model identifier.
        """
        f = self._load_binary_data(bir_model)
        super().__init__(bir_model=f)

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _load_binary_data(bir_model):
        # from https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y
        filepath = os.path.dirname(os.path.realpath(__file__)) + '/birefringence_models/birefringence_' + bir_model + '.npy'
        return np.load(filepath, allow_pickle=True)

class mooresbay_simple(medium_base.IceModelSimple):
    """MB1 model from https://doi.org/10.1088/1475-7516/2018/07/055

    Includes reflective layer at -576 m from https://doi.org/10.3189/2015JoG14J214
    """
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
        super().__init__(
            n_ice = 1.78,
            z_0 = 34.5*units.meter,
            delta_n = 0.46,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom(
            refl_z = -576*units.m,
            refl_coef = 0.82,
            refl_phase_shift = 180*units.deg,
            )


class mooresbay_simple_2(medium_base.IceModelSimple):
    """MB2 model from https://doi.org/10.1088/1475-7516/2018/07/055

    Includes reflective layer at -576 m from https://doi.org/10.3189/2015JoG14J214
    """
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB2 model
        super().__init__(
            n_ice = 1.78,
            z_0 = 37*units.meter,
            delta_n = 0.481,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom(
            refl_z = -576*units.m,
            refl_coef = 0.82,
            refl_phase_shift = 180*units.deg,
            )


class greenland_simple(medium_base.IceModelSimple):
    """Greenland ice model from C. Deaconu, fit to data from Hawley '08, Alley '88"""
    def __init__(self):
        # from C. Deaconu, fit to data from Hawley '08, Alley '88
        # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
        super().__init__(
            z_bottom = -3000*units.meter,
            n_ice = 1.78,
            z_0 = 37.25*units.meter,
            delta_n = 0.51,
            )

class greenland_simple_nils(medium_base.IceModelSimple):

    def __init__(self):
        super().__init__(
            z_bottom = -3000*units.meter,
            n_ice = 1.781,
            z_0 = 45.20*units.meter,
            delta_n = 0.485,
            )
                

class greenland_firn(medium_base.IceModel):
    """
    double exponential ice model at summit, Greenland

    This model can only be used with the radiopropa raytracer.
    Therefor, the model is implemented through radiopropa.
    """
    def __init__(self):
        """
        initiation of a double exponential ice model at summit, Greenland

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.

        The z_shift is a variable introduced to be able to shift the exponential
        up or down along the z direction. For simple models this is almost never
        but it is used to construct more complex ice models which rely on exp.
        profiles also

        Parameters
        ----------
        z_air_boundary: float, NuRadio length units
            z coordinate of the surface of the glacier
        z_bottom: float, NuRadio length units
            z coordinate of the bedrock/bottom of the glacier.
        z_firn: float, NuRadio length units
            z coordinate of the transition from the upper
            exponential profile to the lower one

        The following parameters can be found without (lower)
        and with (upper) the suffix of `_firn`

        n_ice: float, dimensionless
            refractive index of the deep bulk ice
        delta_n: float, NuRadio length units
            difference between n_ice and the refractive index
            of the snow at the surface
        z_0: float, NuRadio length units
            scale depth of the exponential
        z_shift: float, NuRadio length units
            up or down shift od the exponential profile
        """

        if not medium_base.radiopropa_is_imported:
            logger.error('This ice model depends fully on RadioPropa, which was not import, and can therefore not be used.'+
                         '\nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('This ice model depends fully on RadioPropa, which could not be imported')

        super().__init__(z_bottom = -3000*units.meter)
        self.z_firn = -14.9*units.meter

        self._scalarfield = RP.IceModel_Firn(
            z_surface = self.z_air_boundary*RP.meter/units.meter,
            z_firn = self.z_firn*RP.meter/units.meter,
            n_ice = 1.78,
            delta_n = 0.310,
            z_0 = 40.9*RP.meter,
            z_shift = -14.9*RP.meter,
            n_ice_firn = 1.78,
            delta_n_firn = 0.502,
            z_0_firn = 30.8*RP.meter,
            z_shift_firn = 0.*RP.meter,
            )

    def get_index_of_refraction(self, position):
        """
        returns the index of refraction at position.

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n: float
            index of refraction
        """
        position = RP.Vector3d(*(position * RP.meter/units.meter))
        return self._scalarfield.getValue(position)

    def get_average_index_of_refraction(self, position1, position2):
        """
        returns the average index of refraction between two points

        Parameters
        ----------
        position1: 3dim np.array
            point
        position2: 3dim np.array
            point

        Returns
        -------
        n_average: float
            averaged index of refraction between the two points
        """
        position1 = RP.Vector3d(*(position1 * RP.meter/units.meter))
        position2 = RP.Vector3d(*(position2 * RP.meter/units.meter))
        return self._scalarfield.getAverageValue(position1, position2)


    def get_gradient_of_index_of_refraction(self, position):
        """
        returns the gradient of index of refraction at position

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n_nabla: np.array(3,)
            gradient of index of refraction at the point
        """
        pos = RP.Vector3d(*(position * RP.meter/units.meter))
        return self._scalarfield.getGradient(pos) * (1 / (units.meter/RP.meter))


    def _compute_default_ice_model_radiopropa(self):
        """
        Computes a default object holding the radiopropa scalarfield and necessary radiopropa
        moduldes that define the medium in radiopropa. It uses the parameters of the medium
        object to contruct the scalar field (using the firn ice model implementation
        in radiopropa) and some modules (like a discontinuity object for the air boundary).

        Overwrites function of the mother class

        Returns
        -------
        ice_model_radiopropa: RadioPropaIceWrapper
            object holding the radiopropa scalarfield and modules
        """
        return medium_base.RadioPropaIceWrapper(self, self._scalarfield)

class greenland_perturbation(greenland_firn):
    def __init__(self):
        greenland_firn.__init__(self)

    def _compute_default_ice_model_radiopropa(self, discontinuity=False):
        """
        Computes a default object holding the radiopropa scalarfield and necessary radiopropa
        moduldes that define the medium in radiopropa. It uses the parameters of the medium
        object to contruct some modules using the default computation of the firn model.
        An additional module for the perturbation layer is then added to the object.

        Overwrites function of the mother class

        Returns
        -------
        ice_model_radiopropa: RadioPropaIceWrapper
            object holding the radiopropa scalarfield and modules
        """
        ice = greenland_firn._compute_default_ice_model_radiopropa(self)
        #fraction from ArXiv 1805.12576 table IV last row
        perturbation_horz = RP.PerturbationHorizontal(-100*RP.meter,2*RP.meter, fraction=1)
        ice.add_module('horizontal perturbation',perturbation_horz)
        return ice

class greenland_poly5(medium_base.IceModelExponentialPolynomial):
    """
    Fifth-degree exponential polynomial model for Summit Station, Greenland by Oeyen B.

    See https://doi.org/10.5281/zenodo.15067984 for more details
    """
    def __init__(self, density_factor=0.851 * (units.cm**3 / units.gram)):
        """
        initiation of the model based on the fitted coefficient

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.
        """

        super().__init__(
            a=np.array([917, -62.2, 1177, -9051, 14360, -7024]) * (units.kg / units.m**3),
            z_0=74.6 * units.meter,
            density_factor=density_factor,
            z_bottom=-3000 * units.meter
        )


class uniform_ice(medium_base.IceModelSimple):
    """
    uniform ice with refractive index of typical deep ice (1.78)
    """
    def __init__(self, z_bottom=None):
        super().__init__(
            z_bottom = z_bottom,
            n_ice = 1.78,
            z_0 = 1*units.meter,
            delta_n = 0,
            )


def get_ice_model(name):
    """
    function to access the right ice model class by name of the class

    Parameters
    ----------
    name: string
        name of the class of the requested ice model

    Returns
    -------
    ice_model: IceModel object
        object of the class with the name of the requested model
    """
    if globals()[name]() == None:
        logger.error('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
        raise NotImplementedError('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
    else:
        return globals()[name]()


class greenland_firn_layered(medium_base.IceModelExpLayers):
    """
    Two-layer refractive index model.
     
    values taken from greenland_firn and adapted to match the expected medium definition needed for the multi layer analytic raytracer.
    Combination of firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """
    def __init__(self):
        
        layers = [
            {
                "z_min": -14.9,
                "z_max": 0.0,
                "n_ice": 1.78,
                "delta_n": 0.502,
                "z_0": 30.8,
                "region": "firn",
                "region_name": "Firn"
            },
            {
                "z_min": -3000.0,
                "z_max": -14.9,
                "n_ice": 1.78,
                "delta_n": 0.446,
                "z_0": 40.9,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
        )

# Air above Summit Station, Greenland (3216 m a.s.l.), for the Greenland models with an air layer:
# n(z) = 1 + 2.14e-4 * exp(-z / 9 km), z above the snow surface.
# Refractivity N = (n - 1) * 1e6 = 77.6 P / T + 3.732e5 e / T^2 (ITU-R P.453; P, e in hPa, T in K), evaluated on the
# NOAA GML Summit hourly meteorology 2008-2026 (https://gml.noaa.gov/aftp/data/meteorology/in-situ/sum/), April to
# October (the RNO-G data-taking season): mean P = 672.6 hPa, T(2 m) = -21.0 C, water-vapour term 6.8 (from the
# relative humidity) -> N = 213.9 (5-95%: 205.6-223.5; monthly means 212-217).
# Scale height of the air density for the mean T and a lapse rate of 6.5 K/km: T / (g / R_d - 6.5 K/km) = 9.1 km
# (isothermal: 7.4 km). The US standard atmosphere at sea level would give n - 1 = 2.7e-4.
N_AIR_SUMMIT = 1 + 2.14e-4
AIR_SCALE_HEIGHT_SUMMIT = 9 * units.km


class greenland_3exp_density(medium_base.IceModelContinuousExpLayers):
    """
    (3+1) layer refractive index model. Three layer model in ice plus one air layer.
     
    Values for below the ice obtained from a fit to ice core density data.
    Combination of air layer above z=0.0, snow layer,
    firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """

    def __init__(self):
        super().__init__(
            nN = 1.778,
            delta_nN = 1.120,
            ls = [28.606, 73.728, 8.848],
            zs = [-80.5, -14.9],
            n_air_surface = N_AIR_SUMMIT,
            air_scale_height = AIR_SCALE_HEIGHT_SUMMIT
        )

class greenland_3exp_timing_2025(medium_base.IceModelContinuousExpLayers):
    """
    (3+1) layer refractive index model. Three layer model in ice plus one air layer.

    Values for below the ice obtained from a fit to absolute-timing data
    taken during the 2025 Summit field season.
    Combination of air layer above z=0.0, snow layer,
    firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """

    def __init__(self):
        super().__init__(
            nN = 1.778,
            delta_nN = 1.0659,
            ls = [29.3437, 62.2818, 15.88],
            zs = [-80.5, -14.9],
            n_air_surface = N_AIR_SUMMIT,
            air_scale_height = AIR_SCALE_HEIGHT_SUMMIT
        )


#: Default ice model per detector site (lower-case site name as returned by ``Detector.get_site``),
#: used by `get_station_ice_model` when neither the caller nor the detector provides one.
SITE_ICE_MODELS = {
    'summit': 'greenland_3exp_timing_2025',
    'greenland': 'greenland_3exp_timing_2025',
    'southpole': 'southpole_2015',
    'sp': 'southpole_2015',
    'mooresbay': 'mooresbay_simple',
}

#: Sites without ice: the antennas are in air (`get_station_ice_model` returns None).
AIR_SITES = ('lofar', 'auger', 'ska')

#: Refractive index of air at the surface per site, used above the surface by ice models without an air layer.
#: Sites not listed use `medium_base.N_AIR_SEA_LEVEL`.
SITE_N_AIR = {
    'summit': N_AIR_SUMMIT,
    'greenland': N_AIR_SUMMIT,
}

def get_n_air(site):
    """
    Returns the refractive index of air at the surface of a site (`SITE_N_AIR`, else `medium_base.N_AIR_SEA_LEVEL`).

    Parameters
    ----------
    site: str
        Site name (case insensitive), e.g. from ``Detector.get_site``

    Returns
    -------
    n_air: float
    """
    return SITE_N_AIR.get(site.lower(), medium_base.N_AIR_SEA_LEVEL)


@functools.lru_cache(maxsize=None)
def _get_cached_ice_model(name):
    return get_ice_model(name)


def get_station_ice_model(det, station_id, ice_model=None):
    """
    Returns the ice model for a station.

    The model is taken, in this order, from:

    1. the argument ``ice_model`` (name of a model in this module, or an `IceModel` object),
    2. the detector, if it provides ``get_ice_model(station_id)`` and a model is attached to the station,
    3. the default model of the station's site (`SITE_ICE_MODELS`).

    Parameters
    ----------
    det: Detector
        Detector description; uses ``det.get_site(station_id)`` and, if present, ``det.get_ice_model(station_id)``
    station_id: int
        Station id
    ice_model: str or `NuRadioMC.utilities.medium_base.IceModel`, optional
        Overrides the detector and site defaults

    Returns
    -------
    ice_model: `NuRadioMC.utilities.medium_base.IceModel` or None
        None for sites without ice (`AIR_SITES`). Models resolved by name are shared between calls;
        do not modify them.

    Raises
    ------
    ValueError
        If no model is given and the site has no default (e.g. "simulation" or an unknown site)
    """
    if isinstance(ice_model, str):
        return _get_cached_ice_model(ice_model)
    if ice_model is not None:
        return ice_model

    if hasattr(det, "get_ice_model"):
        try:
            det_ice_model = det.get_ice_model(station_id)
        except KeyError:
            det_ice_model = None  # no model attached to this station
        if det_ice_model is not None:
            return det_ice_model

    site = det.get_site(station_id).lower()
    if site in AIR_SITES:
        return None
    if site not in SITE_ICE_MODELS:
        raise ValueError(
            f"No default ice model for site '{site}' (station {station_id}). "
            f"Pass an ice model explicitly (`ice_model`, e.g. one of {sorted(set(SITE_ICE_MODELS.values()))}).")

    return _get_cached_ice_model(SITE_ICE_MODELS[site])


def _has_air_layer(ice_model):
    # Air layers built by `medium_base.exponential_air_layer` (IceModelContinuousExpLayers,
    # IceModelExpLayers.from_IceModel) are physical; other layer definitions may carry an n ~ 1 placeholder.
    return isinstance(ice_model, medium_base.IceModelExpLayers) and any(
        layer.get("region") == "air" for layer in ice_model.layers)


def get_refractive_index(z, ice_model, site):
    """
    Returns the refractive index at height ``z`` of a station.

    Below the surface (``z <= ice_model.z_air_boundary``), this is the ice model's n(z). Above it, the model's
    air layer if it has a physical one (built by `medium_base.exponential_air_layer`), else the site's air value (`get_n_air`);
    most ice models themselves return n = 1 in air.

    Parameters
    ----------
    z: float
        Height relative to the surface (negative: in the ice)
    ice_model: `NuRadioMC.utilities.medium_base.IceModel` or None
        Ice model, e.g. from `get_station_ice_model`. None means no ice (air everywhere).
    site: str
        Site name, used for the air value

    Returns
    -------
    n: float
    """
    if ice_model is None:
        return get_n_air(site)

    if z > ice_model.z_air_boundary and not _has_air_layer(ice_model):
        return get_n_air(site)

    return float(ice_model.get_index_of_refraction(np.array([0, 0, z])))


def get_station_refractive_index(det, station_id, z, ice_model=None):
    """
    Returns the refractive index at height ``z`` of a station.

    Combines `get_station_ice_model` and `get_refractive_index`. If the station's site has no default ice model
    (e.g. "simulation") and none is given, the site's air value (`get_n_air`) is returned above the surface
    (``z > 0``); below it, the `ValueError` of `get_station_ice_model` is raised.

    Parameters
    ----------
    det: Detector
        Detector description (see `get_station_ice_model`)
    station_id: int
        Station id
    z: float
        Height relative to the surface (negative: in the ice)
    ice_model: str or `NuRadioMC.utilities.medium_base.IceModel`, optional
        Overrides the detector and site defaults

    Returns
    -------
    n: float
    """
    site = det.get_site(station_id)
    try:
        station_ice_model = get_station_ice_model(det, station_id, ice_model)
    except ValueError:
        if z > 0:
            return get_n_air(site)
        raise

    return get_refractive_index(z, station_ice_model, site)
