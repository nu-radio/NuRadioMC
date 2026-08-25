"""Atmospheric models and density profiles for LOFAR cosmic-ray reconstruction.

Provides:
    - ``atm_models``: tabulated 5-layer atmospheric model parameters (Linsley / Keilhauer)
    - ``GDASAtmosphere``: reads a GDAS atmosphere data file
    - ``Atmosphere``: unified interface to standard or GDAS models, JAX-compatible

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

from typing import Optional

import numpy as np
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError:
    jax = None
    jnp = None
    lax = None

# Physical constants
_R_EARTH = 6.371e6   # radius of Earth [m]
_H_MAX = 112829.2    # height where overburden vanishes [m]
_EPSILON = 1e-9


# Tabulated 5-layer atmospheric model parameters
atm_models = {  # US standard after Linsley
    1: {
        "a": 1e4 * np.array([-186.555305, -94.919, 0.61289, 0.0, 0.01128292]),
        "b": 1e4 * np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0]),
        "c": 1e-2 * np.array([994186.38, 878153.55, 636143.04, 772170.16, 1.0e9]),
        "h": 1e3 * np.array([4.0, 10.0, 40.0, 100.0]),
    },
    # southpole January after Lipari
    15: {
        "a": 1e4 * np.array([-113.139, -79.0635, -54.3888, 0.0, 0.00421033]),
        "b": 1e4 * np.array([1133.1, 1101.2, 1085.0, 1098.0, 1.0]),
        "c": 1e-2 * np.array([861730.0, 826340.0, 790950.0, 682800.0, 2.6798156e9]),
        "h": 1e3 * np.array([2.67, 5.33, 8.0, 100.0]),
    },
    # US standard after Keilhauer
    17: {
        "a": 1e4
        * np.array([-149.801663, -57.932486, 0.63631894, 4.35453690e-4, 0.01128292]),
        "b": 1e4 * np.array([1183.6071, 1143.0425, 1322.9748, 655.67307, 1.0]),
        "c": 1e-2 * np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9]),
        "h": 1e3 * np.array([7.0, 11.4, 37.0, 100.0]),
    },
    # Malargue January
    18: {
        "a": 1e4
        * np.array(
            [-136.72575606, -31.636643044, 1.8890234035, 3.9201867984e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.8298334, 1204.8233453, 1637.7703583, 735.96095023, 1.0]),
        "c": 1e-2
        * np.array([982815.95248, 754029.87759, 594416.83822, 733974.36972, 1e9]),
        "h": 1e3 * np.array([9.4, 15.3, 31.6, 100.0]),
    },
    # Malargue February
    19: {
        "a": 1e4
        * np.array(
            [-137.25655862, -31.793978896, 2.0616227547, 4.1243062289e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1176.0907565, 1197.8951104, 1646.4616955, 755.18728657, 1.0]),
        "c": 1e-2
        * np.array([981369.6125, 756657.65383, 592969.89671, 731345.88332, 1.0e9]),
        "h": 1e3 * np.array([9.2, 15.4, 31.0, 100.0]),
    },
    # Malargue March
    20: {
        "a": 1e4
        * np.array(
            [-132.36885162, -29.077046629, 2.090501509, 4.3534337925e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.6227784, 1215.3964677, 1617.0099282, 769.51991638, 1.0]),
        "c": 1e-2
        * np.array([972654.0563, 742769.2171, 595342.19851, 728921.61954, 1.0e9]),
        "h": 1e3 * np.array([9.6, 15.2, 30.7, 100.0]),
    },
    # Malargue April
    21: {
        "a": 1e4
        * np.array(
            [-129.9930412, -21.847248438, 1.5211136484, 3.9559055121e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.3291878, 1250.2922774, 1542.6248413, 713.1008285, 1.0]),
        "c": 1e-2
        * np.array([962396.5521, 711452.06673, 603480.61835, 735460.83741, 1.0e9]),
        "h": 1e3 * np.array([10.0, 14.9, 32.6, 100.0]),
    },
    # Malargue May
    22: {
        "a": 1e4
        * np.array(
            [-125.11468467, -14.591235621, 0.93641128677, 3.2475590985e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1169.9511302, 1277.6768488, 1493.5303781, 617.9660747, 1.0]),
        "c": 1e-2
        * np.array([947742.88769, 685089.57509, 609640.01932, 747555.95526, 1.0e9]),
        "h": 1e3 * np.array([10.2, 15.1, 35.9, 100.0]),
    },
    # Malargue June
    23: {
        "a": 1e4
        * np.array(
            [-126.17178851, -7.7289852811, 0.81676828638, 3.1947676891e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1171.0916276, 1295.3516434, 1455.3009344, 595.11713507, 1.0]),
        "c": 1e-2
        * np.array([940102.98842, 661697.57543, 612702.0632, 749976.26832, 1.0e9]),
        "h": 1e3 * np.array([10.1, 16.0, 36.7, 100.0]),
    },
    # Malargue July
    24: {
        "a": 1e4
        * np.array(
            [-126.17216789, -8.6182537514, 0.74177836911, 2.9350702097e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.7340688, 1258.9180079, 1450.0537141, 583.07727715, 1.0]),
        "c": 1e-2
        * np.array([934649.58886, 672975.82513, 614888.52458, 752631.28536, 1.0e9]),
        "h": 1e3 * np.array([9.6, 16.5, 37.4, 100.0]),
    },
    # Malargue August
    25: {
        "a": 1e4
        * np.array(
            [-123.27936204, -10.051493041, 0.84187346153, 3.2422546759e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1169.763036, 1251.0219808, 1436.6499372, 627.42169844, 1.0]),
        "c": 1e-2
        * np.array([931569.97625, 678861.75136, 617363.34491, 746739.16141, 1.0e9]),
        "h": 1e3 * np.array([9.6, 15.9, 36.3, 100.0]),
    },
    # Malargue September
    26: {
        "a": 1e4
        * np.array(
            [-126.94494665, -9.5556536981, 0.74939405052, 2.9823116961e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.8676453, 1251.5588529, 1440.8257549, 606.31473165, 1.0]),
        "c": 1e-2
        * np.array([936953.91919, 678906.60516, 618132.60561, 750154.67709, 1.0e9]),
        "h": 1e3 * np.array([9.5, 15.9, 36.3, 100.0]),
    },
    # Malargue October
    27: {
        "a": 1e4
        * np.array(
            [-133.13151125, -13.973209265, 0.8378263431, 3.111742176e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1176.9833473, 1244.234531, 1464.0120855, 622.11207419, 1.0]),
        "c": 1e-2
        * np.array([954151.404, 692708.89816, 615439.43936, 747969.08133, 1.0e9]),
        "h": 1e3 * np.array([9.5, 15.5, 36.5, 100.0]),
    },
    # Malargue November
    28: {
        "a": 1e4
        * np.array(
            [-134.72208165, -18.172382908, 1.1159806845, 3.5217025515e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1175.7737972, 1238.9538504, 1505.1614366, 670.64752105, 1.0]),
        "c": 1e-2
        * np.array([964877.07766, 706199.57502, 610242.24564, 741412.74548, 1.0e9]),
        "h": 1e3 * np.array([9.6, 15.3, 34.6, 100.0]),
    },
    # Malargue December
    29: {
        "a": 1e4
        * np.array(
            [-135.40825209, -22.830409026, 1.4223453493, 3.7512921774e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.644971, 1227.2753683, 1585.7130562, 691.23389637, 1.0]),
        "c": 1e-2
        * np.array([973884.44361, 723759.74682, 600308.13983, 738390.20525, 1.0e9]),
        "h": 1e3 * np.array([9.6, 15.6, 33.3, 100.0]),
    },
    # South Pole April (De Ridder)
    33: {
        "a": 1e4 * np.array([-69.7259, -2.79781, 0.262692, -0.0000841695, 0.00207722]),
        "b": 1e4 * np.array([1111.70, 1128.64, 1413.98, 587.688, 1]),
        "c": 1e-2 * np.array([766099.0, 641716.0, 588082.0, 693300.0, 5430320300]),
        "h": 1e3 * np.array([7.6, 22.0, 40.4, 100.0]),
    },
}


# GDAS atmosphere reader
class GDASAtmosphere:
    """Read and store a GDAS atmosphere profile for use with ``Atmosphere``.

    Parameters
    ----------
    gdas_file : str
        Path to the GDAS atmosphere file.
    """

    def __init__(self, gdas_file: str) -> None:
        self.gdas_file = gdas_file

        with open(self.gdas_file, "rb") as f:
            lines = f.readlines()

        # skip first entry (0) of the layer boundaries; convert cm → m
        self.layers = jnp.array(lines[1].strip(b"\n").split()[1:], dtype=float) / 100
        self.a = jnp.array(lines[2].strip(b"\n").split(), dtype=float) * 1e4
        self.b = jnp.array(lines[3].strip(b"\n").split(), dtype=float) * 1e4
        self.c = jnp.array(lines[4].strip(b"\n").split(), dtype=float) * 1e-2

        h_grid_np, n_grid_np = np.genfromtxt(self.gdas_file, unpack=True, skip_header=6)
        self.h_grid = jnp.array(h_grid_np)
        self.n_grid = jnp.array(n_grid_np)

    def get_gdas_atm_model(self) -> tuple:
        """Return the pre-loaded 5-layer model parameters ``(a, b, c, layers)``."""
        return self.a, self.b, self.c, self.layers

    def get_refractive_index(self, h: jnp.ndarray) -> jnp.ndarray:
        """Return the refractive index at height *h* (metres) via interpolation."""
        return jnp.interp(x=h, xp=self.h_grid, fp=self.n_grid)


# Unified atmosphere interface
class Atmosphere:
    """JAX-compatible atmosphere model (standard tabulated or GDAS).

    Parameters
    ----------
    model : int, default=17
        Index into ``atm_models`` (only used when *gdas_file* is ``None``).
    n0 : float
        Refractive index at ground level.
    observation_level : float
        Observer altitude in metres.
    gdas_file : str, optional
        Path to a GDAS atmosphere file.  If given, overrides *model*.
    curved : bool
        Reserved for future use (curved atmosphere support).
    """

    def __init__(
        self,
        model: int = 17,
        n0: float = (1 + 292e-6),
        observation_level: float = 0.0,
        gdas_file: Optional[str] = None,
        curved: bool = False,
    ) -> None:
        self._n0 = n0
        self._obs_lvl = observation_level
        self._gdas_model = None
        if gdas_file is not None:
            self._gdas_model = GDASAtmosphere(gdas_file)
            self.a, self.b, self.c, self.layers = self._gdas_model.get_gdas_atm_model()
        else:
            self.a = jnp.array(atm_models[model]["a"])
            self.b = jnp.array(atm_models[model]["b"])
            self.c = jnp.array(atm_models[model]["c"])
            self.layers = jnp.array(atm_models[model]["h"])

    @property
    def obs_lvl(self) -> float:
        return self._obs_lvl

    @obs_lvl.setter
    def obs_lvl(self, observation_level: float = 0.0) -> None:
        self._obs_lvl = observation_level

    def __get_density_from_height(self, height: jnp.ndarray) -> jnp.ndarray:
        height = jnp.asarray(height)

        dens = 0.0 * height

        # Build up from the top layer; earlier assignments are overwritten by
        # deeper layers, giving the correct piecewise-constant structure.
        dens = jnp.where(height <= _H_MAX,
                         self.b[4] / self.c[4],
                         dens)
        dens = jnp.where(height < self.layers[3],
                         self.b[3] * jnp.exp(-height / self.c[3]) / self.c[3],
                         dens)
        dens = jnp.where(height < self.layers[2],
                         self.b[2] * jnp.exp(-height / self.c[2]) / self.c[2],
                         dens)
        dens = jnp.where(height < self.layers[1],
                         self.b[1] * jnp.exp(-height / self.c[1]) / self.c[1],
                         dens)
        dens = jnp.where(height < self.layers[0],
                         self.b[0] * jnp.exp(-height / self.c[0]) / self.c[0],
                         dens)

        return dens

    def get_density(self, grammage: jnp.ndarray, zenith: jnp.ndarray) -> jnp.ndarray:
        """Return atmospheric density at the given slant depth and zenith angle."""
        vert_height = self.get_vertical_height(grammage * jnp.cos(zenith) * 1e4)
        return self.__get_density_from_height(vert_height)

    def get_refractive_index(self, grammage: jnp.ndarray, zenith: jnp.ndarray) -> jnp.ndarray:
        """Return the refractive index at the given slant depth and zenith angle."""
        vert_height = self.get_vertical_height(grammage * jnp.cos(zenith) * 1e4)
        if self._gdas_model is not None:
            return self._gdas_model.get_refractive_index(vert_height)
        density_at_h = self.__get_density_from_height(vert_height)
        density_at_obs = self.__get_density_from_height(jnp.array(self._obs_lvl))
        return (self._n0 - 1) * density_at_h / (density_at_obs + _EPSILON) + 1

    def get_cherenkov_angle(self, grammage: jnp.ndarray, zenith: jnp.ndarray) -> jnp.ndarray:
        """Return the Cherenkov angle at the given slant depth and zenith angle."""
        n = self.get_refractive_index(grammage, zenith)
        return jnp.arccos(jnp.minimum(1.0, 1.0 / n))

    def get_atmosphere(self, height: jnp.ndarray) -> jnp.ndarray:
        """Return the atmospheric grammage (g/cm²·1e4 → g/m²) at a given height."""
        y = jnp.where(height < self.layers[0],
                      self.a[0] + self.b[0] * jnp.exp(-height / self.c[0]),
                      self.a[1] + self.b[1] * jnp.exp(-height / self.c[1]))
        y = jnp.where(height < self.layers[1], y,
                      self.a[2] + self.b[2] * jnp.exp(-height / self.c[2]))
        y = jnp.where(height < self.layers[2], y,
                      self.a[3] + self.b[3] * jnp.exp(-height / self.c[3]))
        y = jnp.where(height < self.layers[3], y,
                      self.a[4] - self.b[4] * height / self.c[4])
        y = jnp.where(height < _H_MAX, y, 0.0)
        return y

    def get_vertical_height(self, at: jnp.ndarray) -> jnp.ndarray:
        """Return the vertical height (m) corresponding to a vertical grammage value."""
        at = jnp.asarray(at)

        bounds = jnp.array([
            self.get_atmosphere(self.layers[0]),
            self.get_atmosphere(self.layers[1]),
            self.get_atmosphere(self.layers[2]),
            self.get_atmosphere(self.layers[3]),
        ])

        # Start with the top layer (linear) and override towards lower layers
        h = -self.c[4] * (at - self.a[4]) / self.b[4]

        h = jnp.where(at > bounds[3],
                      -self.c[3] * jnp.log(_EPSILON + (at - self.a[3]) / self.b[3]),
                      h)
        h = jnp.where(at > bounds[2],
                      -self.c[2] * jnp.log(_EPSILON + (at - self.a[2]) / self.b[2]),
                      h)
        h = jnp.where(at > bounds[1],
                      -self.c[1] * jnp.log(_EPSILON + (at - self.a[1]) / self.b[1]),
                      h)
        h = jnp.where(at > bounds[0],
                      -self.c[0] * jnp.log(_EPSILON + (at - self.a[0]) / self.b[0]),
                      h)

        return h

    def get_geometric_distance_grammage(self, grammage: jnp.ndarray, zenith: jnp.ndarray) -> jnp.ndarray:
        """Return the geometric distance from the observer to a point at the given slant depth."""
        height = self.get_vertical_height(grammage * jnp.cos(zenith) * 1e4) + self._obs_lvl
        r = _R_EARTH + self._obs_lvl
        sqrt_arg = height**2 + 2 * r * height + r**2 * jnp.cos(zenith)**2
        return jnp.sqrt(jnp.maximum(0.0, sqrt_arg)) - r * jnp.cos(zenith)

    def get_xmax_from_distance(self, distance: jnp.ndarray, zenith: jnp.ndarray) -> jnp.ndarray:
        """Return the slant depth X_max (g/cm²) from the geometric distance to the shower maximum."""
        r = _R_EARTH + self._obs_lvl
        x = distance * jnp.sin(zenith)
        y = distance * jnp.cos(zenith) + r
        height_xmax = jnp.sqrt(jnp.maximum(0.0, x**2 + y**2)) - r + self._obs_lvl
        return self.get_atmosphere(height_xmax) * 1e-4 / (jnp.cos(zenith) + _EPSILON)
