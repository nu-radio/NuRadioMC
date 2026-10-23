"""
NIFTy forward model (footprintModel) for the LOFAR IFT cosmic-ray reconstruction.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

import logging
import os

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import numpy.typing as npt
from jax import vmap
from jax.scipy.interpolate import RegularGridInterpolator

from NuRadioReco.utilities.LOFAR import atmosphere as atm
from NuRadioReco.utilities.LOFAR import jaxHelpers as helper

# --- Constants ---
N_SEA_LEVEL = 2.8e-4
SCALE_HEIGHT = 8400.0

# --- Energy calibration constants (E_rad -> E_CR) ---
#
# Radiation energy scales with the cosmic-ray energy following the
# parametrisation of Glaser et al., "Simulation of radiation energy release in
# air showers", JCAP 09 (2016) 024 [arXiv:1606.01641]:
#
#   E_rad = A (sin^2(alpha) + a_CE^2) (B / B_ref)^B_exponent
#           (E_CR / E_scale)^B (1 + c_rho (rho_Xmax - rho_ref))   [MeV]
#
# with alpha the geomagnetic angle and B the local field strength. The charge
# excess term a_CE keeps E_rad finite as alpha -> 0, and the last factor
# corrects for the air density at shower maximum, which is why this is called
# the density calibration; rho_Xmax is evaluated at the effective Xmax in the
# event's own atmosphere.
#
# The coefficients were fitted on LOFAR CoREAS simulations in the 30-80 MHz band, giving
#
#   A     = 10.439 +- 0.080      a_CE  =  0.1635 +- 0.0032
#   B     =  2.039 +- 0.003      c_rho = -1.0812 (held fixed)
#
# with 6.4% residual scatter.
_ERAD_A = 10.438566054041386
_ERAD_B = 2.039088792498413
_ERAD_a_CE = 0.16352934045844977
_ERAD_c_rho = -1.08120000
_ERAD_rho_ref = 0.65
_ERAD_B_ref = 0.24
_ERAD_B_exponent = 1.8
_ERAD_E_scale = 1e18

# Cached US-standard atmosphere for events without a GDAS profile.
_ERAD_FALLBACK_ATMOSPHERE = None


def _erad_atmosphere(atmosphere):
    """The atmosphere rho(X_max) is evaluated in; the event's own if there is one."""
    global _ERAD_FALLBACK_ATMOSPHERE
    if atmosphere is not None:
        return atmosphere
    # The first call can occur while NIFTy traces the model. Keep cached
    # atmosphere arrays concrete rather than leaking JAX tracers into globals.
    with jax.ensure_compile_time_eval():
        if _ERAD_FALLBACK_ATMOSPHERE is None:
            _ERAD_FALLBACK_ATMOSPHERE = atm.Atmosphere()
    return _ERAD_FALLBACK_ATMOSPHERE


# Grid Generation Constants
GRID_PAD = 150.0
TARGET_RESOLUTION = 5.0
MAX_GRID_DIM = 400
MIN_GRID_DIM = 64

# Systematic CF parameters
SYST_CF_ZM = dict(offset_mean=0.00, offset_std=(1e-3, 1e-4))
SYST_TARGET_SIGMA = 0.05
SYST_CF_FL = dict(
    fluctuations=(SYST_TARGET_SIGMA, SYST_TARGET_SIGMA),
    loglogavgslope=(-5.0, 2.5),
    flexibility=(0.1, 0.5),
)
SYST_MULT_MIN = 0.8
SYST_MULT_MAX = 1.2

# Timing Correlated Fields
TIMING_CF_ZM_2 = dict(offset_mean=0.0, offset_std=(1e-10, 1e-11))
TIMING_CLIP_NS = 15.0
TIMING_CF_FLUCTUATION_SCALE = 1.6
TIMING_CF_LOGLOGAVGSLOPE = (-2.3, 1.0)
DEFAULT_LDF_RADIATION_ENERGY_SCATTER = 0.036

DEFAULT_FLUENCE_DXMAX_PRECISION_GPCM2 = 16.3
DEFAULT_TIMING_DXMAX_PRECISION_GPCM2 = 35.0


# Bound the fluence-Xmax nuisance in units of its prior sigma.
# The likelihood constrains the difference between raw Xmax and this offset.
FLUENCE_DXMAX_OFFSET_MAX_SIGMA = 3.0
FLUENCE_DXMAX_OFFSET_CLIP_SOFTNESS = 0.1

# Softness [g/cm^2] of the effective-Xmax bounds within calibrated spline support.
FLUENCE_XMAX_EFFECTIVE_CLIP_SOFTNESS_GPCM2 = 5.0

# Bound the timing-core displacement in units of its prior sigma.
CORE_TIMING_OFFSET_MAX_SIGMA = 3.0
CORE_TIMING_OFFSET_CLIP_SOFTNESS = 0.1

# Wavefront fit constants
WAVEFRONT_B_OFFSET_S = -3e-9
RHO_POLY_C3 = -5.864929e-05
RHO_POLY_C2 = 1.313198e-01
RHO_POLY_C1 = -9.791198e01
RHO_POLY_C0 = 4.748834e04
GAMMA_PRIOR_MEAN = 1.465
GAMMA_PRIOR_STD = 0.292
CONST_RHO_RES_MEAN = 0.0


def _soft_clip(z, lo, hi, softness):
    s = softness
    z = hi - s * jax.nn.softplus((hi - z) / s)  # smooth min(z, hi)
    z = lo + s * jax.nn.softplus((z - lo) / s)  # smooth max(., lo)
    return z


class footprintModel(jft.Model):
    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        magnetic_field_vector=None,
        params_Erad: dict | None = None,
        params_phi: dict | None = None,
        params_theta: dict | None = None,
        params_X_max: dict | None = None,
        params_X_max_timing: dict | None = None,
        params_X: dict | None = None,
        params_Y: dict | None = None,
        params_core_timing_offset: dict | None = None,
        params_t0: dict | None = None,
        noise_floor_per_antenna=None,
        atmosphere_path: str | None = None,
        prefix: str = "",
        timing_std_s: float = 1.5e-9,
        syst_mult_min: float = SYST_MULT_MIN,
        syst_mult_max: float = SYST_MULT_MAX,
        enable_syst_cf: bool = True,
        enable_timing_cf: bool = True,
        enable_ldf_energy_scale_uncertainty: bool = True,
        ldf_energy_scale_fractional_std: float = DEFAULT_LDF_RADIATION_ENERGY_SCATTER,
        enable_fluence_dxmax_precision: bool = True,
        fluence_dxmax_precision_gpcm2: float = DEFAULT_FLUENCE_DXMAX_PRECISION_GPCM2,
        enable_timing_dxmax_precision: bool = True,
        timing_dxmax_precision_gpcm2: float = DEFAULT_TIMING_DXMAX_PRECISION_GPCM2,
        grid_settings: dict | None = None,
    ):

        if magnetic_field_vector is None:
            magnetic_field_vector = np.array([0.004675, 0.186270, -0.456412])
        if params_Erad is None:
            params_Erad = {"mean": np.log(10**6.5), "std": 3.5}
        if params_phi is None:
            params_phi = {"a_min": np.pi, "a_max": 3 * np.pi}
        if params_theta is None:
            params_theta = {"a_min": 0.0, "a_max": np.pi}
        if params_X_max is None:
            params_X_max = {"a_min": 400.0, "a_max": 1200.0}
        if params_X is None:
            params_X = {"mean": 0.0, "std": 200.0}
        if params_Y is None:
            params_Y = {"mean": 0.0, "std": 200.0}
        if params_t0 is None:
            params_t0 = {"mean": 0.0, "std": 1000e-9}

        # Fixed attributes
        self.magnetic_field_vector = magnetic_field_vector
        self.x = x
        self.y = y
        self.atmosphere_path = atmosphere_path
        self.speedoflight = 299792458.0
        self.b_offset_s = WAVEFRONT_B_OFFSET_S

        self.enable_syst_cf = enable_syst_cf
        self.enable_timing_cf = enable_timing_cf
        self.enable_ldf_energy_scale_uncertainty = enable_ldf_energy_scale_uncertainty
        self.ldf_energy_scale_fractional_std = float(ldf_energy_scale_fractional_std)
        self.ldf_energy_scale_log_std = float(
            np.sqrt(np.log1p(self.ldf_energy_scale_fractional_std**2))
        )
        self.enable_fluence_dxmax_precision = enable_fluence_dxmax_precision
        self.fluence_dxmax_precision_gpcm2 = float(fluence_dxmax_precision_gpcm2)
        self.enable_timing_dxmax_precision = enable_timing_dxmax_precision
        self.timing_dxmax_precision_gpcm2 = float(timing_dxmax_precision_gpcm2)

        self.atmosphere = None
        if self.atmosphere_path:
            try:
                if not os.path.exists(self.atmosphere_path):
                    raise FileNotFoundError(self.atmosphere_path)
                self.atmosphere = atm.Atmosphere(gdas_file=self.atmosphere_path)
            except (OSError, ValueError, IndexError) as exc:
                logging.getLogger(__name__).warning(
                    "Cannot load event atmosphere %s; using US standard: %s",
                    self.atmosphere_path,
                    exc,
                )
                self.atmosphere_path = None

        # --- PRIORS ---
        self.log_Erad_prior = jft.prior.LogNormalPrior(
            **params_Erad, shape=(1,), name=prefix + "log_Erad"
        )
        self.phi = jft.prior.UniformPrior(**params_phi, shape=(1,), name=prefix + "phi")
        self.theta = jft.prior.UniformPrior(
            **params_theta, shape=(1,), name=prefix + "theta"
        )

        self.X_max_prior = jft.prior.UniformPrior(
            **params_X_max, shape=(1,), name=prefix + "X_max"
        )

        self.X_max_bounds = (float(params_X_max["a_min"]), float(params_X_max["a_max"]))

        self.X_max_timing_prior = jft.prior.UniformPrior(
            **(params_X_max if params_X_max_timing is None else params_X_max_timing),
            shape=(1,),
            name=prefix + "X_max_timing",
        )

        self.X_core = jft.prior.NormalPrior(**params_X, shape=(1,), name=prefix + "X")
        self.Y_core = jft.prior.NormalPrior(**params_Y, shape=(1,), name=prefix + "Y")
        self.t0 = jft.prior.NormalPrior(**params_t0, shape=(1,), name=prefix + "t0")

        self.core_timing_offset_prior = None
        self.core_timing_offset_std = 0.0
        if (
            params_core_timing_offset is not None
            and params_core_timing_offset["std"] > 0
        ):
            self.core_timing_offset_std = float(params_core_timing_offset["std"])
            self.core_timing_offset_prior = jft.prior.NormalPrior(
                **params_core_timing_offset,
                shape=(2,),
                name=prefix + "core_timing_offset",
            )
        # Per-antenna noise floor: the measured floor of each antenna is added to
        # that antenna's predicted fluence

        self.noise_floor_per_antenna = (
            None
            if noise_floor_per_antenna is None
            else jnp.asarray(np.asarray(noise_floor_per_antenna, dtype=np.float64))
        )
        # The floor is measured per antenna, so it is only meaningful when this
        # model is evaluated at those same antennas. generate_reco_plot rebuilds
        # the model over a plotting grid with the same model_kw; there the vector
        # cannot broadcast, and the right stand-in is the mean floor.
        if (
            self.noise_floor_per_antenna is not None
            and self.noise_floor_per_antenna.size != np.size(x)
        ):
            self.noise_floor_per_antenna = jnp.mean(self.noise_floor_per_antenna)
        self.noise_floor_scale = jft.prior.LogNormalPrior(
            mean=1.0, std=0.05, shape=(1,), name=prefix + "noise_floor_scale"
        )

        # Wavefront priors
        self.gamma_prior = jft.prior.NormalPrior(
            mean=GAMMA_PRIOR_MEAN,
            std=GAMMA_PRIOR_STD,
            shape=(1,),
            name=prefix + "gamma",
        )
        _const_rho_res = jnp.full((1,), CONST_RHO_RES_MEAN, dtype=jnp.float64)
        self.const_rho_residual = lambda x: _const_rho_res
        self.ldf_energy_scale_prior = (
            jft.prior.NormalPrior(
                mean=0.0, std=1.0, shape=(1,), name=prefix + "ldf_energy_scale"
            )
            if self.enable_ldf_energy_scale_uncertainty
            and self.ldf_energy_scale_fractional_std > 0.0
            else None
        )
        self.fluence_dxmax_offset_prior = (
            jft.prior.NormalPrior(
                mean=0.0, std=1.0, shape=(1,), name=prefix + "fluence_dxmax_offset"
            )
            if self.enable_fluence_dxmax_precision
            and self.fluence_dxmax_precision_gpcm2 > 0.0
            else None
        )
        self.timing_dxmax_offset_prior = (
            jft.prior.NormalPrior(
                mean=0.0, std=1.0, shape=(1,), name=prefix + "timing_dxmax_offset"
            )
            if self.enable_timing_dxmax_precision
            and self.timing_dxmax_precision_gpcm2 > 0.0
            else None
        )

        # --- Grid Calculation ---
        if grid_settings is not None:
            self.min_x = grid_settings["min_x"]
            self.min_y = grid_settings["min_y"]
            self.extent = grid_settings["extent"]
            self.dims = grid_settings["dims"]
            self.distances = self.extent / self.dims[0]
        else:
            min_x_data, max_x_data = np.min(x), np.max(x)
            min_y_data, max_y_data = np.min(y), np.max(y)
            center_x = (min_x_data + max_x_data) / 2.0
            center_y = (min_y_data + max_y_data) / 2.0
            span_x = max_x_data - min_x_data
            span_y = max_y_data - min_y_data
            max_span = max(span_x, span_y)
            self.extent = max_span + 2 * GRID_PAD
            self.min_x = center_x - self.extent / 2.0
            self.min_y = center_y - self.extent / 2.0

            calc_dim = int(np.ceil(self.extent / TARGET_RESOLUTION))
            calc_dim = max(MIN_GRID_DIM, min(calc_dim, MAX_GRID_DIM))
            if calc_dim % 2 != 0:
                calc_dim += 1
            self.dims = (calc_dim, calc_dim)
            self.distances = self.extent / self.dims[0]

        # Systematics limits (the fluence correlated field is soft-clipped to this
        # multiplicative window
        self.sys_mult_min = syst_mult_min
        self.sys_mult_max = syst_mult_max
        self._sys_log_min = jnp.log(self.sys_mult_min)
        self._sys_log_max = jnp.log(self.sys_mult_max)

        # --- Handle Correlated Fields ---
        cf_initializers = []

        if self.enable_syst_cf:
            cfm_fluence = jft.CorrelatedFieldMaker(prefix + "syst_cf")
            cfm_fluence.set_amplitude_total_offset(**SYST_CF_ZM)
            cfm_fluence.add_fluctuations(
                self.dims,
                distances=self.distances,
                **SYST_CF_FL,
                prefix="ax1",
                non_parametric_kind="power",
            )
            self.syst_cf_op = cfm_fluence.finalize()
            self.syst_cf_raw = self.syst_cf_op
            cf_initializers.append(self.syst_cf_op.init)
        else:
            zero_grid = jnp.zeros(self.dims)
            self.syst_cf_op = lambda x: zero_grid
            self.syst_cf_raw = self.syst_cf_op

        if self.enable_timing_cf:
            timing_cf_sigma = timing_std_s * TIMING_CF_FLUCTUATION_SCALE
            TIMING_CF_FL_2 = dict(
                fluctuations=(timing_cf_sigma, timing_cf_sigma),
                loglogavgslope=TIMING_CF_LOGLOGAVGSLOPE,
            )
            cfm_timing_2 = jft.CorrelatedFieldMaker(prefix + "timing_cf_2")
            cfm_timing_2.set_amplitude_total_offset(**TIMING_CF_ZM_2)
            cfm_timing_2.add_fluctuations(
                self.dims,
                distances=self.distances,
                **TIMING_CF_FL_2,
                prefix="ax1_time_2",
                non_parametric_kind="power",
            )
            self.timing_cf_op_2 = cfm_timing_2.finalize()
            cf_initializers.append(self.timing_cf_op_2.init)
        else:
            zero_grid = jnp.zeros(self.dims)
            self.timing_cf_op_2 = lambda x: zero_grid

        self.syst_cf = self._get_cherenkov_ring_cf_log

        # Initialize priors
        init = (
            self.log_Erad_prior.init
            | self.phi.init
            | self.theta.init
            | self.X_max_prior.init
            | self.X_max_timing_prior.init
            | self.X_core.init
            | self.Y_core.init
            | self.t0.init
            | self.noise_floor_scale.init
            | self.gamma_prior.init
        )
        if self.ldf_energy_scale_prior is not None:
            init = init | self.ldf_energy_scale_prior.init
        if self.fluence_dxmax_offset_prior is not None:
            init = init | self.fluence_dxmax_offset_prior.init
        if self.timing_dxmax_offset_prior is not None:
            init = init | self.timing_dxmax_offset_prior.init
        if self.core_timing_offset_prior is not None:
            init = init | self.core_timing_offset_prior.init
        for cf_init in cf_initializers:
            init = init | cf_init
        super().__init__(init=init)

    def Erad(self, x):
        return jnp.exp(self.log_Erad_prior(x))

    def calculate_ecr_jax(self, x):
        # Inverts, with rho the air density at X_max in kg/m^3:
        #   E_rad[MeV] = A*(sin²α+a²)*(B/B_ref)^1.8*(E_CR/E_scale)^B
        #                 *(1+c_rho*(rho-<rho>))
        zenith = self.theta(x).squeeze()
        azimuth = self.phi(x).squeeze()
        Erad_eV = (self.Erad(x) / self.get_energy_correction_factor(x)).squeeze()

        B_vect = jnp.array(self.magnetic_field_vector).squeeze()
        B_mag = jnp.linalg.norm(B_vect)

        s_x = jnp.sin(zenith) * jnp.cos(azimuth)
        s_y = jnp.sin(zenith) * jnp.sin(azimuth)
        s_z = jnp.cos(zenith)
        s_vec = jnp.stack([s_x, s_y, s_z]).squeeze()
        b_hat = B_vect / B_mag
        sin_alpha = jnp.clip(jnp.linalg.norm(jnp.cross(s_vec, b_hat)), 0.05, 1.0)

        bc_term = (B_mag / _ERAD_B_ref) ** _ERAD_B_exponent
        ce_term = sin_alpha**2 + _ERAD_a_CE**2

        rho = (
            _erad_atmosphere(self.atmosphere).get_density(
                self.X_max_effective(x).squeeze(), zenith
            )
            / 1.0e3
        )
        rho_term = 1.0 + _ERAD_c_rho * (rho - _ERAD_rho_ref)
        denom = 1e6 * _ERAD_A * bc_term * ce_term * rho_term
        E_cr = _ERAD_E_scale * jnp.power(Erad_eV / denom, 1.0 / _ERAD_B)
        return jnp.reshape(E_cr, (1,)) if self.Erad(x).ndim > 0 else E_cr

    def X_max(self, x):
        xmax_val = self.X_max_prior(x)
        return jnp.reshape(xmax_val, (1,)) if xmax_val.ndim > 0 else xmax_val

    def X_max_timing(self, x):
        xmax_val = self.X_max_timing_prior(x)
        return jnp.reshape(xmax_val, (1,)) if xmax_val.ndim > 0 else xmax_val

    def fluence_dxmax_offset(self, x):
        if self.fluence_dxmax_offset_prior is None:
            return jnp.zeros((1,), dtype=jnp.float64)
        sigma = self.fluence_dxmax_precision_gpcm2
        # Bound the excursion: this offset is degenerate with X_max, so an
        # unclipped latent lets the pair slide along a flat likelihood direction.
        z = _soft_clip(
            self.fluence_dxmax_offset_prior(x),
            -FLUENCE_DXMAX_OFFSET_MAX_SIGMA,
            FLUENCE_DXMAX_OFFSET_MAX_SIGMA,
            FLUENCE_DXMAX_OFFSET_CLIP_SOFTNESS,
        )
        return sigma * z

    def timing_dxmax_offset(self, x):
        if self.timing_dxmax_offset_prior is None:
            return jnp.zeros((1,), dtype=jnp.float64)
        return self.timing_dxmax_precision_gpcm2 * self.timing_dxmax_offset_prior(x)

    def calculate_rho_with_calibration(
        self,
        zenith,
        xmax_for_timing,
        gamma,
        const_rho_res,
        c3=None,
        c2=None,
        c1=None,
        c0=None,
        dxmax_offset=0.0,
    ):
        # Atmosphere returns g/m²
        x_atm_obs_gm2 = _erad_atmosphere(self.atmosphere).get_atmosphere(7.60)

        cos_eps, cos_soft = 0.02, 0.02
        cos_zenith = cos_eps + cos_soft * jax.nn.softplus(
            (jnp.cos(zenith) - cos_eps) / cos_soft
        )

        slant_depth_obs_gm2 = x_atm_obs_gm2 / cos_zenith
        d_xmax = (slant_depth_obs_gm2 - xmax_for_timing * 1e4) * 1e-4 + dxmax_offset

        c3 = RHO_POLY_C3 if c3 is None else c3
        c2 = RHO_POLY_C2 if c2 is None else c2
        c1 = RHO_POLY_C1 if c1 is None else c1
        c0 = RHO_POLY_C0 if c0 is None else c0

        const = c3 * d_xmax**3 + c2 * d_xmax**2 + c1 * d_xmax + c0
        const += const_rho_res

        const_floor, const_soft = 1.0e3, 1.0e2
        const = const_floor + const_soft * jax.nn.softplus(
            (const - const_floor) / const_soft
        )

        return (xmax_for_timing / const) * (cos_zenith**gamma)

    def calculate_rho(self, x, zenith, xmax):
        return self.calculate_rho_with_calibration(
            zenith,
            xmax,
            self.gamma_prior(x),
            self.const_rho_residual(x),
            dxmax_offset=self.timing_dxmax_offset(x),
        )

    def core(self, x):
        return self.X_core(x), self.Y_core(x)

    def core_timing_offset(self, x):
        """Displacement [m] of the wavefront's core from the fluence's, (dx, dy)."""
        if self.core_timing_offset_prior is None:
            return jnp.zeros((2,), dtype=jnp.float64)
        s = self.core_timing_offset_std
        return _soft_clip(
            self.core_timing_offset_prior(x),
            -CORE_TIMING_OFFSET_MAX_SIGMA * s,
            CORE_TIMING_OFFSET_MAX_SIGMA * s,
            CORE_TIMING_OFFSET_CLIP_SOFTNESS * s,
        )

    def core_timing(self, x):
        """The core the wavefront model uses; equals :meth:`core` when disabled."""
        off = self.core_timing_offset(x)
        return self.X_core(x) + off[0:1], self.Y_core(x) + off[1:2]

    def zen_and_az(self, x):
        return self.theta(x), self.phi(x)

    def ldf_dxmax_gpcm2(self, X_max, zenith):
        """Use the same observation altitude and units as the LDF."""
        column = _erad_atmosphere(self.atmosphere).get_atmosphere(0.0)
        return column * 1e-4 / jnp.cos(zenith) - X_max

    def ldf_energy_scale_factor(self, x):
        if self.ldf_energy_scale_prior is None:
            return jnp.array(1.0, dtype=jnp.float64)
        z = jnp.squeeze(self.ldf_energy_scale_prior(x))
        return jnp.exp(self.ldf_energy_scale_log_std * z)

    def X_max_effective(self, x):
        """Return raw Xmax minus its nuisance, bounded by prior and spline support."""
        lo, hi = self.X_max_bounds
        dx_lo, dx_hi = helper.ldf_dxmax_valid_range()
        slant = self.ldf_dxmax_gpcm2(0.0, self.theta(x))
        lo = jnp.maximum(lo, slant - dx_hi)
        hi = jnp.minimum(hi, slant - dx_lo)
        return _soft_clip(
            self.X_max(x) - self.fluence_dxmax_offset(x),
            lo,
            hi,
            FLUENCE_XMAX_EFFECTIVE_CLIP_SOFTNESS_GPCM2,
        )

    def X_max_combined(self, x):
        """The reported Xmax: the bare latent.

        The LDF is evaluated at :meth:`X_max_effective` instead, i.e. this
        latent minus the fluence nuisance and soft-clipped into the spline's
        support. Reporting the clipped value understates the
        posterior width.
        """
        return self.X_max(x)

    def fluence(
        self, E, X_max, azimuth, zenith, x_core, y_core, x, y, dxmax_offset=0.0
    ):
        ldf = helper.LDF(
            x,
            y,
            E,
            X_max - dxmax_offset,
            zenith,
            azimuth,
            [0, 0],
            magnetic_field_vector=self.magnetic_field_vector,
            atmosphere_path=self.atmosphere_path,
        )
        return ldf

    def get_arrival_time_differences(self, x):
        Xmax = self.X_max_timing(x)
        zenith = self.theta(x)
        azimuth = self.phi(x)
        x_core, y_core = self.core_timing(x)
        t0_val = self.t0(x)

        rho = self.calculate_rho(x, zenith, Xmax)

        mycstrafo = helper.CStrafoJAX(
            zenith, azimuth, magnetic_field_vector=self.magnetic_field_vector
        )
        pos_ground = jnp.array([self.x, self.y, jnp.zeros_like(self.x)])
        pos_shower_plane = mycstrafo.transform_to_vxB_vxvxB(
            pos_ground, core=[x_core, y_core, jnp.zeros_like(x_core)]
        )
        d = jnp.sqrt(pos_shower_plane[0] ** 2 + pos_shower_plane[1] ** 2)
        z_s = pos_shower_plane[2]
        term1 = (d * jnp.sin(rho)) ** 2
        term2 = (self.speedoflight * self.b_offset_s) ** 2
        tau_geo = (1 / self.speedoflight) * (
            jnp.sqrt(term1 + term2)
            + z_s * jnp.cos(rho)
            + self.speedoflight * self.b_offset_s
        )
        return t0_val + tau_geo

    def get_signal_fluence_without_cf(self, x, include_ldf_energy_scale=True):
        mycstrafo = helper.CStrafoJAX(
            self.theta(x), self.phi(x), magnetic_field_vector=self.magnetic_field_vector
        )
        pos_array = jnp.array([self.x, self.y, 7.6 * jnp.ones_like(self.x)])
        vxvxB_positions = mycstrafo.transform_to_vxB_vxvxB(
            pos_array,
            core=[self.X_core(x), self.Y_core(x), 7.6 * jnp.ones_like(self.X_core(x))],
        )
        # The offset that lands on the clipped effective Xmax, so the LDF is always
        # evaluated inside the window the prior bounds define.
        dxmax_off = self.X_max(x) - self.X_max_effective(x)
        energy_scale = (
            self.ldf_energy_scale_factor(x) if include_ldf_energy_scale else 1.0
        )

        def single_fluence(x_pos, y_pos):
            op_values = (op(x) for op in self.ops)
            return self.fluence(*op_values, x_pos, y_pos, dxmax_off)[0] * energy_scale

        return vmap(single_fluence)(
            vxvxB_positions[0, :], vxvxB_positions[1, :]
        ).squeeze()

    def get_energy_correction_factor(self, x):
        flu_signal_base = self.get_signal_fluence_without_cf(
            x, include_ldf_energy_scale=False
        )
        flu_signal_scaled = flu_signal_base * self.ldf_energy_scale_factor(x)
        syst_log_grid = self._get_cherenkov_ring_cf_log(x)
        xi = jnp.linspace(0, self.dims[0] - 1, self.dims[0])
        yi = jnp.linspace(0, self.dims[1] - 1, self.dims[1])
        points = jnp.stack(
            [
                (self.x - self.min_x) / self.distances,
                (self.y - self.min_y) / self.distances,
            ],
            axis=-1,
        )
        interp_syst = RegularGridInterpolator(
            (xi, yi), syst_log_grid, method="linear", bounds_error=False, fill_value=0.0
        )
        syst_log_at_pos = interp_syst(points)
        total_multiplier = jnp.exp(syst_log_at_pos)
        flu_with_cf = flu_signal_scaled * total_multiplier
        return jnp.sum(flu_signal_base) / (jnp.sum(flu_with_cf) + 1e-9)

    def _get_cherenkov_ring_cf_log(self, x):
        return _soft_clip(
            self.syst_cf_raw(x), self._sys_log_min, self._sys_log_max, 0.01
        )

    def __call__(self, x):
        mycstrafo = helper.CStrafoJAX(
            self.theta(x), self.phi(x), magnetic_field_vector=self.magnetic_field_vector
        )
        pos_array = jnp.array([self.x, self.y, 7.6 * jnp.ones_like(self.x)])
        vxvxB_positions = mycstrafo.transform_to_vxB_vxvxB(
            pos_array,
            core=[self.X_core(x), self.Y_core(x), 7.6 * jnp.ones_like(self.X_core(x))],
        )
        cf_on_grid_log = self.syst_cf(x)
        xi = jnp.linspace(0, self.dims[0] - 1, self.dims[0])
        yi = jnp.linspace(0, self.dims[1] - 1, self.dims[1])
        points = jnp.stack(
            [
                (self.x - self.min_x) / self.distances,
                (self.y - self.min_y) / self.distances,
            ],
            axis=-1,
        )
        interp_sys_fluence = RegularGridInterpolator(
            (xi, yi),
            cf_on_grid_log,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        total_multiplier = jnp.exp(
            interp_sys_fluence(points)
        ) * self.ldf_energy_scale_factor(x)

        timing_cf_on_grid_2 = self.timing_cf_op_2(x)
        interp_sys_timing_2 = RegularGridInterpolator(
            (xi, yi),
            timing_cf_on_grid_2,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        clipped_timing_correction = _soft_clip(
            interp_sys_timing_2(points),
            -TIMING_CLIP_NS * 1e-9,
            TIMING_CLIP_NS * 1e-9,
            1e-9,
        )

        arrival_times = self.get_arrival_time_differences(x) + clipped_timing_correction
        # Fit the same bounded Xmax used by the signal and energy helpers.
        dxmax_off = self.X_max(x) - self.X_max_effective(x)

        def single_fluence(x_pos, y_pos, mult_val):
            return (
                self.fluence(*(oo(x) for oo in self.ops), x_pos, y_pos, dxmax_off)[0]
                * mult_val
            )

        fluence_values = vmap(single_fluence)(
            vxvxB_positions[0, :], vxvxB_positions[1, :], total_multiplier
        )
        # Add the measured noise floor outside the per-antenna vmap; it is
        # ordered like the antennas, so it broadcasts against the fluences.
        fluence_values = fluence_values.squeeze() + self.noise_floor(x)
        return jnp.stack([fluence_values.squeeze(), arrival_times.squeeze()])

    def noise_floor(self, x):
        """Noise floor added to the predicted fluence of each antenna."""
        if self.noise_floor_per_antenna is None:
            return 0.0
        return self.noise_floor_per_antenna * self.noise_floor_scale(x).squeeze()

    @property
    def ops(self):
        return (self.Erad, self.X_max, self.phi, self.theta, self.X_core, self.Y_core)
