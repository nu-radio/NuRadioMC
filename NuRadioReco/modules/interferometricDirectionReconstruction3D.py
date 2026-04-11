"""
3D interferometric direction reconstruction for RNO-G.

Searches all three cylindrical coordinates (rho, phi, z) simultaneously using
a coarse 3D grid scan followed by L-BFGS-B optimizer refinement from the top-N
coarse peaks.
"""

import numpy as np
import itertools
import os
import sys
import yaml
import logging
import time
from collections import namedtuple
from functools import lru_cache

from scipy.signal import correlate, hilbert, windows
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

TableData = namedtuple('TableData', [
    'interp', 'values', 'r_min', 'z_min', 'dr_inv', 'dz_inv', 'nr', 'nz',
])

from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp

logger = logging.getLogger("reco3d.interferometric_reco_3d")

USE_NUMBA = False
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    pass

USE_CUPY = False
_FUSED_CORR_KERNEL = None
try:
    import cupy as cp
    # Check that a CUDA device is actually visible; cupy can import on
    # CPU-only nodes but fail at first kernel launch.
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_CUPY = True
        logger.info("CuPy GPU backend available (device count=%d)",
                    cp.cuda.runtime.getDeviceCount())

        # Fused CUDA kernel: one thread per grid point, iterates all pairs,
        # single read of each delay cell, single write of each output cell.
        # Pair-major delay_stack layout (n_pairs, n_points) gives coalesced
        # access across warps since adjacent threads handle adjacent points.
        _FUSED_CORR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void fused_correlator(
    const double* __restrict__ delay_stack,
    const double* __restrict__ corr_packed,
    const long long* __restrict__ corr_lens,
    const double* __restrict__ dts,
    const double* __restrict__ offsets,
    const double* __restrict__ pair_weights,
    const double w_sum,
    const int n_pairs,
    const long long n_points,
    const long long corr_stride,
    double* __restrict__ out
) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;

    double acc = 0.0;
    for (int p = 0; p < n_pairs; ++p) {
        double d = delay_stack[(long long)p * n_points + pt];
        if (isnan(d)) continue;
        double kf = (d - offsets[p]) / dts[p];
        long long k = (long long)floor(kf);
        long long clen = corr_lens[p];
        if (k < 0 || k >= clen - 1) continue;
        double alpha = kf - (double)k;
        double y0 = corr_packed[(long long)p * corr_stride + k];
        double y1 = corr_packed[(long long)p * corr_stride + k + 1];
        double v = y0 + (y1 - y0) * alpha;
        acc += v * pair_weights[p];
    }
    out[pt] = (w_sum > 0.0) ? (acc / w_sum) : 0.0;
}
''', 'fused_correlator')
except Exception:
    cp = None

USE_NUMBA_GROUPED = False
try:
    from fast_grouped_multiray import (
        grouped_multiray_numba, perpair_multiray_numba,
        pack_tt_grids, pack_corr_data, build_combo_table,
        _grouped_multiray_kernel,
    )
    if USE_NUMBA:
        USE_NUMBA_GROUPED = True
        logger.info("Numba grouped multiray kernels loaded")
except ImportError:
    pass

if USE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _scalar_grouped_corr_numba(
            tt_vals, tt_valid, corr_packed, corr_lengths,
            corr_dts, corr_offsets, pair_ch1, pair_ch2, pair_weights,
            combo_table, n_combos, n_pairs, w_sum):
        """Numba-compiled scalar grouped correlation at a single point.

        Args:
            tt_vals: float64 array (n_ch, n_rt). Travel times per channel per ray type.
            tt_valid: bool array (n_ch, n_rt). Whether each TT is valid.
            corr_packed: float64 array (n_pairs, max_corr_len). Padded correlations.
            corr_lengths: int64 array (n_pairs,). Actual correlation lengths.
            corr_dts: float64 array (n_pairs,). Sample spacing per pair.
            corr_offsets: float64 array (n_pairs,). Time offset per pair.
            pair_ch1: int64 array (n_pairs,). First channel index per pair.
            pair_ch2: int64 array (n_pairs,). Second channel index per pair.
            pair_weights: float64 array (n_pairs,). Weight per pair.
            combo_table: int64 array (n_combos, n_ch). Ray type per channel per combo.
            n_combos: int. Number of combos.
            n_pairs: int. Number of pairs.
            w_sum: float64. Sum of weights.

        Returns:
            Negative best weighted mean correlation (for minimization).
        """
        best_total = -np.inf
        for ci in range(n_combos):
            total = 0.0
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                rt1 = combo_table[ci, c1]
                rt2 = combo_table[ci, c2]
                if not tt_valid[c1, rt1] or not tt_valid[c2, rt2]:
                    continue
                delay = tt_vals[c1, rt1] - tt_vals[c2, rt2]

                dt = corr_dts[pidx]
                offset = corr_offsets[pidx]
                clen = corr_lengths[pidx]
                kf = (delay - offset) / dt
                k = int(np.floor(kf))
                if k < 0 or k >= clen - 1:
                    val = 0.0
                else:
                    alpha = kf - k
                    val = (corr_packed[pidx, k]
                           + (corr_packed[pidx, k + 1]
                              - corr_packed[pidx, k]) * alpha)
                total += val * pair_weights[pidx]
            if total > best_total:
                best_total = total

        if best_total == -np.inf:
            return 0.0
        return -best_total / w_sum if w_sum > 0.0 else 0.0

    @njit(parallel=True, fastmath=True)
    def _interp_uniform_numba(y, dt, offset, x):
        """Fast uniform-grid linear interpolation via Numba."""
        M = y.shape[0]
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            kf = (x[i] - offset) / dt
            k = int(np.floor(kf))
            if k < 0 or k >= M - 1:
                out[i] = np.nan
            else:
                alpha = kf - k
                out[i] = y[k] + (y[k + 1] - y[k]) * alpha
        return out

    @njit(fastmath=True, cache=True)
    def _scalar_singleray_corr_numba(
            rho, phi_rad, z, pa_x, pa_y,
            ant_xy, td_values, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total):
        """Fused single-point singleray correlation for the optimizer.

        Computes all channel travel times via bilinear table lookup, then
        accumulates the weighted correlation over all pairs, in a single
        Numba kernel. Replaces the per-call Python loops in
        ``_correlation_at_point`` for the non-multiray path.

        Args:
            rho, phi_rad, z: Scalar source coordinates (m, rad, m).
            pa_x, pa_y: PA center absolute coordinates (m).
            ant_xy: (n_ch, 2) float64, channel absolute (x, y).
            td_values: (n_ch, nr_max, nz_max) float64, packed TT tables.
            td_r_min, td_dr_inv: (n_ch,) float64 per-channel grid origin
                and inverse spacing in r.
            td_nr: (n_ch,) int64 per-channel grid size in r.
            td_z_min, td_dz_inv: same in z.
            td_nz: (n_ch,) int64 per-channel grid size in z.
            corr_packed: (n_pairs, max_corr_len) padded.
            corr_lengths, corr_dts, corr_offsets: (n_pairs,).
            pair_ch1, pair_ch2: (n_pairs,) int64.
            pair_weights: (n_pairs,) float64.
            w_total: sum(pair_weights).

        Returns:
            Negative weighted mean correlation (for minimization).
        """
        n_ch = ant_xy.shape[0]
        x_src = rho * np.cos(phi_rad) + pa_x
        y_src = rho * np.sin(phi_rad) + pa_y

        tts = np.empty(n_ch, dtype=np.float64)
        valid = np.zeros(n_ch, dtype=np.bool_)
        for ci in range(n_ch):
            dx = x_src - ant_xy[ci, 0]
            dy = y_src - ant_xy[ci, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < 1.0:
                r = 1.0
            ri = (r - td_r_min[ci]) * td_dr_inv[ci]
            zi = (z - td_z_min[ci]) * td_dz_inv[ci]
            i0 = int(np.floor(ri))
            j0 = int(np.floor(zi))
            nr_ch = td_nr[ci]
            nz_ch = td_nz[ci]
            if i0 < 0 or i0 >= nr_ch - 1 or j0 < 0 or j0 >= nz_ch - 1:
                continue
            fx = ri - i0
            fy = zi - j0
            v = ((1.0 - fx) * (1.0 - fy) * td_values[ci, i0, j0]
                 + fx * (1.0 - fy) * td_values[ci, i0 + 1, j0]
                 + (1.0 - fx) * fy * td_values[ci, i0, j0 + 1]
                 + fx * fy * td_values[ci, i0 + 1, j0 + 1])
            if v > 0.0 and np.isfinite(v):
                tts[ci] = v
                valid[ci] = True

        n_pairs = pair_ch1.shape[0]
        total = 0.0
        for pidx in range(n_pairs):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            if not valid[c1] or not valid[c2]:
                continue
            delay = tts[c1] - tts[c2]
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]
            kf = (delay - offset) / dt
            k = int(np.floor(kf))
            if k < 0 or k >= clen - 1:
                continue
            alpha = kf - k
            v = (corr_packed[pidx, k]
                 + (corr_packed[pidx, k + 1] - corr_packed[pidx, k]) * alpha)
            total += v * pair_weights[pidx]

        if w_total > 0.0:
            return -total / w_total
        return 0.0

    @njit(parallel=True, fastmath=True, cache=True)
    def _all_pairs_corr_numba(delay_T, corr_packed, corr_lengths,
                              corr_dts, corr_offsets, pair_weights):
        """Fused all-pairs weighted correlation sum.

        Parallelizes over grid points (outer prange). Inner sequential loop
        over pairs. Each thread accumulates into its own point, no race.

        Uses a points-major delay layout ``delay_T[pt, pidx]`` so the inner
        pair loop is stride-1 over cache lines. A pair-major layout is
        cache-unfriendly at grid sizes where a single point slice doesn't
        fit in L1, causing 3-5x slowdowns at production grid sizes.

        Args:
            delay_T: (n_points, n_pairs) float64. NaN = skip.
            corr_packed: (n_pairs, M_max) float64, zero-padded.
            corr_lengths: (n_pairs,) int64.
            corr_dts: (n_pairs,) float64.
            corr_offsets: (n_pairs,) float64.
            pair_weights: (n_pairs,) float64.

        Returns:
            (n_points,) float64 weighted-mean correlation.
        """
        n_points, n_pairs = delay_T.shape
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        out = np.empty(n_points, dtype=np.float64)
        inv_w_sum = 1.0 / w_sum if w_sum > 0.0 else 0.0

        for pt in prange(n_points):
            acc = 0.0
            for pidx in range(n_pairs):
                d = delay_T[pt, pidx]
                if not (d == d):  # NaN check
                    continue
                dt = corr_dts[pidx]
                offset = corr_offsets[pidx]
                clen = corr_lengths[pidx]
                kf = (d - offset) / dt
                k = int(np.floor(kf))
                if k < 0 or k >= clen - 1:
                    continue
                alpha = kf - k
                val = (corr_packed[pidx, k]
                       + (corr_packed[pidx, k + 1]
                          - corr_packed[pidx, k]) * alpha)
                acc += val * pair_weights[pidx]
            out[pt] = acc * inv_w_sum
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _bilinear_batch_numba(values, r_min, dr_inv, nr, z_min, dz_inv, nz,
                              r_coords, z_coords):
        """Batch 2D bilinear interpolation on a uniform grid.

        Args:
            values: 2D array of table values, shape (nr, nz).
            r_min: Minimum r value of the grid.
            dr_inv: Inverse of r spacing (1/dr).
            nr: Number of r grid points.
            z_min: Minimum z value of the grid.
            dz_inv: Inverse of z spacing (1/dz).
            nz: Number of z grid points.
            r_coords: 1D array of query r coordinates.
            z_coords: 1D array of query z coordinates.

        Returns:
            1D array of interpolated values. Out-of-bounds returns -inf.
        """
        n = r_coords.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            ri = (r_coords[i] - r_min) * dr_inv
            zi = (z_coords[i] - z_min) * dz_inv
            i0 = int(np.floor(ri))
            j0 = int(np.floor(zi))
            if i0 < 0 or i0 >= nr - 1 or j0 < 0 or j0 >= nz - 1:
                out[i] = -np.inf
            else:
                fx = ri - i0
                fy = zi - j0
                out[i] = ((1 - fx) * (1 - fy) * values[i0, j0]
                          + fx * (1 - fy) * values[i0 + 1, j0]
                          + (1 - fx) * fy * values[i0, j0 + 1]
                          + fx * fy * values[i0 + 1, j0 + 1])
        return out

    @njit(fastmath=True, cache=True)
    def _bilinear_scalar_numba(values, r_min, dr_inv, nr, z_min, dz_inv, nz,
                               r_val, z_val):
        """Single-point 2D bilinear interpolation on a uniform grid.

        Args:
            values: 2D array of table values, shape (nr, nz).
            r_min: Minimum r value of the grid.
            dr_inv: Inverse of r spacing (1/dr).
            nr: Number of r grid points.
            z_min: Minimum z value of the grid.
            dz_inv: Inverse of z spacing (1/dz).
            nz: Number of z grid points.
            r_val: Query r coordinate.
            z_val: Query z coordinate.

        Returns:
            Interpolated value, or -inf if out of bounds.
        """
        ri = (r_val - r_min) * dr_inv
        zi = (z_val - z_min) * dz_inv
        i0 = int(np.floor(ri))
        j0 = int(np.floor(zi))
        if i0 < 0 or j0 < 0:
            return -np.inf
        # Clamp to last valid cell for exact boundary queries
        if i0 >= nr - 1:
            if ri <= nr - 1 + 1e-9:
                i0 = nr - 2
                fx = 1.0
            else:
                return -np.inf
        else:
            fx = ri - i0
        if j0 >= nz - 1:
            if zi <= nz - 1 + 1e-9:
                j0 = nz - 2
                fy = 1.0
            else:
                return -np.inf
        else:
            fy = zi - j0
        return ((1 - fx) * (1 - fy) * values[i0, j0]
                + fx * (1 - fy) * values[i0 + 1, j0]
                + (1 - fx) * fy * values[i0, j0 + 1]
                + fx * fy * values[i0 + 1, j0 + 1])

USE_CPP_EXTENSION = False
try:
    cpp_path = os.path.join(os.path.dirname(__file__), "cpp")
    if cpp_path not in sys.path:
        sys.path.insert(0, cpp_path)
    from fast_delay_matrices_3d import compute_delay_matrices_3d as _compute_delay_matrices_cpp
    USE_CPP_EXTENSION = True
    logger.info("3D C++ extension loaded")
except (ImportError, OSError):
    logger.info("3D C++ extension not available, using Python fallback")


RAY_TYPES = ['direct', 'refracted', 'reflected']
SOLUTION_TYPES = ['solution_0', 'solution_1']
RAY_TYPE_COMBOS = list(itertools.product(RAY_TYPES, RAY_TYPES))


def _build_z_vec(z_min, z_max, n_z, spacing='linear', surface_offset=0.1):
    """Build a z-axis vector with linear or log spacing.

    Log spacing concentrates grid density near the ice surface (z=0) and is
    useful for near-surface sources (CR) when the search volume spans the
    full ice depth. Falls back to linear with a warning if the z range does
    not straddle zero.

    Args:
        z_min: Minimum z (meters, negative, below surface).
        z_max: Maximum z (meters, should be >= 0 for log mode).
        n_z: Number of grid points.
        spacing: 'linear' or 'log'.
        surface_offset: Minimum |z| for the shallowest log bin (meters).
            Only used when spacing='log'.

    Returns:
        np.ndarray of length n_z, sorted ascending.
    """
    if spacing == 'log':
        if z_max >= 0 and z_min < 0:
            z_depths = np.geomspace(surface_offset, -z_min, n_z)
            return -z_depths[::-1]
        logger.warning(
            "z_spacing='log' requires z_min < 0 and z_max >= 0; "
            "got [%s, %s]. Falling back to linear.", z_min, z_max)
    return np.linspace(z_min, z_max, n_z)


class InterferometricReco3D:
    """3D interferometric reconstruction: coarse grid + L-BFGS-B refinement."""

    # Depth groups for grouped combo selection. Channels in the same group
    # are assumed to see the same ray type. Channels not listed get their
    # own individual group.
    DEFAULT_DEPTH_GROUPS = {
        'pa': [0, 1, 2, 3],       # ~94-97m, 1m spacing
        'pa_hpol': [4, 8],        # ~92-93m HPOL on power string
        'helper_b': [9, 10],       # ~96-97m on string B
        'helper_b_hpol': [11],    # ~95m HPOL on string B
        'helper_c': [22, 23],      # ~96-97m on string C
        'helper_c_hpol': [21],    # ~95m HPOL on string C
    }

    def __init__(self):
        """Initialize with empty caches and default settings."""
        self._delay_matrix_cache = {}
        self._gpu_delay_stack_cache = {}
        self._interpolators = {}
        self._multiray_interpolators = {}
        self.ant_locs = None
        self._multi_ray_types = False
        self._multiray_combo_mode = 'per_pair'
        self._active_ray_types = RAY_TYPES
        self._n_ray_slots = len(RAY_TYPES)
        self._use_gpu = False
        # Grids smaller than this fall back to CPU path when GPU is active,
        # since kernel-launch overhead exceeds compute time for tiny grids.
        self._gpu_min_grid_cells = 10000
        # Use the fused all-pairs Numba kernel by default. Can be disabled
        # for debugging via config 'use_fused_correlator: false'.
        self._use_fused_correlator = True

    def _set_station_parameters(self, station, rho, phi, z, corr):
        """Set reconstruction parameters on station, if supported.

        Args:
            station: Station object.
            rho, phi, z, corr: Reconstructed position and correlation.
        """
        if not hasattr(station, 'set_parameter'):
            return
        station.set_parameter(stnp.rec_max_correlation, corr)
        station.set_parameter(stnp.rec_coord_0, rho * units.m)
        station.set_parameter(stnp.rec_coord_1, phi * units.deg)
        station.set_parameter(stnp.rec_coord_2, z * units.m)

    _KNOWN_CONFIG_KEYS = {
        'time_delay_tables', 'station_id', 'channels', 'limits', 'step_sizes',
        'coord_system', 'rec_type', 'fixed_coord',
        'coarse_limits', 'coarse_step_sizes', 'coarse_n_rho', 'coarse_n_z',
        'coarse_n_peaks', 'coarse_peak_separation',
        'n_z', 'z_spacing', 'z_surface_offset',
        'refine_step_sizes', 'refine_window', 'refine_n_peaks', 'refine_radius',
        'refine_levels',
        'pass2_step_sizes', 'pass2_coarse_step_sizes', 'pass2_n_rho',
        'pass2_n_z', 'pass2_coarse_n_z',
        'pass2_window', 'pass2_hierarchical',
        'pass2_coarse_n_peaks', 'pass2_coarse_peak_separation',
        'pass2_refine_window', 'z_profile_step',
        'hilbert_envelope_mode', 'use_hilbert_envelope',
        'apply_hann_window', 'correlation_normalization', 'interp_method',
        'apply_upsampling', 'apply_cw_removal', 'apply_cable_delays',
        'apply_bandpass', 'apply_cable_delay', 'apply_hw_phase_removal',
        'apply_dedispersion',
        'bandpass_band', 'bandpass_order', 'bandpass_filter_type',
        'cw_peak_prominence', 'cw_freq_band',
        'peak_separation_threshold',
        'helper_snr_threshold', 'surf_corr_z_max', 'surf_corr_zen_max',
        'mode', 'hierarchical', 'tdoa_mode',
        'multi_ray_types', 'multiray_combo_mode',
        'multiray_table_name_pattern', 'table_name_pattern', 'table_scheme',
        'optimizer_method', 'optimizer_maxiter', 'n_optimizer_seeds',
        'optimizer_rho_offsets',
        'skip_optimizer', 'use_tdoa_seed',
        'snr_pair_weighting', 'pair_weights',
        'save_results_to', 'detector_file', 'detector_date',
        'interpolation_method', 'table_type',
        'n_peaks_save', 'save_coherent_waveforms', 'n_coherent_waveforms',
        'polarization_groups', 'hpol_weight_scale',
        'validation', 'use_gpu', 'gpu_min_grid_cells', 'use_fused_correlator',
        'post_optimizer_mode', 'rho_scan_step',
        'refinement_envelope_mode', 'refinement_window', 'refinement_maxiter',
        'de_window', 'de_maxiter', 'de_popsize',
        'bh_window', 'bh_niter', 'bh_stepsize',
        'plane_wave_fallback', 'plane_wave_snr_threshold',
    }

    def begin(self, station_id, config, det):
        """Initialize interpolators and antenna positions.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict or str
            Configuration dictionary or path to YAML file.
        det : Detector
            Detector description object.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        self._validate_config(config)
        self._preload_tables(station_id, config)
        self.ant_locs = self._get_ant_locs(station_id, det)
        if 1 in self.ant_locs and 2 in self.ant_locs:
            self._pa_center = (self.ant_locs[1] + self.ant_locs[2]) / 2.0
        else:
            logger.warning(
                "Channels 1 and 2 not in ant_locs (available: %s). "
                "PA center defaulting to origin.", sorted(self.ant_locs.keys()))
            self._pa_center = np.zeros(3)

        want_gpu = bool(config.get('use_gpu', False))
        if want_gpu and not USE_CUPY:
            logger.warning(
                "use_gpu=True requested but CuPy or CUDA device is not "
                "available. Falling back to CPU path.")
            self._use_gpu = False
        else:
            self._use_gpu = want_gpu
        self._gpu_min_grid_cells = int(
            config.get('gpu_min_grid_cells', self._gpu_min_grid_cells))
        self._use_fused_correlator = bool(
            config.get('use_fused_correlator', self._use_fused_correlator))
        if self._use_gpu:
            logger.info(
                "InterferometricReco3D running on GPU (CuPy). "
                "Grids < %d cells fall back to CPU.",
                self._gpu_min_grid_cells)

    def _validate_config(self, config):
        """Check config for common mistakes and warn about unknown keys."""
        unknown = set(config.keys()) - self._KNOWN_CONFIG_KEYS
        if unknown:
            logger.warning(
                "Unknown config keys (ignored): %s. "
                "This module uses cylindrical coordinates (rho, phi, z). "
                "Keys like 'coord_system', 'rec_type', and 'fixed_coord' "
                "have no effect.", sorted(unknown)
            )

        if 'coord_system' in config:
            cs = config['coord_system']
            if cs != 'cylindrical':
                logger.warning(
                    "coord_system='%s' has no effect. This module always "
                    "uses cylindrical coordinates (rho, phi, z). The "
                    "'limits' key is interpreted as "
                    "[rho_min, rho_max, phi_min, phi_max, z_min, z_max]. "
                    "If you intended spherical coordinates, this config "
                    "will produce wrong results.", cs
                )

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_rz_interpolator(table_filename, interpolation_method):
        """Load R-Z travel time interpolator from .npz file.

        Fills interior NaN rows by averaging adjacent z slices. This
        handles the ice surface boundary (z=0) where ray tracers
        produce NaN but z=-1 and z=+1 are valid.

        Parameters
        ----------
        table_filename : str
            Path to .npz file with keys 'r_range_vals', 'z_range_vals', 'data'.
        interpolation_method : str
            Interpolation method for RegularGridInterpolator.

        Returns
        -------
        TableData
            Wraps the SciPy interpolator plus raw grid arrays for fast Numba
            bilinear interpolation.
        """
        f = np.load(table_filename)
        travel_time_table = f['data'].copy()
        r_range = f['r_range_vals']
        z_range = f['z_range_vals']

        nr, nz = travel_time_table.shape
        # Interior: average of neighbors on both sides
        for j in range(1, nz - 1):
            nan_mask = np.isnan(travel_time_table[:, j])
            if not nan_mask.any():
                continue
            below = travel_time_table[:, j - 1]
            above = travel_time_table[:, j + 1]
            fillable = nan_mask & np.isfinite(below) & np.isfinite(above)
            if fillable.any():
                travel_time_table[fillable, j] = (
                    below[fillable] + above[fillable]) / 2.0
        # Top boundary: linear extrapolation from j-2, j-1
        if nz >= 3:
            j = nz - 1
            nan_mask = np.isnan(travel_time_table[:, j])
            if nan_mask.any():
                v1 = travel_time_table[:, j - 2]
                v2 = travel_time_table[:, j - 1]
                fillable = nan_mask & np.isfinite(v1) & np.isfinite(v2)
                if fillable.any():
                    travel_time_table[fillable, j] = (
                        2.0 * v2[fillable] - v1[fillable])

        interp = RegularGridInterpolator(
            (r_range, z_range), travel_time_table,
            method=interpolation_method, bounds_error=False, fill_value=-np.inf
        )
        values = np.ascontiguousarray(travel_time_table, dtype=np.float64)
        dr = r_range[1] - r_range[0]
        dz = z_range[1] - z_range[0]
        return TableData(
            interp=interp,
            values=values,
            r_min=float(r_range[0]),
            z_min=float(z_range[0]),
            dr_inv=1.0 / dr,
            dz_inv=1.0 / dz,
            nr=len(r_range),
            nz=len(z_range),
        )

    def _preload_tables(self, station_id, config):
        """Load travel time interpolators for all channels.

        Loads per-ray-type tables (direct, refracted, reflected) when
        multi_ray_types is enabled. Falls back to single combined table.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict
            Must contain 'channels', 'time_delay_tables', and optionally
            'table_name_pattern' and 'interp_method'.
        """
        interp_method = config.get('interp_method', 'linear')
        table_base = config['time_delay_tables']
        self._multi_ray_types = config.get('multi_ray_types', False)
        self._multiray_combo_mode = config.get('multiray_combo_mode', 'per_pair')

        table_scheme = config.get('table_scheme', 'ray_type')
        if table_scheme == 'solution_ordered':
            self._active_ray_types = SOLUTION_TYPES
        else:
            self._active_ray_types = RAY_TYPES
        self._n_ray_slots = len(self._active_ray_types)

        self._interpolators = {}
        self._multiray_interpolators = {}

        if self._multi_ray_types:
            pattern = config.get(
                'multiray_table_name_pattern',
                'st{station_id}_ch{ch}_rz_table_{ray_type}.npz'
            )
            for ch in config['channels']:
                self._multiray_interpolators[ch] = {}
                for rt in self._active_ray_types:
                    fname = pattern.format(
                        station_id=station_id, ch=ch, ray_type=rt
                    )
                    table_file = os.path.join(
                        table_base, f"station{station_id}", fname
                    )
                    self._multiray_interpolators[ch][rt] = \
                        self._load_rz_interpolator(table_file, interp_method)
            logger.info("Loaded %s tables (%d types) for %d channels",
                        table_scheme, self._n_ray_slots,
                        len(config['channels']))
        else:
            pattern = config.get('table_name_pattern',
                                 'st{station_id}_ch{ch}_rz_table.npz')
            for ch in config['channels']:
                fname = pattern.format(station_id=station_id, ch=ch)
                table_file = os.path.join(
                    table_base, f"station{station_id}", fname
                )
                self._interpolators[ch] = self._load_rz_interpolator(
                    table_file, interp_method
                )

    @staticmethod
    def _get_ant_locs(station_id, det):
        """Get antenna positions with absolute Z.

        Parameters
        ----------
        station_id : int
            Station ID.
        det : Detector
            Detector description.

        Returns
        -------
        dict
            Channel ID -> [x_rel, y_rel, z_abs] array.
        """
        station_abs_pos = det.get_absolute_position(int(station_id))
        station_abs_z = station_abs_pos[2]
        locs = {}
        for ch in range(24):
            rel = np.array(det.get_relative_position(int(station_id), int(ch)))
            rel[2] += station_abs_z
            locs[ch] = rel
        return locs

    _MAX_GRID_POINTS = 50_000_000

    def _generate_coord_arrays(self, config):
        """Generate 1D coordinate arrays from 6-element limits.

        Parameters
        ----------
        config : dict
            Must contain 'limits' (6 elements) and 'step_sizes' (3 elements).

        Returns
        -------
        tuple
            (rho_vec, phi_vec, z_vec) in physical units (m, rad, m).
        """
        rho_min, rho_max, phi_min, phi_max, z_min, z_max = config['limits']
        d_rho, d_phi, d_z = config['step_sizes']

        if z_max > 0:
            raise ValueError(
                f"z_max={z_max} is above the ice surface. Travel time "
                f"tables only cover in-ice positions (z <= 0). "
                f"Use negative z values, e.g. limits: "
                f"[{rho_min}, {rho_max}, {phi_min}, {phi_max}, "
                f"-{abs(z_max)}, {z_min}]"
            )

        rho_min = max(rho_min, 1.0)  # avoid R=0 table edge effects

        z_spacing = config.get('z_spacing', 'linear')
        n_z_cfg = config.get('n_z', 0)

        n_rho = len(np.arange(rho_min, rho_max + d_rho, d_rho))
        n_phi = len(np.arange(phi_min, phi_max, d_phi))
        if n_z_cfg > 0:
            n_z = n_z_cfg
        else:
            n_z = len(np.arange(z_min, z_max + d_z, d_z))
        n_total = n_rho * n_phi * n_z

        if n_total > self._MAX_GRID_POINTS:
            raise ValueError(
                f"Grid has {n_total:,} points "
                f"({n_rho} rho x {n_phi} phi x {n_z} z), which exceeds "
                f"the {self._MAX_GRID_POINTS:,} point limit. This would "
                f"require excessive memory. Use 'coarse_limits' and "
                f"'coarse_step_sizes' with the hierarchical search "
                f"(set hierarchical: true) for large search volumes, "
                f"or increase step_sizes."
            )

        if not config.get('hierarchical', False):
            logger.warning(
                "Running flat grid scan with %s points. Consider using "
                "hierarchical: true with coarse_limits/coarse_step_sizes "
                "for better performance.", f"{n_total:,}"
            )

        rho_vec = np.arange(rho_min, rho_max + d_rho, d_rho)
        phi_vec = np.arange(phi_min, phi_max, d_phi) * (np.pi / 180.0)
        if n_z_cfg > 0:
            z_surf_offset = config.get('z_surface_offset', 0.1)
            z_vec = _build_z_vec(
                z_min, z_max, n_z_cfg, z_spacing, z_surf_offset)
        else:
            z_vec = np.arange(z_min, z_max + d_z, d_z)

        return rho_vec, phi_vec, z_vec

    def _build_source_enu_matrix(self, rho_vec, phi_vec, z_vec):
        """Build 3D matrix of source ENU positions.

        Parameters
        ----------
        rho_vec : array
            Radial distances in meters.
        phi_vec : array
            Azimuths in radians.
        z_vec : array
            Depths in meters (absolute).

        Returns
        -------
        np.ndarray
            Shape (n_rho, n_phi, n_z, 3) with [x, y, z] at each point.
        """
        rho_g, phi_g, z_g = np.meshgrid(rho_vec, phi_vec, z_vec, indexing='ij')
        x = rho_g * np.cos(phi_g) + self._pa_center[0]
        y = rho_g * np.sin(phi_g) + self._pa_center[1]

        return np.stack((x, y, z_g), axis=-1)

    def _compute_rho_and_coords(self, src_enu, channels):
        """Compute per-channel (R, Z) coordinate buffers for table lookups.

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        dict
            Channel ID -> (flat_size, 2) array of [R, Z] coordinates.
        """
        grid_shape = src_enu.shape[:3]
        flat_size = int(np.prod(grid_shape))
        xy_positions = src_enu[..., :2]
        z_grid = src_enu[..., 2]

        dx = np.empty(grid_shape, dtype=np.float64)
        dy = np.empty(grid_shape, dtype=np.float64)

        coords_per_ch = {}
        for ch in channels:
            pos = self.ant_locs[ch]
            np.subtract(xy_positions[..., 0], pos[0], out=dx)
            np.subtract(xy_positions[..., 1], pos[1], out=dy)
            np.multiply(dx, dx, out=dx)
            np.multiply(dy, dy, out=dy)
            np.add(dx, dy, out=dx)
            np.sqrt(dx, out=dx)
            np.maximum(dx, 1.0, out=dx)

            buf = np.empty((flat_size, 2), dtype=np.float64)
            buf[:, 0] = dx.ravel()
            buf[:, 1] = z_grid.ravel()
            coords_per_ch[ch] = buf

        return coords_per_ch

    def _compute_delay_matrices(self, src_enu, channels):
        """Compute pairwise time delay matrices over a 3D grid.

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        list
            One 3D delay matrix per channel pair.
        """
        grid_shape = src_enu.shape[:3]
        coords_per_ch = self._compute_rho_and_coords(src_enu, channels)

        travel_times = {}
        for ch in channels:
            td = self._interpolators[ch]
            if USE_NUMBA:
                tt_flat = _bilinear_batch_numba(
                    td.values, td.r_min, td.dr_inv, td.nr,
                    td.z_min, td.dz_inv, td.nz,
                    coords_per_ch[ch][:, 0], coords_per_ch[ch][:, 1])
                travel_times[ch] = tt_flat.reshape(grid_shape)
            else:
                travel_times[ch] = td.interp(
                    coords_per_ch[ch]
                ).reshape(grid_shape)

        ch_pairs = list(itertools.combinations(channels, 2))
        return [travel_times[c1] - travel_times[c2] for c1, c2 in ch_pairs]

    def _compute_tt_multiray(self, src_enu, channels):
        """Compute per-channel, per-ray-type travel time grids.

        Returns only the travel times (not delay matrices), keeping memory at
        O(n_channels * 3 * grid) instead of O(n_pairs * 9 * grid).

        Parameters
        ----------
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        dict
            Maps channel_id -> {ray_type_name -> travel_time_grid}.
            Only ray types with valid data are included.
        """
        grid_shape = src_enu.shape[:3]
        coords_per_ch = self._compute_rho_and_coords(src_enu, channels)

        tt_all = {}
        for ch in channels:
            tt_all[ch] = {}
            for rt in self._active_ray_types:
                td = self._multiray_interpolators[ch][rt]
                if USE_NUMBA:
                    tt_flat = _bilinear_batch_numba(
                        td.values, td.r_min, td.dr_inv, td.nr,
                        td.z_min, td.dz_inv, td.nz,
                        coords_per_ch[ch][:, 0], coords_per_ch[ch][:, 1])
                    tt = tt_flat.reshape(grid_shape)
                else:
                    tt = td.interp(coords_per_ch[ch]).reshape(grid_shape)
                if np.any(np.isfinite(tt) & (tt > 0)):
                    tt_all[ch][rt] = tt

        return tt_all

    def _get_t_delay_matrices(self, station_id, config, src_enu, channels):
        """Compute pairwise time delay matrices, cached by config.

        Parameters
        ----------
        station_id : int
            Station ID.
        config : dict
            Configuration dictionary with 'limits' and 'step_sizes'.
        src_enu : np.ndarray
            Source ENU matrix, shape (n_rho, n_phi, n_z, 3).
        channels : list
            Channel IDs.

        Returns
        -------
        list
            One 3D delay matrix per channel pair.
        """
        cache_key = (
            station_id,
            tuple(sorted(channels)),
            tuple(config['limits']),
            tuple(config['step_sizes']),
        )
        if cache_key in self._delay_matrix_cache:
            return self._delay_matrix_cache[cache_key]

        delay_matrices = self._compute_delay_matrices(src_enu, channels)
        self._delay_matrix_cache[cache_key] = delay_matrices
        return delay_matrices

    def _correlator(self, corr_data, delay_matrices, pair_weights=None):
        """Compute correlation map over the 3D grid.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair from
            ``_prepare_corr_funcs``.
        delay_matrices : list of array
            3D delay matrices, one per pair.
        pair_weights : list or None
            Per-pair weights.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr, pair_corr_maps)
        """
        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        pair_corr = np.full((n_pairs, *grid_shape), np.nan, dtype=np.float64)

        for pidx in range(n_pairs):
            corr_arr, dt, offset = corr_data[pidx]
            delays = delay_matrices[pidx]
            valid = np.isfinite(delays)

            if np.any(valid):
                flat_delays = delays[valid].ravel().astype(np.float64)
                if USE_NUMBA:
                    vals = _interp_uniform_numba(corr_arr, dt, offset,
                                                 flat_delays)
                else:
                    M = len(corr_arr)
                    time_lags = np.arange(M) * dt + offset
                    vals = np.interp(flat_delays, time_lags, corr_arr)

                pair_corr[pidx][valid] = vals

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64).reshape(-1, 1, 1, 1)
            w_sum = np.nansum(w)
            mean_corr = np.nansum(w * pair_corr, axis=0) / w_sum if w_sum > 0 else np.nanmean(pair_corr, axis=0)
        else:
            mean_corr = np.nansum(pair_corr, axis=0) / n_pairs

        max_corr = float(np.nanmax(mean_corr)) if not np.all(np.isnan(mean_corr)) else np.nan

        return mean_corr, max_corr, pair_corr

    def _interp_delays(self, corr_arr, dt, offset, delays):
        """Interpolate a correlation function at given delay values.

        Parameters
        ----------
        corr_arr : np.ndarray
            1D correlation array.
        dt : float
            Sample spacing.
        offset : float
            Time offset of first sample.
        delays : np.ndarray
            Flat array of delay values to interpolate at.

        Returns
        -------
        np.ndarray
            Interpolated correlation values.
        """
        if USE_NUMBA:
            return _interp_uniform_numba(corr_arr, dt, offset, delays)
        M = len(corr_arr)
        time_lags = np.arange(M) * dt + offset
        return np.interp(delays, time_lags, corr_arr)

    def _correlator_lean(self, corr_data, delay_matrices, pair_weights=None,
                         delay_cache_key=None):
        """Memory-efficient correlator that accumulates in place.

        Unlike ``_correlator``, does not allocate the full (n_pairs, *grid)
        array. Returns only the weighted mean correlation map.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        delay_matrices : list of array
            3D delay matrices, one per pair.
        pair_weights : list or None
            Per-pair weights.
        delay_cache_key : hashable or None
            If provided and GPU path is active, cache the stacked delay
            matrices on the GPU under this key. The caller is responsible
            for ensuring the key uniquely identifies the geometry of
            ``delay_matrices``. Pass None to skip caching (safe default).

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        if self._use_gpu and USE_CUPY:
            # Small grids (refine stages) are dominated by kernel launch
            # overhead; run them on CPU. Threshold chosen so the coarse
            # scan (>=100k cells) always goes to GPU.
            grid_size = int(np.prod(delay_matrices[0].shape))
            if grid_size >= self._gpu_min_grid_cells:
                return self._correlator_lean_gpu(
                    corr_data, delay_matrices, pair_weights=pair_weights,
                    delay_cache_key=delay_cache_key)

        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
            w_sum = float(w.sum())
        else:
            w = np.ones(n_pairs, dtype=np.float64)
            w_sum = float(n_pairs)

        if USE_NUMBA and self._use_fused_correlator:
            # Fused all-pairs kernel: one launch, parallel over cells.
            # Points-major layout so the inner pair loop is cache-friendly.
            n_points = int(np.prod(grid_shape))
            # Cache the stacked delay_T by cache key to avoid rebuilding
            # every call for stable coarse grids. Key None means skip cache.
            delay_T = None
            if delay_cache_key is not None:
                if not hasattr(self, '_cpu_delay_T_cache'):
                    self._cpu_delay_T_cache = {}
                cached = self._cpu_delay_T_cache.get(delay_cache_key)
                if cached is not None and cached.shape[0] == n_points and cached.shape[1] == n_pairs:
                    delay_T = cached
            if delay_T is None:
                delay_T = np.empty((n_points, n_pairs), dtype=np.float64)
                for pidx in range(n_pairs):
                    delay_T[:, pidx] = delay_matrices[pidx].reshape(-1)
                if delay_cache_key is not None:
                    self._cpu_delay_T_cache[delay_cache_key] = delay_T

            corr_lens = np.array([c[0].shape[0] for c in corr_data],
                                 dtype=np.int64)
            M_max = int(corr_lens.max())
            corr_packed = np.zeros((n_pairs, M_max), dtype=np.float64)
            dts = np.empty(n_pairs, dtype=np.float64)
            offsets = np.empty(n_pairs, dtype=np.float64)
            for pidx, (corr_arr, dt, offset) in enumerate(corr_data):
                corr_packed[pidx, :corr_arr.shape[0]] = corr_arr
                dts[pidx] = dt
                offsets[pidx] = offset

            flat = _all_pairs_corr_numba(
                delay_T, corr_packed, corr_lens, dts, offsets, w)
            mean_corr = flat.reshape(grid_shape)
        else:
            mean_corr = np.zeros(grid_shape, dtype=np.float64)
            for pidx in range(n_pairs):
                corr_arr, dt, offset = corr_data[pidx]
                delays = delay_matrices[pidx]
                valid = np.isfinite(delays)

                if not np.any(valid):
                    continue

                flat_delays = delays[valid].ravel().astype(np.float64)
                vals = self._interp_delays(corr_arr, dt, offset, flat_delays)
                np.nan_to_num(vals, copy=False, nan=0.0)
                mean_corr[valid] += vals * w[pidx]

            if w_sum > 0:
                mean_corr /= w_sum

        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _stack_delay_matrices_gpu(self, delay_matrices, cache_key=None):
        """Stack per-pair delay matrices into one GPU array.

        When cache_key is provided, stores the stacked GPU tensor under that
        key and returns the cached value on subsequent calls, skipping the
        host→device transfer. When cache_key is None, transfers fresh every
        call. The caller owns the key: it must be stable across calls that
        share the same geometry (same grid, same pair ordering) and unique
        otherwise.

        Args:
            delay_matrices: List of numpy delay matrices, one per pair.
            cache_key: Optional hashable cache key.

        Returns:
            cupy array of shape (n_pairs, *grid_shape), dtype float64.
        """
        if cache_key is not None:
            cached = self._gpu_delay_stack_cache.get(cache_key)
            if cached is not None and cached.shape[0] == len(delay_matrices):
                return cached
        stacked = np.stack(delay_matrices, axis=0).astype(np.float64)
        stacked_gpu = cp.asarray(stacked)
        if cache_key is not None:
            self._gpu_delay_stack_cache[cache_key] = stacked_gpu
        return stacked_gpu

    def _correlator_lean_gpu(self, corr_data, delay_matrices, pair_weights=None,
                             delay_cache_key=None):
        """Batched GPU correlator: one fused CUDA kernel over all pairs and cells.

        Uses a custom CUDA RawKernel when available (much less HBM traffic
        than the cupy elementwise path), otherwise falls back to vectorized
        cupy. The delay-matrix stack is cached on the GPU across calls so
        only the (small) per-event correlation arrays transfer each time.

        Args:
            corr_data: List of (corr_array, dt, offset) per pair.
            delay_matrices: List of 3D delay arrays, one per pair.
            pair_weights: Per-pair weights or None.

        Returns:
            (mean_corr_map, max_corr) with mean_corr_map as a numpy array.
        """
        n_pairs = len(corr_data)
        grid_shape = delay_matrices[0].shape

        if pair_weights is not None:
            w_np = np.asarray(pair_weights, dtype=np.float64)
        else:
            w_np = np.ones(n_pairs, dtype=np.float64)
        w_sum = float(w_np.sum())

        # Stack delay matrices (optionally cached on GPU across calls).
        delay_gpu = self._stack_delay_matrices_gpu(
            delay_matrices, cache_key=delay_cache_key)

        # Stack corr arrays (varies per event). Pad to max length in case
        # pair trace lengths differ.
        corr_lens = [c[0].shape[0] for c in corr_data]
        M_max = max(corr_lens)
        corr_stack = np.zeros((n_pairs, M_max), dtype=np.float64)
        dts = np.empty(n_pairs, dtype=np.float64)
        offsets = np.empty(n_pairs, dtype=np.float64)
        M_per_pair = np.empty(n_pairs, dtype=np.int64)
        for p, (corr_arr, dt, offset) in enumerate(corr_data):
            corr_stack[p, :corr_arr.shape[0]] = corr_arr
            dts[p] = dt
            offsets[p] = offset
            M_per_pair[p] = corr_arr.shape[0]

        corr_stack_gpu = cp.asarray(corr_stack)
        dts_gpu = cp.asarray(dts)
        offsets_gpu = cp.asarray(offsets)
        M_gpu = cp.asarray(M_per_pair)
        w_gpu = cp.asarray(w_np)

        n_points = int(np.prod(grid_shape))

        if _FUSED_CORR_KERNEL is not None:
            # Ensure contiguous layout required by the RawKernel: delay_gpu
            # must be (n_pairs, n_points) C-contiguous. Reshape the cached
            # stack once.
            delay_flat = delay_gpu.reshape(n_pairs, n_points)
            if not delay_flat.flags.c_contiguous:
                delay_flat = cp.ascontiguousarray(delay_flat)
            mean_corr_gpu = cp.empty(n_points, dtype=cp.float64)
            corr_stride = corr_stack_gpu.strides[0] // corr_stack_gpu.itemsize

            threads = 256
            blocks = (n_points + threads - 1) // threads
            _FUSED_CORR_KERNEL(
                (blocks,), (threads,),
                (delay_flat, corr_stack_gpu, M_gpu,
                 dts_gpu, offsets_gpu, w_gpu,
                 cp.float64(w_sum),
                 np.int32(n_pairs), np.int64(n_points),
                 np.int64(corr_stride),
                 mean_corr_gpu)
            )
            mean_corr_gpu = mean_corr_gpu.reshape(grid_shape)
        else:
            # Fallback: elementwise cupy path (many kernel launches).
            dts_gpu = dts_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            offsets_gpu = offsets_gpu.reshape(
                (n_pairs,) + (1,) * len(grid_shape))
            M_gpu = M_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            w_gpu = w_gpu.reshape((n_pairs,) + (1,) * len(grid_shape))
            kf = (delay_gpu - offsets_gpu) / dts_gpu
            k = cp.floor(kf).astype(cp.int64)
            alpha = kf - k
            in_bounds = (k >= 0) & (k < (M_gpu - 1)) & cp.isfinite(delay_gpu)
            k_safe = cp.where(in_bounds, k, 0)
            pair_idx = cp.arange(n_pairs).reshape(
                (n_pairs,) + (1,) * len(grid_shape))
            y0 = corr_stack_gpu[pair_idx, k_safe]
            y1 = corr_stack_gpu[pair_idx, cp.minimum(k_safe + 1, M_gpu - 1)]
            vals = y0 + (y1 - y0) * alpha
            vals = cp.where(in_bounds, vals, 0.0)
            weighted = vals * w_gpu
            mean_corr_gpu = weighted.sum(axis=0)
            if w_sum > 0:
                mean_corr_gpu /= w_sum

        mean_corr = cp.asnumpy(mean_corr_gpu)
        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    def _correlator_lean_multiray(self, corr_data, tt_all, channels,
                                  pair_weights=None):
        """Multi-ray-type correlator: take max across ray combinations per pair.

        Computes delays inline from per-channel travel times to avoid
        materializing all 9*n_pairs delay matrices simultaneously. Memory
        stays at O(grid) instead of O(n_pairs * 9 * grid).

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        tt_all : dict
            From ``_compute_tt_multiray``. Maps ch -> {ray_type -> grid}.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        ch_pairs = list(itertools.combinations(channels, 2))
        n_pairs = len(ch_pairs)

        grid_shape = None
        for ch in channels:
            for rt in tt_all.get(ch, {}):
                grid_shape = tt_all[ch][rt].shape
                break
            if grid_shape is not None:
                break
        if grid_shape is None:
            return np.zeros(1), np.nan

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
            w_sum = float(w.sum())
        else:
            w = np.ones(n_pairs, dtype=np.float64)
            w_sum = float(n_pairs)

        mean_corr = np.zeros(grid_shape, dtype=np.float64)
        best_corr = np.empty(grid_shape, dtype=np.float64)

        for pidx, (c1, c2) in enumerate(ch_pairs):
            corr_arr, dt, offset = corr_data[pidx]
            tt1_dict = tt_all.get(c1, {})
            tt2_dict = tt_all.get(c2, {})

            if not tt1_dict or not tt2_dict:
                continue

            best_corr[:] = -np.inf

            for rt1, tt1 in tt1_dict.items():
                for rt2, tt2 in tt2_dict.items():
                    delay = tt1 - tt2
                    valid = np.isfinite(delay)
                    if not np.any(valid):
                        continue

                    flat_delays = delay[valid].ravel().astype(np.float64)
                    vals = self._interp_delays(corr_arr, dt, offset,
                                               flat_delays)
                    np.nan_to_num(vals, copy=False, nan=0.0)

                    current = best_corr[valid]
                    np.maximum(current, vals, out=current)
                    best_corr[valid] = current

            no_combo = best_corr == -np.inf
            best_corr[no_combo] = 0.0

            mean_corr += best_corr * w[pidx]

        if w_sum > 0:
            mean_corr /= w_sum

        max_corr = float(np.max(mean_corr)) if mean_corr.size > 0 else np.nan
        return mean_corr, max_corr

    @staticmethod
    def _build_channel_groups(channels, depth_groups=None):
        """Assign each channel to a depth group.

        Channels in the same group share a ray-type assignment.
        Ungrouped channels get their own individual group.

        Parameters
        ----------
        channels : list
            Channel IDs used in reconstruction.
        depth_groups : dict or None
            Maps group name to list of channel IDs. Defaults to
            DEFAULT_DEPTH_GROUPS.

        Returns
        -------
        dict
            Maps channel ID to group index.
        list
            List of group names (for logging).
        """
        if depth_groups is None:
            depth_groups = InterferometricReco3D.DEFAULT_DEPTH_GROUPS

        ch_to_group = {}
        group_names = []
        gidx = 0
        for name, members in depth_groups.items():
            active = [ch for ch in members if ch in channels]
            if active:
                for ch in active:
                    ch_to_group[ch] = gidx
                group_names.append(name)
                gidx += 1
        for ch in channels:
            if ch not in ch_to_group:
                ch_to_group[ch] = gidx
                group_names.append(f'ch{ch}')
                gidx += 1
        return ch_to_group, group_names

    def _correlator_grouped_multiray(self, corr_data, tt_all, channels,
                                     pair_weights=None):
        """Grouped combo multiray correlator.

        Instead of taking per-pair max across 9 ray combos, enumerate all
        depth-group ray-type assignments and take the max of the weighted
        mean correlation across assignments. Channels in the same depth
        group share a ray type.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        tt_all : dict
            Maps ch -> {ray_type -> grid}.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        ch_pairs = list(itertools.combinations(channels, 2))
        n_pairs = len(ch_pairs)

        grid_shape = None
        for ch in channels:
            for rt in tt_all.get(ch, {}):
                grid_shape = tt_all[ch][rt].shape
                break
            if grid_shape is not None:
                break
        if grid_shape is None:
            return np.zeros(1), np.nan

        if pair_weights is not None:
            w = np.asarray(pair_weights, dtype=np.float64)
            w_sum = float(w.sum())
        else:
            w = np.ones(n_pairs, dtype=np.float64)
            w_sum = float(n_pairs)

        ch_to_group, group_names = self._build_channel_groups(channels)
        n_groups = len(group_names)

        # Available ray types per group. Start with intersection; fall back
        # to union if the intersection is empty (e.g. shallow channels that
        # lack some ray types at certain source depths).
        group_ray_types = []
        for gidx in range(n_groups):
            group_chs = [ch for ch in channels if ch_to_group[ch] == gidx]
            rts = set(self._active_ray_types)
            for ch in group_chs:
                rts &= set(tt_all.get(ch, {}).keys())
            if not rts:
                for ch in group_chs:
                    rts |= set(tt_all.get(ch, {}).keys())
            if not rts:
                rts = {self._active_ray_types[0]}
            group_ray_types.append(sorted(rts))

        combos = list(itertools.product(*group_ray_types))
        logger.debug("Grouped multiray: %d groups (%s), %d combos",
                      n_groups, group_names, len(combos))

        best_mean_corr = np.full(grid_shape, -np.inf, dtype=np.float64)
        combo_corr = np.zeros(grid_shape, dtype=np.float64)

        for combo in combos:
            ch_rt = {ch: combo[ch_to_group[ch]] for ch in channels}

            combo_corr[:] = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                rt1 = ch_rt[c1]
                rt2 = ch_rt[c2]

                tt1 = tt_all.get(c1, {}).get(rt1)
                tt2 = tt_all.get(c2, {}).get(rt2)
                if tt1 is None or tt2 is None:
                    continue

                delay = tt1 - tt2
                valid = np.isfinite(delay)
                if not np.any(valid):
                    continue

                flat_delays = delay[valid].ravel().astype(np.float64)
                vals = self._interp_delays(corr_data[pidx][0],
                                           corr_data[pidx][1],
                                           corr_data[pidx][2],
                                           flat_delays)
                np.nan_to_num(vals, copy=False, nan=0.0)
                combo_corr[valid] += vals * w[pidx]

            if w_sum > 0:
                combo_corr /= w_sum

            np.maximum(best_mean_corr, combo_corr, out=best_mean_corr)

        neg_inf = best_mean_corr == -np.inf
        best_mean_corr[neg_inf] = 0.0

        max_corr = float(np.max(best_mean_corr)) if best_mean_corr.size > 0 else np.nan
        return best_mean_corr, max_corr

    def _multiray_correlate(self, corr_data, tt_all, channels,
                            pair_weights=None, force_perpair=False):
        """Dispatch to per-pair or grouped multiray correlator.

        Parameters
        ----------
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        tt_all : dict
            Maps ch -> {ray_type -> grid}.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.
        force_perpair : bool
            Force per-pair mode regardless of combo_mode setting.
            Used for the coarse grid where grouped is too expensive.

        Returns
        -------
        tuple
            (mean_corr_map, max_corr)
        """
        use_grouped = (self._multiray_combo_mode == 'grouped'
                       and not force_perpair)
        if use_grouped and USE_NUMBA_GROUPED:
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            return grouped_multiray_numba(
                corr_data, tt_all, channels, ch_to_group, n_groups,
                pair_weights=pair_weights
            )
        if use_grouped:
            return self._correlator_grouped_multiray(
                corr_data, tt_all, channels, pair_weights=pair_weights
            )
        if USE_NUMBA_GROUPED:
            return perpair_multiray_numba(
                corr_data, tt_all, channels, pair_weights=pair_weights
            )
        return self._correlator_lean_multiray(
            corr_data, tt_all, channels, pair_weights=pair_weights
        )

    def _prepare_corr_funcs(self, times, v_array_pairs, hilbert_envelope_mode=None,
                            apply_hann_window=False,
                            correlation_normalization="normalized"):
        """Pre-compute cross-correlations for use by the point evaluator.

        Parameters
        ----------
        times : list of array
            Time arrays for each channel.
        v_array_pairs : list of tuple
            Voltage trace pairs.
        hilbert_envelope_mode : str or None
            Hilbert envelope mode.
        apply_hann_window : bool
            Apply Hann window.
        correlation_normalization : str
            "pearson" (default): mean-subtract and divide by std, then overlap
            normalization. Values in [-1, 1], amplitude-blind.
            "energy": mean-subtract, divide by sqrt(E1*E2) globally. Preserves
            relative amplitude across pairs.
            "overlap_only": mean-subtract, divide by overlap count only. High-SNR
            pairs naturally produce larger correlation values.
            Legacy alias "normalized" maps to "pearson".

        Returns
        -------
        list of tuple
            Each element is (corr_array, dt, offset) for one channel pair.
        """
        channel_pairs = list(itertools.combinations(range(len(times)), 2))
        dts = np.array([t[1] - t[0] if len(t) > 1 else 1.0 for t in times])
        len_trace = len(v_array_pairs[0][0])
        overlap_norm = np.concatenate([np.arange(1, len_trace + 1),
                                       np.arange(len_trace - 1, 0, -1)],
                                      dtype=np.float64)

        hilbert_cache = {}
        if hilbert_envelope_mode == "traces":
            for pidx in range(len(v_array_pairs)):
                cidx1, cidx2 = channel_pairs[pidx]
                if cidx1 not in hilbert_cache:
                    hilbert_cache[cidx1] = np.abs(hilbert(v_array_pairs[pidx][0]))
                if cidx2 not in hilbert_cache:
                    hilbert_cache[cidx2] = np.abs(hilbert(v_array_pairs[pidx][1]))

        corr_data = []
        for pidx in range(len(v_array_pairs)):
            v1, v2 = v_array_pairs[pidx]
            cidx1, cidx2 = channel_pairs[pidx]
            t1, t2 = times[cidx1], times[cidx2]
            dt = min(dts[cidx1], dts[cidx2])

            if hilbert_envelope_mode == "traces":
                v1 = hilbert_cache[cidx1]
                v2 = hilbert_cache[cidx2]

            norm_mode = correlation_normalization
            if norm_mode == "normalized":
                norm_mode = "pearson"

            v1n = v1 - v1.mean()
            v2n = v2 - v2.mean()
            if norm_mode == "pearson":
                std1, std2 = v1n.std(), v2n.std()
                if std1 > 0 and std2 > 0:
                    v1n = v1n / std1
                    v2n = v2n / std2

            corr = correlate(v1n, v2n, mode='full', method='auto')
            if hilbert_envelope_mode == "correlation":
                corr = np.abs(hilbert(corr))

            if norm_mode == "energy":
                e1 = np.sum(v1n**2)
                e2 = np.sum(v2n**2)
                energy_norm = np.sqrt(e1 * e2)
                if energy_norm > 0:
                    corr /= energy_norm
            else:
                corr /= overlap_norm
            if apply_hann_window:
                corr *= windows.hann(len(corr))

            M = len(corr)
            offset = -(M // 2) * dt + (t1[0] - t2[0])
            corr_data.append((corr.astype(np.float64), float(dt), float(offset)))

        return corr_data

    def _interp_corr_scalar(self, corr_arr, dt, offset, delay):
        """Interpolate correlation at a single delay value.

        Parameters
        ----------
        corr_arr : np.ndarray
            1D correlation array.
        dt : float
            Sample spacing.
        offset : float
            Time offset of first sample.
        delay : float
            Delay to interpolate at.

        Returns
        -------
        float
            Interpolated correlation value.
        """
        M = len(corr_arr)
        kf = (delay - offset) / dt
        k = int(np.floor(kf))
        if k < 0 or k >= M - 1:
            return 0.0
        alpha = kf - k
        return corr_arr[k] + (corr_arr[k + 1] - corr_arr[k]) * alpha

    def _tt_scalar(self, ch, rt, r, z):
        """Look up travel time for a single (R, Z) point.

        Uses Numba scalar bilinear interpolation when available, falling back
        to SciPy RegularGridInterpolator.

        Args:
            ch: Channel ID.
            rt: Ray type string ('direct', 'refracted', 'reflected').
            r: Horizontal distance in meters.
            z: Depth in meters.

        Returns:
            Travel time in ns, or -inf if out of bounds.
        """
        td = self._multiray_interpolators[ch][rt]
        if USE_NUMBA:
            return _bilinear_scalar_numba(
                td.values, td.r_min, td.dr_inv, td.nr,
                td.z_min, td.dz_inv, td.nz, r, z)
        return td.interp(np.array([[r, z]]))[0]

    def _build_optimizer_cache(self, channels, pair_weights, corr_data=None):
        """Pre-compute invariant data for the optimizer objective.

        Event-invariant geometry (channel positions, pair indices, TT tables)
        is stored in ``self._opt_geom_cache`` keyed by the channel tuple so
        repeated calls across events reuse it. Only the per-event pieces
        (pair weights, packed correlation arrays) are rebuilt each call.

        Args:
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            corr_data: Pre-computed correlation data (for Numba packing).

        Returns:
            Dict with cached arrays for fast objective evaluation.
        """
        geom_key = (tuple(channels), bool(self._multi_ray_types),
                    self._multiray_combo_mode)
        if not hasattr(self, '_opt_geom_cache'):
            self._opt_geom_cache = {}
        geom = self._opt_geom_cache.get(geom_key)
        if geom is not None:
            cache = dict(geom)
            n_pairs = cache['n_pairs']
            pw = np.ones(n_pairs, dtype=np.float64)
            if pair_weights is not None:
                pw = np.asarray(pair_weights, dtype=np.float64)
            cache['pw'] = pw
            cache['w_total'] = float(pw.sum())
            if corr_data is not None and USE_NUMBA:
                lengths = np.array([len(cd[0]) for cd in corr_data],
                                   dtype=np.int64)
                max_len = int(lengths.max())
                corr_packed = np.zeros((n_pairs, max_len), dtype=np.float64)
                corr_dts = np.empty(n_pairs, dtype=np.float64)
                corr_offsets = np.empty(n_pairs, dtype=np.float64)
                for i in range(n_pairs):
                    corr_packed[i, :lengths[i]] = corr_data[i][0]
                    corr_dts[i] = corr_data[i][1]
                    corr_offsets[i] = corr_data[i][2]
                cache['corr_packed'] = corr_packed
                cache['corr_lengths'] = lengths
                cache['corr_dts'] = corr_dts
                cache['corr_offsets'] = corr_offsets
            return cache

        ch_pairs = list(itertools.combinations(channels, 2))
        n_pairs = len(ch_pairs)
        pw = np.ones(n_pairs, dtype=np.float64)
        if pair_weights is not None:
            pw = np.asarray(pair_weights, dtype=np.float64)
        w_total = float(pw.sum())

        pa_center = self._pa_center

        ant_pos = np.array([self.ant_locs[ch][:2] for ch in channels])

        ch_idx = list(range(len(channels)))
        pair_ch1 = np.array([p[0] for p in
                             itertools.combinations(ch_idx, 2)],
                            dtype=np.int64)
        pair_ch2 = np.array([p[1] for p in
                             itertools.combinations(ch_idx, 2)],
                            dtype=np.int64)

        # Pack table data for fast scalar TT lookup
        n_ch = len(channels)
        td_list = []
        if self._multi_ray_types:
            for ch in channels:
                ch_tds = []
                for rt in self._active_ray_types:
                    ch_tds.append(self._multiray_interpolators[ch][rt])
                td_list.append(ch_tds)
        else:
            for ch in channels:
                td_list.append([self._interpolators[ch]])

        cache = {
            'ch_pairs': ch_pairs,
            'n_pairs': n_pairs,
            'pw': pw,
            'w_total': w_total,
            'pa_center': pa_center,
            'ant_pos': ant_pos,
            'channels': channels,
            'pair_ch1': pair_ch1,
            'pair_ch2': pair_ch2,
            'td_list': td_list,
            'n_ch': n_ch,
        }

        if self._multi_ray_types and self._multiray_combo_mode == 'grouped':
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            rt_map = {rt: i for i, rt in enumerate(self._active_ray_types)}
            group_rts_all = []
            for gidx in range(n_groups):
                group_chs = [ch for ch in channels
                             if ch_to_group[ch] == gidx]
                rts = set(self._active_ray_types)
                for ch in group_chs:
                    avail = set(self._multiray_interpolators[ch].keys())
                    rts &= avail
                if not rts:
                    for ch in group_chs:
                        rts |= set(self._multiray_interpolators[ch].keys())
                if not rts:
                    rts = {self._active_ray_types[0]}
                group_rts_all.append(sorted(rts))

            combos = list(itertools.product(*group_rts_all))
            n_combos = len(combos)
            ch_group_indices = [ch_to_group[ch] for ch in channels]
            combo_table = np.empty((n_combos, n_ch), dtype=np.int64)
            for ci, combo in enumerate(combos):
                for chi in range(n_ch):
                    combo_table[ci, chi] = rt_map[combo[ch_group_indices[chi]]]
            cache['combo_table'] = combo_table
            cache['n_combos'] = n_combos

            # Also keep string-based combo_rt for fallback
            combo_rt = []
            for combo in combos:
                combo_rt.append([combo[ch_to_group[ch]] for ch in channels])
            cache['combo_rt'] = combo_rt

        # Pack correlation data for Numba kernel
        if corr_data is not None and USE_NUMBA:
            lengths = np.array([len(cd[0]) for cd in corr_data],
                               dtype=np.int64)
            max_len = int(lengths.max())
            corr_packed = np.zeros((n_pairs, max_len), dtype=np.float64)
            corr_dts = np.empty(n_pairs, dtype=np.float64)
            corr_offsets = np.empty(n_pairs, dtype=np.float64)
            for i in range(n_pairs):
                corr_packed[i, :lengths[i]] = corr_data[i][0]
                corr_dts[i] = corr_data[i][1]
                corr_offsets[i] = corr_data[i][2]
            cache['corr_packed'] = corr_packed
            cache['corr_lengths'] = lengths
            cache['corr_dts'] = corr_dts
            cache['corr_offsets'] = corr_offsets

        # Pack per-channel TT tables for the fused singleray kernel
        if USE_NUMBA and not self._multi_ray_types:
            nr_arr = np.array([td_list[ci][0].nr for ci in range(n_ch)],
                              dtype=np.int64)
            nz_arr = np.array([td_list[ci][0].nz for ci in range(n_ch)],
                              dtype=np.int64)
            nr_max = int(nr_arr.max())
            nz_max = int(nz_arr.max())
            td_values_packed = np.zeros((n_ch, nr_max, nz_max),
                                        dtype=np.float64)
            td_r_min = np.empty(n_ch, dtype=np.float64)
            td_dr_inv = np.empty(n_ch, dtype=np.float64)
            td_z_min = np.empty(n_ch, dtype=np.float64)
            td_dz_inv = np.empty(n_ch, dtype=np.float64)
            for ci in range(n_ch):
                td = td_list[ci][0]
                nr, nz = td.nr, td.nz
                td_values_packed[ci, :nr, :nz] = td.values
                td_r_min[ci] = td.r_min
                td_dr_inv[ci] = td.dr_inv
                td_z_min[ci] = td.z_min
                td_dz_inv[ci] = td.dz_inv
            cache['td_values_packed'] = td_values_packed
            cache['td_nr'] = nr_arr
            cache['td_nz'] = nz_arr
            cache['td_r_min'] = td_r_min
            cache['td_dr_inv'] = td_dr_inv
            cache['td_z_min'] = td_z_min
            cache['td_dz_inv'] = td_dz_inv
            cache['pa_x'] = float(pa_center[0])
            cache['pa_y'] = float(pa_center[1])

        # Store event-invariant geometry for reuse across events. The
        # per-event parts (pw, w_total, corr_packed, corr_dts, corr_offsets)
        # will be overwritten on subsequent calls.
        geom = {k: v for k, v in cache.items()
                if k not in ('pw', 'w_total', 'corr_packed',
                             'corr_lengths', 'corr_dts', 'corr_offsets')}
        self._opt_geom_cache[geom_key] = geom

        return cache

    def _correlation_at_point(self, params, corr_data, channels,
                              pair_weights=None, _cache=None):
        """Evaluate negative mean correlation at a single (rho, phi_deg, z) point.

        When multi_ray_types is enabled, tests all 9 ray type combinations per
        pair and takes the maximum.

        Parameters
        ----------
        params : array-like
            [rho, phi_deg, z] in meters and degrees.
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        channels : list
            Channel IDs.
        pair_weights : list or None
            Per-pair weights.
        _cache : dict or None
            Pre-computed invariants from _build_optimizer_cache.

        Returns
        -------
        float
            Negative mean correlation (for minimization).
        """
        rho, phi_deg, z = params
        phi_rad = phi_deg * (np.pi / 180.0)

        if _cache is not None:
            pa_center = _cache['pa_center']
            ch_pairs = _cache['ch_pairs']
            w_total = _cache['w_total']
            pw = _cache['pw']
            ant_pos = _cache['ant_pos']
        else:
            pa_center = self._pa_center
            ch_pairs = list(itertools.combinations(channels, 2))
            if pair_weights is not None:
                pw = np.array(pair_weights, dtype=np.float64)
            else:
                pw = np.ones(len(ch_pairs))
            w_total = float(pw.sum())
            ant_pos = None

        # Fast path: fused singleray Numba kernel
        if (not self._multi_ray_types and USE_NUMBA and _cache is not None
                and 'td_values_packed' in _cache
                and 'corr_packed' in _cache):
            return _scalar_singleray_corr_numba(
                float(rho), float(phi_rad), float(z),
                _cache['pa_x'], _cache['pa_y'],
                _cache['ant_pos'],
                _cache['td_values_packed'],
                _cache['td_r_min'], _cache['td_dr_inv'], _cache['td_nr'],
                _cache['td_z_min'], _cache['td_dz_inv'], _cache['td_nz'],
                _cache['corr_packed'], _cache['corr_lengths'],
                _cache['corr_dts'], _cache['corr_offsets'],
                _cache['pair_ch1'], _cache['pair_ch2'], pw, w_total)

        x = rho * np.cos(phi_rad) + pa_center[0]
        y = rho * np.sin(phi_rad) + pa_center[1]

        if self._multi_ray_types:
            n_ch = _cache['n_ch'] if _cache else len(channels)

            n_rt = self._n_ray_slots
            tt_vals = np.full((n_ch, n_rt), -np.inf, dtype=np.float64)
            tt_valid = np.zeros((n_ch, n_rt), dtype=np.bool_)
            for ci in range(n_ch):
                ch = channels[ci]
                if ant_pos is not None:
                    dx = x - ant_pos[ci, 0]
                    dy = y - ant_pos[ci, 1]
                else:
                    pos = self.ant_locs[ch]
                    dx = x - pos[0]
                    dy = y - pos[1]
                r = max(np.sqrt(dx * dx + dy * dy), 1.0)
                if _cache is not None:
                    td_ch = _cache['td_list'][ci]
                    for rti in range(n_rt):
                        td = td_ch[rti]
                        if USE_NUMBA:
                            tt = _bilinear_scalar_numba(
                                td.values, td.r_min, td.dr_inv, td.nr,
                                td.z_min, td.dz_inv, td.nz, r, z)
                        else:
                            tt = td.interp(np.array([[r, z]]))[0]
                        if np.isfinite(tt) and tt > 0:
                            tt_vals[ci, rti] = tt
                            tt_valid[ci, rti] = True
                else:
                    for rti, rt in enumerate(self._active_ray_types):
                        tt = self._tt_scalar(ch, rt, r, z)
                        if np.isfinite(tt) and tt > 0:
                            tt_vals[ci, rti] = tt
                            tt_valid[ci, rti] = True

            # Fast Numba path for grouped mode
            if (self._multiray_combo_mode == 'grouped' and USE_NUMBA
                    and _cache is not None and 'corr_packed' in _cache):
                return _scalar_grouped_corr_numba(
                    tt_vals, tt_valid,
                    _cache['corr_packed'], _cache['corr_lengths'],
                    _cache['corr_dts'], _cache['corr_offsets'],
                    _cache['pair_ch1'], _cache['pair_ch2'], pw,
                    _cache['combo_table'], _cache['n_combos'],
                    _cache['n_pairs'], w_total)

            # Python fallback for grouped mode
            if self._multiray_combo_mode == 'grouped':
                ch_tt = {}
                for ci, ch in enumerate(channels):
                    ch_tt_ch = {}
                    for rti, rt in enumerate(self._active_ray_types):
                        if tt_valid[ci, rti]:
                            ch_tt_ch[rt] = tt_vals[ci, rti]
                    ch_tt[ch] = ch_tt_ch
                return self._correlation_at_point_grouped(
                    ch_tt, corr_data, channels, ch_pairs, pw, w_total,
                    _cache=_cache
                )

            # Per-pair multiray mode
            total = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                best_val = 0.0
                corr_arr, dt, offset = corr_data[pidx]
                ci1 = channels.index(c1)
                ci2 = channels.index(c2)
                for rti1 in range(n_rt):
                    if not tt_valid[ci1, rti1]:
                        continue
                    for rti2 in range(n_rt):
                        if not tt_valid[ci2, rti2]:
                            continue
                        delay = tt_vals[ci1, rti1] - tt_vals[ci2, rti2]
                        val = self._interp_corr_scalar(
                            corr_arr, dt, offset, delay)
                        if val > best_val:
                            best_val = val
                total += pw[pidx] * best_val
            return -total / w_total if w_total > 0 else 0.0

        # Single-table mode
        travel_times = {}
        for ci, ch in enumerate(channels):
            if ant_pos is not None:
                dx = x - ant_pos[ci, 0]
                dy = y - ant_pos[ci, 1]
            else:
                pos = self.ant_locs[ch]
                dx = x - pos[0]
                dy = y - pos[1]
            r = max(np.sqrt(dx * dx + dy * dy), 1.0)
            td = self._interpolators[ch]
            if USE_NUMBA:
                travel_times[ch] = _bilinear_scalar_numba(
                    td.values, td.r_min, td.dr_inv, td.nr,
                    td.z_min, td.dz_inv, td.nz, r, z)
            else:
                travel_times[ch] = td.interp(np.array([[r, z]]))[0]

        total = 0.0
        for pidx, (c1, c2) in enumerate(ch_pairs):
            t1 = travel_times[c1]
            t2 = travel_times[c2]
            if not np.isfinite(t1) or not np.isfinite(t2):
                continue
            delay = t1 - t2
            corr_arr, dt, offset = corr_data[pidx]
            val = self._interp_corr_scalar(corr_arr, dt, offset, delay)
            total += pw[pidx] * val

        return -total / w_total if w_total > 0 else 0.0

    def _correlation_at_point_grouped(self, ch_tt, corr_data, channels,
                                      ch_pairs, pair_weights, w_total,
                                      _cache=None):
        """Grouped-combo scalar correlation for L-BFGS-B optimizer.

        Enumerates depth-group ray-type assignments and returns the negative
        of the best weighted mean correlation.

        Parameters
        ----------
        ch_tt : dict
            Maps ch -> {ray_type: travel_time} at this point.
        corr_data : list of tuple
            Pre-computed (corr_array, dt, offset) per pair.
        channels : list
            Channel IDs.
        ch_pairs : list of tuple
            Channel pairs.
        pair_weights : array-like
            Per-pair weights.
        w_total : float
            Sum of pair weights.
        _cache : dict or None
            Pre-computed combo_rt from _build_optimizer_cache.

        Returns
        -------
        float
            Negative mean correlation (for minimization).
        """
        if _cache is not None and 'combo_rt' in _cache:
            combo_rt_list = _cache['combo_rt']
        else:
            ch_to_group, _ = self._build_channel_groups(channels)
            n_groups = max(ch_to_group.values()) + 1
            group_ray_types = []
            for gidx in range(n_groups):
                group_chs = [ch for ch in channels
                             if ch_to_group[ch] == gidx]
                rts = set(self._active_ray_types)
                for ch in group_chs:
                    rts &= set(ch_tt.get(ch, {}).keys())
                if not rts:
                    for ch in group_chs:
                        rts |= set(ch_tt.get(ch, {}).keys())
                if not rts:
                    rts = {self._active_ray_types[0]}
                group_ray_types.append(sorted(rts))
            combos = list(itertools.product(*group_ray_types))
            combo_rt_list = []
            for combo in combos:
                combo_rt_list.append(
                    [combo[ch_to_group[ch]] for ch in channels])

        best_total = -np.inf
        ch_idx = {ch: i for i, ch in enumerate(channels)}
        for combo_rt in combo_rt_list:
            total = 0.0
            for pidx, (c1, c2) in enumerate(ch_pairs):
                rt1 = combo_rt[ch_idx[c1]]
                rt2 = combo_rt[ch_idx[c2]]
                tt1 = ch_tt.get(c1, {}).get(rt1)
                tt2 = ch_tt.get(c2, {}).get(rt2)
                if tt1 is None or tt2 is None:
                    continue
                delay = tt1 - tt2
                corr_arr, dt, offset = corr_data[pidx]
                val = self._interp_corr_scalar(corr_arr, dt, offset, delay)
                w = pair_weights[pidx] if not isinstance(pair_weights, (int, float)) else pair_weights
                total += w * val
            if total > best_total:
                best_total = total

        if best_total == -np.inf:
            return 0.0
        return -best_total / w_total if w_total > 0 else 0.0

    @staticmethod
    def _compute_snr_pair_weights(volt_arrays, channels):
        """Compute per-pair weights from channel SNRs.

        Each pair is weighted by the geometric mean of the two channels' SNRs,
        normalized so the maximum is 1. All pairs contribute; low-SNR channels
        are naturally downweighted.

        Parameters
        ----------
        volt_arrays : list of array
            Voltage traces, one per channel (same order as channels).
        channels : list
            Channel IDs (for logging only).

        Returns
        -------
        list of float
            Per-pair weights, one per combination(channels, 2).
        """
        from NuRadioReco.utilities.trace_utilities import (
            get_split_trace_noise_RMS, get_signal_to_noise_ratio)

        snrs = []
        for v in volt_arrays:
            noise_rms = get_split_trace_noise_RMS(v)
            snrs.append(get_signal_to_noise_ratio(v, noise_rms)
                        if noise_rms > 0 else 0.0)

        channel_snrs = dict(zip(channels, snrs))

        ch_pairs = list(itertools.combinations(range(len(channels)), 2))
        weights = [np.sqrt(snrs[i] * snrs[j]) for i, j in ch_pairs]

        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            weights = [w / max_w for w in weights]

        logger.debug("SNR pair weights: %d pairs", len(weights))
        return weights, channel_snrs

    def _extract_top_n_peaks(self, corr_map, rho_vec, phi_vec_deg, z_vec, n,
                             separation):
        """Find top-N peaks in the 3D correlation map with minimum separation.

        Parameters
        ----------
        corr_map : np.ndarray
            3D correlation map (n_rho, n_phi, n_z).
        rho_vec : array
            Rho values in meters.
        phi_vec_deg : array
            Phi values in degrees.
        z_vec : array
            Z values in meters.
        n : int
            Number of peaks to find.
        separation : list
            [d_rho, d_phi, d_z] minimum separation.

        Returns
        -------
        list of tuple
            Each element is (rho, phi_deg, z, corr_value).
        """
        work = corr_map.copy()
        peaks = []
        d_rho, d_phi, d_z = separation

        for _ in range(n):
            if np.all(np.isnan(work)):
                break
            idx = np.unravel_index(np.nanargmax(work), work.shape)
            val = work[idx]
            if np.isnan(val):
                break

            rho_peak = rho_vec[idx[0]]
            phi_peak = phi_vec_deg[idx[1]]
            z_peak = z_vec[idx[2]]
            peaks.append((rho_peak, phi_peak, z_peak, float(val)))

            rho_mask = np.abs(rho_vec - rho_peak) < d_rho
            z_mask = np.abs(z_vec - z_peak) < d_z
            phi_diff = np.abs(phi_vec_deg - phi_peak)
            phi_diff = np.minimum(phi_diff, 360.0 - phi_diff)
            phi_mask = phi_diff < d_phi

            work[np.ix_(rho_mask, phi_mask, z_mask)] = np.nan

        return peaks

    def _compute_travel_times_single_point(self, rho, phi_deg, z, channels):
        """Compute per-channel travel times for a single source position.

        Args:
            rho: Horizontal distance from PA center (meters).
            phi_deg: Azimuth in degrees.
            z: Depth (meters, negative below surface).
            channels: List of channel IDs.

        Returns:
            Dict mapping channel ID to travel time (seconds), or NaN.
        """
        phi_rad = phi_deg * (np.pi / 180.0)
        x = rho * np.cos(phi_rad) + self._pa_center[0]
        y = rho * np.sin(phi_rad) + self._pa_center[1]

        travel_times = {}
        for ch in channels:
            pos = self.ant_locs[ch]
            dx = x - pos[0]
            dy = y - pos[1]
            r = max(np.sqrt(dx * dx + dy * dy), 1.0)
            td = self._interpolators[ch]
            if USE_NUMBA:
                tt = _bilinear_scalar_numba(
                    td.values, td.r_min, td.dr_inv, td.nr,
                    td.z_min, td.dz_inv, td.nz, r, z)
            else:
                tt = td.interp(np.array([[r, z]]))[0]
            travel_times[ch] = tt
        return travel_times

    def _compute_coherent_waveform(self, rho, phi_deg, z,
                                   volt_arrays, time_arrays, channels):
        """Form coherent delay-and-stack waveform at a source direction.

        Shifts each channel trace by the relative travel time delay and
        sums. Uses linear interpolation for sub-sample shifting.

        Args:
            rho, phi_deg, z: Source position in cylindrical coordinates.
            volt_arrays: List of voltage traces per channel.
            time_arrays: List of time arrays per channel.
            channels: List of channel IDs.

        Returns:
            (times, coherent_trace) where times is the output time array
            and coherent_trace is the delay-and-stack sum, normalized by
            the number of contributing channels. Returns (None, None) if
            travel times are unavailable.
        """
        travel_times = self._compute_travel_times_single_point(
            rho, phi_deg, z, channels)

        valid_tt = {ch: tt for ch, tt in travel_times.items()
                    if np.isfinite(tt) and tt > 0}
        if len(valid_tt) < 2:
            return None, None

        t_ref = min(valid_tt.values())
        delays = {ch: tt - t_ref for ch, tt in valid_tt.items()}

        ref_idx = channels.index(list(valid_tt.keys())[0])
        out_times = time_arrays[ref_idx].copy()
        n_out = len(out_times)
        coherent = np.zeros(n_out, dtype=np.float64)
        n_contributing = 0

        for ci, ch in enumerate(channels):
            if ch not in valid_tt:
                continue
            trace = volt_arrays[ci]
            times = time_arrays[ci]
            delay = delays[ch]

            shifted_times = out_times + delay
            shifted_trace = np.interp(shifted_times, times, trace,
                                      left=0.0, right=0.0)
            coherent += shifted_trace
            n_contributing += 1

        if n_contributing > 0:
            coherent /= n_contributing

        return out_times, coherent

    def _compute_map_snr(self, corr_map, peak_idx, exclusion_bins=3):
        """Compute map SNR: peak correlation / RMS of map away from peak.

        Args:
            corr_map: 3D correlation map (n_rho, n_phi, n_z).
            peak_idx: Tuple (i_rho, i_phi, i_z) of the peak bin.
            exclusion_bins: Number of bins to exclude around peak in each dim.

        Returns:
            Map SNR (float), or NaN if map RMS is zero.
        """
        mask = np.ones(corr_map.shape, dtype=bool)
        ir, ip, iz = peak_idx
        nr, nphi, nz = corr_map.shape
        r_lo = max(0, ir - exclusion_bins)
        r_hi = min(nr, ir + exclusion_bins + 1)
        p_lo = max(0, ip - exclusion_bins)
        p_hi = min(nphi, ip + exclusion_bins + 1)
        z_lo = max(0, iz - exclusion_bins)
        z_hi = min(nz, iz + exclusion_bins + 1)
        mask[r_lo:r_hi, p_lo:p_hi, z_lo:z_hi] = False

        away = corr_map[mask]
        away = away[np.isfinite(away)]
        if len(away) == 0:
            return np.nan
        rms = np.std(away)
        if rms < 1e-12:
            return np.nan
        return float(corr_map[peak_idx]) / rms

    def _find_peak_bin(self, rho, phi_deg, z, rho_vec, phi_vec_deg, z_vec):
        """Find the nearest bin index in the coarse grid for a peak position.

        Args:
            rho, phi_deg, z: Peak position.
            rho_vec, phi_vec_deg, z_vec: Coarse grid vectors.

        Returns:
            Tuple (i_rho, i_phi, i_z).
        """
        ir = int(np.argmin(np.abs(rho_vec - rho)))
        phi_diff = np.abs(phi_vec_deg - phi_deg)
        phi_diff = np.minimum(phi_diff, 360.0 - phi_diff)
        ip = int(np.argmin(phi_diff))
        iz = int(np.argmin(np.abs(z_vec - z)))
        return (ir, ip, iz)

    def _deduplicate_peaks(self, peaks, d_rho=10, d_phi=5, d_z=10):
        """Remove duplicate peaks that are within separation thresholds.

        Keeps the highest-correlation peak from each cluster.

        Args:
            peaks: List of (rho, phi_deg, z, corr) tuples, sorted by corr desc.
            d_rho, d_phi, d_z: Minimum separation thresholds.

        Returns:
            Filtered list of peaks.
        """
        kept = []
        for peak in peaks:
            rho_p, phi_p, z_p, corr_p = peak
            is_dup = False
            for rho_k, phi_k, z_k, _ in kept:
                if abs(rho_p - rho_k) < d_rho and abs(z_p - z_k) < d_z:
                    dphi = abs(phi_p - phi_k)
                    dphi = min(dphi, 360 - dphi)
                    if dphi < d_phi:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(peak)
        return kept

    def _optimize_from_seed(self, seed, corr_data, channels, bounds,
                            pair_weights=None, method='L-BFGS-B',
                            maxiter=30, _cache=None):
        """Run local optimization from a single seed point.

        Parameters
        ----------
        seed : tuple
            (rho, phi_deg, z) starting point.
        corr_data : list of tuple
            Pre-computed correlation data.
        channels : list
            Channel IDs.
        bounds : list of tuple
            [(rho_min, rho_max), (phi_min, phi_max), (z_min, z_max)].
        pair_weights : list or None
            Per-pair weights.
        method : str
            Scipy minimize method. 'L-BFGS-B' or 'Nelder-Mead'.
        maxiter : int
            Maximum optimizer iterations.
        _cache : dict or None
            Pre-computed invariants from _build_optimizer_cache.

        Returns
        -------
        tuple
            (rho, phi_deg, z, corr_value) at optimum.
        """
        rho0, phi0, z0 = seed

        phi_shift = phi0 - 180.0
        phi_start = 180.0

        def objective(params):
            rho, phi_shifted, z = params
            phi_actual = (phi_shifted + phi_shift) % 360.0
            return self._correlation_at_point(
                [rho, phi_actual, z], corr_data, channels, pair_weights,
                _cache=_cache
            )

        x0 = np.array([rho0, phi_start, z0])
        shifted_bounds = [
            bounds[0],
            (0.0, 360.0),
            bounds[2],
        ]

        if method == 'Nelder-Mead':
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 0.1, 'fatol': 1e-8}
            )
        else:
            result = minimize(
                objective, x0, method='L-BFGS-B', bounds=shifted_bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10}
            )

        rho_opt, phi_shifted_opt, z_opt = result.x
        if method == 'Nelder-Mead':
            rho_opt = np.clip(rho_opt, bounds[0][0], bounds[0][1])
            z_opt = np.clip(z_opt, bounds[2][0], bounds[2][1])
        phi_opt = (phi_shifted_opt + phi_shift) % 360.0
        corr_opt = -result.fun

        return rho_opt, phi_opt, z_opt, corr_opt

    def _extract_peak_delays(self, corr_data):
        """Extract peak delay from each cross-correlation function.

        Args:
            corr_data: List of (corr_array, dt, offset) per pair.

        Returns:
            1D array of peak delay values (ns) for each pair.
        """
        delays = np.empty(len(corr_data), dtype=np.float64)
        for pidx, (corr_arr, dt, offset) in enumerate(corr_data):
            peak_idx = np.argmax(corr_arr)
            delays[pidx] = offset + peak_idx * dt
        return delays

    def _chan_ho_initialize(self, corr_data, channels, pair_weights=None):
        """Estimate source position via Chan-Ho TDOA linearization.

        Assumes straight-line propagation at average ice velocity for an
        initial estimate. The bias from ray curvature is acceptable as
        a starting point for iterative refinement.

        Args:
            corr_data: Pre-computed correlation data per pair.
            channels: Channel IDs.
            pair_weights: Per-pair weights (used to select strongest pairs).

        Returns:
            (rho, phi_deg, z) initial estimate, or None if solve fails.
        """
        n_ch = len(channels)
        if n_ch < 4:
            return None

        c_ice = 0.3 / 1.55  # average in-ice velocity ~0.1935 m/ns

        ant_pos_3d = np.array([self.ant_locs[ch] for ch in channels])
        pa_center = self._pa_center

        # Extract per-channel delays relative to first channel using
        # pairwise correlations. Build a per-channel TDOA vector.
        pair_delays = self._extract_peak_delays(corr_data)
        ch_pairs = list(itertools.combinations(range(n_ch), 2))

        # Use weighted average of pairwise delays to get per-channel TDOAs
        # relative to channel 0
        tdoa = np.zeros(n_ch)
        tdoa_count = np.zeros(n_ch)

        if pair_weights is not None:
            pw = np.asarray(pair_weights, dtype=np.float64)
        else:
            pw = np.ones(len(ch_pairs))

        for pidx, (ci, cj) in enumerate(ch_pairs):
            w = pw[pidx]
            # delay = t_ci - t_cj (positive means ci signal arrives later)
            delay = pair_delays[pidx]
            if ci == 0:
                tdoa[cj] += -delay * w
                tdoa_count[cj] += w
            elif cj == 0:
                tdoa[ci] += delay * w
                tdoa_count[ci] += w

        # For channels without direct pair to ch0, use transitive delays
        for ci in range(1, n_ch):
            if tdoa_count[ci] == 0:
                for pidx, (ca, cb) in enumerate(ch_pairs):
                    w = pw[pidx]
                    if ca == ci and tdoa_count[cb] > 0:
                        tdoa[ci] += (pair_delays[pidx] +
                                     tdoa[cb] / tdoa_count[cb]) * w
                        tdoa_count[ci] += w
                    elif cb == ci and tdoa_count[ca] > 0:
                        tdoa[ci] += (-pair_delays[pidx] +
                                     tdoa[ca] / tdoa_count[ca]) * w
                        tdoa_count[ci] += w

        for ci in range(1, n_ch):
            if tdoa_count[ci] > 0:
                tdoa[ci] /= tdoa_count[ci]

        # Chan-Ho linear system: A * [x, y, z, d0] = b
        # Reference receiver is channel 0
        r0 = ant_pos_3d[0]
        r0_sq = np.dot(r0, r0)

        n_eq = n_ch - 1
        A = np.zeros((n_eq, 4))
        b = np.zeros(n_eq)

        for i in range(n_eq):
            ri = ant_pos_3d[i + 1]
            di0 = c_ice * tdoa[i + 1]  # range difference in meters

            A[i, :3] = 2.0 * (ri - r0)
            A[i, 3] = 2.0 * di0
            b[i] = r0_sq - np.dot(ri, ri) - di0 * di0

        # Weighted least squares
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        x, y, z, d0 = result

        # Convert to cylindrical relative to PA center
        dx = x - pa_center[0]
        dy = y - pa_center[1]
        rho = max(np.sqrt(dx**2 + dy**2), 1.0)
        phi_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

        rho = np.clip(rho, 1.0, 1500.0)
        z = np.clip(z, -1500.0, 0.0)

        return rho, phi_deg, z

    def _tdoa_solve(self, corr_data, channels, pair_weights=None,
                    initial_guess=None, cache=None):
        """Solve for source position using TDOA least-squares.

        Extracts observed time delays from cross-correlation peaks, then
        finds the position that minimizes weighted delay residuals.

        Args:
            corr_data: Pre-computed correlation data per pair.
            channels: Channel IDs.
            pair_weights: Per-pair weights or None.
            initial_guess: (rho, phi_deg, z) starting point, or None.
            cache: Pre-computed cache from _build_optimizer_cache.

        Returns:
            (rho, phi_deg, z, residual) or None if solve fails.
        """
        from scipy.optimize import least_squares

        observed_delays = self._extract_peak_delays(corr_data)
        n_pairs = len(corr_data)

        if cache is not None:
            pa_center = cache['pa_center']
            ant_pos = cache['ant_pos']
            pair_ch1 = cache['pair_ch1']
            pair_ch2 = cache['pair_ch2']
            td_list = cache['td_list']
            n_ch = cache['n_ch']
            pw = cache['pw']
        else:
            pa_center = self._pa_center
            ant_pos = np.array([self.ant_locs[ch][:2] for ch in channels])
            ch_idx = list(range(len(channels)))
            pair_ch1 = np.array([p[0] for p in
                                 itertools.combinations(ch_idx, 2)],
                                dtype=np.int64)
            pair_ch2 = np.array([p[1] for p in
                                 itertools.combinations(ch_idx, 2)],
                                dtype=np.int64)
            n_ch = len(channels)
            td_list = []
            for ch in channels:
                ch_tds = [self._multiray_interpolators[ch][rt]
                          for rt in self._active_ray_types]
                td_list.append(ch_tds)
            pw = np.ones(n_pairs, dtype=np.float64)
            if pair_weights is not None:
                pw = np.asarray(pair_weights, dtype=np.float64)

        sqrt_weights = np.sqrt(pw)

        if initial_guess is None:
            ch_init = self._chan_ho_initialize(
                corr_data, channels, pair_weights)
            if ch_init is not None:
                initial_guess = ch_init
            else:
                initial_guess = (500.0, 180.0, -500.0)
        rho0, phi0, z0 = initial_guess

        best_result = None
        best_cost = np.inf

        if cache is not None and 'combo_table' in cache:
            combo_table = cache['combo_table']
            n_combos = cache['n_combos']
        else:
            combo_table = np.zeros((1, n_ch), dtype=np.int64)
            n_combos = 1

        for ci in range(n_combos):
            rt_indices = combo_table[ci]

            def residuals(params):
                rho, phi_deg, z = params
                phi_rad = phi_deg * (np.pi / 180.0)
                x = rho * np.cos(phi_rad) + pa_center[0]
                y = rho * np.sin(phi_rad) + pa_center[1]

                tt = np.full(n_ch, np.nan)
                for chi in range(n_ch):
                    dx = x - ant_pos[chi, 0]
                    dy = y - ant_pos[chi, 1]
                    r = max(np.sqrt(dx * dx + dy * dy), 1.0)
                    rti = int(rt_indices[chi])
                    td = td_list[chi][rti]
                    if USE_NUMBA:
                        val = _bilinear_scalar_numba(
                            td.values, td.r_min, td.dr_inv, td.nr,
                            td.z_min, td.dz_inv, td.nz, r, z)
                    else:
                        val = td.interp(np.array([[r, z]]))[0]
                    if np.isfinite(val) and val > 0:
                        tt[chi] = val

                resid = np.zeros(n_pairs)
                for pidx in range(n_pairs):
                    c1, c2 = int(pair_ch1[pidx]), int(pair_ch2[pidx])
                    if np.isfinite(tt[c1]) and np.isfinite(tt[c2]):
                        predicted = tt[c1] - tt[c2]
                        resid[pidx] = (observed_delays[pidx] - predicted) \
                            * sqrt_weights[pidx]
                    else:
                        resid[pidx] = 0.0
                return resid

            try:
                result = least_squares(
                    residuals, [rho0, phi0, z0],
                    bounds=([1.0, -180.0, -1500.0],
                            [1500.0, 540.0, 0.0]),
                    method='trf', max_nfev=50,
                    ftol=1e-6, xtol=1e-4,
                )
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result.x
            except Exception:
                continue

        if best_result is None:
            return None

        rho_sol, phi_sol, z_sol = best_result
        phi_sol = phi_sol % 360.0

        # Evaluate correlation at TDOA solution to get comparable metric
        corr_val = -self._correlation_at_point(
            [rho_sol, phi_sol, z_sol], corr_data, channels, pair_weights,
            _cache=cache
        )

        return rho_sol, phi_sol, z_sol, corr_val

    def run_tdoa(self, evt, station, det, config):
        """TDOA-based 3D reco: Chan-Ho initialization + iterative refinement + optimizer.

        Bypasses the grid search entirely. Uses cross-correlation peak delays
        to estimate source position via TDOA least-squares, then refines with
        L-BFGS-B on the full correlation objective.

        Args:
            evt: NuRadioReco Event object.
            station: Station object containing channel data.
            det: Detector description.
            config: Configuration dictionary.

        Returns:
            Reconstruction results dict, or None on failure.
        """
        station_id = station.get_id()
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        apply_hann = config.get('apply_hann_window', False)
        corr_norm = config.get('correlation_normalization', 'normalized')

        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())

        pair_weights = None
        if config.get('snr_pair_weighting', False):
            pair_weights, _ = self._compute_snr_pair_weights(
                volt_arrays, channels
            )

        v_pairs = list(itertools.combinations(volt_arrays, 2))
        corr_data = self._prepare_corr_funcs(
            time_arrays, v_pairs,
            hilbert_envelope_mode=hilbert_mode,
            apply_hann_window=apply_hann,
            correlation_normalization=corr_norm,
        )

        full_limits = config.get('limits', [1, 1500, 0, 360, -1500, 0])
        opt_method = config.get('optimizer_method', 'L-BFGS-B')
        opt_maxiter = config.get('optimizer_maxiter', 30)

        t0 = time.time()
        opt_cache = self._build_optimizer_cache(channels, pair_weights,
                                                corr_data)

        # TDOA solve (Chan-Ho init + iterative refinement)
        tdoa_result = self._tdoa_solve(
            corr_data, channels, pair_weights, cache=opt_cache
        )
        t_tdoa = time.time() - t0

        if tdoa_result is None:
            return None

        rho_tdoa, phi_tdoa, z_tdoa, corr_tdoa = tdoa_result

        t0_opt = time.time()
        bounds = [
            (max(full_limits[0], 1.0), full_limits[1]),
            (full_limits[2], full_limits[3]),
            (full_limits[4], full_limits[5]),
        ]
        rho_best, phi_best, z_best, corr_best = self._optimize_from_seed(
            (rho_tdoa, phi_tdoa, z_tdoa), corr_data, channels, bounds,
            pair_weights, method=opt_method, maxiter=opt_maxiter,
            _cache=opt_cache
        )
        t_opt = time.time() - t0_opt

        phi_best = phi_best % 360.0

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        return {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'tdoa_time': t_tdoa,
            'opt_time': t_opt,
            'tdoa_rho': rho_tdoa,
            'tdoa_phi': phi_tdoa,
            'tdoa_z': z_tdoa,
            'tdoa_corr': corr_tdoa,
        }

    def run_hierarchical(self, evt, station, det, config):
        """Hierarchical 3D reco: coarse log-grid + refined linear grid + optimizer.

        Stage 1 uses a logarithmic rho grid (finer resolution at close range)
        with linear phi and z. Stage 2 refines around the top coarse peaks with
        a local linear grid. Stage 3 runs L-BFGS-B from the refined peaks.

        Parameters
        ----------
        evt : Event
            NuRadioReco Event object.
        station : Station
            Station object containing channel data.
        det : Detector
            Detector description.
        config : dict
            Configuration dictionary (already loaded).

        Returns
        -------
        dict
            Reconstruction results.
        """
        station_id = station.get_id()
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        apply_hann = config.get('apply_hann_window', False)
        corr_norm = config.get('correlation_normalization', 'normalized')

        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())

        pair_weights = None
        channel_snrs = {}
        if config.get('snr_pair_weighting', False) or config.get('validation', False):
            pair_weights, channel_snrs = self._compute_snr_pair_weights(
                volt_arrays, channels
            )
            if not config.get('snr_pair_weighting', False):
                pair_weights = None

        v_pairs = list(itertools.combinations(volt_arrays, 2))
        corr_data = self._prepare_corr_funcs(
            time_arrays, v_pairs,
            hilbert_envelope_mode=hilbert_mode,
            apply_hann_window=apply_hann,
            correlation_normalization=corr_norm,
        )

        # Stage 1: Coarse scan with log rho grid
        coarse_limits = config.get('coarse_limits', [1, 1500, 0, 360, -1500, 0])
        coarse_steps = config.get('coarse_step_sizes', [30, 5, 30])
        n_rho_coarse = config.get('coarse_n_rho', 50)

        rho_min_c = max(coarse_limits[0], 1.0)
        rho_max_c = coarse_limits[1]
        phi_min_c, phi_max_c = coarse_limits[2], coarse_limits[3]
        z_min_c, z_max_c = coarse_limits[4], coarse_limits[5]

        if n_rho_coarse > 0:
            rho_vec_c = np.geomspace(rho_min_c, rho_max_c, n_rho_coarse)
        else:
            rho_vec_c = np.arange(rho_min_c, rho_max_c + coarse_steps[0],
                                  coarse_steps[0])
        phi_vec_c = np.arange(phi_min_c, phi_max_c, coarse_steps[1]) * (np.pi / 180.0)

        n_z_coarse = config.get('coarse_n_z', 0)
        z_spacing = config.get('z_spacing', 'linear')
        z_surf_offset = config.get('z_surface_offset', 0.1)
        if n_z_coarse > 0:
            z_vec_c = _build_z_vec(
                z_min_c, z_max_c, n_z_coarse, z_spacing, z_surf_offset)
        else:
            z_vec_c = np.arange(z_min_c, z_max_c + coarse_steps[2], coarse_steps[2])

        coarse_cache_key = (
            'coarse', station_id, tuple(sorted(channels)),
            n_rho_coarse, rho_min_c, rho_max_c,
            coarse_steps[1], phi_min_c, phi_max_c,
            coarse_steps[2], z_min_c, z_max_c,
            n_z_coarse, z_spacing, float(z_surf_offset),
        )

        if coarse_cache_key in self._delay_matrix_cache:
            delay_data_c = self._delay_matrix_cache[coarse_cache_key]
        else:
            src_enu_c = self._build_source_enu_matrix(rho_vec_c, phi_vec_c, z_vec_c)
            if self._multi_ray_types:
                delay_data_c = self._compute_tt_multiray(
                    src_enu_c, channels
                )
            else:
                delay_data_c = self._compute_delay_matrices(src_enu_c, channels)
            self._delay_matrix_cache[coarse_cache_key] = delay_data_c

        t0 = time.time()
        if self._multi_ray_types:
            mean_corr_c, max_corr_c = self._multiray_correlate(
                corr_data, delay_data_c, channels,
                pair_weights=pair_weights, force_perpair=True
            )
        else:
            mean_corr_c, max_corr_c = self._correlator_lean(
                corr_data, delay_data_c, pair_weights=pair_weights,
                delay_cache_key=coarse_cache_key,
            )
        t_coarse = time.time() - t0

        n_coarse_peaks = config.get('coarse_n_peaks', 3)
        coarse_sep = config.get('coarse_peak_separation', [60, 15, 60])
        phi_vec_deg_c = phi_vec_c * (180.0 / np.pi)

        coarse_peaks = self._extract_top_n_peaks(
            mean_corr_c, rho_vec_c, phi_vec_deg_c, z_vec_c,
            n_coarse_peaks, coarse_sep
        )

        if not coarse_peaks:
            logger.warning("No coarse peaks found")
            self._set_station_parameters(
                station, np.nan, np.nan, np.nan, np.nan)
            return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                    'max_corr': np.nan}

        # Stage 2: Refined linear scan around each coarse peak
        full_limits = config.get('limits', coarse_limits)

        refine_levels = config.get('refine_levels', None)
        if refine_levels is None:
            refine_levels = [{
                'window': config.get('refine_window', [150, 20, 150]),
                'steps': config.get('refine_step_sizes', [5, 1, 5]),
                'n_peaks': config.get('coarse_n_peaks', 3),
            }]

        t0_ref = time.time()
        current_peaks = coarse_peaks

        for level_idx, level in enumerate(refine_levels):
            level_window = level['window']
            level_steps = level['steps']
            level_n_peaks = level.get('n_peaks', 3)
            level_sep = level.get('peak_separation',
                                  config.get('peak_separation_threshold',
                                             [10, 5, 10]))

            level_peaks = []
            for rho_p, phi_p, z_p, corr_p in current_peaks:
                rho_lo = max(max(full_limits[0], 1.0),
                             rho_p - level_window[0])
                rho_hi = min(full_limits[1], rho_p + level_window[0])
                phi_lo = phi_p - level_window[1]
                phi_hi = phi_p + level_window[1]
                z_lo = max(full_limits[4], z_p - level_window[2])
                z_hi = min(full_limits[5], z_p + level_window[2])

                rho_vec_r = np.arange(max(rho_lo, 1.0),
                                      rho_hi + level_steps[0],
                                      level_steps[0])
                phi_vec_r = np.arange(phi_lo, phi_hi + level_steps[1],
                                      level_steps[1]) * (np.pi / 180.0)
                z_vec_r = np.arange(z_lo, z_hi + level_steps[2],
                                    level_steps[2])

                if (len(rho_vec_r) == 0 or len(phi_vec_r) == 0
                        or len(z_vec_r) == 0):
                    continue

                src_enu_r = self._build_source_enu_matrix(
                    rho_vec_r, phi_vec_r, z_vec_r)

                if self._multi_ray_types:
                    tt_data_r = self._compute_tt_multiray(
                        src_enu_r, channels
                    )
                    mean_corr_r, _ = self._multiray_correlate(
                        corr_data, tt_data_r, channels,
                        pair_weights=pair_weights
                    )
                else:
                    delay_data_r = self._compute_delay_matrices(
                        src_enu_r, channels
                    )
                    mean_corr_r, _ = self._correlator_lean(
                        corr_data, delay_data_r,
                        pair_weights=pair_weights
                    )

                phi_vec_deg_r = phi_vec_r * (180.0 / np.pi)
                n_extract = config.get('n_optimizer_seeds', 3)

                local_peaks = self._extract_top_n_peaks(
                    mean_corr_r, rho_vec_r, phi_vec_deg_r, z_vec_r,
                    n_extract, level_sep
                )
                level_peaks.extend(local_peaks)

            if not level_peaks:
                level_peaks = current_peaks

            level_peaks.sort(key=lambda x: x[3], reverse=True)
            current_peaks = level_peaks[:level_n_peaks]

        t_refine = time.time() - t0_ref
        refined_peaks = current_peaks

        n_seeds = config.get('n_optimizer_seeds', 3)
        refined_peaks.sort(key=lambda x: x[3], reverse=True)
        refined_peaks = refined_peaks[:n_seeds]

        # Stage 3: optimizer (optional)
        skip_optimizer = config.get('skip_optimizer', False)
        opt_method = config.get('optimizer_method', 'L-BFGS-B')
        opt_maxiter = config.get('optimizer_maxiter', 30)
        use_tdoa = config.get('use_tdoa_seed', False)
        t0_opt = time.time()

        if skip_optimizer:
            all_optimized = [(p[0], p[1] % 360.0, p[2], p[3])
                             for p in refined_peaks]
            rho_best, phi_best, z_best, corr_best = all_optimized[0]
        else:
            bounds = [
                (max(full_limits[0], 1.0), full_limits[1]),
                (full_limits[2], full_limits[3]),
                (full_limits[4], full_limits[5]),
            ]
            opt_cache = self._build_optimizer_cache(channels, pair_weights,
                                                       corr_data)

            # Add TDOA seed from top coarse peak
            if use_tdoa and self._multi_ray_types:
                top_peak = refined_peaks[0]
                tdoa_result = self._tdoa_solve(
                    corr_data, channels, pair_weights,
                    initial_guess=(top_peak[0], top_peak[1], top_peak[2]),
                    cache=opt_cache,
                )
                if tdoa_result is not None:
                    refined_peaks.append(tdoa_result)

            # Add rho-perturbed seeds to explore rho space
            rho_offsets = config.get('optimizer_rho_offsets', None)
            if rho_offsets:
                top = refined_peaks[0]
                for frac in rho_offsets:
                    rho_seed = max(1.0, top[0] * (1 + frac))
                    refined_peaks.append(
                        (rho_seed, top[1], top[2], top[3] * 0.99)
                    )

            all_optimized = []
            for rho_p, phi_p, z_p, corr_p in refined_peaks:
                phi_seed = phi_p % 360.0
                rho_opt, phi_opt, z_opt, corr_opt = self._optimize_from_seed(
                    (rho_p, phi_seed, z_p), corr_data, channels, bounds,
                    pair_weights, method=opt_method, maxiter=opt_maxiter,
                    _cache=opt_cache
                )
                all_optimized.append(
                    (rho_opt, phi_opt % 360.0, z_opt, corr_opt))

            all_optimized.sort(key=lambda x: x[3], reverse=True)
            all_optimized = self._deduplicate_peaks(all_optimized)
            rho_best, phi_best, z_best, corr_best = all_optimized[0]

        t_opt = time.time() - t0_opt

        # Stage 4: Post-optimizer refinement
        t0_post = time.time()
        post_mode = config.get('post_optimizer_mode', None)

        if post_mode == 'rho_scan' and not skip_optimizer:
            rho_step = config.get('rho_scan_step', 5.0)
            rho_scan = np.arange(
                max(full_limits[0], 1.0), full_limits[1] + rho_step, rho_step)
            best_scan_corr = corr_best
            best_scan_rho = rho_best
            for rho_s in rho_scan:
                neg_corr = self._correlation_at_point(
                    [rho_s, phi_best, z_best], corr_data, channels,
                    pair_weights, _cache=opt_cache)
                if -neg_corr > best_scan_corr:
                    best_scan_corr = -neg_corr
                    best_scan_rho = rho_s
            if abs(best_scan_rho - rho_best) > rho_step:
                r2, p2, z2, c2 = self._optimize_from_seed(
                    (best_scan_rho, phi_best, z_best),
                    corr_data, channels, bounds, pair_weights,
                    method=opt_method, maxiter=opt_maxiter, _cache=opt_cache)
                if c2 > corr_best:
                    rho_best, phi_best, z_best, corr_best = r2, p2, z2, c2

        elif post_mode == 'differential_evolution' and not skip_optimizer:
            from scipy.optimize import differential_evolution
            de_window = config.get('de_window', [200, 10, 100])
            de_bounds = [
                (max(full_limits[0], max(1.0, rho_best - de_window[0])),
                 min(full_limits[1], rho_best + de_window[0])),
                (phi_best - de_window[1], phi_best + de_window[1]),
                (max(full_limits[4], z_best - de_window[2]),
                 min(full_limits[5], z_best + de_window[2])),
            ]

            def de_obj(params):
                rho, phi_deg, z = params
                phi_actual = phi_deg % 360.0
                return self._correlation_at_point(
                    [rho, phi_actual, z], corr_data, channels,
                    pair_weights, _cache=opt_cache)

            de_result = differential_evolution(
                de_obj, de_bounds,
                maxiter=config.get('de_maxiter', 50),
                popsize=config.get('de_popsize', 10),
                tol=1e-6, seed=42, polish=True,
                init='sobol',
            )
            de_corr = -de_result.fun
            if de_corr > corr_best:
                rho_best = de_result.x[0]
                phi_best = de_result.x[1] % 360.0
                z_best = de_result.x[2]
                corr_best = de_corr

        elif post_mode == 'basinhopping' and not skip_optimizer:
            from scipy.optimize import basinhopping
            bh_window = config.get('bh_window', [200, 10, 100])
            bh_bounds = [
                (max(full_limits[0], max(1.0, rho_best - bh_window[0])),
                 min(full_limits[1], rho_best + bh_window[0])),
                (phi_best - bh_window[1], phi_best + bh_window[1]),
                (max(full_limits[4], z_best - bh_window[2]),
                 min(full_limits[5], z_best + bh_window[2])),
            ]

            def bh_obj(params):
                rho, phi_deg, z = params
                phi_actual = phi_deg % 360.0
                return self._correlation_at_point(
                    [rho, phi_actual, z], corr_data, channels,
                    pair_weights, _cache=opt_cache)

            bh_kwargs = {'method': 'L-BFGS-B', 'bounds': bh_bounds,
                         'options': {'maxiter': 20}}
            bh_result = basinhopping(
                bh_obj, [rho_best, phi_best, z_best],
                minimizer_kwargs=bh_kwargs,
                niter=config.get('bh_niter', 20),
                stepsize=config.get('bh_stepsize', 50.0),
                seed=42,
            )
            bh_corr = -bh_result.fun
            if bh_corr > corr_best:
                rho_best = bh_result.x[0]
                phi_best = bh_result.x[1] % 360.0
                z_best = bh_result.x[2]
                corr_best = bh_corr

        t_post = time.time() - t0_post

        # Stage 5: Raw-correlation refinement (hybrid envelope approach)
        # Rebuild correlation functions without envelope and re-optimize
        # from the current best seed for sharper peak localization.
        t0_raw = time.time()
        refine_envelope = config.get('refinement_envelope_mode', 'UNSET')
        if refine_envelope != 'UNSET' and refine_envelope != hilbert_mode \
                and not skip_optimizer:
            raw_corr_data = self._prepare_corr_funcs(
                time_arrays, v_pairs,
                hilbert_envelope_mode=refine_envelope,
                apply_hann_window=apply_hann,
                correlation_normalization=corr_norm,
            )
            raw_cache = self._build_optimizer_cache(
                channels, pair_weights, raw_corr_data)

            raw_window = config.get('refinement_window', [30, 3, 30])
            raw_bounds = [
                (max(full_limits[0], max(1.0, rho_best - raw_window[0])),
                 min(full_limits[1], rho_best + raw_window[0])),
                (phi_best - raw_window[1], phi_best + raw_window[1]),
                (max(full_limits[4], z_best - raw_window[2]),
                 min(full_limits[5], z_best + raw_window[2])),
            ]
            raw_maxiter = config.get('refinement_maxiter', 30)

            r_raw, p_raw, z_raw, c_raw = self._optimize_from_seed(
                (rho_best, phi_best, z_best),
                raw_corr_data, channels, raw_bounds, pair_weights,
                method=opt_method, maxiter=raw_maxiter, _cache=raw_cache,
            )
            # Use raw-refined position but keep Hilbert correlation as
            # quality metric (raw correlation has different normalization)
            corr_hilbert = corr_best
            rho_best, phi_best, z_best = r_raw, p_raw, z_raw
            corr_best = corr_hilbert

        t_raw_refine = time.time() - t0_raw
        phi_best = phi_best % 360.0

        # Stage 6: Multi-peak quality metrics and coherent waveforms
        t0_peaks = time.time()
        n_peaks_save = config.get('n_peaks_save', 1)
        save_coh_wf = config.get('save_coherent_waveforms', False)
        n_coh_wf = config.get('n_coherent_waveforms', 1)

        phi_vec_deg_c = phi_vec_c * (180.0 / np.pi)
        saved_peaks = all_optimized[:n_peaks_save]

        peak_map_snrs = []
        for rho_p, phi_p, z_p, corr_p in saved_peaks:
            pidx = self._find_peak_bin(
                rho_p, phi_p, z_p, rho_vec_c, phi_vec_deg_c, z_vec_c)
            peak_map_snrs.append(self._compute_map_snr(mean_corr_c, pidx))

        coherent_waveforms = []
        coherent_times = None
        if save_coh_wf and not self._multi_ray_types:
            for i, (rho_p, phi_p, z_p, corr_p) in enumerate(saved_peaks):
                if i >= n_coh_wf:
                    break
                t_coh, v_coh = self._compute_coherent_waveform(
                    rho_p, phi_p, z_p, volt_arrays, time_arrays, channels)
                if t_coh is not None:
                    if coherent_times is None:
                        coherent_times = t_coh
                    coherent_waveforms.append(v_coh)

        t_peaks = time.time() - t0_peaks

        logger.debug(
            "3D hierarchical: coarse=%.3fs, refine=%.3fs, opt=%.3fs, "
            "post=%.3fs, raw=%.3fs, peaks=%.3fs, rho=%.1f phi=%.1f z=%.1f "
            "corr=%.4f, n_saved=%d",
            t_coarse, t_refine, t_opt, t_post, t_raw_refine, t_peaks,
            rho_best, phi_best, z_best, corr_best, len(saved_peaks)
        )

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        result = {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'coarse_time': t_coarse,
            'refine_time': t_refine,
            'opt_time': t_opt,
            'post_time': t_post if post_mode else 0.0,
            'raw_refine_time': t_raw_refine if refine_envelope != 'UNSET' else 0.0,
            'peak_time': t_peaks,
            'n_coarse_peaks': len(coarse_peaks),
            'n_refined_peaks': len(refined_peaks),
            'n_saved_peaks': len(saved_peaks),
            'coarse_peaks': coarse_peaks,
        }

        for i, (rho_p, phi_p, z_p, corr_p) in enumerate(saved_peaks):
            result[f'peak_{i}_rho'] = rho_p
            result[f'peak_{i}_phi'] = phi_p
            result[f'peak_{i}_z'] = z_p
            result[f'peak_{i}_corr'] = corr_p
            result[f'peak_{i}_map_snr'] = peak_map_snrs[i]

        for i, wf in enumerate(coherent_waveforms):
            result[f'coherent_wf_{i}'] = wf
        if coherent_times is not None:
            result['coherent_times'] = coherent_times

        if config.get('validation', False):
            val = self._compute_validation_metrics(
                mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c,
                channel_snrs, coarse_peaks, config)
            result.update(val)

        return result

    def _compute_validation_metrics(self, mean_corr, rho_vec, phi_vec, z_vec,
                                       channel_snrs, coarse_peaks, config):
        """Compute per-channel SNR summaries, surface correlation, and peak isolation.

        Args:
            mean_corr: 3D coarse correlation array.
            rho_vec: Coarse rho grid (meters).
            phi_vec: Coarse phi grid (radians).
            z_vec: Coarse z grid (meters).
            channel_snrs: Dict mapping channel_id to SNR.
            coarse_peaks: List of (rho, phi, z, corr) tuples.
            config: Reco config dict.

        Returns:
            Dict of validation metrics.
        """
        result = {}

        for ch, snr in channel_snrs.items():
            result[f'ch{ch}_snr'] = snr

        pa = self.DEFAULT_DEPTH_GROUPS['pa']
        hb = self.DEFAULT_DEPTH_GROUPS['helper_b']
        hc = self.DEFAULT_DEPTH_GROUPS['helper_c']
        helpers = hb + hc
        all_chs = pa + hb + hc

        def _safe_stat(chs, func):
            vals = [channel_snrs.get(ch, 0.0) for ch in chs]
            return float(func(vals)) if vals else 0.0

        threshold = config.get('helper_snr_threshold', 5.0)
        result['pa_avg_snr'] = _safe_stat(pa, np.mean)
        result['pa_max_snr'] = _safe_stat(pa, np.max)
        result['helper_b_max_snr'] = _safe_stat(hb, np.max)
        result['helper_b_min_snr'] = _safe_stat(hb, np.min)
        result['helper_c_max_snr'] = _safe_stat(hc, np.max)
        result['helper_c_min_snr'] = _safe_stat(hc, np.min)
        helper_snrs = [channel_snrs.get(ch, 0.0) for ch in helpers]
        all_snrs = [channel_snrs.get(ch, 0.0) for ch in all_chs]
        result['n_helpers_above'] = int(sum(1 for s in helper_snrs if s > threshold))
        result['n_channels_above'] = int(sum(1 for s in all_snrs if s > threshold))
        result['has_helper_signal'] = result['n_helpers_above'] > 0

        # Surface correlation
        pa_center_z = self._pa_center[2]
        surf_z_max = config.get('surf_corr_z_max', -10.0)
        z_mask = z_vec >= surf_z_max
        result['surf_corr_z'] = (float(np.nanmax(mean_corr[:, :, z_mask]))
                                 if z_mask.any() else np.nan)
        surf_zen_max = config.get('surf_corr_zen_max', 65.0)
        rho_g, _, z_g = np.meshgrid(rho_vec, phi_vec, z_vec, indexing='ij')
        zen_grid = np.degrees(np.arctan2(rho_g, z_g - pa_center_z))
        zen_mask = zen_grid <= surf_zen_max
        result['surf_corr_zen'] = (float(np.nanmax(mean_corr[zen_mask]))
                                   if zen_mask.any() else np.nan)

        # Peak isolation
        if len(coarse_peaks) >= 2:
            corrs = sorted([p[3] for p in coarse_peaks], reverse=True)
            top_mean = np.mean(corrs[:5])
            result['peak_isolation_ratio'] = (
                float(corrs[0] / top_mean) if top_mean > 0 else np.nan)
        else:
            result['peak_isolation_ratio'] = np.nan

        return result

    def _run_per_polarization(self, evt, station, det, config):
        """Run independent reconstruction per polarization group.

        Each polarization gets its own correlation map, peak finding, and
        optimizer. No cross-pol pairs are ever formed. The first group
        (typically VPOL) provides the primary result; subsequent groups
        add supplementary fields with a group-name suffix.

        Parameters
        ----------
        evt : Event
            NuRadioReco Event object.
        station : Station
            Station object.
        det : Detector
            Detector description.
        config : dict
            Must contain 'polarization_groups' mapping group names to
            channel lists.

        Returns
        -------
        dict
            Primary result from first group, plus per-group results
            keyed as '{field}_{group_name}'.
        """
        pol_groups = config['polarization_groups']
        all_channels = config['channels']
        results = {}
        primary_group = None

        for group_name, group_chs in pol_groups.items():
            active = [ch for ch in all_channels if ch in group_chs]
            if len(active) < 2:
                logger.info("Pol group '%s': %d channels, skipping (need >= 2)",
                            group_name, len(active))
                continue

            group_config = dict(config)
            group_config['channels'] = active
            group_config.pop('polarization_groups', None)
            group_config.pop('hpol_weight_scale', None)

            logger.info("Running %s reco: %d channels %s",
                         group_name, len(active), active)

            grp_result = self.run(evt, station, det, group_config)

            for key, val in grp_result.items():
                results[f'{key}_{group_name}'] = val

            if primary_group is None:
                primary_group = group_name
                results.update(grp_result)

        if primary_group is None:
            logger.warning("No polarization group had >= 2 channels")
            return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                    'max_corr': np.nan}

        return results

    def run(self, evt, station, det, config):
        """Run 3D interferometric reconstruction on one event.

        Performs a coarse 3D grid scan, extracts the top-N peaks, then refines
        each with L-BFGS-B. Sets station parameters with the best result.

        Parameters
        ----------
        evt : Event
            NuRadioReco Event object.
        station : Station
            Station object containing channel data.
        det : Detector
            Detector description.
        config : dict or str
            Configuration dictionary or path to YAML file.

        Returns
        -------
        dict
            Reconstruction results with keys 'rho', 'phi', 'z', 'max_corr'.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        if config.get('polarization_groups', None) is not None:
            return self._run_per_polarization(evt, station, det, config)

        if config.get('tdoa_mode', False):
            return self.run_tdoa(evt, station, det, config)

        if config.get('hierarchical', False):
            return self.run_hierarchical(evt, station, det, config)

        station_id = station.get_id()
        channels = config['channels']
        hilbert_mode = config.get('hilbert_envelope_mode', None)
        apply_hann = config.get('apply_hann_window', False)
        corr_norm = config.get('correlation_normalization', 'normalized')
        pair_weights = config.get('pair_weights', None)

        rho_vec, phi_vec, z_vec = self._generate_coord_arrays(config)
        phi_vec_deg = phi_vec * (180.0 / np.pi)
        src_enu = self._build_source_enu_matrix(rho_vec, phi_vec, z_vec)

        volt_arrays = []
        time_arrays = []
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())

        if (config.get('snr_pair_weighting', False) or config.get('validation', False)) and pair_weights is None:
            pair_weights, _ = self._compute_snr_pair_weights(
                volt_arrays, channels
            )
            if not config.get('snr_pair_weighting', False):
                pair_weights = None

        v_pairs = list(itertools.combinations(volt_arrays, 2))

        corr_data = self._prepare_corr_funcs(
            time_arrays, v_pairs,
            hilbert_envelope_mode=hilbert_mode,
            apply_hann_window=apply_hann,
            correlation_normalization=corr_norm,
        )

        t0 = time.time()
        if self._multi_ray_types:
            tt_data = self._compute_tt_multiray(src_enu, channels)
            mean_corr, max_corr_grid = self._multiray_correlate(
                corr_data, tt_data, channels, pair_weights=pair_weights,
            )
        else:
            delay_matrices = self._get_t_delay_matrices(
                station_id, config, src_enu, channels
            )
            mean_corr, max_corr_grid, _ = self._correlator(
                corr_data, delay_matrices, pair_weights=pair_weights,
            )
        t_grid = time.time() - t0

        n_seeds = config.get('n_optimizer_seeds', 3)
        sep = config.get('peak_separation_threshold', [10, 10, 10])
        peaks = self._extract_top_n_peaks(
            mean_corr, rho_vec, phi_vec_deg, z_vec, n_seeds, sep
        )

        if not peaks:
            logger.warning("No peaks found in coarse grid")
            self._set_station_parameters(
                station, np.nan, np.nan, np.nan, np.nan)
            return {'rho': np.nan, 'phi': np.nan, 'z': np.nan,
                    'max_corr': np.nan}

        bounds = [
            (max(config['limits'][0], 1.0), config['limits'][1]),
            (config['limits'][2], config['limits'][3]),
            (config['limits'][4], config['limits'][5]),
        ]

        t0_opt = time.time()
        best = None
        for rho_p, phi_p, z_p, corr_p in peaks:
            rho_opt, phi_opt, z_opt, corr_opt = self._optimize_from_seed(
                (rho_p, phi_p, z_p), corr_data, channels, bounds, pair_weights
            )
            if best is None or corr_opt > best[3]:
                best = (rho_opt, phi_opt, z_opt, corr_opt)
        t_opt = time.time() - t0_opt

        rho_best, phi_best, z_best, corr_best = best

        logger.debug(
            "3D reco: grid=%.3fs, opt=%.3fs, "
            "rho=%.1f phi=%.1f z=%.1f corr=%.4f",
            t_grid, t_opt, rho_best, phi_best, z_best, corr_best
        )

        self._set_station_parameters(
            station, rho_best, phi_best, z_best, corr_best)

        return {
            'rho': rho_best,
            'phi': phi_best,
            'z': z_best,
            'max_corr': corr_best,
            'grid_time': t_grid,
            'opt_time': t_opt,
            'n_peaks_found': len(peaks),
            'coarse_peaks': peaks,
        }

    def end(self):
        """Clean up caches."""
        self._delay_matrix_cache.clear()
