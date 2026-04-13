"""
Compute kernels for 3D interferometric direction reconstruction.

Pure compute functions with no class dependency; the main reconstruction
module imports and dispatches to them.
"""

import itertools
import logging
import os
import sys

import numpy as np

logger = logging.getLogger("reco3d.kernels")

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
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_CUPY = True
        logger.info("CuPy GPU backend available (device count=%d)",
                    cp.cuda.runtime.getDeviceCount())

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

        _FUSED_MULTIRAY_CORR_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void fused_multiray_correlator(
    const double* __restrict__ tt_packed,
    const double* __restrict__ corr_packed,
    const long long* __restrict__ corr_lens,
    const double* __restrict__ dts,
    const double* __restrict__ offsets,
    const double* __restrict__ pair_weights,
    const double w_sum,
    const int* __restrict__ pair_ch1,
    const int* __restrict__ pair_ch2,
    const int n_pairs,
    const int n_ch,
    const int n_rt,
    const long long n_points,
    const long long corr_stride,
    double* __restrict__ out
) {
    long long pt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;

    double acc = 0.0;
    for (int pidx = 0; pidx < n_pairs; ++pidx) {
        int c1 = pair_ch1[pidx];
        int c2 = pair_ch2[pidx];
        double best_val = -1e30;

        for (int rt1 = 0; rt1 < n_rt; ++rt1) {
            double tt1 = tt_packed[((long long)c1 * n_rt + rt1) * n_points + pt];
            if (isnan(tt1) || tt1 <= 0.0) continue;
            for (int rt2 = 0; rt2 < n_rt; ++rt2) {
                double tt2 = tt_packed[((long long)c2 * n_rt + rt2) * n_points + pt];
                if (isnan(tt2) || tt2 <= 0.0) continue;
                double d = tt1 - tt2;
                double kf = (d - offsets[pidx]) / dts[pidx];
                long long k = (long long)floor(kf);
                long long clen = corr_lens[pidx];
                if (k < 0 || k >= clen - 1) continue;
                double alpha = kf - (double)k;
                double y0 = corr_packed[(long long)pidx * corr_stride + k];
                double y1 = corr_packed[(long long)pidx * corr_stride + k + 1];
                double v = y0 + (y1 - y0) * alpha;
                if (v > best_val) best_val = v;
            }
        }
        if (best_val > -1e29) {
            acc += best_val * pair_weights[pidx];
        }
    }
    out[pt] = (w_sum > 0.0) ? (acc / w_sum) : 0.0;
}
''', 'fused_multiray_correlator')
except Exception:
    cp = None
    _FUSED_MULTIRAY_CORR_KERNEL = None

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

USE_CPP_EXTENSION = False
try:
    cpp_path = os.path.join(os.path.dirname(__file__), "cpp")
    if cpp_path not in sys.path:
        sys.path.insert(0, cpp_path)
    from fast_delay_matrices_3d import (
        compute_delay_matrices_3d as _compute_delay_matrices_cpp,
    )
    USE_CPP_EXTENSION = True
    logger.info("3D C++ extension loaded")
except (ImportError, OSError):
    logger.info("3D C++ extension not available, using Python fallback")

RAY_TYPES = ['direct', 'refracted', 'reflected']
SOLUTION_TYPES = ['solution_0', 'solution_1']
RAY_TYPE_COMBOS = list(itertools.product(RAY_TYPES, RAY_TYPES))


if USE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _scalar_grouped_corr_numba(
            tt_vals, tt_valid, corr_packed, corr_lengths,
            corr_dts, corr_offsets, pair_ch1, pair_ch2, pair_weights,
            combo_table, n_combos, n_pairs, w_sum):
        """Scalar grouped correlation at a single grid point.

        Args:
            tt_vals: float64 array (n_ch, n_rt).
            tt_valid: bool array (n_ch, n_rt).
            corr_packed: float64 array (n_pairs, max_corr_len).
            corr_lengths: int64 array (n_pairs,).
            corr_dts: float64 array (n_pairs,).
            corr_offsets: float64 array (n_pairs,).
            pair_ch1: int64 array (n_pairs,).
            pair_ch2: int64 array (n_pairs,).
            pair_weights: float64 array (n_pairs,).
            combo_table: int64 array (n_combos, n_ch).
            n_combos: int.
            n_pairs: int.
            w_sum: float64.

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
        """Fast uniform-grid linear interpolation."""
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

        Args:
            rho, phi_rad, z: Scalar source coordinates (m, rad, m).
            pa_x, pa_y: PA center absolute coordinates (m).
            ant_xy: (n_ch, 2) float64, channel absolute (x, y).
            td_values: (n_ch, nr_max, nz_max) float64, packed TT tables.
            td_r_min, td_dr_inv: (n_ch,) float64.
            td_nr: (n_ch,) int64.
            td_z_min, td_dz_inv: same in z.
            td_nz: (n_ch,) int64.
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
            if i0 < 0 or j0 < 0:
                continue
            if i0 >= nr_ch - 1:
                if ri <= nr_ch - 1 + 1e-9:
                    i0 = nr_ch - 2
                    fx = 1.0
                else:
                    continue
            else:
                fx = ri - i0
            if j0 >= nz_ch - 1:
                if zi <= nz_ch - 1 + 1e-9:
                    j0 = nz_ch - 2
                    fy = 1.0
                else:
                    continue
            else:
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

        Parallelizes over grid points (outer prange). Uses points-major
        delay layout ``delay_T[pt, pidx]`` for cache-friendly access.

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
                if np.isnan(d):
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
            values: 2D array (nr, nz).
            r_min, dr_inv, nr, z_min, dz_inv, nz: Grid parameters.
            r_coords, z_coords: 1D query arrays.

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
            values: 2D array (nr, nz).
            r_min, dr_inv, nr, z_min, dz_inv, nz: Grid parameters.
            r_val, z_val: Query coordinates.

        Returns:
            Interpolated value, or -inf if out of bounds.
        """
        ri = (r_val - r_min) * dr_inv
        zi = (z_val - z_min) * dz_inv
        i0 = int(np.floor(ri))
        j0 = int(np.floor(zi))
        if i0 < 0 or j0 < 0:
            return -np.inf
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


if USE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _fused_multiray_grid_numba(
            rho_vec, phi_vec_rad, z_vec,
            pa_x, pa_y,
            ant_xy, n_ch,
            td_values, td_r_min, td_dr_inv, td_nr,
            td_z_min, td_dz_inv, td_nz,
            n_rt,
            corr_packed, corr_lengths, corr_dts, corr_offsets,
            pair_ch1, pair_ch2, pair_weights, w_total):
        """Fused multiray grid correlator: TT lookup + combo evaluation in one kernel.

        For each grid point, computes per-channel per-ray-type travel times
        via inline bilinear table lookup, then evaluates all ray-type combos
        and returns the best weighted mean correlation.

        Args:
            rho_vec: (n_rho,) float64.
            phi_vec_rad: (n_phi,) float64, in radians.
            z_vec: (n_z,) float64.
            pa_x, pa_y: PA center absolute coordinates.
            ant_xy: (n_ch, 2) float64, antenna positions.
            n_ch: int.
            td_values: (n_ch, n_rt, nr_max, nz_max) float64, packed TT tables.
            td_r_min, td_dr_inv: (n_ch, n_rt) float64.
            td_nr: (n_ch, n_rt) int64.
            td_z_min, td_dz_inv: (n_ch, n_rt) float64.
            td_nz: (n_ch, n_rt) int64.
            n_rt: int.
            corr_packed: (n_pairs, max_corr_len) float64.
            corr_lengths: (n_pairs,) int64.
            corr_dts, corr_offsets: (n_pairs,) float64.
            pair_ch1, pair_ch2: (n_pairs,) int64.
            pair_weights: (n_pairs,) float64.
            w_total: float64.

        Returns:
            (n_rho * n_phi * n_z,) float64 per-pair-max correlation at each point.
        """
        n_rho = rho_vec.shape[0]
        n_phi = phi_vec_rad.shape[0]
        n_z = z_vec.shape[0]
        n_points = n_rho * n_phi * n_z
        n_pairs = pair_ch1.shape[0]
        inv_w = 1.0 / w_total if w_total > 0.0 else 0.0

        out = np.empty(n_points, dtype=np.float64)

        for pt in prange(n_points):
            ir = pt // (n_phi * n_z)
            rem = pt % (n_phi * n_z)
            ip = rem // n_z
            iz = rem % n_z

            rho = rho_vec[ir]
            phi = phi_vec_rad[ip]
            z = z_vec[iz]
            x_src = rho * np.cos(phi) + pa_x
            y_src = rho * np.sin(phi) + pa_y

            tts = np.empty((n_ch, n_rt), dtype=np.float64)
            tt_valid = np.zeros((n_ch, n_rt), dtype=np.bool_)

            for ci in range(n_ch):
                dx = x_src - ant_xy[ci, 0]
                dy = y_src - ant_xy[ci, 1]
                r = np.sqrt(dx * dx + dy * dy)
                if r < 1.0:
                    r = 1.0

                for ri in range(n_rt):
                    r_idx = (r - td_r_min[ci, ri]) * td_dr_inv[ci, ri]
                    z_idx = (z - td_z_min[ci, ri]) * td_dz_inv[ci, ri]
                    i0 = int(np.floor(r_idx))
                    j0 = int(np.floor(z_idx))
                    nr_ch = td_nr[ci, ri]
                    nz_ch = td_nz[ci, ri]
                    if (i0 < 0 or i0 >= nr_ch - 1
                            or j0 < 0 or j0 >= nz_ch - 1):
                        continue
                    fx = r_idx - i0
                    fy = z_idx - j0
                    v = ((1.0 - fx) * (1.0 - fy) * td_values[ci, ri, i0, j0]
                         + fx * (1.0 - fy) * td_values[ci, ri, i0 + 1, j0]
                         + (1.0 - fx) * fy * td_values[ci, ri, i0, j0 + 1]
                         + fx * fy * td_values[ci, ri, i0 + 1, j0 + 1])
                    if v > 0.0 and np.isfinite(v):
                        tts[ci, ri] = v
                        tt_valid[ci, ri] = True

            # Per-pair mode: for each pair, try all n_rt^2 ray combos
            # and keep the max correlation. Then sum across pairs.
            total = 0.0
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                best_pair = -np.inf
                for rt1 in range(n_rt):
                    if not tt_valid[c1, rt1]:
                        continue
                    for rt2 in range(n_rt):
                        if not tt_valid[c2, rt2]:
                            continue
                        delay = tts[c1, rt1] - tts[c2, rt2]
                        dt = corr_dts[pidx]
                        offset = corr_offsets[pidx]
                        clen = corr_lengths[pidx]
                        kf = (delay - offset) / dt
                        k = int(np.floor(kf))
                        if k < 0 or k >= clen - 1:
                            continue
                        alpha = kf - k
                        val = (corr_packed[pidx, k]
                               + (corr_packed[pidx, k + 1]
                                  - corr_packed[pidx, k]) * alpha)
                        if val > best_pair:
                            best_pair = val
                if best_pair > -np.inf:
                    total += best_pair * pair_weights[pidx]

            out[pt] = total * inv_w
        return out


def _build_z_vec(z_min, z_max, n_z, spacing='linear', surface_offset=0.1):
    """Build a z-axis vector with linear or log spacing.

    Log spacing concentrates grid density near the ice surface (z=0).

    Args:
        z_min: Minimum z (meters, negative).
        z_max: Maximum z (meters, should be >= 0 for log mode).
        n_z: Number of grid points.
        spacing: 'linear' or 'log'.
        surface_offset: Minimum |z| for the shallowest log bin (meters).

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
