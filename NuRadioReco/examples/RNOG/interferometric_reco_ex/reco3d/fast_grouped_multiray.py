"""Numba-accelerated grouped multiray correlator.

Fuses the combo/pair/grid-point loops into a single compiled kernel,
avoiding Python loop overhead and temporary array allocations.
"""
import numpy as np
import itertools

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


RAY_TYPES = ['direct', 'refracted', 'reflected']


def pack_tt_grids(tt_all, channels, grid_shape):
    """Pack per-channel per-ray-type travel time grids into a contiguous 3D array.

    Parameters
    ----------
    tt_all : dict
        Maps channel_id -> {ray_type_name -> 3D grid}.
    channels : list
        Channel IDs.
    grid_shape : tuple
        Shape of each travel time grid.

    Returns
    -------
    np.ndarray
        Shape (n_channels, 3, n_points). Ray type indices: 0=direct,
        1=refracted, 2=reflected. Missing ray types filled with NaN.
    list
        Per-channel list of available ray type indices.
    """
    n_ch = len(channels)
    n_points = int(np.prod(grid_shape))
    rt_map = {rt: i for i, rt in enumerate(RAY_TYPES)}

    tt_packed = np.full((n_ch, 3, n_points), np.nan, dtype=np.float64)
    ch_available_rts = []

    for ci, ch in enumerate(channels):
        avail = []
        for rt_name, rt_idx in rt_map.items():
            if rt_name in tt_all.get(ch, {}):
                tt_packed[ci, rt_idx, :] = tt_all[ch][rt_name].ravel()
                avail.append(rt_idx)
        ch_available_rts.append(avail)

    return tt_packed, ch_available_rts


def pack_tt_grids_transposed(tt_all, channels, grid_shape):
    """Pack travel time grids with point-major layout for cache locality.

    For single-threaded kernels, this layout puts all channel/ray-type data
    for a single grid point in contiguous memory (n_ch * 3 * 8 = 264 bytes
    for 11 channels), fitting in ~4 cache lines instead of requiring 33
    scattered reads across 211 MB.

    Parameters
    ----------
    tt_all : dict
        Maps channel_id -> {ray_type_name -> 3D grid}.
    channels : list
        Channel IDs.
    grid_shape : tuple
        Shape of each travel time grid.

    Returns
    -------
    np.ndarray
        Shape (n_points, n_channels, 3), C-contiguous.
    list
        Per-channel list of available ray type indices.
    """
    n_ch = len(channels)
    n_points = int(np.prod(grid_shape))
    rt_map = {rt: i for i, rt in enumerate(RAY_TYPES)}

    tt_t = np.full((n_points, n_ch, 3), np.nan, dtype=np.float64)
    ch_available_rts = []

    for ci, ch in enumerate(channels):
        avail = []
        for rt_name, rt_idx in rt_map.items():
            if rt_name in tt_all.get(ch, {}):
                tt_t[:, ci, rt_idx] = tt_all[ch][rt_name].ravel()
                avail.append(rt_idx)
        ch_available_rts.append(avail)

    return tt_t, ch_available_rts


def pack_corr_data(corr_data, n_pairs):
    """Pack variable-length correlation arrays into padded contiguous arrays.

    Parameters
    ----------
    corr_data : list of tuple
        (corr_array, dt, offset) per pair.
    n_pairs : int
        Number of pairs.

    Returns
    -------
    np.ndarray
        Padded correlation arrays, shape (n_pairs, max_len).
    np.ndarray
        Lengths of each correlation array, shape (n_pairs,).
    np.ndarray
        dt values, shape (n_pairs,).
    np.ndarray
        offset values, shape (n_pairs,).
    """
    lengths = np.array([len(cd[0]) for cd in corr_data], dtype=np.int64)
    max_len = int(lengths.max())
    dts = np.array([cd[1] for cd in corr_data], dtype=np.float64)
    offsets = np.array([cd[2] for cd in corr_data], dtype=np.float64)

    corr_packed = np.zeros((n_pairs, max_len), dtype=np.float64)
    for i in range(n_pairs):
        corr_packed[i, :lengths[i]] = corr_data[i][0]

    return corr_packed, lengths, dts, offsets


def build_combo_table(channels, ch_to_group, n_groups, ch_available_rts):
    """Build the combo enumeration table.

    Parameters
    ----------
    channels : list
        Channel IDs.
    ch_to_group : dict
        Maps channel ID to group index.
    n_groups : int
        Number of depth groups.
    ch_available_rts : list
        Per-channel list of available ray type indices.

    Returns
    -------
    np.ndarray
        Shape (n_combos, n_channels). Each row gives the ray type index
        for each channel under that combo.
    """
    group_rts = []
    for gidx in range(n_groups):
        group_chs_ci = [ci for ci, ch in enumerate(channels)
                        if ch_to_group[ch] == gidx]
        rts = set(range(3))
        for ci in group_chs_ci:
            rts &= set(ch_available_rts[ci])
        if not rts:
            # No common ray type across all channels in this group.
            # Fall back to the union so the product is never empty;
            # invalid travel times produce NaN delays which are skipped.
            for ci in group_chs_ci:
                rts |= set(ch_available_rts[ci])
        if not rts:
            rts = {0}  # direct as last resort
        group_rts.append(sorted(rts))

    combos = list(itertools.product(*group_rts))
    n_combos = len(combos)
    n_ch = len(channels)

    ch_group_indices = np.array([ch_to_group[ch] for ch in channels],
                                dtype=np.int64)
    combo_table = np.empty((n_combos, n_ch), dtype=np.int64)
    for ci_combo, combo in enumerate(combos):
        for ci_ch in range(n_ch):
            combo_table[ci_combo, ci_ch] = combo[ch_group_indices[ci_ch]]

    return combo_table


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _grouped_multiray_kernel(tt_packed, corr_packed, corr_lengths,
                                 corr_dts, corr_offsets,
                                 pair_ch1, pair_ch2, pair_weights,
                                 combo_table, n_points):
        """Core kernel: evaluate all combos and return the best weighted mean.

        Parameters
        ----------
        tt_packed : float64 array (n_ch, 3, n_points)
            Travel times per channel per ray type.
        corr_packed : float64 array (n_pairs, max_corr_len)
            Padded correlation arrays.
        corr_lengths : int64 array (n_pairs,)
            Actual lengths of each correlation array.
        corr_dts : float64 array (n_pairs,)
            Sample spacing per pair.
        corr_offsets : float64 array (n_pairs,)
            Time offset per pair.
        pair_ch1 : int64 array (n_pairs,)
            First channel index per pair.
        pair_ch2 : int64 array (n_pairs,)
            Second channel index per pair.
        pair_weights : float64 array (n_pairs,)
            Weight per pair.
        combo_table : int64 array (n_combos, n_ch)
            Ray type index per channel per combo.
        n_points : int
            Number of grid points.

        Returns
        -------
        float64 array (n_points,)
            Best weighted mean correlation across combos at each point.
        """
        n_combos = combo_table.shape[0]
        n_pairs = pair_ch1.shape[0]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        best = np.full(n_points, -np.inf)

        for pt in prange(n_points):
            local_best = -np.inf
            for ci_combo in range(n_combos):
                total = 0.0
                for pidx in range(n_pairs):
                    c1 = pair_ch1[pidx]
                    c2 = pair_ch2[pidx]
                    rt1 = combo_table[ci_combo, c1]
                    rt2 = combo_table[ci_combo, c2]

                    tt1 = tt_packed[c1, rt1, pt]
                    tt2 = tt_packed[c2, rt2, pt]
                    delay = tt1 - tt2

                    if not np.isfinite(delay):
                        continue

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

                if w_sum > 0.0:
                    total /= w_sum
                if total > local_best:
                    local_best = total

            best[pt] = local_best

        return best


    @njit(parallel=True, fastmath=True, cache=True)
    def _grouped_multiray_kernel_pairmajor(tt_packed, corr_packed,
                                            corr_lengths, corr_dts,
                                            corr_offsets, pair_ch1,
                                            pair_ch2, pair_weights,
                                            combo_table, n_points):
        """Grouped multiray kernel with pair-major inner loop.

        For each combo, accumulates pair contributions across all points
        one pair at a time, keeping the correlation array in cache.
        Uses tt_packed (n_ch, n_rt, n_points) layout.

        Parameters
        ----------
        tt_packed : float64 array (n_ch, n_rt, n_points)
        corr_packed : float64 array (n_pairs, max_corr_len)
        corr_lengths : int64 array (n_pairs,)
        corr_dts : float64 array (n_pairs,)
        corr_offsets : float64 array (n_pairs,)
        pair_ch1 : int64 array (n_pairs,)
        pair_ch2 : int64 array (n_pairs,)
        pair_weights : float64 array (n_pairs,)
        combo_table : int64 array (n_combos, n_ch)
        n_points : int

        Returns
        -------
        float64 array (n_points,)
        """
        n_combos = combo_table.shape[0]
        n_pairs = pair_ch1.shape[0]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        best = np.full(n_points, -np.inf, dtype=np.float64)
        combo_accum = np.zeros(n_points, dtype=np.float64)

        for ci_combo in range(n_combos):
            for pt in range(n_points):
                combo_accum[pt] = 0.0

            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                rt1 = combo_table[ci_combo, c1]
                rt2 = combo_table[ci_combo, c2]
                w = pair_weights[pidx]
                dt = corr_dts[pidx]
                offset = corr_offsets[pidx]
                clen = corr_lengths[pidx]

                for pt in prange(n_points):
                    tt1 = tt_packed[c1, rt1, pt]
                    tt2 = tt_packed[c2, rt2, pt]
                    delay = tt1 - tt2

                    if not np.isfinite(delay):
                        continue

                    kf = (delay - offset) / dt
                    k = int(np.floor(kf))
                    if k < 0 or k >= clen - 1:
                        val = 0.0
                    else:
                        alpha = kf - k
                        val = (corr_packed[pidx, k]
                               + (corr_packed[pidx, k + 1]
                                  - corr_packed[pidx, k]) * alpha)

                    combo_accum[pt] += val * w

            if w_sum > 0.0:
                for pt in prange(n_points):
                    combo_accum[pt] /= w_sum

            for pt in prange(n_points):
                if combo_accum[pt] > best[pt]:
                    best[pt] = combo_accum[pt]

        return best


    @njit(parallel=True, fastmath=True, cache=True)
    def _perpair_multiray_kernel(tt_packed, corr_packed, corr_lengths,
                                 corr_dts, corr_offsets,
                                 pair_ch1, pair_ch2, pair_weights,
                                 ch_rt_mask, n_points):
        """Per-pair multiray kernel: max across 9 ray combos per pair.

        Accelerated version of _correlator_lean_multiray for comparison.

        Parameters
        ----------
        tt_packed : float64 array (n_ch, 3, n_points)
        corr_packed : float64 array (n_pairs, max_corr_len)
        corr_lengths : int64 array (n_pairs,)
        corr_dts : float64 array (n_pairs,)
        corr_offsets : float64 array (n_pairs,)
        pair_ch1 : int64 array (n_pairs,)
        pair_ch2 : int64 array (n_pairs,)
        pair_weights : float64 array (n_pairs,)
        ch_rt_mask : bool array (n_ch, 3)
            Which ray types are available per channel.
        n_points : int

        Returns
        -------
        float64 array (n_points,)
        """
        n_pairs = pair_ch1.shape[0]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        result = np.zeros(n_points)

        for pt in prange(n_points):
            total = 0.0
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                best_val = 0.0

                for rt1 in range(3):
                    if not ch_rt_mask[c1, rt1]:
                        continue
                    tt1 = tt_packed[c1, rt1, pt]
                    if not np.isfinite(tt1):
                        continue
                    for rt2 in range(3):
                        if not ch_rt_mask[c2, rt2]:
                            continue
                        tt2 = tt_packed[c2, rt2, pt]
                        delay = tt1 - tt2
                        if not np.isfinite(delay):
                            continue

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

                        if val > best_val:
                            best_val = val

                total += best_val * pair_weights[pidx]

            if w_sum > 0.0:
                total /= w_sum
            result[pt] = total

        return result


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _perpair_multiray_kernel_pairmajor(tt_packed, corr_packed,
                                            corr_lengths, corr_dts,
                                            corr_offsets, pair_ch1,
                                            pair_ch2, pair_weights,
                                            ch_rt_mask, n_points):
        """Pair-major per-pair multiray kernel for better cache locality.

        Outer loop over pairs, inner loop over points. For each pair, the
        correlation array and pair metadata stay in cache while sweeping
        all grid points. Uses tt_packed (n_ch, n_rt, n_points) layout so
        travel times for a given (ch, rt) are contiguous across points.

        Parameters
        ----------
        tt_packed : float64 array (n_ch, n_rt, n_points)
        corr_packed : float64 array (n_pairs, max_corr_len)
        corr_lengths : int64 array (n_pairs,)
        corr_dts : float64 array (n_pairs,)
        corr_offsets : float64 array (n_pairs,)
        pair_ch1 : int64 array (n_pairs,)
        pair_ch2 : int64 array (n_pairs,)
        pair_weights : float64 array (n_pairs,)
        ch_rt_mask : bool array (n_ch, n_rt)
        n_points : int

        Returns
        -------
        float64 array (n_points,)
        """
        n_pairs = pair_ch1.shape[0]
        n_rt = tt_packed.shape[1]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        result = np.zeros(n_points, dtype=np.float64)

        for pidx in range(n_pairs):
            c1 = pair_ch1[pidx]
            c2 = pair_ch2[pidx]
            w = pair_weights[pidx]
            dt = corr_dts[pidx]
            offset = corr_offsets[pidx]
            clen = corr_lengths[pidx]

            for pt in prange(n_points):
                best_val = 0.0
                for rt1 in range(n_rt):
                    if not ch_rt_mask[c1, rt1]:
                        continue
                    tt1 = tt_packed[c1, rt1, pt]
                    if not np.isfinite(tt1):
                        continue
                    for rt2 in range(n_rt):
                        if not ch_rt_mask[c2, rt2]:
                            continue
                        tt2 = tt_packed[c2, rt2, pt]
                        delay = tt1 - tt2
                        if not np.isfinite(delay):
                            continue

                        kf = (delay - offset) / dt
                        k = int(np.floor(kf))
                        if k < 0 or k >= clen - 1:
                            val = 0.0
                        else:
                            alpha = kf - k
                            val = (corr_packed[pidx, k]
                                   + (corr_packed[pidx, k + 1]
                                      - corr_packed[pidx, k]) * alpha)

                        if val > best_val:
                            best_val = val

                result[pt] += best_val * w

        if w_sum > 0.0:
            for pt in prange(n_points):
                result[pt] /= w_sum

        return result


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _perpair_multiray_kernel_t(tt_t, corr_packed, corr_lengths,
                                    corr_dts, corr_offsets,
                                    pair_ch1, pair_ch2, pair_weights,
                                    ch_rt_mask, n_points):
        """Per-pair multiray kernel with point-major TT layout.

        Same algorithm as _perpair_multiray_kernel but tt_t has shape
        (n_points, n_ch, 3) for better cache locality when iterating
        over grid points.

        Parameters
        ----------
        tt_t : float64 array (n_points, n_ch, 3)
            Travel times, point-major layout.
        corr_packed : float64 array (n_pairs, max_corr_len)
        corr_lengths : int64 array (n_pairs,)
        corr_dts : float64 array (n_pairs,)
        corr_offsets : float64 array (n_pairs,)
        pair_ch1 : int64 array (n_pairs,)
        pair_ch2 : int64 array (n_pairs,)
        pair_weights : float64 array (n_pairs,)
        ch_rt_mask : bool array (n_ch, 3)
        n_points : int

        Returns
        -------
        float64 array (n_points,)
        """
        n_pairs = pair_ch1.shape[0]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        result = np.zeros(n_points)

        for pt in prange(n_points):
            total = 0.0
            for pidx in range(n_pairs):
                c1 = pair_ch1[pidx]
                c2 = pair_ch2[pidx]
                best_val = 0.0

                for rt1 in range(3):
                    if not ch_rt_mask[c1, rt1]:
                        continue
                    tt1 = tt_t[pt, c1, rt1]
                    if not np.isfinite(tt1):
                        continue
                    for rt2 in range(3):
                        if not ch_rt_mask[c2, rt2]:
                            continue
                        tt2 = tt_t[pt, c2, rt2]
                        delay = tt1 - tt2
                        if not np.isfinite(delay):
                            continue

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

                        if val > best_val:
                            best_val = val

                total += best_val * pair_weights[pidx]

            if w_sum > 0.0:
                total /= w_sum
            result[pt] = total

        return result


    @njit(parallel=True, fastmath=True, cache=True)
    def _grouped_multiray_kernel_t(tt_t, corr_packed, corr_lengths,
                                    corr_dts, corr_offsets,
                                    pair_ch1, pair_ch2, pair_weights,
                                    combo_table, n_points):
        """Grouped combo kernel with point-major TT layout.

        Parameters
        ----------
        tt_t : float64 array (n_points, n_ch, 3)
        corr_packed : float64 array (n_pairs, max_corr_len)
        corr_lengths : int64 array (n_pairs,)
        corr_dts : float64 array (n_pairs,)
        corr_offsets : float64 array (n_pairs,)
        pair_ch1 : int64 array (n_pairs,)
        pair_ch2 : int64 array (n_pairs,)
        pair_weights : float64 array (n_pairs,)
        combo_table : int64 array (n_combos, n_ch)
        n_points : int

        Returns
        -------
        float64 array (n_points,)
        """
        n_combos = combo_table.shape[0]
        n_pairs = pair_ch1.shape[0]
        w_sum = 0.0
        for p in range(n_pairs):
            w_sum += pair_weights[p]

        best = np.full(n_points, -np.inf)

        for pt in prange(n_points):
            local_best = -np.inf
            for ci_combo in range(n_combos):
                total = 0.0
                for pidx in range(n_pairs):
                    c1 = pair_ch1[pidx]
                    c2 = pair_ch2[pidx]
                    rt1 = combo_table[ci_combo, c1]
                    rt2 = combo_table[ci_combo, c2]

                    tt1 = tt_t[pt, c1, rt1]
                    tt2 = tt_t[pt, c2, rt2]
                    delay = tt1 - tt2

                    if not np.isfinite(delay):
                        continue

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

                if w_sum > 0.0:
                    total /= w_sum
                if total > local_best:
                    local_best = total

            best[pt] = local_best

        return best


def grouped_multiray_numba(corr_data, tt_all, channels, ch_to_group,
                           n_groups, pair_weights=None):
    """Numba-accelerated grouped multiray correlator.

    Parameters
    ----------
    corr_data : list of tuple
        Pre-computed (corr_array, dt, offset) per pair.
    tt_all : dict
        Maps ch -> {ray_type -> grid}.
    channels : list
        Channel IDs.
    ch_to_group : dict
        Maps channel ID to group index.
    n_groups : int
        Number of groups.
    pair_weights : list or None
        Per-pair weights.

    Returns
    -------
    tuple
        (mean_corr_map, max_corr) with mean_corr_map reshaped to grid_shape.
    """
    import itertools as _it

    grid_shape = None
    for ch in channels:
        for rt in tt_all.get(ch, {}):
            grid_shape = tt_all[ch][rt].shape
            break
        if grid_shape is not None:
            break
    if grid_shape is None:
        return np.zeros(1), np.nan

    n_points = int(np.prod(grid_shape))
    ch_pairs = list(_it.combinations(range(len(channels)), 2))
    n_pairs = len(ch_pairs)

    tt_t, ch_available_rts = pack_tt_grids_transposed(
        tt_all, channels, grid_shape)
    corr_packed, corr_lengths, dts, offsets = pack_corr_data(corr_data, n_pairs)
    combo_table = build_combo_table(channels, ch_to_group, n_groups,
                                    ch_available_rts)

    pair_ch1 = np.array([p[0] for p in ch_pairs], dtype=np.int64)
    pair_ch2 = np.array([p[1] for p in ch_pairs], dtype=np.int64)

    if pair_weights is not None:
        pw = np.asarray(pair_weights, dtype=np.float64)
    else:
        pw = np.ones(n_pairs, dtype=np.float64)

    result_flat = _grouped_multiray_kernel_t(
        tt_t, corr_packed, corr_lengths, dts, offsets,
        pair_ch1, pair_ch2, pw, combo_table, n_points
    )

    neg_inf = result_flat == -np.inf
    result_flat[neg_inf] = 0.0

    result = result_flat.reshape(grid_shape)
    max_corr = float(np.max(result)) if result.size > 0 else np.nan
    return result, max_corr


def perpair_multiray_numba(corr_data, tt_all, channels, pair_weights=None):
    """Numba-accelerated per-pair multiray correlator.

    Drop-in replacement for _correlator_lean_multiray.

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
    import itertools as _it

    grid_shape = None
    for ch in channels:
        for rt in tt_all.get(ch, {}):
            grid_shape = tt_all[ch][rt].shape
            break
        if grid_shape is not None:
            break
    if grid_shape is None:
        return np.zeros(1), np.nan

    n_points = int(np.prod(grid_shape))
    ch_pairs = list(_it.combinations(range(len(channels)), 2))
    n_pairs = len(ch_pairs)

    tt_t, ch_available_rts = pack_tt_grids_transposed(
        tt_all, channels, grid_shape)
    corr_packed, corr_lengths, dts, offsets = pack_corr_data(corr_data, n_pairs)

    n_ch = len(channels)
    ch_rt_mask = np.zeros((n_ch, 3), dtype=np.bool_)
    for ci in range(n_ch):
        for rt_idx in ch_available_rts[ci]:
            ch_rt_mask[ci, rt_idx] = True

    pair_ch1 = np.array([p[0] for p in ch_pairs], dtype=np.int64)
    pair_ch2 = np.array([p[1] for p in ch_pairs], dtype=np.int64)

    if pair_weights is not None:
        pw = np.asarray(pair_weights, dtype=np.float64)
    else:
        pw = np.ones(n_pairs, dtype=np.float64)

    result_flat = _perpair_multiray_kernel_t(
        tt_t, corr_packed, corr_lengths, dts, offsets,
        pair_ch1, pair_ch2, pw, ch_rt_mask, n_points
    )

    result = result_flat.reshape(grid_shape)
    max_corr = float(np.max(result)) if result.size > 0 else np.nan
    return result, max_corr
