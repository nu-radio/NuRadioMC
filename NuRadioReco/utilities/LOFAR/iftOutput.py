#!/usr/bin/env python3
"""
iftOutput.py — reconstruction diagnostic plot for the NuRadioReco LOFAR IFT pipeline.

Produces a multi-panel summary figure showing the lateral distribution function,
timing residuals, and posterior parameter distributions from the IFT reconstruction.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""
import logging
import os
import numpy as np
import matplotlib

# The backend must be selected before pyplot is imported, so the imports below
# deliberately do not sit at the top of the file (PEP 8 E402).
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

try:
    import jax.numpy as jnp  # noqa: E402
except ImportError:
    jnp = None

from NuRadioReco.utilities import units  # noqa: E402

logger = logging.getLogger("NuRadioReco.LOFAR.iftOutput")

_FLUENCE_RELATIVE_SYSTEMATIC_ERROR = 0.15
_TIMING_CLIP_NS = 15.0
_LOFAR_B_FIELD = np.array([0.004675, 0.186270, -0.456412])

PLOT_STYLE = {
    "text.usetex": False, "font.family": "serif", "font.size": 24,
    "axes.labelsize": 32, "axes.titlesize": 30, "axes.titleweight": "bold",
    "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 28, "figure.dpi": 300,
}
PLOT_PARAMS = {
    'fg_color': '#ffae00', 'fg_edge': '#cc8b00', 'old_color': 'olivedrab',
    'cmap_fl_map': 'plasma_r', 'cmap_tm_map': 'viridis',
    'cmap_fl_res': 'inferno_r', 'cmap_tm_res': 'YlGn_r',
    'cmap_fl_cf_mean': 'cividis_r', 'cmap_fl_cf_std': 'cividis_r',
    'cmap_tm_cf_mean': 'GnBu_r', 'cmap_tm_cf_std': 'bone',
    'scatter_size': 40,
}


def _setup_ax_fancy(ax, title, extent, xlabel="Easting [m]", ylabel="Northing [m]",
                    show_xlabel=True, show_ylabel=True):
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")
    if show_xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")


def _plot_kde_fancy(ax, data, label, truth_mean=None, truth_std=None):
    from scipy.stats import gaussian_kde, norm as sp_norm
    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    if len(data) < 3:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        kde = gaussian_kde(data)
        mu = np.mean(data)
        sigma = np.std(data)
        x_min, x_max = np.min(data), np.max(data)
        if truth_mean is not None and truth_std is not None:
            x_min = min(x_min, truth_mean - 4 * truth_std)
            x_max = max(x_max, truth_mean + 4 * truth_std)
        span = x_max - x_min
        x_grid = np.linspace(x_min - 0.2 * span, x_max + 0.2 * span, 500)
        y_kde = kde(x_grid)
        ax.plot(x_grid, y_kde, color=PLOT_PARAMS['fg_edge'], lw=3.0,
                label=f"{label}: {mu:.2f} ± {sigma:.2f}")
        ax.fill_between(x_grid, y_kde, color=PLOT_PARAMS['fg_color'], alpha=0.2)
        ax.vlines(mu, 0, np.max(y_kde), color=PLOT_PARAMS['fg_edge'], linestyle='--', lw=2.0)
        if truth_mean is not None and truth_std is not None:
            y_truth = sp_norm.pdf(x_grid, loc=truth_mean, scale=truth_std)
            ax.plot(x_grid, y_truth, color=PLOT_PARAMS['old_color'], linestyle=':', lw=3.0,
                    label=f"LORA: {truth_mean:.2f} ± {truth_std:.2f}")
            ax.fill_between(x_grid, y_truth, color=PLOT_PARAMS['old_color'], alpha=0.1)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=18, frameon=False)
    except Exception as e:
        logger.warning(f"KDE plot failed: {e}")
    ax.set_yticks([])
    ax.set_ylim(bottom=0)


def _plot_2d_contour_fancy(ax, x_data, y_data, old_point=None, x_label='', y_label='', title=''):
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    LOFAR_GREEN = '#405d3a'
    LOFAR_CMAP = LinearSegmentedColormap.from_list("LofarGreen", ["white", LOFAR_GREEN])
    x_data = np.asarray(x_data).flatten()
    y_data = np.asarray(y_data).flatten()
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data, y_data = x_data[valid], y_data[valid]
    if len(x_data) < 5:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
    else:
        try:
            xy = np.vstack([x_data, y_data])
            kde = gaussian_kde(xy)
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            x_std = max(np.std(x_data), 1e-9)
            y_std = max(np.std(y_data), 1e-9)
            x_pad = max(0.5 * (x_max - x_min), 4.0 * x_std) if x_max != x_min else 1.0
            y_pad = max(0.5 * (y_max - y_min), 4.0 * y_std) if y_max != y_min else 1.0
            x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 200)
            y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 200)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            Z_sorted = np.sort(Z.flatten())[::-1]
            Z_cum = np.cumsum(Z_sorted)
            Z_cum /= Z_cum[-1]
            level_1sig = Z_sorted[np.searchsorted(Z_cum, 0.6827)]
            level_2sig = Z_sorted[np.searchsorted(Z_cum, 0.9545)]
            level_3sig = Z_sorted[np.searchsorted(Z_cum, 0.9973)]
            ax.contourf(X, Y, Z, levels=100, cmap=LOFAR_CMAP, alpha=0.8)
            ax.contour(X, Y, Z, levels=sorted([level_3sig, level_2sig, level_1sig]),
                       colors=LOFAR_GREEN, linewidths=[0.8, 1.2, 1.8], alpha=0.9)
        except Exception:
            ax.scatter(x_data, y_data, s=3, color='gray', alpha=0.5, rasterized=True)
    if old_point is not None:
        ax.scatter(*old_point, marker='*', s=150, c=PLOT_PARAMS['old_color'],
                   zorder=10, label='LORA')
    ax.scatter(np.mean(x_data), np.mean(y_data), marker='+', s=120, c='#405d3a',
               linewidth=2.0, zorder=9, label='IFT Mean')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=20, frameon=False)


def generate_reco_plot(samples, samples_ecr, all_data, output_dir, event_id,
                       ref_params, ref_label='LORA', att_truth=None, cf_stats=None,
                       signal_response=None, noise_mean=0.0, model_kw=None,
                       noise_level=None, b_field=None):
    """
    Publication-quality reconstruction summary plot.

    Parameters
    ----------
    samples : list
        Filtered posterior samples.
    samples_ecr : list
        Cosmic-ray energy (eV) for each sample.
    all_data : dict
        Keys: pos_x, pos_y, fluences, times, is_signal.
    output_dir : str
        Directory for the output PNG.
    event_id : int or str
        Event identifier (used in filename).
    ref_params : dict
        LORA reference: zenith (rad), azimuth (rad), core ([x,y] m), energy (eV, optional).
    signal_response : footprintModel
        The NIFTy model object.
    noise_mean : float
        Estimated noise floor (eV/m²).
    model_kw : dict
        Kwargs used to construct signal_response (used to build a grid model).
    noise_level : float or None
        Noise std used for fluence residuals; defaults to noise_mean.
    """
    from NuRadioReco.modules.LOFAR.utilities.iftModel import footprintModel

    plt.rcParams.update(PLOT_STYLE)

    pos_x = np.asarray(all_data['pos_x'])
    pos_y = np.asarray(all_data['pos_y'])
    pos = np.array([pos_x, pos_y])
    fluences = np.asarray(all_data['fluences'])
    times = np.asarray(all_data['times'])
    is_sig = np.asarray(all_data['is_signal'], dtype=bool)

    old_core = ref_params['core']
    old_zenith_deg = np.rad2deg(ref_params['zenith'])
    old_azimuth_deg = np.rad2deg(ref_params['azimuth']) % 360.0

    if att_truth and ('core_x' in att_truth) and ('core_y' in att_truth):
        cmp_core = (att_truth['core_x'], att_truth['core_y'])
    else:
        cmp_core = (old_core[0], old_core[1])
    cmp_zenith_deg = (np.rad2deg(att_truth['zenith'])
                      if (att_truth and 'zenith' in att_truth) else old_zenith_deg)
    cmp_azimuth_deg = ((np.rad2deg(att_truth['azimuth']) % 360.0)
                       if (att_truth and 'azimuth' in att_truth) else old_azimuth_deg)

    model_min_x = signal_response.min_x
    model_min_y = signal_response.min_y
    model_extent_val = signal_response.extent
    model_dims = signal_response.dims
    extent = [model_min_x, model_min_x + model_extent_val,
              model_min_y, model_min_y + model_extent_val]

    grid_ax_x = np.linspace(model_min_x, model_min_x + model_extent_val, model_dims[0])
    grid_ax_y = np.linspace(model_min_y, model_min_y + model_extent_val, model_dims[1])
    GX, GY = np.meshgrid(grid_ax_x, grid_ax_y, indexing='ij')
    flat_gx, flat_gy = GX.flatten(), GY.flatten()

    if model_kw is not None:
        grid_kw = model_kw.copy()
        grid_kw['enable_syst_cf'] = False
        grid_kw['enable_timing_cf'] = False
        if 'noise_mean' in grid_kw:
            grid_kw['noise_mean'] = float(np.asarray(grid_kw['noise_mean']).ravel()[0])
        grid_kw.pop('grid_settings', None)
        bfield_arr = jnp.array(b_field if b_field is not None else _LOFAR_B_FIELD)
        grid_model = footprintModel(flat_gx, flat_gy, bfield_arr, **grid_kw)
    else:
        grid_model = None

    fl_acc, tm_acc, nm_acc = 0.0, 0.0, 0.0
    for s in samples:
        if grid_model is not None:
            res = grid_model(s)
            fl_acc += np.asarray(res[0]).flatten()
            tm_acc += np.asarray(res[1]).flatten()
        try:
            nm_acc += float(np.mean(np.asarray(signal_response.noise_mean(s))))
        except Exception:
            nm_acc += noise_mean

    noise_mean_model = nm_acc / len(samples)
    if grid_model is not None:
        map_fl_raw = (fl_acc / len(samples)).reshape(GX.shape).T
        map_fl = np.maximum(map_fl_raw - noise_mean_model, 0)
        map_fl[map_fl < 0.05 * noise_mean] = 0.0
        map_tm_raw = (tm_acc / len(samples)).reshape(GX.shape).T
    else:
        map_fl = np.zeros(GX.shape).T
        map_tm_raw = np.zeros(GX.shape).T

    fl_pts_vis = np.maximum(fluences - noise_mean, 0)

    fl_pred_acc = np.zeros(len(pos[0]))
    tm_pred_acc = np.zeros(len(pos[0]))
    for s in samples:
        res_at_antennas = signal_response(s)
        fl_pred_acc += np.asarray(res_at_antennas[0]).flatten()
        tm_pred_acc += np.asarray(res_at_antennas[1]).flatten()
    pred_fl_with_cf = fl_pred_acc / len(samples)
    pred_tm_with_cf = tm_pred_acc / len(samples)

    res_tm_ns = (times - pred_tm_with_cf) * units.s / units.ns
    _noise_sigma = noise_level if noise_level is not None else noise_mean
    sigma_fl = np.sqrt(_noise_sigma ** 2 +
                       (fluences * _FLUENCE_RELATIVE_SYSTEMATIC_ERROR) ** 2)
    res_fl = (fluences - pred_fl_with_cf) / sigma_fl

    if np.sum(is_sig) > 0:
        t0_reference = np.median(pred_tm_with_cf[is_sig])
    else:
        t0_reference = np.median(pred_tm_with_cf)
    map_tm = (map_tm_raw - t0_reference) * units.s / units.ns
    tm_pts_vis_centered = (times - t0_reference) * units.s / units.ns

    vmax_fl = max(
        np.nanmax(map_fl) if not np.all(np.isnan(map_fl)) else 1.0,
        np.nanmax(fl_pts_vis) if fl_pts_vis.size > 0 else 1.0,
        1.0,
    )
    norm_fl = mcolors.Normalize(vmin=0, vmax=vmax_fl)
    tm_data_vals = tm_pts_vis_centered[is_sig]
    if tm_data_vals.size > 0:
        vmin_tm, vmax_tm = np.nanpercentile(tm_data_vals, [1, 99])
    else:
        vmin_tm, vmax_tm = -10, 10
    norm_tm = mcolors.Normalize(vmin=vmin_tm, vmax=vmax_tm)

    size = PLOT_PARAMS['scatter_size']

    fig = plt.figure(figsize=(22, 22), constrained_layout=True)
    gs = gridspec.GridSpec(4, 3, figure=fig, width_ratios=[1, 1, 1.3])

    # --- Column 0: Fluence ---
    ax_f0 = fig.add_subplot(gs[0, 0])
    im_f0 = ax_f0.imshow(map_fl, origin='lower', extent=extent,
                         cmap=PLOT_PARAMS['cmap_fl_map'], norm=norm_fl)
    ax_f0.scatter(pos[0], pos[1], c=fl_pts_vis, cmap=PLOT_PARAMS['cmap_fl_map'],
                  s=size, norm=norm_fl, edgecolors='w', lw=0.5)
    ax_f0.plot(old_core[0], old_core[1], '*', ms=18,
               mec='k', mfc=PLOT_PARAMS['old_color'], mew=1.5, label=ref_label)
    _setup_ax_fancy(ax_f0, "Fluence", extent, show_xlabel=False)
    ax_f0.legend(loc='upper left', fontsize=20, frameon=False)
    fig.colorbar(im_f0, ax=ax_f0, label=r"Signal [eV/m$^2$]")

    ax_f1 = fig.add_subplot(gs[1, 0], sharex=ax_f0, sharey=ax_f0)
    if res_fl.size > 0 and not np.all(np.isnan(res_fl)):
        vm_fl = max(abs(np.nanpercentile(res_fl, 1)), abs(np.nanpercentile(res_fl, 99)), 1.0)
    else:
        vm_fl = 1.0
    sc_f1 = ax_f1.scatter(pos[0], pos[1], c=res_fl, cmap=PLOT_PARAMS['cmap_fl_res'],
                          s=size, vmin=-vm_fl, vmax=vm_fl, edgecolors='none')
    _setup_ax_fancy(ax_f1, "Fluence Residuals", extent, show_xlabel=False)
    fig.colorbar(sc_f1, ax=ax_f1, label=r"(Obs$-$Pred)/$\sigma$")

    ax_f2 = fig.add_subplot(gs[2, 0], sharex=ax_f0, sharey=ax_f0)
    from matplotlib.ticker import ScalarFormatter
    if cf_stats and 'syst_mean' in cf_stats:
        im_f2 = ax_f2.imshow(cf_stats['syst_mean'].T, origin='lower', extent=extent,
                             cmap=PLOT_PARAMS['cmap_fl_cf_mean'])
        cb_f2 = fig.colorbar(im_f2, ax=ax_f2, label="Multiplier")
        cb_f2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        cb_f2.ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    else:
        ax_f2.text(0.5, 0.5, "No CF", ha='center', va='center',
                   transform=ax_f2.transAxes, fontsize=20)
    _setup_ax_fancy(ax_f2, "Fluence CF Mean", extent, show_xlabel=False)

    ax_f3 = fig.add_subplot(gs[3, 0], sharex=ax_f0, sharey=ax_f0)
    if cf_stats and 'syst_std' in cf_stats:
        im_f3 = ax_f3.imshow(cf_stats['syst_std'].T, origin='lower', extent=extent,
                             cmap=PLOT_PARAMS['cmap_fl_cf_std'])
        cb_f3 = fig.colorbar(im_f3, ax=ax_f3, label="Std")
        cb_f3.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        cb_f3.ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    else:
        ax_f3.text(0.5, 0.5, "No CF", ha='center', va='center',
                   transform=ax_f3.transAxes, fontsize=20)
    _setup_ax_fancy(ax_f3, r"Fluence CF $\sigma$", extent)

    # --- Column 1: Timing ---
    ax_t0 = fig.add_subplot(gs[0, 1], sharex=ax_f0, sharey=ax_f0)
    im_t0 = ax_t0.imshow(map_tm, origin='lower', extent=extent,
                         cmap=PLOT_PARAMS['cmap_tm_map'], norm=norm_tm)
    if np.sum(is_sig) > 0:
        ax_t0.scatter(pos[0][is_sig], pos[1][is_sig],
                      c=tm_pts_vis_centered[is_sig], cmap=PLOT_PARAMS['cmap_tm_map'],
                      s=size, norm=norm_tm, edgecolors='w', lw=0.5)
    ax_t0.scatter(pos[0][~is_sig], pos[1][~is_sig],
                  marker='.', color='gray', s=15, alpha=0.5)
    _setup_ax_fancy(ax_t0, "Timing", extent, show_xlabel=False, show_ylabel=False)
    fig.colorbar(im_t0, ax=ax_t0, label="Time [ns]")

    ax_t1 = fig.add_subplot(gs[1, 1], sharex=ax_f0, sharey=ax_f0)
    if np.sum(is_sig) > 0:
        valid_res = res_tm_ns[is_sig]
        vm_tm = max(abs(np.nanpercentile(valid_res, 5)), abs(np.nanpercentile(valid_res, 95)), 5)
        sc_t1 = ax_t1.scatter(pos[0][is_sig], pos[1][is_sig], c=valid_res,
                              cmap=PLOT_PARAMS['cmap_tm_res'], s=size,
                              vmin=-vm_tm, vmax=vm_tm, edgecolors='none')
        ax_t1.scatter(pos[0][~is_sig], pos[1][~is_sig],
                      marker='x', color='gray', s=15, alpha=0.5)
        fig.colorbar(sc_t1, ax=ax_t1, label="Obs$-$Pred [ns]")
    _setup_ax_fancy(ax_t1, "Timing Residuals", extent, show_xlabel=False, show_ylabel=False)

    ax_t2 = fig.add_subplot(gs[2, 1], sharex=ax_f0, sharey=ax_f0)
    if cf_stats and 'timing_mean' in cf_stats:
        t_map = cf_stats['timing_mean']
        t_vlim = max(abs(np.nanmin(t_map)), abs(np.nanmax(t_map)), 1.0) if t_map.size > 0 else 1.0
        im_t2 = ax_t2.imshow(t_map.T, origin='lower', extent=extent,
                             cmap=PLOT_PARAMS['cmap_tm_cf_mean'],
                             vmin=-t_vlim, vmax=t_vlim)
        fig.colorbar(im_t2, ax=ax_t2, label="Correction [ns]")
    else:
        ax_t2.text(0.5, 0.5, "No CF", ha='center', va='center',
                   transform=ax_t2.transAxes, fontsize=20)
    _setup_ax_fancy(ax_t2, "Timing CF Mean", extent, show_xlabel=False, show_ylabel=False)

    ax_t3 = fig.add_subplot(gs[3, 1], sharex=ax_f0, sharey=ax_f0)
    if cf_stats and 'timing_std' in cf_stats:
        im_t3 = ax_t3.imshow(cf_stats['timing_std'].T, origin='lower', extent=extent,
                             cmap=PLOT_PARAMS['cmap_tm_cf_std'])
        fig.colorbar(im_t3, ax=ax_t3, label="Std [ns]")
    else:
        ax_t3.text(0.5, 0.5, "No CF", ha='center', va='center',
                   transform=ax_t3.transAxes, fontsize=20)
    _setup_ax_fancy(ax_t3, r"Timing CF $\sigma$", extent, show_ylabel=False)

    # --- Column 2: Posteriors ---
    ax_xm = fig.add_subplot(gs[0, 2])
    xmax_combined = np.array([
        float(np.asarray(signal_response.X_max_combined(s)).item()) for s in samples
    ])
    _plot_kde_fancy(ax_xm, xmax_combined, "IFT",
                    truth_mean=att_truth['xmax'] if att_truth else None,
                    truth_std=att_truth['xmax_std'] if att_truth else None)
    ax_xm.set_title(r"$X_{\mathrm{max}}$ Posterior")
    ax_xm.set_xlabel(r"$X_{\mathrm{max}}$ [g/cm$^2$]")

    ax_en = fig.add_subplot(gs[1, 2])
    log_e_samples = np.log10(np.asarray(samples_ecr).flatten())
    truth_e = (att_truth['logE'] if att_truth else
               (np.log10(ref_params['energy']) if ref_params.get('energy', 0) > 0 else 17.0))
    truth_e_std = att_truth['logE_std'] if att_truth else 0.086
    _plot_kde_fancy(ax_en, log_e_samples, "IFT",
                    truth_mean=truth_e, truth_std=truth_e_std)
    ax_en.set_title("Energy Posterior")
    ax_en.set_xlabel(r"$\log_{10}(E / \mathrm{eV})$")

    ax_core = fig.add_subplot(gs[2, 2])
    core_x_s = [float(np.asarray(signal_response.core(s)[0]).item()) for s in samples]
    core_y_s = [float(np.asarray(signal_response.core(s)[1]).item()) for s in samples]
    _plot_2d_contour_fancy(ax_core, core_x_s, core_y_s, old_point=cmp_core,
                           x_label='Core X [m]', y_label='Core Y [m]',
                           title='Core Position')

    ax_dir = fig.add_subplot(gs[3, 2])
    az_samples = (np.rad2deg([float(np.asarray(signal_response.zen_and_az(s)[1]).item())
                              for s in samples]) % 360.0)
    zen_samples = np.rad2deg([float(np.asarray(signal_response.zen_and_az(s)[0]).item())
                              for s in samples])
    _plot_2d_contour_fancy(ax_dir, az_samples, zen_samples,
                           old_point=(cmp_azimuth_deg, cmp_zenith_deg),
                           x_label='Azimuth [deg]', y_label='Zenith [deg]',
                           title='Arrival Direction')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"reco_plot_{event_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved reconstruction plot to %s", out_path)
    print(f"Reconstruction plot saved to {out_path}", flush=True)
    return out_path
