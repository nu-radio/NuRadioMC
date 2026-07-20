#!/usr/bin/env python3
"""Load and plot results from `deep_cr_reco`.

This script reads a saved NumPy archive produced by `deep_cr_reco.py` and
produces a set of standard diagnostic plots.

Example:
    python plot_deep_cr_reco_results.py --run-number 1

Or point directly to a file:
    python plot_deep_cr_reco_results.py --input /path/to/results_run_number_1.npz

Plots are written next to the input file by default.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import units


def plot_hist(ax, data, label, **kwargs):
    ax.hist(data, **kwargs, histtype="step", linewidth=2, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot deep_cr_reco results from a saved run number")
    parser.add_argument("output_folder", type=str, default="results", help="Load results from folder with this name")
    parser.add_argument("--show", action="store_true", help="Show the plots interactively")
    args = parser.parse_args()

    # Match the same output directory structure used in deep_cr_reco.py
    here = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(here, "results", f"{args.output_folder}")
    input_file = os.path.join(output_dir, "results.npz")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Results file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    data = np.load(input_file)

    # extracted arrays (names used in deep_cr_reco.py)
    snr = data.get("snr")
    polarization_true = data.get("polarization_true")
    polarization_llh = data.get("polarization_llh")
    fluence = data.get("fluence")
    zenith_initial = data.get("zenith_initial")
    azimuth_initial = data.get("azimuth_initial")
    zenith_reco = data.get("zenith_reco")
    azimuth_reco = data.get("azimuth_reco")
    fluence_uf = data.get("fluence_uf")
    polarization_uf = data.get("polarization_uf")
    fluence_uf_all = data.get("fluence_uf_all")
    polarization_uf_all = data.get("polarization_uf_all")
    polarization_uf_error = data.get("polarization_uf_error")
    params = data.get("params")
    llh = data.get("llh")
    polarization_error = data.get("polarization_error")
    fluence_error = data.get("fluence_error")
    p_value = data.get("p_value")

    valid_indicies = polarization_true!=0
    snr = snr[valid_indicies]
    polarization_true = polarization_true[valid_indicies]
    #polarization_true = abs(polarization_true) + 2 * (90 * units.deg - abs(polarization_true))
    polarization_llh = polarization_llh[valid_indicies]
    fluence = fluence[valid_indicies]
    zenith_initial = zenith_initial[valid_indicies]
    azimuth_initial = azimuth_initial[valid_indicies]
    polarization_uf = polarization_uf[valid_indicies]
    polarization_uf_error = polarization_uf_error[valid_indicies]
    fluence_uf = fluence_uf[valid_indicies]
    params = params[valid_indicies]
    llh = llh[valid_indicies]
    polarization_error = polarization_error[valid_indicies]
    fluence_error = fluence_error[valid_indicies]
    p_value = p_value[valid_indicies]

    polarization_true_90 = abs(polarization_true)
    selection = polarization_true_90 > 90 * units.deg
    polarization_true_90[selection] = 180 * units.deg - polarization_true_90[selection]

    # Basic checks
    n_events = len(polarization_true)
    print(f"Loaded {n_events} events from {input_file}")


    # Change polarity/polarization convention:
    f_theta = np.abs(params[:,0])
    f_phi = np.abs(params[:,1])
    fluence = f_theta + f_phi
    # polarization = np.arctan2(np.sqrt(f_phi), np.sqrt(f_theta))
    A_theta = np.sign(params[:,0]) * f_theta**0.5
    A_phi = np.sign(params[:,1]) * f_phi**0.5
    polarization_llh = np.arctan2(A_phi, A_theta)
    phi = params[:,3]
    shift = np.pi/4
    for i in range(n_events):
        if (phi[i] + shift) % (2 * np.pi) > np.pi:
            if polarization_llh[i] > 0:
                polarization_llh[i] -= 180 * units.deg
            elif polarization_llh[i] <= 0:
                polarization_llh[i] += 180 * units.deg

    # ------------------------------------------------------------------
    # Polarization comparisons
    # ------------------------------------------------------------------
    plt.figure(figsize=[6, 4])
    plt.hist(polarization_true/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='-', label="True polarization")
    #plt.hist(polarization_ref_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='--', label="Refracted")
    # plt.hist(polarization_h1_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='--', label="H1")
    # plt.hist(polarization_h2_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls=':', label="H2")
    plt.xlabel("Polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("True polarization distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_true_distribution.png"))
    plt.show()
    plt.close()



    # Unfolding results:
    plt.figure(figsize=[6, 4])
    plt.hist(polarization_true/units.deg, bins=30, range=[-180, 180], histtype='step', linewidth=2, ls='-', label="True polarization")
    plt.hist(polarization_uf/units.deg, bins=30, range=[-180, 180], histtype='step', linewidth=2, ls='--', label="Unfolded polarization")
    plt.xlabel("Polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("True vs Unfolded polarization distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_uf.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[6, 4])
    plt.scatter(polarization_true/units.deg, polarization_uf/units.deg, label="Unfolded", alpha=0.5)
    plt.xlabel("Polarization angle 0 [deg]")
    plt.ylabel("Polarization angle unfolded [deg]")
    plt.legend()
    plt.title("True vs Unfolded polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_uf_2D.png"))
    plt.show()
    plt.close()


    # Limit to 0-90 deg:
    plt.hist(polarization_true_90/units.deg, bins=30, range=[0, 90], histtype='step', linewidth=2, ls='-', label="True polarization")
    plt.hist(polarization_uf/units.deg, bins=30, range=[0, 90], histtype='step', linewidth=2, ls='--', label="Unfolded polarization")
    plt.xlabel("Polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("True vs Unfolded polarization distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_uf_alt.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[5, 4])
    x = polarization_true_90/units.deg
    y = polarization_uf/units.deg
    xy_min = 0
    xy_max = 91
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", label = "1:1")
    plt.scatter(x, y, c=np.log10(snr), s=3, label="UF reco", alpha=1)
    plt.axis([xy_min, xy_max, xy_min, xy_max])
    plt.xlabel("True polarization [deg]")
    plt.ylabel("Reconstructed polarization (UF) [deg]")
    plt.colorbar(label=r"log10(SNR$_{max}$)")
    plt.legend()
    plt.title("True vs UF reco polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_uf_2D_alt.png"))
    plt.show()

    plt.figure(figsize=[6, 4])
    x = abs(polarization_uf/units.deg) - abs(polarization_true_90/units.deg)
    sigma_68 = (np.quantile(x, 0.84) - np.quantile(x, 0.16))/2
    plt.hist(x, bins=25, range=[-180, 180], histtype='step', linewidth=2, ls='-', color="r", label=f"Unfolding - True polarization, $\sigma_{{68\%}} = {str(np.round(sigma_68, 1))}$ deg")
    plt.xlabel("Delta polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend(loc=1)
    plt.title("Unfolding vs True polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_uf_delta_alt.png"))
    plt.show()
    plt.close()


    # LLH reconstruction results:
    plt.figure(figsize=[6, 4])
    plt.hist(polarization_true/units.deg, bins=25, range=[-180, 180], histtype='step', linewidth=2, ls='-', label="True polarization")
    plt.hist(polarization_llh/units.deg, bins=25, range=[-180, 180], histtype='step', linewidth=2, ls='--', label="LLH reco polarization")
    plt.xlabel("Polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("True vs LLH reco polarization distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[6, 4])
    plt.hist(abs(polarization_true/units.deg), bins=25, range=[0, 180], histtype='step', linewidth=2, ls='-', label="True polarization") #label=""""True" polarization""")
    plt.hist(abs(polarization_llh/units.deg), bins=25, range=[0, 180], histtype='step', linewidth=2, ls='--', label="LLH reco polarization")
    plt.xlabel("Polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("True vs LLH reco polarization distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_alt.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[6, 4])
    x = polarization_llh/units.deg - polarization_true/units.deg
    sigma_68 = (np.quantile(x, 0.84) - np.quantile(x, 0.16))/2
    plt.hist(x, bins=25, range=[-180, 180], histtype='step', linewidth=2, ls='-', color="r", label=f"LLH reco - True polarization, $\sigma_{{68\%}} = {str(np.round(sigma_68, 1))}$ deg")
    plt.xlabel("Delta polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend(loc=1)
    plt.title("LLH plarization reconstruction minus true distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_delta.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[6, 4])
    x = abs(polarization_llh/units.deg) - abs(polarization_true/units.deg)
    sigma_68 = (np.quantile(x, 0.84) - np.quantile(x, 0.16))/2
    plt.hist(x, bins=25, range=[-180, 180], histtype='step', linewidth=2, ls='-', color="r", label=f"LLH reco - True polarization, $\sigma_{{68\%}} = {str(np.round(sigma_68, 1))}$ deg")
    plt.xlabel("Delta polarization angle [deg]")
    plt.ylabel("Counts")
    plt.legend(loc=1)
    plt.title("LLH plarization reconstruction minus true distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_delta_alt.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=[5, 4])
    x = polarization_true/units.deg
    y = polarization_llh/units.deg
    xy_min = min([min(x), min(y)]) * 1.2
    xy_max = max([max(x), max(y)]) * 1.2
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", label = "1:1")
    plt.scatter(x, y, s=2, label="LLH reco", alpha=0.5)
    plt.axis([xy_min, xy_max, xy_min, xy_max])
    plt.xlabel("True polarization [deg]")
    plt.ylabel("Reconstructed polarization (LLH) [deg]")
    plt.legend()
    plt.title("True vs LLH reco polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_2D.png"))
    plt.show()

    plt.figure(figsize=[5, 4])
    x = abs(polarization_true)/units.deg
    y = abs(polarization_llh)/units.deg
    xy_min = 0
    xy_max = 180
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", label = "1:1")
    plt.scatter(x, y, c=np.log10(snr), s=3, label="LLH reco", alpha=1)
    plt.axis([xy_min, xy_max, xy_min, xy_max])
    plt.xlabel("True polarization [deg]")
    plt.ylabel("Reconstructed polarization (LLH) [deg]")
    plt.colorbar(label=r"log10(SNR$_{max}$)")
    plt.legend()
    plt.title("True vs LLH reco polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_2D_alt.png"))
    plt.show()

    # LLH and P value distributions:
    fig, ax = plt.subplots(2, 1, figsize=[4, 6])
    ax[0].hist(llh, label="LLH distribution", bins=30)
    ax[0].set_xlabel("LLH")
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    ax[0].grid()

    ax[1].hist(p_value, bins=30, range=[0, 1], histtype='step', linewidth=2, ls='-', label="P value distribution")
    ax[1].set_xlabel("P value")
    ax[1].set_ylabel("Counts")
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"p_value_distribution.png"))
    plt.show()

    # Pull plots:
    fig, ax = plt.subplots(1, 2, figsize=[8, 4])
    x = (abs(polarization_llh) - abs(polarization_true))
    selection = abs(x) < 45 * units.deg
    pull = x[selection] / polarization_error[selection]
    std = np.std(pull)
    ax[0].hist(pull, bins=15, histtype='step', linewidth=2, ls='-', label=f"Pull distribution, std = {str(np.round(std, 3))}") #, range=[-5, 5]
    ax[0].set_xlabel("Pull (reco - true)/uncertainty")
    ax[0].set_ylabel("Counts")
    ax[0].legend(loc=1)
    ax[0].grid()
    ax[0].set_title("LLH reco polarization pull")
    x = (abs(polarization_uf) - abs(polarization_true_90))
    selection = abs(x) < 90 * units.deg
    std = np.std(x[selection])
    ax[1].hist(x[selection], bins=15, histtype='step', linewidth=2, ls='-', label=f"Pull distribution, std = {str(np.round(std, 3))}") #, range=[-5, 5]
    ax[1].set_xlabel("Pull (reco - true)/uncertainty")
    ax[1].set_ylabel("Counts")
    ax[1].legend(loc=1)
    ax[1].grid()
    ax[1].set_title("Unfolded polarization pull")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_pull.png"))

    print(abs(polarization_llh))
    print(abs(polarization_true))
    print(polarization_error)
    print(abs(polarization_llh) - abs(polarization_true))

    plt.figure()
    #plt.errorbar(abs(polarization_true), abs(polarization_llh), yerr=polarization_error, fmt='o', markersize=2, alpha=0.5)
    #plt.scatter(abs(polarization_true), abs(polarization_llh), c=np.log10(snr), s=3, label="LLH reco", alpha=1)
    plt.scatter((abs(polarization_true) - abs(polarization_llh))/polarization_error, polarization_error, c=np.log10(snr), s=3, label="LLH reco", alpha=1)
    plt.savefig("debug.png")
    quit()


    # Plot llh vs unfolding correlation
    plt.figure(figsize=[6, 4])
    polarization_llh_90 = abs(polarization_llh)
    selection = polarization_llh_90 > 90 * units.deg
    polarization_llh_90[selection] = 180 * units.deg - polarization_llh_90[selection]
    plt.scatter(polarization_llh_90/units.deg, polarization_uf/units.deg, c=np.log10(snr), s=5,alpha=1)
    plt.xlabel("LLH reco polarization [deg]")
    plt.ylabel("Unfolded polarization [deg]")
    plt.colorbar(label=r"log10(SNR$_{max}$)")
    plt.axis([0, 91, 0, 91])
    plt.plot([0, 91], [0, 91], "k--", label = "1:1")
    plt.title("LLH reco vs Unfolded polarization")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_vs_uf_2D.png"))
    plt.show()


    # Plot reconstructed paramters:
    n_paramters = 6 if (params[:,5] != 0).any() else 5
    parameter_names = ["f_theta", "f_phi", "slope", "phase", "time", "2nd order"]
    fig, ax = plt.subplots(n_paramters, n_paramters, figsize=[n_paramters*2, n_paramters*2])
    for i in range(n_paramters):
        for j in range(n_paramters):
            if i == j:
                ax[i,j].hist(params[:,i], bins=30)
            if i > j:
                ax[i,j].scatter(params[:,j], params[:,i], s=2, alpha=0.5)
            if i == n_paramters-1:
                ax[i,j].set_xlabel(parameter_names[j])
            if j == 0 and i > 0:
                ax[i,j].set_ylabel(parameter_names[i])
            if i == 0 and j == 0:
                ax[i,j].set_ylabel("Counts")
            if j > i:
                ax[i,j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"polarization_llh_params_2D.png"))
