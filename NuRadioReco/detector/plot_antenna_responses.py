import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from NuRadioReco.utilities import units
from NuRadioReco.detector.antennapattern import AntennaPattern, AntennaPatternAnalytic

frequencies = np.linspace(0, 1, 512) * units.GHz

antenna_pattern = AntennaPattern("createLPDA_100MHz_InfFirn_n1.4")

antenna_pattern_analytic = AntennaPatternAnalytic("analytic_LPDA")

phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
theta_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = antenna_response_analytic[0]
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = antenna_response_analytic[1]

fig, ax = plt.subplots(n_theta, n_phi, figsize=(3*n_theta, 2*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (createLPDA)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (createLPDA)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]), ':', color="C0", label='V_theta (analytic)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]), ':', color="C1", label='V_phi (analytic)')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_LPDA.png")
plt.close()

fig, ax = plt.subplots(n_theta, n_phi, figsize=(3*n_theta, 2*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        ax[i_theta, i_phi].plot(frequencies, np.angle((antenna_response_NuRadio_array[i_theta, i_phi, :, 0])), '--', color="C0", label='V_theta (createLPDA) phase')
        ax[i_theta, i_phi].plot(frequencies, np.angle((antenna_response_NuRadio_array[i_theta, i_phi, :, 1])), '--', color="C1", label='V_phi (createLPDA) phase')
        ax[i_theta, i_phi].plot(frequencies, np.angle((antenna_response_analytic_array[i_theta, i_phi, :, 0])), ':', color="C0", label='V_theta (analytic) phase')
        ax[i_theta, i_phi].plot(frequencies, np.angle((antenna_response_analytic_array[i_theta, i_phi, :, 1])), ':', color="C1", label='V_phi (analytic) phase')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_LPDA_complex.png")
plt.close()


# full circular plots
phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg]
theta_array = np.linspace(0, 180 * units.deg, 180)
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = antenna_response_analytic[0]
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = antenna_response_analytic[1]


fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).max()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).max()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"createLPDA")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"createLPDA")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Max. V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Max. V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_LPDA_circular_max.png")
plt.close()



fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).sum()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).sum()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"createLPDA")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"createLPDA")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Sum V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Sum V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_LPDA_circular_sum.png")
plt.close()





# Repeat for VPol
antenna_pattern = AntennaPattern("RNOG_vpol_v3_5inch_center_n1.74") #"RNOG_vpol_4inch_center_n1.73")
antenna_pattern_analytic = AntennaPatternAnalytic("analytic_VPol")

phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
theta_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = antenna_response_analytic[0]
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = antenna_response_analytic[1]

fig, ax = plt.subplots(n_theta, n_phi, figsize=(4*n_theta, 3*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_VPol)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_VPol)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]), ':', color="C0", label='V_theta (analytic)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]), ':', color="C1", label='V_phi (analytic)')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_VPol.png")
plt.close()


fig, ax = plt.subplots(n_theta, n_phi, figsize=(3*n_theta, 2*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_VPol) phase')
        #ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_quadslot) imag')
        #ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_quadslot) real')
        #ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_quadslot) imag')
        ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_analytic_array[i_theta, i_phi, :, 0]), ':', color="C0", label='V_theta (analytic) phase')
        # ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_analytic_array[i_theta, i_phi, :, 1]), ':', color="C1", label='V_phi (analytic)')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_VPol_complex.png")
plt.close()



# full circular plots
phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg]
theta_array = np.linspace(0, 180 * units.deg, 180)
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = abs(antenna_response_analytic[0])
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = abs(antenna_response_analytic[1])



fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).max()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).max()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"RNOG_VPol")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"RNOG_VPol")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Max. V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Max. V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_VPol_circular_max.png")
plt.close()



fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_array = np.zeros(len(theta_array))
    V_phi_max_array = np.zeros(len(theta_array))
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).sum()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).sum()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"RNOG_VPol")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"RNOG_VPol")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Sum V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Sum V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_VPol_circular_sum.png")
plt.close()




# Repeat for HPol

antenna_pattern = AntennaPattern("RNOG_hpol_v4_8inch_center_n1.74") #"RNOG_quadslot_v3_air_rescaled_to_n1.74")
antenna_pattern_analytic = AntennaPatternAnalytic("analytic_HPol")

phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
theta_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg, 115 * units.deg, 180 * units.deg]
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = antenna_response_analytic[0]
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = antenna_response_analytic[1]

fig, ax = plt.subplots(n_theta, n_phi, figsize=(4*n_theta, 3*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_quadslot)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_quadslot)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]), ':', color="C0", label='V_theta (analytic)')
        ax[i_theta, i_phi].plot(frequencies, abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]), ':', color="C1", label='V_phi (analytic)')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_HPol.png")
plt.close()

fig, ax = plt.subplots(n_theta, n_phi, figsize=(3*n_theta, 2*n_theta+1), sharex=True, sharey=True)
for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):
        #ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_VPol) real')
        #ax[i_theta, i_phi].plot(frequencies, np.imag(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]), '--', color="C0", label='V_theta (RNOG_VPol) imag')
        ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_quadslot) phase')
        # plt.figure()
        # a = np.diff(np.angle(antenna_response_NuRadio_array[i_theta, i_phi, :, 1])) / frequencies[1]
        # plt.hist(a[a<0], bins=100)
        # print(np.median(a[a<0]))
        # plt.show()
        # quit()
        #ax[i_theta, i_phi].plot(frequencies, np.imag(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]), '--', color="C1", label='V_phi (RNOG_VPol) imag')
        # ax[i_theta, i_phi].plot(frequencies, np.imag(antenna_response_analytic_array[i_theta, i_phi, :, 0]), ':', color="C0", label='V_theta (analytic)')
        ax[i_theta, i_phi].plot(frequencies, np.angle(antenna_response_analytic_array[i_theta, i_phi, :, 1]), ':', color="C1", label='V_phi (analytic)')
        ax[i_theta, i_phi].set_title(f"Theta={theta/units.deg:.0f} deg, Phi={phi/units.deg:.0f} deg")
        if i_theta == n_theta-1:
            ax[i_theta, i_phi].set_xlabel("Frequency [GHz]")
        if i_phi == 0:
            ax[i_theta, i_phi].set_ylabel("Antenna response [a.u.]")
        if i_theta == 0 and i_phi == n_phi-1:
            ax[i_theta, i_phi].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_HPol_complex.png")
plt.close()



# full circular plots
phi_array = [0 * units.deg, 22.5 * units.deg, 45 * units.deg, 90 * units.deg]
theta_array = np.linspace(0, 180 * units.deg, 180)
n_theta = len(theta_array)
n_phi = len(phi_array)

antenna_response_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_NuRadio_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2], dtype=np.complex_)
antenna_response_analytic_array = np.zeros([len(theta_array), len(phi_array), frequencies.shape[0], 2])

for i_theta, theta in enumerate(theta_array):
    for i_phi, phi in enumerate(phi_array):

        # NuRadioMC antenna pattern:
        antenna_response_NuRadioMC = antenna_pattern._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_NuRadio_array[i_theta, i_phi, :, 0] = antenna_response_NuRadioMC[0]
        antenna_response_NuRadio_array[i_theta, i_phi, :, 1] = antenna_response_NuRadioMC[1]

        # Analytic antenna pattern:
        antenna_response_analytic = antenna_pattern_analytic._get_antenna_response_vectorized_raw(frequencies, theta, phi)
        antenna_response_analytic_array[i_theta, i_phi, :, 0] = abs(antenna_response_analytic[0])
        antenna_response_analytic_array[i_theta, i_phi, :, 1] = abs(antenna_response_analytic[1])


fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).max()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).max()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).max()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"RNOG_quadslot")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"RNOG_quadslot")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Max. V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Max. V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_HPol_circular_max.png")
plt.close()



fig, ax = plt.subplots(2, n_phi, figsize=(10,5), sharex=True, sharey=True)
for i_phi, phi in enumerate(phi_array):
    V_theta_max_NuRadio_array = np.zeros(len(theta_array))
    V_phi_max_NuRadio_array = np.zeros(len(theta_array))
    V_theta_max_analytic_array = np.zeros(len(theta_array))
    V_phi_max_analytic_array = np.zeros(len(theta_array))

    for i_theta, theta in enumerate(theta_array):
        V_theta_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_NuRadio_array[i_theta] = abs(antenna_response_NuRadio_array[i_theta, i_phi, :, 1]).sum()
        V_theta_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 0]).sum()
        V_phi_max_analytic_array[i_theta] = abs(antenna_response_analytic_array[i_theta, i_phi, :, 1]).sum()

    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_NuRadio_array, '--', label=f"RNOG_quadslot")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_NuRadio_array, '--', label=f"RNOG_quadslot")
    ax[0, i_phi].plot(theta_array/units.deg, V_theta_max_analytic_array, ':', label=f"Analytic")
    ax[1, i_phi].plot(theta_array/units.deg, V_phi_max_analytic_array, ':', label=f"Analytic")
    ax[0, i_phi].set_title(f"Phi={phi/units.deg:.0f} deg")

    if i_phi == 0: ax[0, i_phi].set_ylabel("Sum V_theta [a.u.]")
    if i_phi == 0: ax[1, i_phi].set_ylabel("Sum V_phi [a.u.]")
    ax[1, i_phi].set_xlabel("Theta [deg]")
    #ax[1, i_phi].legend()

ax[0, 2].legend()
plt.tight_layout()
plt.savefig("antenna_pattern_HPol_circular_sum.png")
plt.close()