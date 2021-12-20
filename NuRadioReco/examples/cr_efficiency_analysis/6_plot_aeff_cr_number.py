import numpy as np
import json
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import NuRadioReco.utilities.cr_flux as hcr

filename = 'results/air_shower/dict_air_shower_pb_80_180_e7_z9_d2_4000.json'

with open(filename, 'r') as fp:
    data = json.load(fp)

energy_bins_low = np.array(data['energy_bins_low'])
energy_bins_high = np.array(data['energy_bins_high'])
zenith_bins_low = np.array(data['zenith_bins_low'])
zenith_bins_high = np.array(data['zenith_bins_high'])

trigger_effective_area = np.array(data['trigger_effective_area'])
trigger_effective_area_err = np.array(data['trigger_effective_area_err'])
trigger_weight = np.array(data['triggered_trigger_weight'])

### choose only some energy and zenith bins for plotting
chosen_energy_bins = [0, -1]
chosen_zenith_bins = [0, 8]

trigger_effective_area = trigger_effective_area[chosen_energy_bins[0]:chosen_energy_bins[1],chosen_zenith_bins[0]:chosen_zenith_bins[1]]
trigger_effective_area_err = trigger_effective_area_err[chosen_energy_bins[0]:chosen_energy_bins[1],chosen_zenith_bins[0]:chosen_zenith_bins[1]]
trigger_weight = trigger_weight[chosen_energy_bins[0]:chosen_energy_bins[1],chosen_zenith_bins[0]:chosen_zenith_bins[1]]
energy_bins_low = energy_bins_low[chosen_energy_bins[0]:chosen_energy_bins[1]]
energy_bins_high = energy_bins_high[chosen_energy_bins[0]:chosen_energy_bins[1]]
zenith_bins_low = zenith_bins_low[chosen_zenith_bins[0]:chosen_zenith_bins[1]]
zenith_bins_high = zenith_bins_high[chosen_zenith_bins[0]:chosen_zenith_bins[1]]

steradian = []
weight_det = []
for zenith_start, zenith_stop in zip(zenith_bins_low, zenith_bins_high):
    # weighting due to cos distribution visible for detector without solid angle
    weighting_zenith_interval_det = 0.5*(np.cos(zenith_start) + np.cos(zenith_stop))
    sr = 2 * np.pi * ((1 - np.cos(zenith_stop)) - (1 - np.cos(zenith_start)))

    weight_det.append(weighting_zenith_interval_det)
    steradian.append(sr)

# effective area of det for zenith intervals
aeff_det_zenith = trigger_effective_area * weight_det # sum over all zenith bins to get number of cr for each energy bin
total_aeff_det = np.nansum(aeff_det_zenith, axis=1)
aeff_det_zenith_err = trigger_effective_area_err * np.array(weight_det)
total_aeff_det_err = np.sqrt(np.nansum(aeff_det_zenith_err**2, axis=1))

# effective area of det for zenith intervals * sr
aeff_det_zenith_sr = trigger_effective_area * weight_det * steradian
total_aeff_det_sr = np.nansum(aeff_det_zenith_sr, axis=1)
aeff_det_zenith_sr_err = trigger_effective_area_err * np.array(weight_det) * np.array(steradian)
total_aeff_det_sr_err = np.sqrt(np.nansum(np.array(aeff_det_zenith_sr_err**2), axis=1))

zen = zenith_bins_high - zenith_bins_low

# plot effective area as function of zenith
fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.errorbar(energy_bins_low, total_aeff_det/units.km**2, yerr=total_aeff_det_err/units.km**2, marker='x')
for it in range(len(energy_bins_low)):
    ax2.errorbar(zen, ((aeff_det_zenith[it,:])/units.km**2), yerr=((aeff_det_zenith_err[it,:]))/units.km**2, marker= 'x', label=r'Energy {:.2e}, total = ({:.2f} $\pm$ {:.2f}) $km^2$'.format(energy_bins_low[it], float(total_aeff_det[it]/units.km**2), float(total_aeff_det_err[it]/units.km**2)))
ax1.set_xlabel('Energy [eV]', fontsize=18)
ax1.set_ylabel(r'Effective area $[km^2]$', fontsize=18)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xlabel('Zenith $[^\circ]$', fontsize=18)
ax2.set_ylabel(r'Effective area $[km^2]$', fontsize=18)
ax2.set_yscale('log')
ax2.set_xlim(0, 90)
plt.legend()
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.show()
plt.close()

# plot number of cr as function of zenith
fig2, (ax3, ax4) = plt.subplots(1, 2)
for it in range(len(energy_bins_low)):
    auger_flux_int = hcr.get_flux_per_energy_bin(np.log10(energy_bins_low[it]), np.log10(energy_bins_high[it]))
    num_cr = np.array(aeff_det_zenith_sr[it, :]) * auger_flux_int/units.second * 86400
    num_cr_err = np.sqrt((np.array(aeff_det_zenith_sr_err[it, :]) ** 2 * (auger_flux_int/units.second * 86400) ** 2))

    total_num_cr = np.sum(num_cr, axis=0)  # sum over all zenith bins to get number of cr for each energy bin
    total_num_cr_err = np.sqrt(np.sum(num_cr_err** 2, axis=0))

    ax3.errorbar(energy_bins_low[it], total_num_cr, yerr=total_num_cr_err, marker='x', label='total number of cr')
    ax4.errorbar(zen, num_cr, yerr=num_cr_err, marker='x', label=r'Energy {:.2e}, total = {:.2f} $\pm$ {:.2f}'.format(energy_bins_low[it], float(total_num_cr), float(total_num_cr_err)))

ax3.set_xlabel('Energy [eV]', fontsize=18)
ax3.set_ylabel(r'CR [1/day]', fontsize=18)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax4.set_xlabel('Zenith $[^\circ]$', fontsize=18)
ax4.set_ylabel(r'CR [1/day]', fontsize=18)
ax4.set_yscale('log')
plt.legend()
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(0, 90)
plt.tight_layout()
plt.show()