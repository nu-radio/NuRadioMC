import numpy as np
import json
import matplotlib.pyplot as plt

filename = 'results/air_shower/dict_air_shower_pb_80_180_e7_z9_d2_4000.json'

with open(filename, 'r') as fp:
    data = json.load(fp)

distance_bins_low = np.array(data['distance_bins_low'])
energy_bins_low = np.array(data['energy_bins_low'])
zenith_bins_low = np.array(data['zenith_bins_low'])
distance_bins_high = np.array(data['distance_bins_high'])
energy_bins_high = np.array(data['energy_bins_high'])
zenith_bins_high = np.array(data['zenith_bins_high'])

energy_center = (energy_bins_low + energy_bins_high) / 2

trigger_efficiency = np.array(data['trigger_efficiency'])
triggered_trigger = np.array(data['triggered_trigger'])
masked_events = np.array(data['trigger_masked_events'])


### choose only some bins
trigger_efficiency = np.nan_to_num(trigger_efficiency[:,0:8,:])
zenith_bins_low = zenith_bins_low[0:8]
zenith_bins_high = zenith_bins_high[0:8]
triggered_trigger = np.nan_to_num(triggered_trigger[1:3,0:8,:])
masked_events = np.nan_to_num(masked_events[1:3,0:8,:])

triggered_trigger_ez = np.sum(triggered_trigger, axis=2)
masked_events_ez = np.sum(masked_events, axis=2)
trigger_efficiency_ez = triggered_trigger_ez / masked_events_ez

plt.plot(energy_center[:5], trigger_efficiency[:5, 0, 0], marker='x')
plt.hist(energy_center[:5], bins=np.logspace(15, 19, 9), weights=trigger_efficiency[:5, 0, 0], edgecolor='k', color='lightgrey')
plt.xscale('log')
plt.xlim(1e15, 1e20)
plt.xlabel('Cosmic ray energy [eV]', fontsize=18)
plt.ylabel('Trigger efficiency', fontsize=18)
plt.show()
plt.close()


zen = (zenith_bins_low + zenith_bins_high) /2
energy = energy_bins_low

int = np.linspace(0, 1, len(distance_bins_low)+1)
for i_energy in range(len(energy)):
    plt.plot(zen, trigger_efficiency_ez[i_energy, :], marker='x', label=r'Energy {}, all distances'.format(energy[i_energy]), color='k', linestyle='dashed' )
    for i_dist in range(len(distance_bins_low)):
        if i_dist == len(distance_bins_low) - 1:
            label = r'd < {} m'.format(distance_bins_high[i_dist])
            plt.plot(zen, trigger_efficiency[i_energy, :, i_dist], marker='x', label=label, color='k')
        else:
            label = r'd < {} m'.format(distance_bins_high[i_dist])
            plt.plot(zen, trigger_efficiency[i_energy,:,i_dist], marker='x', label=label)
    plt.xlabel('Zenith $[^\circ]$', fontsize=18)
    plt.ylabel(r'Trigger efficiency', fontsize=18)
    plt.title('CR energy {:.0e} $-$ {:.0e}'.format(energy_bins_low[i_energy], energy_bins_high[i_energy]))
    plt.xlim(0,90)
    plt.ylim(-0.01,1.01)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.show()
