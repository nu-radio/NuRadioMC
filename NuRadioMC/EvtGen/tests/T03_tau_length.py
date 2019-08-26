from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt
import json
plt.tight_layout()

"""
This file calculates some percentiles for the tau length and energy distributions
given by the CSDA, plots them, and saves them to a json file.

Just run:
    python T03_tau_length.py
"""

def get_p(p):

    return -tau_rest_lifetime * np.log(1-p)

def make_dict(keylist, valuelist):

    outdict = {}
    for key, value in zip(keylist, valuelist):
        outdict[key] = value

    return outdict


#user_times = np.array([])*units.fs
ps = [0.1, 0.5, 1-1/np.e, 0.9]
user_times = get_p(np.array(ps))

colourlist = ['orange', 'blue', 'red', 'black']
colours = make_dict(user_times, colourlist)

linestylelist = ['-','-','-.','-']
linestyles = make_dict(user_times, linestylelist)

labels = make_dict(user_times, [r'10%', 'Median', 'Mean', r'90%'])

energies = np.linspace(15.1, 20, 40)
energies = 10**energies
print(energies)

lengths = {}
lengths_nolosses = {}
tau_energies = {}

for user_time in user_times:
    lengths[user_time] = []
    lengths_nolosses[user_time] = []
    tau_energies[user_time] = []
    for energy in energies:
        print(energy)
        times = get_decay_time_losses(energy, 1000*units.km, average=True, compare=True, user_time=user_time)
        lengths[user_time].append(times[0]*cspeed)
        lengths_nolosses[user_time].append(times[1]*cspeed)
        tau_energies[user_time].append(times[2])

    lengths[user_time] = np.array(lengths[user_time])
    lengths_nolosses[user_time] = np.array(lengths_nolosses[user_time])


plt.loglog(energies, lengths[user_times[2]]/units.km, linestyle='-', color=colours[user_times[2]], label='Mean, with PN losses')
plt.loglog(energies, lengths_nolosses[user_times[2]]/units.km, linestyle='--', color=colours[user_times[2]], label='Mean, without losses')

plt.gcf().subplots_adjust(bottom=0.13)
fontsize=16
plt.tick_params(labelsize=12)
plt.fill_between(energies, lengths[user_times[0]]/units.km, lengths[user_times[-1]]/units.km, facecolor='0.75', interpolate=True, label=r'10% to 90% quantiles')
#plt.loglog([],[], linestyle='', label='P.N. losses - Solid\nNo losses - Dashed')
#plt.loglog([],[], linestyle='', label='No losses - Dashed')
plt.xlabel('Tau energy [eV]', size=fontsize)
plt.ylabel('Tau track length [km]', size=fontsize)
plt.legend(fontsize=12)
plt.savefig('tau_decay_length.png', format='png')
plt.show()

plt.gcf().subplots_adjust(bottom=0.13)
plt.tick_params(labelsize=12)
#for user_time in user_times:
#    plt.loglog(energies, tau_energies[user_time], linestyle='-', color=colours[user_time], label=labels[user_time])
#    if (user_time == user_times[2]):
#        plt.loglog(energies, tau_energies[user_time], linestyle=linestyles[user_time], color=colours[user_time], label='Mean, with PN losses')
plt.loglog(energies, tau_energies[user_times[2]], linestyle='-', color=colours[user_times[2]], label='Mean, with PN losses')
plt.loglog(energies, energies, linestyle='--', color=colours[user_times[2]], label="Mean, without losses")
plt.fill_between(energies, tau_energies[user_times[0]], tau_energies[user_times[-1]], facecolor='0.75', interpolate=True, label=r'10% to 90% quantiles')
#plt.loglog([],[], linestyle='', label='P.N. losses - Solid\nNo losses - Dashed')
#plt.loglog([],[], linestyle='', label='No losses - Dashed')
plt.xlabel('Tau initial energy [eV]', size=fontsize)
plt.ylabel('Tau decay energy [eV]', size=fontsize)
plt.legend(fontsize=12)
plt.savefig('tau_decay_energy.png', format='png')
plt.show()

output = {}
output['energies'] = list(energies)
output['lengths_10'] = list(lengths[user_times[0]])
output['lengths_90'] = list(lengths[user_times[-1]])
output['lengths_mean'] = list(lengths[user_times[2]])
output['tau_energies_10'] = list(tau_energies[user_times[0]])
output['tau_energies_90'] = list(tau_energies[user_times[-1]])
output['tau_energies_mean'] = list(tau_energies[user_times[2]])

with open('continuous_decay.json', 'w+') as data_file:
    json.dump(output, data_file, sort_keys=True, indent=4)
