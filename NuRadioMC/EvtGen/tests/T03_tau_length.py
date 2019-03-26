from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt

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
colours = {}
for user_time, colour in zip(user_times, colourlist):
    colours[user_time] = colour

colours = make_dict(user_times, colourlist)

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

    plt.loglog(energies, lengths[user_time]/units.km, linestyle='-', color=colours[user_time], label=labels[user_time])
    plt.loglog(energies, lengths_nolosses[user_time]/units.km, linestyle='--', color=colours[user_time])
plt.loglog([],[], linestyle='', label='Photonuclear losses - Solid\nNo losses - Dashed')
plt.xlabel('Tau energy [eV]')
plt.ylabel('Tau track length [km]')
plt.legend()
plt.savefig('tau_decay_length.png', format='png')
plt.show()

for user_time in user_times:
    plt.loglog(energies, tau_energies[user_time], linestyle='-', color=colours[user_time], label=labels[user_time])
plt.loglog(energies, energies, linestyle='--', color='cadetblue')
plt.loglog([],[], linestyle='', label='Photonuclear losses - Solid\nNo losses - Dashed')
plt.xlabel('Tau initial energy [eV]')
plt.ylabel('Tau decay energy [eV]')
plt.legend()
plt.savefig('tau_decay_energy.png', format='png')
plt.show()
