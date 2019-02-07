from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt

energies = np.linspace(15.1, 20, 40)
energies = 10**energies
print(energies)
lengths = []
lengths_nolosses = []
for energy in energies:
    print(energy)
    times = get_decay_time_losses(energy, 1000*units.km, average=True, compare=True)
    lengths.append(times[0]*cspeed)
    lengths_nolosses.append(times[1]*cspeed)

lengths = np.array(lengths)
lengths_nolosses = np.array(lengths_nolosses)

plt.loglog(energies, lengths/units.km, label='With photonuclear losses')
plt.loglog(energies, lengths_nolosses/units.km, label='Without losses')
plt.xlabel('Energy [eV]')
plt.ylabel('Tau track length [km]')
plt.legend()
plt.show()
