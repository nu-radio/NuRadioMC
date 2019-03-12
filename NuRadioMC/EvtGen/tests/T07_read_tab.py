from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt

def get_p(p):

    return -tau_rest_lifetime * np.log(1-p)

table = create_interp('../decay_library.hdf5')

time = tau_rest_lifetime
times = np.linspace(0, 10*tau_rest_lifetime, 50)
energies = np.linspace(15,20,50)
energies = 10**energies

ps = [0.1, 0.5, 1-1/np.e, 0.9]
#ps = np.linspace(0.5,0.6,7)
times = get_p(np.array(ps))

exact = True
if exact:
    decay_times = []
    decay_energies = []
    for time in times:
        row_times = []
        row_energies = []
        for energy in energies:
            res = get_decay_time_losses(energy, 1000*units.km, average=False, compare=False, user_time=time)
            row_times.append(res[0])
            row_energies.append(res[1])
        decay_times.append(row_times)
        decay_energies.append(row_energies)

for itime, time in enumerate(times):
    plt.loglog(energies, table[0](time, energies)[0])
    if exact:
        plt.loglog(energies, decay_times[itime], linestyle='--')
plt.show()

for itime, time in enumerate(times):
    plt.loglog(energies, table[1](time, energies)[0])
    if exact:
        plt.loglog(energies, decay_energies[itime], linestyle='--')
plt.show()
