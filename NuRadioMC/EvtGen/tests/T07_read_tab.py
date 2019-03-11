from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt

def get_p(p):

    return -tau_rest_lifetime * np.log(1-p)

fin = load_input_hdf5('decay_library.hdf5')

table = create_interp('decay_library.hdf5')

time = tau_rest_lifetime
times = np.linspace(0, 10*tau_rest_lifetime, 100)
energies = np.linspace(15,20,100)
energies = 10**energies

ps = [0.1, 0.5, 1-1/np.e, 0.9]
times = get_p(np.array(ps))

for time in times:
    plt.loglog(energies, table[0](time, np.log10(energies)))
plt.show()

for time in times:
    plt.loglog(energies, table[1](time, np.log10(energies)))
plt.show()
