from NuRadioMC.EvtGen.generator import *

filename = 'decay_library.hdf5'
fout = h5py.File(filename, 'w')

#times = np.linspace(1e-3*tau_rest_lifetime, 10*tau_rest_lifetime, 200)
times = np.linspace(-3,10,100)
times = 10**times * tau_rest_lifetime
energies = np.linspace(15, 20, 100)
energies = 10**energies * units.eV

# "Clever" way of looping. However, we can't see the progress with this.
#tables = [ [ get_decay_time_losses(energy, 1000*units.km, average=True, compare=True, user_time=time)
#            for time in times ] for energy in energies ]

tables = []
for itime, time in enumerate(times):
    row = []
    for ienergy, energy in enumerate(energies):
        print(itime, ienergy)
        row.append( get_decay_time_losses(energy, 1000*units.km, average=True, compare=False, user_time=time) )
    tables.append(row)

tables = np.array(tables)

fout['decay_times'] = tables[:,:,0]
fout['decay_energies'] = tables[:,:,1]
fout['rest_times'] = times
fout['initial_energies'] = energies

fout.close()
