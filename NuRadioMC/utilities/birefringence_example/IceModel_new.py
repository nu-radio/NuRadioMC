import numpy as np
import matplotlib.pyplot as plt
from NuRadioMC.utilities.medium import birefringence_index

pos = np.array([2000,-100,-2500])
a = birefringence_index()
  
n = a.get_index_of_refraction(pos)
print(n)

n_all = a.get_index_of_refraction_all(pos)
print(n_all)

znew = np.arange(pos[2], 0, 1)

plt.plot(znew, n_all[:,0], label = 'nx - model')
plt.plot(znew, n_all[:,1], label = 'ny - model')
plt.plot(znew, n_all[:,2], label = 'nz - model')

plt.title('Refractive index model for birefringence')
plt.xlabel('depth [m]')
plt.ylabel('Refractive index')
#plt.legend()
#plt.xlim(100, 200)
#plt.ylim(1.77, 1.782)
#plt.grid()
plt.show()


