import numpy as np
import matplotlib.pyplot as plt
from NuRadioMC.utilities.medium import birefringence_index

depth = -2500
a = birefringence_index(depth)
print(a)
  
n = a.index()
print(n)

n_all = a.index_all()
print(n_all)

xnew = np.arange(depth, 0, 1)

plt.plot(xnew, n_all[:,0], label = 'n1 - model')
plt.plot(xnew, n_all[:,1], label = 'n2 - model')
plt.plot(xnew, n_all[:,2], label = 'n3 - model')

plt.title('Refractive index model for birefringence')
plt.xlabel('depth [m]')
plt.ylabel('Refractive index')
plt.legend()
#plt.xlim(100, 200)
plt.ylim(1.77, 1.782)
plt.grid()
plt.show()


