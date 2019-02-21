import json
import numpy as np
import matplotlib.pyplot as plt

with open('interp_1e-3.json', 'r') as f:
    interp = json.load(f)

colours = ['red', 'black', 'blue', 'orange', 'cadetblue']
colour_dict = {}

for iR, R in enumerate(interp):

    colour_dict[R] = colours[iR]

for R in interp:

    thetas = interp[R].keys().sort()
    thetas = [ float(theta) for theta in interp[R].keys() ]
    thetas = np.sort(np.array(thetas))
    thetas_plot = np.array(thetas)

    interp_factors = [ interp[R][theta] for theta in interp[R].keys() ]
    interp_factors = np.array(interp_factors)

    plt.subplot(2,1,1)
    plt.plot(thetas_plot, interp_factors[:,0], label=R+' m, interp 1',
            color = colour_dict[R], linestyle = '-', marker='o')
    plt.subplot(2,1,2)
    plt.plot(thetas_plot, interp_factors[:,1], label=R+' m, interp 2',
            color = colour_dict[R], linestyle = '--', marker='o')

plt.subplot(2,1,1)
plt.ylabel('Interp factor 1')
plt.legend()
plt.subplot(2,1,2)
plt.ylabel('Interp factor 2')
plt.xlabel('Angle [deg]')
plt.savefig('interpolation_plot.png', format='png')
plt.show()
