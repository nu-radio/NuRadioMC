import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy


def pol(x, a0, a1, a2, a3):

    y = a0
    y += a1 * x
    y += a2 * x ** 2
    y += a3 * x ** 3

    return y


with open ('dist_vs_edge_Greenland_1.5sigma.json', 'r') as f:
    dist_dict = json.load(f)

left_bins = np.array(dist_dict['left'])
right_bins = np.array(dist_dict['right'])
dists = np.array(dist_dict['dists'])

log_centres = (np.log10(left_bins) + np.log10(right_bins)) * 0.5
log_dists = np.log10(dists)

results = curve_fit(pol, log_centres, log_dists)[0]
print(f"coefficients of a polynomial with 1.0x cover factor {results}")
results2 = copy.copy(results)
results2[0] += np.log10(1.5)
print(f"coefficients of a polynomial with 1.5x cover factor {results2}")
poly = np.polynomial.polynomial.Polynomial(results2)

slope = (np.log10(900) - np.log10(100)) / 1
intercept = np.log10(150) - slope * 15
print(f"coefficients of a straight line fit {intercept}, {slope}")
xx = np.linspace(14, 20, 1000)
plt.loglog(10 ** log_centres, 10 ** log_dists, "o", label='Maximum distances to vertex')
plt.loglog(10 ** xx, 10 ** pol(xx, *results), label='Fit')
plt.loglog(10 ** xx, 1.5 * 10 ** pol(xx, *results), label='Fit with 50% cover factor')
plt.loglog(10 ** xx, 10 ** poly(xx), '--', label='Fit with 50% cover factor')
plt.loglog(10 ** xx, 10 ** pol(xx, intercept, slope, 0, 0), label='Linear cut')
plt.xlabel(r'Shower energy[eV]')
plt.ylabel(r'Maximum distance [m]')
plt.ylim(1, 1e4)
plt.legend()
plt.savefig('fit_max_distances.pdf', format='pdf')
plt.show()
