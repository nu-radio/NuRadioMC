from NuRadioMC.EvtGen.generator import *
import matplotlib.pyplot as plt

tau_energy = 1.e19 * units.eV
branch = 'tau_pi'

def f_1(y,r):
    return -(2*y+1+r)*(1-2*r)/(1-r)**2/(1+2*r)
def f_0(y,r):
    return 1/(1-r)
def f(y,r):
    if ( y < 0 or y > 1-r**2 ):
        return 0
    else:
        return f_1(y,r)+f_0(y,r)

def f_pi(y,r):
    if ( y < 0 or y > 1-r**2 ):
        return 0
    else:
        return f_0(y,r)-(2*y-1+r)/(1-r**2)**2

r = rho770_mass/tau_mass
print(r)
x = np.linspace(0,1,100)
distr = [f(x0,r) for x0 in x]
plt.plot(x, distr)
plt.show()

energies = []
for i in range(10000):
    products = products_from_tau_decay(tau_energy, branch)
    energies.append(products[16])

plt.hist(energies)
plt.show()
