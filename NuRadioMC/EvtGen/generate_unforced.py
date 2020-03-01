import numpy as np
from NuRadioReco.utilities import units
from matplotlib import pyplot as plt
from radiotools import plthelpers as php
from radiotools import helper as hp
from NuRadioMC.utilities import cross_sections as cs
from NuRadioMC.utilities import earth_attenuation
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy import constants
from shapely.geometry import LineString
from shapely.geometry import Point
R_earth = 6357390 * units.m
earth = earth_attenuation.PREM()

# define cylinder by two points and the radius
pt1 = np.array([0, 0, R_earth])
pt2 = np.array([0, 0, R_earth - 2.7 * units.km])
r_cylinder = 5 * units.km
h_cylinder = 2.7 * units.km
d = 5 * units.km  # width of area

phimin = 0
phimax = 360 * units.deg
thetamin = 0
thetamax = 180 * units.deg


def get_exit_point(zen, az, X):
    angle = 180 * units.deg - zen  # convert zenith angle to nadir
    depth = -X[2]
    v = -hp.spherical_to_cartesian(zen, az)
    # Starting point (x0, z0)
    x0 = 0
    z0 = R_earth - depth
    # Find exit point (x1, z1)
    if angle == 0:
        x1 = 0
        z1 = -R_earth
    else:
        m = -np.cos(angle) / np.sin(angle)
        a = z0 - m * x0
        b = 1 + m ** 2
        if angle < 0:
            x1 = -m * a / b - np.sqrt(m ** 2 * a ** 2 / b ** 2
                                  -(a ** 2 - R_earth ** 2) / b)
        else:
            x1 = -m * a / b + np.sqrt(m ** 2 * a ** 2 / b ** 2
                                  -(a ** 2 - R_earth ** 2) / b)
        z1 = z0 + m * (x1 - x0)


def get_R(t, v, X):
    """"
    calculate distance to center of Earth as a function of travel distance
    
    Parameters
    -----------
    t: 3dim array
        travel distance
    v: 3dim array
        direction
    X: 3dim array
        start point
    """
    return np.linalg.norm(v * t + X)


def get_density(t, v, X):
    """
    calculates density as a function of travel distance
    
    Parameters
    -----------
    t: 3dim array
        travel distance
    v: 3dim array
        direction
    X: 3dim array
        start point
    """
    return  earth.density(get_R(t, v, X))


def slant_depth(t, v, X):
    """
    calculates slant depth (grammage) as a function of travel distance
    
    Parameters
    -----------
    t: 3dim array
        travel distance
    v: 3dim array
        direction
    X: 3dim array
        start point
    """
    res = quad(get_density, 0, t, args=(v, X), limit=200)
    return res[0]


def obj_dist_to_surface(t, v, X):
    return get_R(t, v, X) - R_earth


def obj(t, v, X):
    """
    objective function to determine at which travel distance we reached the interaction point
    """
    return slant_depth(t, v, X) - Lint[i]


# precalculate the maximum slant depth to the detector
zens = np.linspace(0, 180 * units.deg, 100)
Lint_max = np.zeros_like(zens)
Lint_min = np.zeros_like(zens)
Xs = np.array([[0, -0.5 * r_cylinder, -h_cylinder + R_earth],
               [0, -0.5 * r_cylinder, R_earth],
               [0, 0.5 * r_cylinder, -h_cylinder + R_earth],
               [0, 0.5 * r_cylinder, R_earth]]
               )
v_tmps = -hp.spherical_to_cartesian(zens, np.zeros_like(zens))  # neutrino direction
for i in range(len(v_tmps)):
    v = v_tmps[i]
    sdepth_tmp = np.zeros(4)
    for j, X in enumerate(Xs):
        if((X[2] == R_earth) and (zens[i] <= 90 * units.deg)):
            t = 0
            sdepth_tmp[j] = 0
        else:
            t = brentq(obj_dist_to_surface, 100, 2 * R_earth, args=(-v, X))
            sdepth_tmp[j] = slant_depth(t, -v, X)
#         print(i, zens[i] / units.deg, X, sdepth_tmp[j])
#     exit_point = X + (-v * t)
    Lint_max[i] = np.max(sdepth_tmp)
    Lint_min[i] = np.min(sdepth_tmp)

get_Lmax = interp1d(zens, Lint_max)
get_Lmin = interp1d(zens, Lint_min)

fig, a = plt.subplots(1, 1)
a.plot(zens / units.deg, Lint_max / units.g * units.cm ** 2, label="max possible Lint")
a.plot(zens / units.deg, Lint_min / units.g * units.cm ** 2, label="min possible Lint")
a.hlines(cs.get_interaction_length(.1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180, label="0.1 EeV", colors='C2')
a.hlines(cs.get_interaction_length(1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="1 EeV", colors='C3')
a.hlines(cs.get_interaction_length(10 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="10 EeV", colors='C4')
a.semilogy(True)
a.set_ylim(5e5)
a.legend()
fig.tight_layout()
fig.savefig("Lvszen.png")
plt.show()

n_events = int(1e6)
Enu = np.ones(n_events) * 1 * units.EeV
az = np.random.uniform(phimin, phimax, n_events)
zen = np.arccos(np.random.uniform(-1, 1, n_events))
# generate random positions on an area perpendicular do neutrino direction
ax, ay = np.random.uniform(-0.5 * d, 0.5 * d, (2, n_events))
az = np.ones(n_events) * (R_earth - .5 * h_cylinder)  # move plane to the center of the cylinder

# calculate grammage (g/cm^2) after which neutrino interacted
Lint = np.random.exponential(cs.get_interaction_length(Enu, 1, 12, "total"), n_events)

mask = (Lint < get_Lmax(zen)) & (Lint > get_Lmin(zen))
print(f"{np.sum(mask)}/{n_events} = {np.sum(mask)/n_events:.2g} interact in simulation volume")

# calculate position where neutrino interacts
# calculate rotation matrix to transform position on area to 3D
R = np.array([hp.get_rotation(np.array([0, 0, 1]), x) for x in hp.spherical_to_cartesian(zen, az)])
for i in range(n_events):
    v = -hp.spherical_to_cartesian(zen[i], az[i])  # neutrino direction
    X = np.matmul(R[i], np.array([ax[i], ay[i], az[i]]))

    # calculate point where neutrino enters Earth
    t = brentq(obj_dist_to_surface, -2 * max(r_cylinder, h_cylinder), 2 * R_earth, args=(-v, X))
    exit_point = X + (-v * t)

    # check if event interacts at all
    if(Lint[i] > slant_depth(2 * R_earth, v, X)):
        print("neutrino does not interact in Earth, skipping to next event")
        continue

    # calculate interaction point by inegrating the density of Earth along the neutrino path until we wind the interaction length
    t = brentq(obj, 0, R_earth, args=(v, exit_point))
    Xint = X + v * t  # calculate interaction point

    def points_in_cylinder(pt1, pt2, r, q):
        """
        determines if point lies within a cylinder
        
        Parameters
        -----------
        pt1: 3dim array
            lowest point on cylinder axis
        pt2: 3dim array
            highest point on cylinder axis
        r: float
            radius of cylinder
        q: 3dim array
            point under test
        
        Returns True/False
        """
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return len(np.where(np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const)[0]) > 0

    is_in_cylinder = points_in_cylinder(pt1, pt2, r_cylinder, Xint)
    if(not is_in_cylinder):
        print("neutrino does not interact in simulation volume")
    else:
        print("neutrino interaction")

#     a = 1 / 0
# logger.debug("generating vertex positions")
# rr_full = np.random.triangular(full_rmin, full_rmax, full_rmax, n_events)
# phiphi = np.random.uniform(0, 2 * np.pi, n_events)
# data_sets["xx"] = rr_full * np.cos(phiphi)
# data_sets["yy"] = rr_full * np.sin(phiphi)
# data_sets["zz"] = np.random.uniform(full_zmin, full_zmax, n_events)
