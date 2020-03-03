import numpy as np
from NuRadioReco.utilities import units
from matplotlib import pyplot as plt
from radiotools import helper as hp
from NuRadioMC.utilities import cross_sections as cs
from NuRadioMC.utilities import earth_attenuation
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from NuRadioMC.simulation.simulation import pretty_time_delta
from NuRadioMC.EvtGen.generator import write_events_to_hdf5
import pickle
import os
import time
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

R_earth = 6357390 * units.m
earth = earth_attenuation.PREM()

# define cylinder by two points and the radius
h_cylinder = 2.7 * units.km
pt1 = np.array([0, 0, R_earth])
pt2 = np.array([0, 0, R_earth - h_cylinder])
r_cylinder = 5 * units.km

# calculate maximum width of projected area
theta_max = np.arctan(h_cylinder / 2 / r_cylinder)
d = 2 * r_cylinder * np.cos(theta_max) + h_cylinder * np.sin(theta_max)  # width of area

print(f"cylinder r = {r_cylinder/units.km:.1f}km, h = {h_cylinder/units.km:.1f}km -> dmax = {d/units.km:.1f}km")

phimin = 0
phimax = 360 * units.deg
thetamin = 0
thetamax = 180 * units.deg


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
    res = quad(get_density, 0, t, args=(v, X), limit=50)
    return res[0]


def slant_depth_num(t, v, X, step=50 * units.m):
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
    tt = np.linspace(0, t, t / step)
    rr = np.linalg.norm(X + np.outer(tt, v), axis=1)
    res = np.trapz(earth.density(rr), tt)
    return res


def obj_dist_to_surface(t, v, X):
    return get_R(t, v, X) - R_earth


def obj(t, v, X, Lint):
    """
    objective function to determine at which travel distance we reached the interaction point
    """
    return slant_depth_num(t, v, X) - Lint


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


# precalculate the maximum slant depth to the detector
if(not os.path.exists("buffer_Llimit.pkl")):
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
                sdepth_tmp[j] = slant_depth_num(t, -v, X)
    #         print(i, zens[i] / units.deg, X, sdepth_tmp[j])
    #     exit_point = X + (-v * t)
        Lint_max[i] = np.max(sdepth_tmp)
        Lint_min[i] = np.min(sdepth_tmp)
    pickle.dump([zens, Lint_max, Lint_min], open("buffer_Llimit.pkl", "wb"), protocol=4)
else:
    zens, Lint_max, Lint_min = pickle.load(open("buffer_Llimit.pkl", "rb"))

get_Lmax = interp1d(zens, Lint_max, kind='next')
get_Lmin = interp1d(zens, Lint_min, kind='previous')

if 0:
    fig, a = plt.subplots(1, 1)
    ztmp = np.linspace(0, 180 * units.deg, 10000)
    a.plot(ztmp / units.deg, get_Lmax(ztmp) / units.g * units.cm ** 2, 'C0-')
    a.plot(zens / units.deg, Lint_max / units.g * units.cm ** 2, 'oC0', label="max possible Lint")
    a.plot(ztmp / units.deg, get_Lmin(ztmp) / units.g * units.cm ** 2, 'C1-')
    a.plot(zens / units.deg, Lint_min / units.g * units.cm ** 2, 'dC1', label="min possible Lint")
    a.hlines(cs.get_interaction_length(.1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180, label="0.1 EeV", colors='C2')
    a.hlines(cs.get_interaction_length(1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="1 EeV", colors='C3')
    a.hlines(cs.get_interaction_length(10 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="10 EeV", colors='C4')
    a.set_xlabel("zenith angle [deg]")
    a.set_ylabel("slant depth [g/cm^2]")
    a.semilogy(True)
    a.set_ylim(5e5)
    a.legend()
    fig.tight_layout()
    fig.savefig("Lvszen.png")
    plt.show()

n_events = int(1e7)
failed = 0
Enu = np.ones(n_events) * 1 * units.EeV
az = np.random.uniform(phimin, phimax, n_events)
zen = np.arccos(np.random.uniform(-1, 1, n_events))
# generate random positions on an area perpendicular do neutrino direction
ax, ay = np.random.uniform(-0.5 * d, 0.5 * d, (2, n_events))
# az = np.ones(n_events) * (R_earth - .5 * h_cylinder)  # move plane to the center of the cylinder

# calculate grammage (g/cm^2) after which neutrino interacted
Lint = np.random.exponential(cs.get_interaction_length(Enu, 1, 12, "total"), n_events)

mask = (Lint < get_Lmax(zen)) & (Lint > get_Lmin(zen))
print(f"{np.sum(mask)}/{n_events} = {np.sum(mask)/n_events:.2g} can potentially interact in simulation volume")

# calculate position where neutrino interacts
data_sets = {'xx': [],
            'yy': [],
            'zz': [],
            'azimuths': [],
            'zeniths': []}
# calculate rotation matrix to transform position on area to 3D
mask_int = np.zeros_like(mask, dtype=np.bool)
t0 = time.perf_counter()
for j, i in enumerate(np.arange(n_events, dtype=np.int)[mask]):
    if(j % 1000 == 0):
        eta = (time.perf_counter() - t0) * (n_events - i) / i
        logger.info(f"{i}/{n_events} interacting = {np.sum(mask_int)}, failed = {failed}, eta = {pretty_time_delta(eta)}")
#     print(f"calculating interaction point of event {i}")
    R = hp.get_rotation(np.array([0, 0, 1]), hp.spherical_to_cartesian(zen[i], az[i]))
    v = -hp.spherical_to_cartesian(zen[i], az[i])  # neutrino direction
    X = np.matmul(R, np.array([ax[i], ay[i], 0])) + np.array([0, 0, R_earth - 0.5 * h_cylinder])

    # check if trajectory passes through cylinder
    if(not points_in_cylinder(pt1, pt2, r_cylinder, X)):
        # if point is not in cylinder, check if trajectory passes through
        x = (X[0] ** 2 + X[1] ** 2) ** 0.5
        if(x > r_cylinder):
            continue
        # we reduce it to a 2D problem in (x**2+y**2)0.5, z
        alpha = min(zen[i], 180 * units.deg - zen[i])
        z = X[2] - R_earth
        # case 1: z > 0
        if(z > 0):
            d = np.tan(alpha) * z + x
            if(d > r_cylinder):
                continue
        elif(z < h_cylinder):
            d = np.tan(alpha) * (-z - h_cylinder) + x
            if(d > r_cylinder):
                continue
    # calculate point where neutrino enters Earth
    try:
        if(X[2] > R_earth):
            s = (X[2] - R_earth) / np.cos(min(zen[i], 180 * units.deg - zen[i]))
            if(zen[i] > 90 * units.deg):
                t = brentq(obj_dist_to_surface, 0.8 * s - 100, 1.2 * s + 100, args=(-v, X))
            else:
                t = brentq(obj_dist_to_surface, 0.8 * s - 100, 1.2 * s + 100, args=(v, X))
        else:
            t = brentq(obj_dist_to_surface, 0, 2 * R_earth, args=(-v, X))
    except:
        logger.warning("failed to converge, skipping event")
        failed += 1
        continue
    exit_point = X + (-v * t)
#     logger.debug(f"zen = {zen[i]/units.deg:.0f}deg, trajectory enters Earth at {exit_point[0]:.1f}, {exit_point[0]:.1f}, {exit_point[0]:.1f}. Dist to core = {np.linalg.norm(exit_point)/R_earth:.5f}, dist to (0,0,R) = {np.linalg.norm(exit_point - np.array([0,0,R_earth]))/R_earth:.4f}")

#     # check if event interacts at all
    if(Lint[i] > slant_depth_num(2 * R_earth, v, X)):
#         logger.debug("neutrino does not interact in Earth, skipping to next event")
        continue

    try:
        # calculate interaction point by inegrating the density of Earth along the neutrino path until we wind the interaction length
        t = brentq(obj, 0, 2 * R_earth, args=(v, exit_point, Lint[i]), maxiter=500)
    except:
        logger.warning("failed to converge, skipping event")
        failed += 1
        continue
    Xint = X + v * t  # calculate interaction point

    is_in_cylinder = points_in_cylinder(pt1, pt2, r_cylinder, Xint)
    mask_int[i] = is_in_cylinder
    if(is_in_cylinder):
        logger.debug(f"event {i}, interaction point ({Xint[0]:.1f}, {Xint[1]:.1f}, {Xint[2]-R_earth:.1f}), in cylinder {is_in_cylinder}")
        data_sets['xx'].append(Xint[0])
        data_sets['yy'].append(Xint[1])
        data_sets['zz'].append(Xint[2] - R_earth)
        data_sets['zeniths'].append(zen[i])
        data_sets['azimuths'].append(az[i])

data_sets['event_ids'] = range(np.sum(mask_int))
data_sets['inelasticity'] = np.ones(np.sum(mask_int))
data_sets['flavors'] = np.ones(np.sum(mask_int))
data_sets['interaction_type'] = np.ones(np.sum(mask_int))
attributes = {'n_events': n_events}

logger.info(f"{np.sum(mask_int)} event interacted in simulation volume")

write_events_to_hdf5("test.hdf5", data_sets, attributes)
