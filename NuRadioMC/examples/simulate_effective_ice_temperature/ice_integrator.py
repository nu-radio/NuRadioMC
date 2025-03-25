from NuRadioMC.utilities import attenuation, medium
from NuRadioReco.utilities import units

import os
import sys
import h5py
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

try:
    from AntPosCal.ice_model import icemodel
except ImportError:
    print("Import of \"AntPosCal.ice_model import icemodel\" failed. "
          "Please install this repo https://github.com/RNO-G/antenna-positioning")
    sys.exit()

try:
    import radiopropa
except ImportError:
    # You might need to install the branch ice_model/exponential_polynomial
    print("Import of \"radiopropa\" failed. "
          "Please install this repo https://github.com/nu-radio/RadioPropa")

# See PR https://github.com/nu-radio/NuRadioMC/pull/834 for some more information.
# this value comes from the GRIP temperature data sheet NuRadioMC/utilities/data/griptemp.txt
z_min = -3027.6

def get_angles_rhos(file):
    """ Getting a simulated ray-traced path from a hdf5 file

    Parameters
    ----------
    file : str
        path to the hdf5 file

    Returns
    -------
    theta0: float
        starting angle of the ray in degrees
    x: np.array
        x-coordinates of the ray
    z: np.array
        z-coordinates of the ray
    distance: np.array
        distance along the ray (in meters)
    power: np.array
        power fraction along the ray (1 if no reflection, <1 if reflection).
        does not include the attenuation factor
    have_reflection: bool
        True if the ray has a reflection at the surface, False if it has not hit a reflection layer
    """
    f = h5py.File(file)
    d = f['Trajectory3D']

    unique_sn, counts = np.unique(d["SN"], return_counts=True)
    if len(unique_sn) > 1:
        assert counts[0] > counts[1], "Something went wrong, thre reflected ray should be longer than throughpassing one"

    have_reflection = len(unique_sn) > 1
    # select only the reflected ray, not the throughpassing one (through the surface layer)
    d = d[d["SN"] == d["SN"][0]]

    # ray path
    x = d['X']
    z = d['Z']

    # starting angle from first step
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    theta = abs(np.arctan(dx / dz))

    distance = d['D']
    power = (d['Ax']**2 + d['Ay']**2 + d['Az']**2)

    return theta, x, z, distance, power, have_reflection


def get_trace_file(theta, z_antenna, freq=403 * 1e6, in_file=False, filename='__output_tempfile.h5'):
    """
    Perfrom ray tracing from an antenna at z_antenna in the direction defined by theta in the Z-X plane.
    The ray tracing is stopped when the ray reaches the surface (actually a layer 10m above the surface)
    or the maximum depth of the ice model.
    """

    # use fifth order polynomial ice model here, I checked that index of refraction is 1. at 0+epsilon and 1.27 at 0-epsilon
    ice = icemodel.greenland_poly5()
    iceModelScalar = ice.get_ice_model_radiopropa().get_scalar_field()

    # simulation setup
    sim = radiopropa.ModuleList()
    sim.add(radiopropa.PropagationCK(iceModelScalar, 1E-8, .001, 1.))

    # add a discontinuity
    firnLayer = radiopropa.Discontinuity(
        radiopropa.Plane(
            radiopropa.Vector3d(0, 0, -0.001), radiopropa.Vector3d(0, 0, 1)),
        iceModelScalar.getValue(radiopropa.Vector3d(0, 0, -0.001)), 1.)

    # add a reflective layer at the surface
    reflective = radiopropa.ReflectiveLayer(radiopropa.Plane(radiopropa.Vector3d(0, 0, -0.001), radiopropa.Vector3d(0, 0, 1)),1)
    sim.add(firnLayer)

    # Define the surfaces at which the ray tracing is stopped
    obs2 = radiopropa.Observer()
    obsz2 = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0, 0, z_min), radiopropa.Vector3d(0, 0, 1)))
    obs2.add(obsz2)
    obs2.setDeactivateOnDetection(True)
    sim.add(obs2)

    obs3 = radiopropa.Observer()
    obsz3 = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0, 0, 10), radiopropa.Vector3d(0, 0, 1)))
    obs3.add(obsz3)
    obs3.setDeactivateOnDetection(True)
    sim.add(obs3)

    if in_file:
        # Output
        if os.path.isfile(filename):
            os.remove(filename)

        output = radiopropa.HDF5Output(filename, radiopropa.Output.Trajectory3D)
        output.setLengthScale(radiopropa.meter)
        #output.enable(radiopropa.Output.CurrentAmplitudeColumn)
        output.enable(radiopropa.Output.SerialNumberColumn)
        sim.add(output)

    # Source - Antenna at z_antenna at which the ray tracing is started
    source = radiopropa.Source()
    source.add(radiopropa.SourcePosition(radiopropa.Vector3d(0, 0, z_antenna)))
    source.add(radiopropa.SourceAmplitude(1))
    source.add(radiopropa.SourceFrequency(freq))
    z = np.cos(theta / units.deg * radiopropa.deg)
    x = np.sin(theta / units.deg * radiopropa.deg)
    source.add(radiopropa.SourceDirection(radiopropa.Vector3d(x, 0 , z)))
    sim.setShowProgress(False)

    if in_file:
        sim.run(source, 1)
        return filename
    else:
        ray = source.getCandidate()
        sim.run(ray, True)
        # Do something with the ray ...


def get_trace(theta, z_antenna, freq=403 * 1e6):
    """ Wrapper around get_trace_file """
    file = get_trace_file(theta, z_antenna, in_file=True, freq=freq)
    return get_angles_rhos(file)


def get_ray_depth_profile(zenith, z_antenna):
    """ Return the depth profile of the ray path as simulated with radiopropa """
    # get the track for respective zenith direction
    # (distance vs depth)
    _, radius_raytracing, depth_raytracing, distance_raytracing, reflected_power, _ = get_trace(zenith, z_antenna)

    # Using extrapolation here is not ideal, but we will "correct" for this in the temperature_integral function
    depth_interp = interp1d(distance_raytracing, depth_raytracing, fill_value="extrapolate")
    power_interp = interp1d(distance_raytracing, reflected_power, fill_value="extrapolate")
    radius_interp = interp1d(distance_raytracing, radius_raytracing, fill_value="extrapolate")

    distance = np.linspace(0, 30000, 10000) * units.m

    depth = depth_interp(distance)
    reflection_coef = power_interp(distance)
    radius = radius_interp(distance)

    return distance, radius, depth, reflection_coef


def temperature_integral(zenith, z_antenna, freq=400 * units.MHz, model="GL3"):
    """ Return the integral of the effective temperature along the ray path.

    This integral takes into accout the attenuation of the signal along the ray path.
    The effective temperature is calculated as the sum of the temperature at each
    point along the ray path weighted by the attenuation factor at that point (emissivity).

    Parameters
    ----------
    zenith : float
        Zenith angle of the ray
    z_antenna : float
        Depth of the antenna in meters
    freq : float
        Frequency of the signal
    model : str
        Ice attenuation model to use

    Returns
    -------
    eff_temperature : float
        Effective temperature at the antenna
    """

    distance, _, depth, reflection_coef = get_ray_depth_profile(zenith, z_antenna)
    d_distance = distance[1] - distance[0]

    l_att = attenuation.get_attenuation_length(depth, frequency=freq, model=model)
    meaned_l_att = np.cumsum(l_att) / np.cumsum(np.ones_like(distance))

    att_factor = np.exp(-distance / meaned_l_att)

    assert model.startswith("GL"), "Only the Greenland ice model is supported for now (because the GRIP temperature model is hardcoded.)"
    temp_env = attenuation.get_grip_temperature(np.abs(depth))  # already in kelvin, this function takes the depth as a positive value

    # We assume that we do not receive radiation from deep in the ice,
    # hence, we are setting the att_factor to zero everything below z_min.
    # However, we will add a black body radiation term at the end...
    att_factor[depth < z_min] = 0

    # l_att is the emissivity of the volume element. att_factor is the attenuation factor of the proagation of
    # the signal from the volume element to the antenna. The reflection coefficient is the fraction of the signal
    # being lost at the surface. The temperature is the temperature of the volume element.
    t_eff_at_antenna = np.sum(reflection_coef * att_factor * temp_env / l_att * d_distance)

    # Adding the black body radiation ...
    ground_idx = np.argmin(np.abs(depth - z_min))
    if depth[ground_idx] < z_min:
        ground_idx -= 1  # get the index which is just above z_min

    # print(f"Temperature at the ground: {temp_env[ground_idx]} K, Attenuation factor: {att_factor[ground_idx]}")
    t_eff_at_antenna += temp_env[ground_idx] * att_factor[ground_idx]  * reflection_coef[ground_idx]

    return t_eff_at_antenna


def get_eff_temperature(z_antenna=-100, n_theta=100, plot=False, attenuation_model="GL3", fname=None):
    import time
    t0 = time.time()
    thetas = np.linspace(0, np.pi, n_theta)

    eff_temperatures = []
    for theta in thetas:
        eff_temperatures.append(float(temperature_integral(theta, z_antenna, model=attenuation_model)))

    data = {
        "z_antenna": z_antenna,
        "theta": thetas.tolist(),
        "eff_temperature": eff_temperatures,
        "attenuation_model": attenuation_model
    }

    if fname is None:
        fname = f"eff_temperature_{z_antenna}_ntheta{n_theta}.json"

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)

    print(time.time() - t0)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(thetas), eff_temperatures)
        ax.set_xlabel("theta / deg")
        ax.set_ylabel("temperature / K")
        plt.show()


def plot_ray_paths(z_antenna=-100, n_theta=100):
    thetas = np.linspace(0, np.pi, n_theta)

    import matplotlib as mpl
    cmap = plt.get_cmap('plasma')

    norm = mpl.colors.Normalize(vmin=0, vmax=180)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    for theta in thetas:
        _, x, z, _, _, _ = get_trace(theta, z_antenna)
        ax.plot(x, z, color=sm.to_rgba(np.rad2deg(theta)), alpha=0.8)
    cb = plt.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("theta / deg")

    ax.set_xlabel("x / m")
    ax.set_ylabel("z / m")
    ax.set_xlim(-10, 1000)
    fig.tight_layout()
    plt.savefig("plot.pdf")


def plot_ray_paths_attenuation(z_antenna=-100, n_theta=10, model="GL3"):
    thetas = np.linspace(np.pi / 2, np.pi, n_theta)
    # thetas = [np.pi]
    import matplotlib as mpl
    cmap = plt.get_cmap('plasma')

    norm = mpl.colors.LogNorm(vmin=5e-3, vmax=1)
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()

    for theta in thetas:
        distance, radius, depth, reflection_coef = get_ray_depth_profile(theta, z_antenna)

        d_distance = distance[1] - distance[0]
        l_att = attenuation.get_attenuation_length(depth, frequency=400 * units.MHz, model="GL3")

        meaned_l_att = np.cumsum(l_att) / np.cumsum(np.ones_like(distance))
        att_factor = np.exp(-distance / meaned_l_att)

        assert model.startswith("GL"), "Only the Greenland ice model is supported for now (because the GRIP temperature model is hardcoded.)"
        temp_env = attenuation.get_grip_temperature(np.abs(depth))  # already in kelvin, depth is positivly defined

        mask = depth < z_min

        # We assume that we do not receive radiation from deep in the rock ...
        eff_temp = (reflection_coef * att_factor * temp_env / l_att * d_distance)[~mask]

        # ... but adding a black body radiator for the rock.
        ground_idx = np.argmin(np.abs(depth - z_min))
        if depth[ground_idx] < z_min:
            ground_idx -= 1  # get the index which is just above z_min

        # print(f"Temperature at the ground: {temp_env[ground_idx]} K, Attenuation factor: {att_factor[ground_idx]}")
        eff_temp[-1] += temp_env[ground_idx] * att_factor[ground_idx] * reflection_coef[ground_idx]

        eff_temp = 1 - np.cumsum(eff_temp) / np.sum(eff_temp)

        ax.scatter(radius[~mask], depth[~mask], c=eff_temp, cmap=cmap, norm=norm, alpha=0.8, marker='.')

    cb = plt.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label(r"1 - $\sum T(x, z) / T_{eff}$")

    ax.set_xlabel("x / m")
    ax.set_ylabel("z / m")

    # ax.set_ylim(-4000, 10)
    ax.axhline(z_min, color='k', linestyle='--')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate the effective temperature at the antenna. ')
    parser.add_argument('--z_antenna', type=float, default=-100, help='Depth of the antenna in meters')
    parser.add_argument('--n_theta', type=int, default=100, help='Number of angles to consider (equidistant in theta from 0 to pi)')
    parser.add_argument('--plot', action='store_true', help='Plot the effective temperature as a function of the angle')

    parser.add_argument('--attenuation_model', type=str, default="GL3", help='Specify the attenuation model to use')
    parser.add_argument('--fname', type=str, default=None, help='Filename to save the data to')

    args = parser.parse_args()

    fname = args.fname or f"eff_temperature_{args.z_antenna}m_ntheta{args.n_theta}_{args.attenuation_model}.json"

    get_eff_temperature(args.z_antenna, args.n_theta, args.plot, args.attenuation_model, fname)
    # plot_ray_paths_attenuation(args.z_antenna, args.n_theta)