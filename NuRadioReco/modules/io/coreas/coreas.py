import numpy as np

from radiotools import helper as hp
from radiotools import coordinatesystems

from NuRadioReco.utilities import units
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.radio_shower

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp

import logging
logger = logging.getLogger('coreas')


conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter


def get_angles(corsika):
    """
    Converting angles in corsika coordinates to local coordinates
    """
    zenith = np.deg2rad(corsika['inputs'].attrs["THETAP"][0])
    azimuth = hp.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))

    Bx, Bz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(Bz, Bx)

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5 * units.micro * units.tesla

    # in local coordinates north is + 90 deg
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(np.pi * 0.5 + B_inclination, 0 + np.pi * 0.5)

    return zenith, azimuth, magnetic_field_vector


def calculate_simulation_weights(positions):
    """Calculate weights according to the area that one simulated position represents.
    Weights are therefore given in units of area.
    Note: The volume of a 2d convex hull is the area."""

    import scipy.spatial as spatial
    weights = np.zeros_like(positions[:, 0])
    vor = spatial.Voronoi(positions[:, :2])  # algorithm will find no solution if flat simulation is given in 3d.
    for p in range(0, weights.shape[0]):
        weights[p] = spatial.ConvexHull(vor.vertices[vor.regions[vor.point_region[p]]]).volume
    return weights


def make_sim_station(station_id, corsika, observer, channel_ids, weight=None):
    """
    creates an NuRadioReco sim station from the observer object of the coreas hdf5 file

    Parameters
    ----------
    station_id : station id
        the id of the station to create
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    observer : hdf5 observer object
    channel_ids :
    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        ARIANNA simulated station object
    """
    # loop over all coreas stations, rotate to ARIANNA CS and save to simulation branch
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)

    position = observer.attrs['position']

    data = np.copy(observer)
    data[:, 1], data[:, 2] = -observer[:, 2], observer[:, 1]

    # convert to SI units
    data[:, 0] *= units.second
    data[:, 1] *= conversion_fieldstrength_cgs_to_SI
    data[:, 2] *= conversion_fieldstrength_cgs_to_SI
    data[:, 3] *= conversion_fieldstrength_cgs_to_SI

    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)
    efield = cs.transform_from_magnetic_to_geographic(data[:, 1:].T)
    efield = cs.transform_from_ground_to_onsky(efield)

    # prepend trace with zeros to not have the pulse directly at the start
    n_samples_prepend = efield.shape[1]
    efield2 = np.zeros((3, n_samples_prepend + efield.shape[1]))
    efield2[0] = np.append(np.zeros(n_samples_prepend), efield[0])
    efield2[1] = np.append(np.zeros(n_samples_prepend), efield[1])
    efield2[2] = np.append(np.zeros(n_samples_prepend), efield[2])

    sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids)
    electric_field.set_trace(efield2, sampling_rate)
    electric_field.set_parameter(efp.ray_path_type, 'direct')
    electric_field.set_parameter(efp.zenith, zenith)
    electric_field.set_parameter(efp.azimuth, azimuth)
    sim_station.add_electric_field(electric_field)
    sim_station.set_parameter(stnp.azimuth, azimuth)
    sim_station.set_parameter(stnp.zenith, zenith)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_station.set_parameter(stnp.cr_energy, energy)
    sim_station.set_magnetic_field_vector(magnetic_field_vector)
    sim_station.set_parameter(stnp.cr_xmax, corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
    try:
        sim_station.set_parameter(stnp.cr_energy_em, corsika["highlevel"].attrs["Eem"])
    except:
        logger.warning("No high-level quantities in HDF5 file, not setting EM energy")
    sim_station.set_is_cosmic_ray()
    sim_station.set_simulation_weight(weight)
    return sim_station


def make_sim_shower(corsika, observer=None, detector=None, station_id=None):
    sim_shower = NuRadioReco.framework.radio_shower.RadioShower()

    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    sim_shower.set_parameter(shp.zenith, zenith)
    sim_shower.set_parameter(shp.azimuth, azimuth)
    sim_shower.set_parameter(shp.magnetic_field_vector, magnetic_field_vector)

    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_shower.set_parameter(shp.energy, energy)
    # We can only set the shower core relative to the station if we know its position
    if observer is not None and detector is not None and station_id is not None:
        station_position = detector.get_absolute_position(station_id)
        position = observer.attrs['position']
        cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)
        observer_position = np.zeros(3)
        observer_position[0], observer_position[1], observer_position[2] = -position[1] * units.cm, position[0] * units.cm, position[2] * units.cm
        observer_position = cs.transform_from_magnetic_to_geographic(observer_position)
        core_position = (-observer_position + station_position)
        core_position[2] = 0
        sim_shower.set_parameter(shp.core, core_position)

    sim_shower.set_parameter(shp.shower_maximum, corsika['CoREAS'].attrs['DepthOfShowerMaximum'] * units.g / units.cm2)
    sim_shower.set_parameter(shp.refractive_index_at_ground, corsika['CoREAS'].attrs["GroundLevelRefractiveIndex"])
    sim_shower.set_parameter(shp.magnetic_field_rotation,
                             corsika['CoREAS'].attrs["RotationAngleForMagfieldDeclination"] * units.degree)
    sim_shower.set_parameter(shp.distance_shower_maximum_geometric,
                             corsika['CoREAS'].attrs["DistanceOfShowerMaximum"] * units.cm)

    sim_shower.set_parameter(shp.observation_level, corsika["inputs"].attrs["OBSLEV"] * units.cm)
    sim_shower.set_parameter(shp.primary_particle, corsika["inputs"].attrs["PRMPAR"])
    if 'ATMOD' in corsika['inputs'].attrs.keys():
        sim_shower.set_parameter(shp.atmospheric_model, corsika["inputs"].attrs["ATMOD"])

    try:
        sim_shower.set_parameter(shp.electromagnetic_energy, corsika["highlevel"].attrs["Eem"] * units.eV)
    except:
        logger.warning("No high-level quantities in HDF5 file, not setting EM energy")

    return sim_shower
