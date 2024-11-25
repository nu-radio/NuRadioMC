import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from radiotools import helper as hp
from radiotools import coordinatesystems
from NuRadioReco.utilities import units
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.event
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
import cr_pulse_interpolator.interpolation_fourier
import cr_pulse_interpolator.signal_interpolation_fourier
import logging
import copy
import h5py

logger = logging.getLogger('NuRadioReco.coreas')
logger.setLevel(logging.INFO)

warning_printed_coreas_py = False

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter


def get_angles(corsika, declination):
    """
    Converting angles in corsika coordinates to local coordinates. 
    
    Corsika positive x-axis points to the magnetic north, NRR coordinates positive x-axis points to the geographic east.
    Corsika positive y-axis points to the west, NRR coordinates positive y-axis points to the geographic north.
    Corsika z-axis points upwards, NuRadio z-axis points upwards.

    Corsika's zenith angle of a particle trajectory is defined between the particle momentum vector and the negative
    z-axis, meaning that the particle is described in the direction where it is going to. The azimuthal angle is
    described between the positive x-axis and the horizontal component of the particle momentum vector
    (i.e. with respect to the magnetic north) proceeding counterclockwise.
    
    NRR describes the particle zenith and azimuthal angle in the direction where the particle is coming from.
    Therefore the zenith angle is the same, but the azimuthal angle has to be shifted by 180 + 90 degrees.
    The north has to be shifted by 90 degrees plus difference between geomagetic and magnetic north.


    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    declination : float
        declination of the magnetic field, in internal units

    Returns
    -------
    zenith : float
        zenith angle
    azimuth : float
        azimuth angle
    magnetic_field_vector : np.ndarray
        magnetic field vector
    """
    zenith = corsika['inputs'].attrs["THETAP"][0] * units.deg
    azimuth = hp.get_normalized_angle(
        3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]) + declination / units.rad
    ) * units.rad  # TODO: check if sign of declination is correct is correct

    Bx, Bz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(Bz, Bx)

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5 * units.micro * units.tesla

    # in local coordinates north is + 90 deg
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(
        B_inclination, declination / units.rad
    ) * units.rad + np.pi / 2 # TODO: check if sign of declination is correct is correct

    return zenith, azimuth, magnetic_field_vector


def get_geomagnetic_angle(zenith, azimuth, magnetic_field_vector):
    """
    Calculates the angle between the geomagnetic field and the shower axis defined by `zenith` and `azimuth`.

    Parameters
    ----------
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        The magnetic field vector in the NRR coordinate system (x points East, y points North, z points up)

    Returns
    -------
    geomagnetic_angle : float
        geomagnetic angle
    """
    shower_axis_vector = hp.spherical_to_cartesian(zenith / units.rad, azimuth / units.rad)
    geomagnetic_angle = hp.get_angle(magnetic_field_vector, shower_axis_vector) * units.rad

    return geomagnetic_angle


def convert_obs_to_nuradio_efield(observer, zenith, azimuth, magnetic_field_vector):
    """
    Converts the electric field from one CoREAS observer to NuRadio units and the on-sky coordinate system.

    The on-sky CS in NRR has basis vectors eR, eTheta, ePhi.
    Before going to the on-sky CS, we account for the magnetic field which does not point strictly North.
    To get the zenith, azimuth and magnetic field vector, one can use `get_angles()`.
    The `observer` array should have the shape (n_samples, 4) with the columns (time, Ey, -Ex, Ez),
    where (x, y, z) is the NuRadio CS.

    Parameters
    ----------
    observer : np.ndarray
        The observer as in the HDF5 file, e.g. list(corsika['CoREAS']['observers'].values())[i].
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        magnetic field vector
    
    Returns
    -------
    efield: np.array (n_samples, 3)
        Electric field in the on-sky CS (r, theta, phi)
    efield_times: np.array (n_samples)
        The time values corresponding to the electric field samples

    """
    cs = coordinatesystems.cstrafo(
        zenith / units.rad, azimuth / units.rad,
        magnetic_field_vector  # the magnetic field vector is used to find showerplane, so only direction is important
    )

    efield_times = observer[:, 0] * units.second
    efield = np.array([
        observer[:, 2] * -1,  # CORSIKA y-axis points West
        observer[:, 1],
        observer[:, 3]
    ]) * conversion_fieldstrength_cgs_to_SI

    # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
    efield_geographic = cs.transform_from_magnetic_to_geographic(efield)
    efield_on_sky = cs.transform_from_ground_to_onsky(efield_geographic)

    return efield_on_sky.T, efield_times


def convert_obs_positions_to_nuradio_on_ground(observer, zenith, azimuth, magnetic_field_vector):
    """
    Convert observer positions from the CORSIKA CS to the NRR ground CS.

    First, the observer position is converted to the NRR coordinate conventions (i.e. x pointing East,
    y pointing North, z pointing up). Then, the observer position is corrected for magnetic north
    (as CORSIKA only has two components to its magnetic field vector) and put in the geographic CS.
    To get the zenith, azimuth and magnetic field vector, one can use the `get_angles()` function.
    If multiple observers are to be converted, the `observer` array should have the shape (n_observers, 3).

    Parameters
    ----------
    observer : np.ndarray
        The observer's position as extracted from the HDF5 file, e.g. corsika['CoREAS']['my_observer']['position'].
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        magnetic field vector
  
    Returns
    -------
    obs_positions_geo: np.ndarray
        observer positions in geographic coordinates, shaped as (n_observers, 3).
    
    """
    cs = coordinatesystems.cstrafo(
        zenith / units.rad, azimuth / units.rad,
        magnetic_field_vector
    )

    # If single position is given, make sure it has the right shape (3,) -> (1, 3)
    if observer.ndim == 1:
        observer = observer[np.newaxis, :]

    obs_positions = np.array([
        observer[:, 1] * -1,
        observer[:, 0],
        observer[:, 2]
    ]) * units.cm

    # second to last dimension has to be 3 for the transformation
    obs_positions_geo = cs.transform_from_magnetic_to_geographic(obs_positions)

    return obs_positions_geo.T


def convert_obs_positions_to_vxB_vxvxB(observer, zenith, azimuth, magnetic_field_vector):
    """
    Convert observer position from CORSIKA CS to the showerplane CS.

    The showerplane CS has the basis (vxB, vxvxB, v).
    The position is first transformed to the NRR ground CS, using `convert_obs_positions_to_nuradio_on_ground()`.
    Then the radiotools function `coordinatesystems.cstrafo.transform_to_vxB_vxvxB()` is used to go to the showerplane.
    If multiple observers are to be converted, the `observer` array should have the shape (n_observers, 3).

    Parameters
    ----------
    observer : np.ndarray
        The observer's position as extracted from the HDF5 file, e.g. corsika['CoREAS']['my_observer']['position'].
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        magnetic field vector
  
    Returns
    -------
    obs_positions_vxB_vxvxB: np.ndarray
        observer positions in (vxB, vxvxB, v) CS, shaped as (n_observers, 3).
    
    """
    cs = coordinatesystems.cstrafo(
        zenith / units.rad, azimuth / units.rad,
        magnetic_field_vector
    )

    obs_positions_geo = convert_obs_positions_to_nuradio_on_ground(
        observer, zenith, azimuth, magnetic_field_vector
    )  # This will have shape (n_observers, 3)

    # transforms the coreas observer positions into the vxB, vxvxB shower plane
    obs_positions_vxB_vxvxB = cs.transform_to_vxB_vxvxB(obs_positions_geo)

    return obs_positions_vxB_vxvxB.T


def read_CORSIKA7(input_file, declination=None):
    """
    this function reads the corsika hdf5 file and returns a sim_station with all relevent information from the file

    Note that the function assumes the energy has been fixed to a single value.

    Parameters
    ----------
    input_file : string
        path to the corsika hdf5 file

    Returns
    -------
    evt : Event
        event object containing the corsika information
    """
    if declination is None:
        declination = 0
        logger.status(
            "No declination given, assuming 0 degrees. This might need to incorrect electric field polarizations."
        )

    corsika = h5py.File(input_file, "r")

    sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
    zenith, azimuth, magnetic_field_vector = get_angles(corsika, declination)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV  # Assume fixed energy

    # Create RadioShower to store simulation parameters in Event
    sim_shower = NuRadioReco.framework.radio_shower.RadioShower()

    sim_shower.set_parameter(shp.primary_particle, corsika["inputs"].attrs["PRMPAR"])
    sim_shower.set_parameter(shp.observation_level, corsika["inputs"].attrs["OBSLEV"] * units.cm)

    sim_shower.set_parameter(shp.zenith, zenith)
    sim_shower.set_parameter(shp.azimuth, azimuth)
    sim_shower.set_parameter(shp.magnetic_field_vector, magnetic_field_vector)
    sim_shower.set_parameter(shp.energy, energy)

    sim_shower.set_parameter(
        shp.core_coordinate_vertical, corsika['CoREAS'].attrs["CoreCoordinateVertical"] * units.cm
    )
    sim_shower.set_parameter(
        shp.coreas_GPSSecs, corsika['CoREAS'].attrs["GPSSecs"]
    )
    sim_shower.set_parameter(
        shp.shower_maximum, corsika['CoREAS'].attrs['DepthOfShowerMaximum'] * units.g / units.cm2
    )
    sim_shower.set_parameter(
        shp.distance_shower_maximum_geometric, corsika['CoREAS'].attrs["DistanceOfShowerMaximum"] * units.cm
    )
    sim_shower.set_parameter(
        shp.refractive_index_at_ground, corsika['CoREAS'].attrs["GroundLevelRefractiveIndex"]
    )
    sim_shower.set_parameter(
        shp.magnetic_field_rotation, corsika['CoREAS'].attrs["RotationAngleForMagfieldDeclination"] * units.degree
    )

    if 'ATMOD' in corsika['inputs'].attrs.keys():
        sim_shower.set_parameter(shp.atmospheric_model, corsika["inputs"].attrs["ATMOD"])

    if 'highlevel' in corsika.keys():
        sim_shower.set_parameter(shp.electromagnetic_energy, corsika["highlevel"].attrs["Eem"] * units.eV)

    # The traces are stored in a SimStation
    sim_station = NuRadioReco.framework.sim_station.SimStation(0)  # set sim station id to 0

    sim_station.set_parameter(stnp.azimuth, azimuth)
    sim_station.set_parameter(stnp.zenith, zenith)
    sim_station.set_parameter(stnp.cr_energy, energy)
    sim_station.set_magnetic_field_vector(magnetic_field_vector)

    sim_station.set_is_cosmic_ray()

    for j_obs, observer in enumerate(corsika['CoREAS']['observers'].values()):
        obs_positions_geo = convert_obs_positions_to_nuradio_on_ground(observer, zenith, azimuth, magnetic_field_vector)
        efield, efield_time = convert_obs_to_nuradio_efield(observer, zenith, azimuth, magnetic_field_vector)

        electric_field = NuRadioReco.framework.electric_field.ElectricField(
            np.array([j_obs]), position=obs_positions_geo
        )
        electric_field.set_trace(efield.T, sampling_rate)
        electric_field.set_trace_start_time(efield_time[0])

        electric_field.set_parameter(efp.ray_path_type, 'direct')
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)

        sim_station.add_electric_field(electric_field)

    evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])
    stn = NuRadioReco.framework.station.Station(0)  # set station id to 0
    stn.set_sim_station(sim_station)
    evt.set_station(stn)
    evt.add_sim_shower(sim_shower)
    corsika.close()

    return evt


def plot_footprint_onsky(sim_station, fig=None, ax=None):
    """
    plots the footprint of the observer positions in the (vxB, vxvxB) shower plane

    Parameters
    ----------
    sim_station : sim station
        simulated station object
    """
    obs_positions = []
    zz = []
    for efield in sim_station.get_electric_fields():
        obs_positions.append(efield.get_position_onsky())
        zz.append(np.sum(efield[efp.signal_energy_fluence]))
    obs_positions = np.array(obs_positions)
    zz = np.array(zz)
    if fig is None:
        fig, ax = plt.subplots()
    ax.set_aspect('equal')
    sc = ax.scatter(obs_positions[:, 0], obs_positions[:, 1], c=zz, cmap=cm.gnuplot2_r, marker='o', edgecolors='k')
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('fluence [eV/m^2]')
    ax.set_xlabel('vxB [m]')
    ax.set_ylabel('vx(vxB) [m]')
    return fig, ax


def make_sim_station(station_id, corsika, weight=None):
    """
    deprecated as it uses coreas hdf5 file as input, use set_sim_station() instead.

    creates an NuRadioReco sim station with the information from the coreas hdf5 file.
    To add an electric field the function add_electric_field_to_sim_station() has to be used.

    Parameters
    ----------
    station_id : station id
        the id of the station to create
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        simulated station object
    """
    logger.warning("make_sim_station() is deprecated, use create_sim_station() instead.")
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    sim_station.set_parameter(stnp.azimuth, azimuth)
    sim_station.set_parameter(stnp.zenith, zenith)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_station.set_parameter(stnp.cr_energy, energy)
    sim_station.set_magnetic_field_vector(magnetic_field_vector)
    sim_station.set_parameter(stnp.cr_xmax, corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
    try:
        sim_station.set_parameter(stnp.cr_energy_em, corsika["highlevel"].attrs["Eem"])
    except:
        global warning_printed_coreas_py
        if (not warning_printed_coreas_py):
            logger.warning(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    sim_station.set_is_cosmic_ray()
    sim_station.set_simulation_weight(weight)
    return sim_station


def create_sim_station(station_id, evt, weight=None):
    """
    creates an NuRadioReco sim station with the information from an event object created with e.g. read_CORSIKA7().
    An weight per station can be added.
    To add an electric field the function add_electric_field_to_sim_station() has to be used.

    Parameters
    ----------
    station_id : station id
        the id of the station to create
    evt : Event object
        event object containing the CoREAS output, e.g. created with read_CORSIKA7()
    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        simulated station object
    """
    coreas_sta = evt.get_station(station_id=0)  # read_coreas has only station id 0
    coreas_sim_station = coreas_sta.get_sim_station()

    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    sim_station.set_parameter(stnp.azimuth, coreas_sim_station.get_parameter(stnp.azimuth))
    sim_station.set_parameter(stnp.zenith, coreas_sim_station.get_parameter(stnp.zenith))
    sim_station.set_parameter(stnp.cr_energy, coreas_sim_station.get_parameter(stnp.cr_energy))
    sim_station.set_parameter(stnp.cr_xmax, coreas_sim_station.get_parameter(stnp.cr_xmax))
    magnetic_field_vector = coreas_sim_station.get_magnetic_field_vector()
    sim_station.set_magnetic_field_vector(magnetic_field_vector)
    try:
        sim_station.set_parameter(stnp.cr_energy_em, coreas_sim_station.get_parameter(stnp.cr_energy_em))
    except:
        global warning_printed_coreas_py
        if (not warning_printed_coreas_py):
            logger.warning(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    if coreas_sim_station.is_cosmic_ray():
        sim_station.set_is_cosmic_ray()

    sim_station.set_simulation_weight(weight)
    return sim_station


def make_sim_shower(corsika, observer=None, detector=None, station_id=None):
    """
    deprecated as it uses coreas hdf5 file as input, use set_sim_shower instead

    creates an NuRadioReco sim shower from the coreas hdf5 file, the core positions are set such that the detector station is on top of
    each coreas observer position

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    observer : hdf5 observer object
    detector : detector object
    station_id : station id of the station relativ to which the shower core is given

    Returns
    -------
    sim_shower: sim shower
        simulated shower object
    """
    logger.warning("make_sim_shower() is deprecated, use set_sim_shower() instead.")

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
        observer_position = convert_obs_positions_to_nuradio_on_ground(position, zenith, azimuth, magnetic_field_vector)
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
        global warning_printed_coreas_py
        if (not warning_printed_coreas_py):
            logger.warning(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True

    return sim_shower


def create_sim_shower(evt, detector=None, station_id=None):
    """
    creates an NuRadioReco sim shower from an event object created with e.g. read_CORSIKA7(),
    the core positions are set such that the detector station is on top of
    each coreas observer position

    Parameters
    ----------
    evt : Event object
        event object containing the CoREAS output, e.g. created with read_CORSIKA7()
    detector : detector object
    station_id : int
        station id of the station relative to which the shower core is given

    Returns
    -------
    sim_shower: sim shower
        simulated shower object
    """
    sim_shower = copy.copy(evt.get_first_sim_shower())

    efields = evt.get_station(0).get_sim_station().get_electric_fields()

    # We can only set the shower core relative to the station if we know its position
    if efields is not None and detector is not None and station_id is not None:
        efield_pos = []
        for efield in efields:
            efield_pos.append(efield.get_position())
        efield_pos = np.array(efield_pos)
        station_position = detector.get_absolute_position(station_id)
        core_position = (-efield_pos + station_position)
        core_position[2] = 0
        sim_shower.set_parameter(shp.core, core_position)

    return sim_shower


def add_electric_field_to_sim_station(sim_station, channel_ids, efield, efield_times, zenith, azimuth, sampling_rate,
                                      fluence=None):
    """
    adds an electric field trace to an existing sim station

    Parameters
    ----------
    sim_station : sim station object
         simulated station object, e.g. from make_empty_sim_station()
    channel_ids : list or int
        channel_ids for which the efield is to be used
    efield : 3d array (3, n_samples)
        efield with three polarizations (r, theta, phi) for channel
    efield_times : 1d array (n_samples)
        time array for efield
    zenith : float
        zenith angle in radians
    azimuth : float
        azimuth angle in radians
    sampling_rate : float
        sampling rate of the efield
    fluence : float
        inserted fluence of the signal, if None, the fluence is not set
    """
    if type(channel_ids) is not list:
        channel_ids = [channel_ids]

    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids)
    electric_field.set_trace(efield, sampling_rate)
    electric_field.set_trace_start_time(efield_times[0])
    electric_field.set_parameter(efp.ray_path_type, 'direct')
    electric_field.set_parameter(efp.zenith, zenith)
    electric_field.set_parameter(efp.azimuth, azimuth)
    if fluence is not None:
        electric_field.set_parameter(efp.signal_energy_fluence, fluence)
    sim_station.add_electric_field(electric_field)


def calculate_simulation_weights(positions, zenith, azimuth, site='summit', debug=False):
    """
    Calculate weights according to the area that one simulated position in readCoreasStation represents.
    Weights are therefore given in units of area.
    Note: The volume of a 2d convex hull is the area.
    
    Parameters
    ----------
    positions : list
        station position with [x, y, z] on ground
    zenith : float
        zenith angle of the shower
    azimuth : float
        azimuth angle of the shower
    site : str
        site of the simulation, default is 'summit'
    debug : bool
        if true, plots are created for debugging
    
    Returns
    -------
    weights : np.array
        weights of the observer position
    """

    import scipy.spatial as spatial

    positions = np.array(positions)
    cs = coordinatesystems.cstrafo(zenith=zenith, azimuth=azimuth, magnetic_field_vector=None,
                                   site=site)
    x_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([1, 0, 0]))
    y_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([0, 1, 0]))
    z_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([0, 0, 1]))

    # voronoi has to be calculated in the shower plane due to symmetry reasons
    shower = cs.transform_to_vxB_vxvxB(station_position=positions)
    vor = spatial.Voronoi(shower[:, :2])  # algorithm will find no solution if flat simulation is given in 3d.

    if debug:
        fig1 = plt.figure(figsize=[12, 4])
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        spatial.voronoi_plot_2d(vor, ax1)
        ax1.set_aspect('equal')
        ax1.set_title('In shower plane, zenith = {:.2f}'.format(zenith / units.degree))
        ax1.set_xlabel(r'Position in $\vec{v} \times \vec{B}$ - direction [m]')
        ax1.set_ylabel(r'Position in $\vec{v} \times \vec{v} \times \vec{B}$ - direction [m]')

    weights = np.zeros_like(positions[:, 0])
    for p in range(0, weights.shape[0]):  # loop over all observer positions
        vertices_shower_2d = vor.vertices[vor.regions[vor.point_region[p]]]

        x_vertice_shower = vertices_shower_2d[:, 0]
        y_vertice_shower = vertices_shower_2d[:, 1]
        z_vertice_shower = -(
                x_trafo_from_shower[2] * x_vertice_shower + y_trafo_from_shower[2] * y_vertice_shower
        ) / z_trafo_from_shower[2]

        vertices_shower_3d = np.column_stack((x_vertice_shower, y_vertice_shower, z_vertice_shower))

        vertices_ground = cs.transform_from_vxB_vxvxB(station_position=vertices_shower_3d)

        n_arms = 8  # mask last observer position of each arm
        length_shower = np.sqrt(shower[:, 0] ** 2 + shower[:, 1] ** 2)
        ind = np.argpartition(length_shower, -n_arms)[-n_arms:]

        weight = spatial.ConvexHull(vertices_ground[:, :2])
        weights[p] = weight.volume  # volume of a 2d dataset is the area, area of a 2d data set is the perimeter
        weights[ind] = 0

        if debug:
            ax2.plot(vertices_ground[:, 0], vertices_ground[:, 1], c='grey', zorder=1)
            ax2.scatter(vertices_ground[:, 0], vertices_ground[:, 1], c='tab:orange', zorder=2)
    if debug:
        ax2.scatter(positions[:, 0], positions[:, 1], c='tab:blue', s=10, label='Position of observer')
        ax2.scatter(vertices_ground[:, 0], vertices_ground[:, 1], c='tab:orange', label='Vertices of cell')
        ax2.set_aspect('equal')
        ax2.set_title('On ground, total area {:.2f} $km^2$'.format(sum(weights) / units.km ** 2))
        ax2.set_xlabel('East [m]')
        ax2.set_ylabel('West [m]')
        ax2.set_xlim(-5000, 5000)
        ax2.set_ylim(-5000, 5000)
        plt.legend()
        plt.show()

        fig3 = plt.figure(figsize=[12, 4])
        ax4 = fig3.add_subplot(121)
        ax5 = fig3.add_subplot(122)

        ax4.hist(weights)
        ax4.set_title('Weight distribution')
        ax4.set_xlabel(r'Weights (here area) $[m^2]$')
        ax4.set_ylabel(r'Number of observer')

        ax5.scatter(length_shower ** 2, weights)
        ax5.set_xlabel(r'$Length^2 [m^2]$')
        ax5.set_ylabel('Weight $[m^2]$')
        plt.show()

    return weights


class coreasInterpolator:
    """
    The functions in this class are used to interpolate the fluence and signal shape from coreas files

    Parameters
    ----------
    corsika_evt : NuRadio event object
            use read_CORSIKA7() to create the event object containing the CoREAS output
    """

    def __init__(self, corsika_evt):
        self.sampling_rate = None
        self.electric_field_on_sky = None
        self.efield_times = None
        self.obs_positions_geo = None
        self.obs_positions_vxB_vxvxB = None
        self.empty_efield = None
        self.max_coreas_efield = None
        self.star_radius = None
        self.geo_star_radius = None

        self.corsika_evt = corsika_evt
        self.sim_station = corsika_evt.get_station(0).get_sim_station()
        self.shower = corsika_evt.get_first_sim_shower()  # there should only be one simulated shower
        self.star_shape_initialized = False
        self.efield_interpolator_initialized = False
        self.fluence_interpolator_initialized = False
        self.zenith = self.shower.get_parameter(shp.zenith)
        self.azimuth = self.shower.get_parameter(shp.azimuth)

        self.interp_lowfreq = None
        self.interp_highfreq = None
        self.efield_interpolator = None
        self.fluence_interpolator = None

        self.cs = coordinatesystems.cstrafo(self.zenith, self.azimuth, self.shower[shp.magnetic_field_vector])

        self.initialize_star_shape()

    def initialize_star_shape(self):
        """
        Initializes the star shape pattern for interpolation, e.g. creates the arrays with the observer positions
        in the shower plane and the electric field.
        """
        obs_positions_geo = []
        obs_positions_onsky = []
        electric_field_on_sky = []
        efield_times = []

        for j_obs, efield in enumerate(self.sim_station.get_electric_fields()):
            obs_positions_geo.append(efield.get_position())
            obs_positions_onsky.append(efield.get_position_onsky())
            electric_field_on_sky.append(efield.get_trace().T)
            efield_times.append(efield.get_times())

        # shape: (n_observers, n_samples, (eR, eTheta, ePhi))
        self.electric_field_on_sky = np.array(electric_field_on_sky)
        self.efield_times = np.array(efield_times)
        self.sampling_rate = 1. / (self.efield_times[0][1] - self.efield_times[0][0])
        self.obs_positions_geo = np.array(obs_positions_geo)
        self.obs_positions_vxB_vxvxB = np.array(obs_positions_onsky)

        self.max_coreas_efield = np.max(np.abs(self.electric_field_on_sky))
        self.empty_efield = np.zeros_like(self.electric_field_on_sky[0, :, :])

        self.star_radius = np.max(np.linalg.norm(self.obs_positions_vxB_vxvxB[:, :-1], axis=-1))
        self.geo_star_radius = np.max(np.linalg.norm(self.obs_positions_geo[:-1, :], axis=0))
        logger.info(
            f'Initialize star shape pattern for interpolation. '
            f'The shower arrives at zenith={self.zenith / units.deg:.0f}deg, '
            f'azimuth={self.azimuth / units.deg:.0f}deg with radius {self.star_radius:.0f}m in the shower plane '
            f'and {self.geo_star_radius:.0f}m on ground. ')
        self.star_shape_initialized = True

    def get_sampling_rate(self):
        """
        returns the sampling rate of the electric field
        """
        return self.sampling_rate

    def get_empty_efield(self):
        """
        returns the an array of zeros in the shape of the electric field on the sky
        """
        if not self.star_shape_initialized:
            logger.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.empty_efield

    def get_max_efield(self):
        """
        returns the maximum value of the electric field provided by coreas
        """
        if not self.star_shape_initialized:
            logger.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.max_coreas_efield

    def get_star_radius(self):
        """
        returns the maximal radius of the star shape pattern in the shower plane
        """
        if not self.star_shape_initialized:
            logger.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.star_radius

    def get_geo_star_radius(self):
        """
        returns the maximal radius of the star shape pattern on ground
        """
        if not self.star_shape_initialized:
            logger.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.geo_star_radius

    def initialize_efield_interpolator(self, interp_lowfreq, interp_highfreq):
        """
        Initialises the efield interpolator object. The efield will be interpolated in the shower plane for
        geometrical reasons. If the geomagnetic angle is smaller than 15deg, no interpolator object is returned.

        Parameters
        ----------
        interp_lowfreq : float
            lower frequency for the bandpass filter in interpolation in GHz
        interp_highfreq : float
            higher frequency for the bandpass filter in interpolation in GHz

        Returns
        -------
        efield_interpolator : interpolator object

        """
        self.efield_interpolator_initialized = True
        self.interp_lowfreq = interp_lowfreq
        self.interp_highfreq = interp_highfreq

        geomagnetic_angle = get_geomagnetic_angle(self.zenith, self.azimuth, self.shower[shp.magnetic_field_vector])

        if geomagnetic_angle < 15 * units.deg:
            logger.warning(
                f'geomagnetic angle is {geomagnetic_angle / units.deg:.2f} deg, '
                f'which is smaller than 15deg, which is the lower limit for the signal interpolation. '
                f'The closest obersever is used instead.')
            self.efield_interpolator = -1
        else:
            logger.info(
                f'electric field interpolation with lowfreq {interp_lowfreq / units.MHz} MHz '
                f'and highfreq {interp_highfreq / units.MHz} MHz')

            self.efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
                self.obs_positions_vxB_vxvxB[:, 0],
                self.obs_positions_vxB_vxvxB[:, 1],
                self.electric_field_on_sky,
                signals_start_times=self.efield_times[:, 0],
                lowfreq=(interp_lowfreq - 0.01) / units.MHz,
                highfreq=(interp_highfreq + 0.01) / units.MHz,
                sampling_period=1 / self.sampling_rate / units.s,  # interpolator wants sampling period in seconds
                phase_method="phasor",
                radial_method='cubic',
                upsample_factor=5,
                coherency_cutoff_threshold=0.9,
                ignore_cutoff_freq_in_timing=False,
                verbose=False
            )
        return self.efield_interpolator

    def initialize_fluence_interpolator(self, quantity=efp.signal_energy_fluence, debug=False):
        """
        initilized fluence interpolator object.

        Parameters
        ----------
        quantity : electric field parameter
            quantity to interpolate, e.g. efp.signal_energy_fluence
            The quantity needs to be available as parameter in the electric field object!!! You might
            need to run the electricFieldSignalReconstructor. The default is efp.signal_energy_fluence.

        Returns
        -------
        fluence_interpolator : interpolator object
        """
        fluence_per_position = [
            np.sum(efield[quantity]) for efield in self.sim_station.get_electric_fields()
        ]  # the fluence is calculated per polarization, so we need to sum them up

        logger.info(f'fluence interpolation')
        self.fluence_interpolator = cr_pulse_interpolator.interpolation_fourier.interp2d_fourier(
            self.obs_positions_vxB_vxvxB[:, 0],
            self.obs_positions_vxB_vxvxB[:, 1],
            fluence_per_position
        )
        self.fluence_interpolator_initialized = True

        if debug:
            max_efield = []
            for i in range(len(self.electric_field_on_sky[:, 0, 1])):
                max_efield.append(np.max(np.abs(self.electric_field_on_sky[i, :, :])))
            plt.scatter(self.obs_positions_vxB_vxvxB[:, 0], self.obs_positions_vxB_vxvxB[:, 1], c=max_efield,
                        cmap='viridis', marker='o', edgecolors='k')
            cbar = plt.colorbar()
            cbar.set_label('max amplitude')
            plt.xlabel('v x B [m]')
            plt.ylabel('v x v x B [m]')
            plt.show()
            plt.close()

        return self.fluence_interpolator

    def plot_footprint_fluence(self, radius=300):
        """
        plots the interpolated values of the fluence in the shower plane

        Parameters
        ----------
        radius : float
            radius around shower core which should be plotted

        Returns
        -------
        fig : figure object

        ax : axis object
        """

        # Make color plot of f(x, y), using a meshgrid
        ti = np.linspace(-radius, radius, 500)
        XI, YI = np.meshgrid(ti, ti)

        # Get interpolated values at each grid point, calling the instance of interp2d_fourier
        ZI = self.fluence_interpolator(XI, YI)

        # And plot it
        maxp = np.max(ZI)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolor(XI, YI, ZI, vmax=maxp, vmin=0, cmap=cm.gnuplot2_r)
        mm = cm.ScalarMappable(cmap=cm.gnuplot2_r)
        mm.set_array([0.0, maxp])
        cbar = plt.colorbar(mm, ax=ax)
        cbar.set_label(r'energy fluence [eV/m^2]', fontsize=14)
        ax.set_xlabel(r'$\vec{v} \times \vec{B} [m]', fontsize=16)
        ax.set_ylabel(r'$\vec{v} \times (\vec{v} \times \vec{B})$ [m]', fontsize=16)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect('equal')
        return fig, ax

    def get_interp_efield_value(self, position_on_ground, core):
        """
        Accesses the interpolated electric field given the position of the detector on ground. For the interpolation,
        the pulse will be projected in the shower plane. If the geomagnetic angle is smaller than 15deg, the electric
        field of the closest observer position is returned.

        Parameters
        ----------
        position_on_ground : np.array (3)
            position of the antenna on ground

        core : np.array (3)
            position of the core on ground

        Returns
        -------
        efield_interp : float
            interpolated efield value or efield of clostest observer

        trace_start_time : float
            start time of the trace
        """
        logger.debug(
            f"get interpolated efield for antenna position {position_on_ground} on ground and core position {core}")
        antenna_position = copy.copy(position_on_ground)

        # core and antenna need to be in the same z plane
        antenna_position[2] = core[2]

        # transform antenna position into shower plane with respect to core position, core position is set to 0,0 in shower plane
        antenna_pos_vBvvB = self.cs.transform_to_vxB_vxvxB(antenna_position, core=core)
        logger.debug(f"antenna position in shower plane {antenna_pos_vBvvB}")

        # calculate distance between core position at(0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB[:-1])
        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > self.star_radius:
            efield_interp = self.empty_efield
            trace_start_time = None
            logger.debug(
                f'antenna position with distance {dcore_vBvvB:.2f} to core is outside of star pattern '
                f'with radius {self.star_radius:.2f} on ground {self.geo_star_radius:.2f}, '
                f'set efield to zero')
        else:
            if self.efield_interpolator == -1:
                efield = self.get_closest_observer_efield(antenna_pos_vBvvB)
                efield_interp = efield
                trace_start_time = None

            else:
                efield_interp, trace_start_time, abs_spectrum, phasespectrum = self.efield_interpolator(
                    antenna_pos_vBvvB[0], antenna_pos_vBvvB[1],
                    lowfreq=self.interp_lowfreq / units.MHz,
                    highfreq=self.interp_highfreq / units.MHz,
                    filter_up_to_cutoff=False,
                    account_for_timing=True,
                    pulse_centered=True,
                    full_output=True)

        #check if interpolation is within expected range
        if np.max(np.abs(efield_interp)) > self.max_coreas_efield:
            logger.warning(
                f'interpolated efield {np.max(np.abs(efield_interp)):.2f} is larger than '
                f'the maximum coreas efield {self.max_coreas_efield:.2f}')

        return efield_interp, trace_start_time

    def get_interp_fluence_value(self, position_on_ground, core):
        """
        Accesses the interpolated fluence for a given position on ground

        Parameters
        ----------
        position_on_ground : np.array (3)
            position of the antenna on ground

        core : np.array (3)
            position of the core on ground

        Returns
        -------
        fluence_interp : float
            interpolated fluence value
        """
        antenna_position = position_on_ground
        z_plane = core[2]

        #core and antenna need to be in the same z plane
        antenna_position[2] = z_plane

        # transform antenna position into shower plane with respect to core position, core position is set to 0,0 in shower plane
        antenna_pos_vBvvB = self.cs.transform_to_vxB_vxvxB(antenna_position, core=core)

        # calculate distance between core position at(0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB[:-1])
        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > self.star_radius:
            fluence_interp = 0
            logger.debug(
                f'antenna position with distance {dcore_vBvvB:.2f} to core is outside of star pattern '
                f'with radius {self.star_radius:.2f} on ground {self.geo_star_radius:.2f}, '
                f'fluence to zero')
        else:
            fluence_interp = self.fluence_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1])

        return fluence_interp

    def get_closest_observer_efield(self, antenna_pos_vBvvB):
        """
        Returns the electric field of the closest observer position for an antenna position in the shower plane.

        Parameters
        ----------
        antenna_pos_vBvvB : np.array (3)
            antenna position in the shower plane

        Returns
        -------
        efield : array of floats
            electric field

        """
        distances = np.linalg.norm(antenna_pos_vBvvB[:2] - self.obs_positions_vBvvB[:, :2], axis=1)
        index = np.argmin(distances)
        distance = distances[index]
        efield = self.electric_field_on_sky[index, :, :]
        logger.info(f'antenna position with distance {distance:.2f} to closest observer is used')
        return efield
