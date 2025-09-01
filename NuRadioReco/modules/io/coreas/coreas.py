import copy
import h5py
import numpy as np
import matplotlib.pyplot as plt
from radiotools import helper as hp
from radiotools import coordinatesystems

from NuRadioReco.utilities import units
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.event
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp

import logging
logger = logging.getLogger('NuRadioReco.coreas')

warning_printed_coreas_py = False

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter


# DEPRECATED FUNCTIONS
def make_sim_shower(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to `create_sim_shower_from_hdf5()`, however its functionality has been
    modified heavily. You will probably want to move to `create_sim_shower()` instead, which has an easier
    interface. Please refer to the documentation of `create_sim_shower()` for more information.
    """
    raise DeprecationWarning("This function has been deprecated since version 3.0. "
                             "You will probably want to move to create_sim_shower() instead.")


def make_sim_station(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to `create_sim_station()`, however its functionality has been modified. Please
    refer to the documentation of `create_sim_station()` for more information.
    """
    raise DeprecationWarning("This function has been deprecated since version 3.0. "
                             "You will probably want to move to create_sim_station() instead.")


# UTILITY FUNCTIONS
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
    Therefore, the zenith angle is the same, but the azimuthal angle has to be shifted by 180 + 90 degrees.
    The north has to be shifted by 90 degrees plus difference between geomagnetic and magnetic north.

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

    Examples
    --------
    The declinations can be obtained using the functions in the radiotools helper package, if you
    have the magnetic field for the site you are interested in.

    >>> magnet = hp.get_magnetic_field_vector('mooresbay')
    >>> dec = hp.get_declination(magnet)
    >>> evt = h5py.File('NuRadioReco/examples/example_data/example_data.hdf5', 'r')
    >>> get_angles(corsika, dec)[2] / units.gauss
    array([ 0.05646405, -0.08733734,  0.614     ])
    >>> magnet
    array([ 0.058457, -0.09042 ,  0.61439 ])
    """
    zenith = corsika['inputs'].attrs["THETAP"][0] * units.deg
    azimuth = hp.get_normalized_angle(
        3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]) + declination / units.rad
    ) * units.rad

    # in CORSIKA convention, the first component points North (y in NRR) and the second component points down (minus z)
    By, minBz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(minBz, By)  # angle from y-axis towards negative z-axis

    B_strength = np.sqrt(By ** 2 + minBz ** 2) * units.micro * units.tesla

    # zenith of the magnetic field vector is 90 deg + inclination, as inclination proceeds downwards from horizontal
    # azimuth of the magnetic field vector is 90 deg - declination, as declination proceeds clockwise from North
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(
        np.pi / 2 + B_inclination, np.pi / 2 - declination / units.rad
    )

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
    efield: np.array (3, n_samples)
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

    return efield_on_sky, efield_times


def convert_obs_positions_to_nuradio_on_ground(observer_pos, declination=0):
    """
    Convert observer positions from the CORSIKA CS to the NRR ground CS.

    First, the observer position is converted to the NRR coordinate conventions (i.e. x pointing East,
    y pointing North, z pointing up). Then, the observer position is corrected for magnetic north
    (as CORSIKA only has two components to its magnetic field vector) and put in the geographic CS.
    To get the zenith, azimuth and magnetic field vector, one can use the `get_angles()` function.
    If multiple observers are to be converted, the `observer` array should have the shape (n_observers, 3).

    Parameters
    ----------
    observer_pos : np.ndarray
        The observer's position as extracted from the HDF5 file, e.g. corsika['CoREAS']['my_observer'].attrs['position']
    declination : float (default: 0)
        Declination of the magnetic field.

    Returns
    -------
    obs_positions_geo: np.ndarray
        observer positions in geographic coordinates, shaped as (n_observers, 3).
    """
    # If single position is given, make sure it has the right shape (3,) -> (1, 3)
    if observer_pos.ndim == 1:
        observer_pos = observer_pos[np.newaxis, :]

    obs_positions = np.array([
        observer_pos[:, 1] * -1,
        observer_pos[:, 0],
        observer_pos[:, 2]
    ]) * units.cm

    obs_positions = hp.rotate_vector_in_2d(obs_positions, -declination).T

    return np.squeeze(obs_positions)

# READER FUNCTIONS
def read_CORSIKA7(input_file, declination=None, site=None):
    """
    This function reads in a CORSIKA/CoREAS HDF5 file and returns an Event object with all relevant information
    from the file. This Event object can then be used to create an interpolator, for example.

    The structure of the Event is as follows. It contains one station with ID equal to 0, which hosts a SimStation.
    The SimStation stores the simulated ElectricField objects in on-sky coordinates (ie eR, eTheta, ePhi).
    They are equipped with a position attribute that contains the position of the simulated antenna
    (in the NRR ground coordinate system) and are associated to a channel with an ID equal to the index of the
    channel as it was read from the HDF5 file.

    Next to the (Sim)Station, the Event also contains a SimShower object, which stores the CORSIKA input parameters.
    For a list of stored parameters, see the `create_sim_shower_from_hdf5()` function.

    Note that the function assumes the energy has been fixed to a single value, as is typical with a CoREAS simulation.

    Parameters
    ----------
    input_file: str
        Path to the CORSIKA HDF5 file
    declination: float, default=0
        The declination to use for the magnetic field, in internal units
    site: str, default=None
        Instead of declination, a site name can be given to retrieve the declination using the magnetic field
        as obtained from the radiotools.helper.get_magnetic_field_vector() function

    Returns
    -------
    evt: NuRadioReco.framework.event.Event
        Event object containing the CORSIKA information
    """
    if declination is None:
        if site is not None:
            try:
                magnet = hp.get_magnetic_field_vector(site)
                declination = hp.get_declination(magnet)
                logger.info(
                    f"Declination obtained from site information, is set to {declination / units.degree:.2f} deg"
                )
            except KeyError:
                declination = 0
                logger.warning(
                    "Site is not recognised by radiotools. Defaulting to 0 degrees declination. "
                    "This might lead to unexpected electric field polarizations."
                )
        else:
            declination = 0
            logger.warning(
                "No declination or site given, assuming 0 degrees. "
                "This might lead to unexpected electric field polarizations."
            )

    corsika = h5py.File(input_file, "r")

    sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
    zenith, azimuth, magnetic_field_vector = get_angles(corsika, declination)

    # The traces are stored in a SimStation
    sim_station = NuRadioReco.framework.sim_station.SimStation(0)  # set sim station id to 0

    sim_station.set_is_cosmic_ray()

    for j_obs, observer in enumerate(corsika['CoREAS']['observers'].values()):
        obs_positions_geo = convert_obs_positions_to_nuradio_on_ground(
            observer.attrs['position'], declination
        )

        efield, efield_time = convert_obs_to_nuradio_efield(
            observer, zenith, azimuth, magnetic_field_vector
        )

        add_electric_field_to_sim_station(
            sim_station, [j_obs],
            efield, efield_time[0],
            zenith, azimuth,
            sampling_rate,
            efield_position=obs_positions_geo
        )

    evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])
    evt.set_event_time(corsika['CoREAS'].attrs['GPSSecs'], format="gps")

    stn = NuRadioReco.framework.station.Station(0)  # set station id to 0
    stn.set_sim_station(sim_station)
    evt.set_station(stn)

    sim_shower = create_sim_shower_from_hdf5(corsika)
    evt.add_sim_shower(sim_shower)

    corsika.close()

    return evt


def create_sim_shower_from_hdf5(corsika, declination=0):
    """
    Creates an NuRadioReco `RadioShower` from a CoREAS HDF5 file, which contains the simulation inputs shower parameters.
    These include

    - the primary particle type
    - the observation level
    - the zenith and azimuth angles
    - the magnetic field vector
    - the energy of the primary particle

    The following parameters are retrieved from the REAS file:

    - the core position
    - the depth of the shower maximum (in g/cm2, converted to internal units)
    - the distance of the shower maximum (in cm, converted to internal units)
    - the refractive index at ground level
    - the declination of the magnetic field

    Lastly, these parameters are also saved IF they are available in the HDF5 file:

    - the atmospheric model used for the simulation
    - the electromagnetic energy of the shower (only present in high-level quantities are present)

    This function is used in the `read_CORSIKA7()` function to create the SimShower object. In order to copy a
    `SimShower` object from an Event object, use the `create_sim_shower()` method.

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    declination : float

    Returns
    -------
    sim_shower: RadioShower
        simulated shower object
    """
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
        shp.core, np.array([
            corsika['CoREAS'].attrs["CoreCoordinateWest"] * -1,
            corsika['CoREAS'].attrs["CoreCoordinateNorth"],
            corsika['CoREAS'].attrs["CoreCoordinateVertical"]
        ]) * units.cm
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

    if 'ATMOD' in corsika['inputs'].attrs:  # this can be false is left on default or when using GDAS atmosphere
        sim_shower.set_parameter(shp.atmospheric_model, corsika["inputs"].attrs["ATMOD"])

    if 'highlevel' in corsika:
        sim_shower.set_parameter(shp.electromagnetic_energy, corsika["highlevel"].attrs["Eem"] * units.eV)
    else:
        global warning_printed_coreas_py
        if not warning_printed_coreas_py:
            logger.info(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True

    return sim_shower


def create_sim_shower(evt, core_shift=None):
    """
    Create an NuRadioReco `SimShower` from an Event object created with e.g. read_CORSIKA7(),
    and apply a core shift if desired. If no core shift is given, the returned SimShower will
    have the same core as used in the CoREAS simulation.

    Parameters
    ----------
    evt: Event
        event object containing the CoREAS output, e.g. created with read_CORSIKA7()
    core_shift: np.ndarray, default=None
        The core shift to apply to the core position, in internal units. Must be 3D array.

    Returns
    -------
    sim_shower: RadioShower
        simulated shower object
    """
    sim_shower = copy.deepcopy(evt.get_first_sim_shower())  # this has the core set to the one defined in the REAS file

    # We can only set the shower core relative to the station if we know its position
    if core_shift is not None:
        sim_shower.set_parameter(shp.core, sim_shower.get_parameter(shp.core) + core_shift)

    return sim_shower


def create_sim_station(station_id, evt, weight=None):
    """
    Creates an NuRadioReco `SimStation` with the information from an `Event` object created with e.g. read_CORSIKA7().

    Optionally, station can be assigned a weight. Note that the station is empty. To add an
    electric field the function add_electric_field_to_sim_station() has to be used.

    Parameters
    ----------
    station_id : int
        The id to assign to the new station
    evt : Event
        event object containing the CoREAS output
    weight : float
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: SimStation
        simulated station object
    """
    coreas_station = evt.get_station(station_id=0)  # read_coreas has only station id 0
    coreas_shower = evt.get_first_sim_shower()
    coreas_sim_station = coreas_station.get_sim_station()

    # Make the SimStation and store the parameters extracted from the SimShower
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)

    sim_station.set_parameter(stnp.azimuth, coreas_shower.get_parameter(shp.azimuth))
    sim_station.set_parameter(stnp.zenith, coreas_shower.get_parameter(shp.zenith))

    sim_station.set_parameter(stnp.cr_energy, coreas_shower.get_parameter(shp.energy))
    sim_station.set_parameter(stnp.cr_xmax, coreas_shower.get_parameter(shp.shower_maximum))

    sim_station.set_magnetic_field_vector(
        coreas_shower.get_parameter(shp.magnetic_field_vector)
    )

    # Check if the high-level attribute is present in the Event
    try:
        sim_station.set_parameter(stnp.cr_energy_em, coreas_shower.get_parameter(shp.electromagnetic_energy))
    except KeyError:
        global warning_printed_coreas_py
        if not warning_printed_coreas_py:
            logger.warning(
                "No high-level quantities in Event, not setting EM energy, this warning will be only printed once"
            )
            warning_printed_coreas_py = True

    if coreas_sim_station.is_cosmic_ray():
        sim_station.set_is_cosmic_ray()

    # Set simulation weight
    sim_station.set_simulation_weight(weight)

    return sim_station

# HELPER FUNCTIONS
def add_electric_field_to_sim_station(
        sim_station, channel_ids, efield, efield_start_time, zenith, azimuth, sampling_rate, efield_position=None
):
    """
    Adds an electric field trace to an existing SimStation, with the provided attributes.

    All variables should be provided in internal units.

    Parameters
    ----------
    sim_station : SimStation
         The simulated station object, e.g. from make_empty_sim_station()
    channel_ids : list of int
        Channel IDs to associate to the electric field
    efield : np.ndarray
        Electric field trace, shaped as (3, n_samples)
    efield_start_time : float
        Start time of the electric field trace
    zenith : float
        Zenith angle
    azimuth : float
        Azimuth angle
    sampling_rate : float
        Sampling rate of the trace
    efield_position : np.ndarray or list of float
        Position to associate to the electric field
    """
    if type(channel_ids) is not list:
        channel_ids = [channel_ids]

    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids, position=efield_position)

    electric_field.set_trace(efield, sampling_rate)
    electric_field.set_trace_start_time(efield_start_time)

    electric_field.set_parameter(efp.ray_path_type, 'direct')
    electric_field.set_parameter(efp.zenith, zenith)
    electric_field.set_parameter(efp.azimuth, azimuth)

    sim_station.add_electric_field(electric_field)


def calculate_simulation_weights(positions, zenith, azimuth, site='summit', debug=False):
    """
    Calculate weights according to the area that one observer position in a starshape pattern represents.
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

def set_fluence_of_efields(function, sim_station, quantity=efp.signal_energy_fluence):
    """
    This helper function is used to set the fluence quantity of all electric fields in a SimStation.
    Use this to calculate the fluences to use for interpolation.

    One option to use as `function` is `trace_utilities.get_electric_field_energy_fluence()`.

    Parameters
    ----------
    function: callable
        The function to apply to the traces in order to calculate the fluence. Should take in a (3, n_samples) shaped
        array and return a float (or an array with 3 elements if you want the fluence per polarisation).
    sim_station: SimStation
        The simulated station object
    quantity: electric field parameter, default=efp.signal_energy_fluence
        The parameter where to store the result of the fluence calculation
    """
    for electric_field in sim_station.get_electric_fields():
        fluence = function(electric_field.get_trace())
        electric_field.set_parameter(quantity, fluence)
