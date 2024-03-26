import numpy as np
import matplotlib.pyplot as plt
from radiotools import helper as hp
from radiotools import coordinatesystems
from NuRadioReco.utilities import units
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.radio_shower
import radiotools.coordinatesystems
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
import logging
import h5py
import pickle
import os
logger = logging.getLogger('coreas')

warning_printed_coreas_py = False

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter

def get_angles(corsika):
    """
    Converting angles in corsika coordinates to local coordinates. 
    
    Corsika positiv x-axis points to the magnetic north, NuRadio coordinates positiv x-axis points to the geographic east.
    Corsika positiv y-axis points to the west, and the z-axis upwards. NuRadio coordinates positiv y-axis points to the geographic north, and the z-axis upwards.
    Corsikas zenith angle of a particle trajectory is defined between the particle momentum vector and the negativ z-axis, meaning that
    the particle is described in the direction where it is going to. The azimuthal angle is described between the positive X-axis and the horizontal component of the particle momentum vector (i.e. with respect to North) 
    proceeding counterclockwise. NuRadio describes the particle zenith and azimuthal angle in the direction where the particle is coming from. Therefore 
    the zenith angle is the same, but the azimuthal angle has to be shifted by 180 + 90 degrees. The north has to be shifted by 90 degrees plus difference in geomagetic and magnetic north.
    """
    zenith = np.deg2rad(corsika['inputs'].attrs["THETAP"][0])
    azimuth = hp.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))
    Bx, Bz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(Bz, Bx)

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5 * units.micro * units.tesla
    # in local coordinates north is + 90 deg
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(np.pi * 0.5 + B_inclination, 0 + np.pi * 0.5)
    return zenith, azimuth, magnetic_field_vector

def coreas_observer_to_nuradio_efield(corsika, out_dir=None, save_dict=False):
    """
    converts the electric field from the corsika file to NuRadio units

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    outdir: str
        Directory to save the resulting dictionary. Default is None. Outputname is based on hdf5_file
    save_dict: bool
        Whether to save the resulting dictionary to the specified output file. Default is False.

    Returns
    -------
    efield: np.array (3, n_samples)
        efield with three polarizations (r, theta, phi)
    efield_times: np.array (n_samples)
    """
    n_observer = len(corsika["CoREAS"]['observers'].items())
    group = (corsika["CoREAS"]['observers'])
    keys = list(group.keys())
    n_trace_samples = len(corsika["CoREAS"]['observers'][keys[0]][:,0])

    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)
    
    electric_field = np.zeros((n_observer, n_trace_samples, 4))

    obs_positions = []
    electric_field_on_sky = []
    for j_obs, observer in enumerate(corsika['CoREAS']['observers'].values()):
        #account for different coordinate systems (see get_angles function)
        obs_positions.append(np.array([-observer.attrs['position'][1], observer.attrs['position'][0], 0]) * units.cm)

        efield = np.array([observer[()][:,0]*units.second,
                            -observer[()][:,2]*conversion_fieldstrength_cgs_to_SI,
                            observer[()][:,1]*conversion_fieldstrength_cgs_to_SI,
                            observer[()][:,3]*conversion_fieldstrength_cgs_to_SI])

        efield_geo = cs.transform_from_magnetic_to_geographic(efield[1:,:])
        # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
        efield_on_sky = cs.transform_from_ground_to_onsky(efield_geo)
        # insert time column before efield values
        electric_field_on_sky.append(np.insert(efield_on_sky.T, 0, efield[0,:], axis = 1))
    # shape: (n_observers, n_samples, (time, eR, eTheta, ePhi))
    electric_field[:,:,:] = np.array(electric_field_on_sky)

    efield = observer[1:, :]
    efield_times = observer[0, :]

    if save_dict:
        dic = {}
        dic['energy'] = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
        dic['zenith'] = zenith
        dic['azimuth'] = azimuth
        dic['efield'] = electric_field
        dic['sampling_rates'] =  1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)

        run_nr = corsika['inputs'].attrs['RUNNR']
        outfile = f'coreas_SIM_{run_nr:06}_efield.pickle'

        with open(os.path.join(out_dir, outfile), 'wb') as pickle_out:
            pickle.dump(dic, pickle_out)

    return efield, efield_times

def coreas_observer_to_nuradio_positions(corsika, out_dir=None, save_dict=False):
    """
    Converts observer positions from the corsika file to NuRadio units, on ground and in the shower plane.

    coreas: x-axis pointing to the magnetic north, the positive y-axis to the west, and the z-axis upwards.
    NuRadio: x-axis pointing to the east, the positive y-axis geographical north, and the z-axis upwards.
    NuRadio_x = -coreas_y, NuRadio_y = coreas_x, NuRadio_z = coreas_z and then correct for mag north

    Parameters
    ----------
    corsika : hdf5 file object
    outdir: str
        Directory to save the resulting dictionary. Default is None. Outputname is based on hdf5_file
    save_dict: bool
        Whether to save the resulting dictionary to the specified output file. Default is False.

    Returns
    -------
    obs_positions_geo: np.array (n_observers, 3)
        observer positions in geographic coordinates
    obs_positions_vBvvB: np.array (n_observers, 3)
        observer positions in shower plane coordinates
    
    """
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)
    obs_positions = []
    for j_obs, observer in enumerate(corsika['CoREAS']['observers'].values()):
        obs_positions.append(np.array([-observer.attrs['position'][1], observer.attrs['position'][0], 0]) * units.cm)
    
    obs_positions = np.array(obs_positions)
    # second to last dimension has to be 3 for the transformation
    obs_positions_geo = cs.transform_from_magnetic_to_geographic(obs_positions.T)
    # transforms the coreas observer positions into the vxB, vxvxB shower plane
    obs_positions_vBvvB = cs.transform_to_vxB_vxvxB(obs_positions_geo).T
    
    if save_dict:
        dic = {}
        dic['energy'] = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
        dic['zenith'] = zenith
        dic['azimuth'] = azimuth
        dic['obs_positions_geo'] = obs_positions_geo
        dic['obs_positions_vBvvB'] = obs_positions_vBvvB

        run_nr = corsika['inputs'].attrs['RUNNR']
        outfile = f'coreas_SIM_{run_nr:06}_pos.pickle'

        with open(os.path.join(out_dir, outfile), 'wb') as pickle_out:
            pickle.dump(dic, pickle_out)
    return obs_positions_geo, obs_positions_vBvvB

def calculate_simulation_weights(positions, zenith, azimuth, site='summit', debug=False):
    """
    Calculate weights according to the area that one simulated position in readCoreasStation represents.
    Weights are therefore given in units of area.
    Note: The volume of a 2d convex hull is the area.
    
    Parameters
    ----------
    positions : list
        station position with [x, y, z]
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
    cstrafo = radiotools.coordinatesystems.cstrafo(zenith=zenith, azimuth=azimuth, magnetic_field_vector=None,
                                                   site=site)
    x_trafo_from_shower = cstrafo.transform_from_vxB_vxvxB(station_position=np.array([1, 0, 0]))
    y_trafo_from_shower = cstrafo.transform_from_vxB_vxvxB(station_position=np.array([0, 1, 0]))
    z_trafo_from_shower = cstrafo.transform_from_vxB_vxvxB(station_position=np.array([0, 0, 1]))

    # voronoi has to be calculated in the shower plane due to symmetry reasons
    shower = cstrafo.transform_to_vxB_vxvxB(station_position=positions)
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

        # x_vertice_ground = x_trafo_from_shower[0] * x_vertice_shower + y_trafo_from_shower[0] * y_vertice_shower + z_trafo_from_shower[0] * z_vertice_shower
        # y_vertice_ground = x_trafo_from_shower[1] * x_vertice_shower + y_trafo_from_shower[1] * y_vertice_shower + z_trafo_from_shower[1] * z_vertice_shower
        # z_vertice_ground = x_trafo_from_shower[2] * x_vertice_shower + y_trafo_from_shower[2] * y_vertice_shower + z_trafo_from_shower[2] * z_vertice_shower

        x_vertice_shower = vertices_shower_2d[:, 0]
        y_vertice_shower = vertices_shower_2d[:, 1]
        z_vertice_shower = -(x_trafo_from_shower[2] * x_vertice_shower + y_trafo_from_shower[2] * y_vertice_shower) / z_trafo_from_shower[2]

        vertices_shower_3d = []
        for iter in range(len(x_vertice_shower)):
            vertices_shower_3d.append([x_vertice_shower[iter], y_vertice_shower[iter], z_vertice_shower[iter]])
        vertices_shower_3d = np.array(vertices_shower_3d)
        vertices_ground = cstrafo.transform_from_vxB_vxvxB(station_position=vertices_shower_3d)

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

def make_sim_station(station_id, corsika, observer, channel_ids, fluence=None, weight=None, observer_in_nuradio_units=False):
    """
    creates an NuRadioReco sim station with the same (interpolated) observer object of the coreas hdf5 file
    for all channel ids.

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    observer : hdf5 observer object or 4d array
        in case of observer = None, an electric field with zeros is created
    channel_ids : list or int
        channel_ids for which the efield is to be used if None, an station without electric field is created
    weight : weight of individual station
        weight corresponds to area covered by station
    observer_in_nuradio_units : bool
        indicates if observer is in nuradio units or coreas units
        in case of observer_in_nuradio_units, observer has to be a 4d array with 
        efield_times = observer[0, n_trace_samples]
        efield_r = observer[1, n_trace_samples]
        efield_theta = observer[2, n_trace_samples]
        efield_phi = observer[3, n_trace_samples]

    Returns
    -------
    efield: np.array (3, n_samples)
        efield with three polarizations (r, theta, phi)
    efield_times: np.array (n_samples)
    """
    zenith, azimuth, magnetic_field_vector = get_angles(corsika)

    if(observer is None):
        observer = np.zeros((512, 4))
        observer[:, 0] = np.arange(0, 512) * units.ns / units.second
        efield = observer[:, 1:]
        efield_times = observer[:, 0]

    elif observer_in_nuradio_units==False and observer is not None:
        # assume coreas observers as input e.g. from readCoreasStation or readCoreasShower
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
        efield_times = data[:, 0]

        # prepend trace with zeros to not have the pulse directly at the start
        n_samples_prepend = efield.shape[1]
        efield2 = np.zeros((3, n_samples_prepend + efield.shape[1]))
        efield2[0] = np.append(np.zeros(n_samples_prepend), efield[0])
        efield2[1] = np.append(np.zeros(n_samples_prepend), efield[1])
        efield2[2] = np.append(np.zeros(n_samples_prepend), efield[2])

    elif observer_in_nuradio_units and observer is not None:
        efield = observer[1:, :]
        efield_times = observer[0, :]
    
    sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
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
        if(not warning_printed_coreas_py):
            logger.warning("No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    sim_station.set_is_cosmic_ray()
    sim_station.set_simulation_weight(weight)
    if channel_ids is not None:
        if type(channel_ids) is not list:
            channel_ids = [channel_ids]
        electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids)
        electric_field.set_trace(efield2, sampling_rate)
        electric_field.set_trace_start_time(efield_times[0])
        electric_field.set_parameter(efp.ray_path_type, 'direct')
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        if fluence is not None:
            electric_field.set_parameter(efp.signal_energy_fluence, fluence)
        sim_station.add_electric_field(electric_field)
   
    return sim_station

def add_electric_field(sim_station, channel_ids, efield, efield_times, corsika, fluence=None):
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
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file

    """
    if type(channel_ids) is not list:
        channel_ids = [channel_ids]

    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids)
    electric_field.set_trace(efield, sampling_rate)
    electric_field.set_trace_start_time(efield_times[0])
    electric_field.set_parameter(efp.ray_path_type, 'direct')
    electric_field.set_parameter(efp.zenith, zenith)
    electric_field.set_parameter(efp.azimuth, azimuth)
    if fluence is not None:
        electric_field.set_parameter(efp.signal_energy_fluence, fluence)
    sim_station.add_electric_field(electric_field)

def make_sim_shower(corsika, observer=None, detector=None, station_id=None):
    """
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
        global warning_printed_coreas_py
        if(not warning_printed_coreas_py):
            logger.warning("No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True

    return sim_shower
