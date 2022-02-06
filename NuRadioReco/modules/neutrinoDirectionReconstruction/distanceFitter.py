from NuRadioReco.utilities import geometryUtilities as geo_utl
import scipy.optimize as opt
import numpy as np
from radiotools import helper as hp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
from NuRadioReco.framework.parameters import showerParameters as shp
import matplotlib.pyplot as plt
import pickle
import NuRadioReco.utilities.io_utilities


class distanceFitter:
    " Fits the direction using plane wave fit to channels "

    def __init__(self):
        pass


    def begin(self, det, lookup_table_path, channel_ids = [0, 3, 9, 10], template = None):
        self.__channel_ids = channel_ids
        self.__lookup_table_location = lookup_table_path
        self.__header = {}
        self.__detector = det
        pass


    def run(self, evt, station, det, template = None, debug = True, debugplots_path = None, fixed_depth = None, method = 'raytracing' ):
    
        """
        Reconstruct the vertex position of the neutrino
        
        Parameters
        ----------
        evt, station, det
            Event, Station, Detector
        template: array of floats
            neutrino voltage template
        debug: Boolean
            if True, debub plots are created
        debugplots_path: str
            path to store debug plots
        fixed_depth: float
            if argument is passed,  the vertex position is determined at the fixed depth
        method: 'raytracing' or 'look-up'
            raytracing uses the analytic raytracing module to calculate time-delays. 'look-up' uses Christophs look-up tables to determine the time-delays
        """
        
        
        for channel in station.get_sim_station().iter_channels():
            if channel.get_id() == self.__channel_ids[0]:
                print("		channel id", channel.get_id())
                shower_id = channel.get_shower_id()


        receive_pickle, launch_pickle, solution_pickle, zenith_vertex_pickle = pickle.load(open('/lustre/fs22/group/radio/plaisier/software/simulations/planeWaveFit/receive_launch.pkl', 'rb'))

        ice = medium.get_ice_model('greenland_simple')
        prop = propagation.get_propagation_module('analytic')

        self.__channel_pairs = []
        self.__relative_positions = []
        station_id = station.get_id()
        self.__station_id = station_id
        for i in range(len(self.__channel_ids) - 1):
            for j in range(i + 1, len(self.__channel_ids)):
                relative_positions = det.get_relative_position(station_id, self.__channel_ids[i]) - det.get_relative_position(station_id, self.__channel_ids[j])
                self.__relative_positions.append(relative_positions)

                self.__channel_pairs.append([self.__channel_ids[i], self.__channel_ids[j]])
        self.__lookup_table = {}
        for channel_id in self.__channel_ids:
            channel_z = abs(det.get_relative_position(station_id, channel_id)[2])
            if channel_z not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[int(channel_z)] = f['header']
                self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]

        self.__sampling_rate = station.get_channel(0).get_sampling_rate()
        self.__template = template

        if debug:
            fig, axs = plt.subplots( len(self.__channel_pairs), 2, figsize = (10, len(self.__channel_pairs)*2))
 
        def get_signal_travel_time(d_hor, z, ray_type, channel_id):
            """
            Calculate the signal travel time between a position and the
            channel
            Parameters:
            ------------
            d_hor, z: numbers or arrays of numbers
                Coordinates of the point from which to calculate the
                signal travel times. Correspond to (r, z) coordinates
                in cylindrical coordinates.
            ray_type: string
                Ray type for which to calculate the travel times. Options
                are direct, reflected and refracted
            channel_id: int
                ID of the channel to which the travel time shall be calculated
            """
            channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
            channel_type = int(abs(channel_pos[2]))
            travel_times = np.zeros_like(d_hor)
            mask = np.ones_like(travel_times).astype(bool)
            i_z = np.array(np.round((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
            i_x_1 = np.array(np.floor((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
            cell_dist_1 = i_x_1 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
            mask[i_x_1 > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
            mask[i_z > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
            i_x_1[~mask] = 0
            i_z[~mask] = 0
            travel_times_1 = self.__lookup_table[channel_type][ray_type][(i_x_1, i_z)]
            i_x_2 = np.array(np.ceil((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
            cell_dist_2 = i_x_2 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
            i_x_2[~mask] = 0
            travel_times_2 = self.__lookup_table[channel_type][ray_type][(i_x_2, i_z)]
            slopes = np.zeros_like(travel_times_1)
            slopes[i_x_2 > i_x_1] = (travel_times_1 - travel_times_2)[i_x_2 > i_x_1] / (cell_dist_1 - cell_dist_2)[i_x_2 > i_x_1]

            travel_times = (d_hor - cell_dist_1) * slopes + travel_times_1

            if travel_times_1 == False: 
                travel_times = False
           
            return travel_times_1
        

        def likelihood(vertex, sim = False, rec = False, minimize = True):

            timings = np.zeros((len(self.__channel_ids), 3))
            solutiontype = np.zeros((len(self.__channel_ids), 2))
            for i_ch, channel_id in enumerate(self.__channel_ids):
       
                    x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
                    if method == 'look-up':
                      
                        for i_type, ray_type in enumerate(['direct', 'refracted', 'reflected']):
                            d_hor = np.sqrt((vertex[0] - x2[0])**2 + (vertex[1] - x2[1])**2)
                            timings[i_ch, i_type] = get_signal_travel_time(d_hor, vertex[2], ray_type, channel_id)


                    if method == 'raytracing':
                       
                        r = prop( ice, 'GL1')
                        r.set_start_and_end_point(vertex, x2)

                        r.find_solutions()
                        for iS in range(r.get_number_of_solutions()):
                            d_hor = np.sqrt((vertex[0] - x2[0])**2 + (vertex[1] - x2[1])**2)
                            i_ray = r.get_solution_type(iS)
                            i_type = i_ray - 1

                            timings[i_ch, i_type] = r.get_travel_time(iS)
                            
            corr = 0
            corrs = []
           
            for ich, ch_pair in enumerate(self.__channel_pairs):
                index_1 = self.__channel_ids.index(ch_pair[0])
                index_2 = self.__channel_ids.index(ch_pair[1])
               
                k = 0
                for t1 in [0,1,2]:# for each ray type
                    for t2 in [0,1,2]:# for each ray type
                
                        if timings[index_1,t1]:#if solution type exist
                            if timings[index_2,t2]: #if solution type exist
                                tmp = timings[index_2,t2 ] - timings[index_1, t1 ]## calculate timing
       
                                n_samples = tmp * self.__sampling_rate
                 
                                pos = int(len(self.__correlation[ich]) / 2 - n_samples)
                                corr += self.__correlation[ich, pos]
                                corrs.append(self.__correlation[ich, pos])
                                if sim and debug:
                                
                                    if not k:
                                        axs[ich, 1].plot(station.get_channel(ch_pair[0]).get_times(), station.get_channel(ch_pair[0]).get_trace())
                                        axs[ich, 1].plot(station.get_channel(ch_pair[1]).get_times(), station.get_channel(ch_pair[1]).get_trace())
                                        axs[ich,1].title.set_text("channel pair {}".format( ch_pair))
                                        axs[ ich, 0].plot(self.__correlation[ich], color = 'blue')
                                        axs[ich, 0].axvline(pos, color = 'green', lw = 1, label = 'simulation')
                                        axs[ich, 0].axvline(int(len(self.__correlation[ich]) / 2), lw = 2, color = 'black', label = 'len(corr)/2')
                                    axs[ich, 0].legend()
                    
                                    axs[ich, 0].axvline(pos, color = 'green', lw = 1)
                                if rec and debug:
                   
                                    axs[ich, 0].set_ylim((0, max(self.__correlation[ich])))
         
                                    axs[ich, 0].axvline(pos, linestyle = '--',color = 'red', lw = 1)
         
                                    if not k:
                                      
                                        axs[ich, 0].axvline(pos, ls = '--',color = 'red', lw = 1, label = 'reconstruction')
                                        axs[ich, 0].legend()
                                        axs[ich, 0].grid()
                                        axs[ich, 1].grid()
                                k += 1
            if rec and debug:         
                 fig.tight_layout()
                 fig.savefig("{}/corr_vertex.pdf".format(debugplots_path))

            likelihood = corr
            if not minimize:
                return corrs
            return -1*likelihood


        trace = np.copy(station.get_channel(self.__channel_pairs[0][0]).get_trace())
        self.__correlation = np.zeros((len(self.__channel_pairs), len(hp.get_normalized_xcorr(trace, self.__template))) )
        for ich, ch_pair in enumerate(self.__channel_pairs):
            trace1 = np.copy(station.get_channel(self.__channel_pairs[ich][0]).get_trace())
            trace2 =np.copy(station.get_channel(self.__channel_pairs[ich][1]).get_trace())
        
            corr_1 = hp.get_normalized_xcorr(trace1, self.__template)
            corr_2 = hp.get_normalized_xcorr(trace2, self.__template)
 
            sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
            toffset = sample_shifts / station.get_channel(0).get_sampling_rate()
            for i_shift, shift_sample in enumerate(sample_shifts):
                if (np.isnan(corr_2).any()):
                    self.__correlation[ich, i_shift] = 0
                elif (np.isnan(corr_1).any()):
                    self.__correlation[ich, i_shift] = 0

                else:
                    self.__correlation[ich, i_shift] = np.max(corr_1 * np.roll(corr_2, shift_sample)) 

        #### get receive zenith from planewave
        receive_zenith = station[stnp.planewave_zenith]
	#### translate receive zenith to launch vector
        zenith_vertex = zenith_vertex_pickle[np.argmin(abs(np.rad2deg(receive_pickle) - np.rad2deg(receive_zenith)))] ## deze aanpassen aan diepte #full distance inladen.
        
        if station.has_sim_station():
            #print("	simulated corrs", likelihood(evt.get_sim_shower(shower_id)[shp.vertex], minimize = False))

            print("	minimization value for simulated values:", likelihood(evt.get_sim_shower(shower_id)[shp.vertex], sim = True))
            print("	simulated vertex:", evt.get_sim_shower(shower_id)[shp.vertex])
            print("	azimuth angle of sim vertex:", np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[1]))
            print("	zenith angle of sim vertex:", np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[0]))
         
       
        
        print("		reconstructed planewave azimuth:", np.rad2deg(station[stnp.planewave_azimuth]))
        print("		reconstructed planewave zenith:", np.rad2deg(station[stnp.planewave_zenith]))

        print("		fixed depth", fixed_depth)
        range_vertices = []
        depths = np.arange(200, 2500, 10)#
        if fixed_depth:
            depths = [fixed_depth]
        for depth in depths:
            diff_tmp = np.inf
            if depth > 400:
                delta = .2
                diff = 4
                diff_az = 2
            else: 
                diff = 10
                delta = .2
        
            for zen in np.arange(zenith_vertex -diff, zenith_vertex + diff, delta):
                azimuth_tmp = station[stnp.planewave_azimuth]
                for az in np.arange(np.rad2deg(azimuth_tmp) - diff, np.rad2deg(azimuth_tmp) + diff, delta):
                
                    cart = hp.spherical_to_cartesian(np.deg2rad(zen), np.deg2rad(az))
                    R = -1*depth/cart[2]
                    x1 = cart * R
                    if (np.sqrt(x1[0]**2 + x1[1]**2 + x1[2]**2) < 4000):
                
                        range_vertices.append(x1)


        
 	#### get for a series of depth the vertex position that corresopnds to receive zenith
        
        likelihood_values = []
        for vertex in range_vertices:
        #        print("reconstruction for vertex", vertex)
            likelihood_values.append(likelihood(vertex))
        #if debug:
           # fig1 =  plt.figure()
           # ax1 = fig1.add_subplot(111)
           # ax1.plot(np.array(range_vertices)[:,2], likelihood_values, 'o', markersize = 3, color = 'blue')
            #ax1.plot(np.array(range_vertices)[:,2], likelihood_values, color = 'blue')
           # ax1.set_xlabel("vertex z [m]")
           # ax1.set_ylabel("minimization value")
           # ax1.axvline(evt.get_sim_shower(shower_id)[shp.vertex][2], label = 'simulated', color = 'green')
           # ax1.axvline(range_vertices[np.argmin(likelihood_values)][2], label = 'reconstructed depth', color = 'red')
           # ax1.axhline(likelihood(evt.get_sim_shower(shower_id)[shp.vertex]), color = 'green')
           # ax1.legend()
           # fig1.tight_layout()
           # fig1.savefig("{}/vertex_likelihood.pdf".format(debugplots_path))
     
        if 0:#debug:
            fig1 =  plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(np.array(range_vertices)[:,2], likelihood_values, 'o', markersize = 3, color = 'blue')
            
            ax1.pcolor(xx, yy, zz)#ax1.plot(np.array(range_vertices)[:,2], likelihood_values, color = 'blue')
            ax1.set_xlabel("vertex z [m]")
            ax1.set_ylabel("minimization value")
            ax1.axvline(evt.get_sim_shower(shower_id)[shp.vertex][2], label = 'simulated', color = 'green')
            ax1.axvline(range_vertices[np.argmin(likelihood_values)][2], label = 'reconstructed depth', color = 'red')
            ax1.axhline(likelihood(evt.get_sim_shower(shower_id)[shp.vertex], sim = True), color = 'green')
            ax1.legend()
            fig1.savefig("{}/vertex_map.pdf".format(debugplots))

        station[stnp.nu_vertex] = range_vertices[np.argmin(likelihood_values)]
        print("		reconstructed vertex", station[stnp.nu_vertex])
        print("		reconstructed value", likelihood(station[stnp.nu_vertex], rec = True))
        #print("reconstructed corrs", likelihood(station[stnp.nu_vertex], minimize = False))
        print("		len reconstructed corrs", len(likelihood(station[stnp.nu_vertex], minimize = False)))
  
    def end(self):
            pass


