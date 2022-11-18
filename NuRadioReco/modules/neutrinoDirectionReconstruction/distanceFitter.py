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
from NuRadioReco.utilities import units

class distanceFitter:
    " Fits the direction using plane wave fit to channels "

    def __init__(self):
        pass


    def begin(self, det, lookup_table_path, channel_ids = [0, 3, 9, 10], template = None, zenith_table_path='./receive_launch.pkl'):
        self.__channel_ids = channel_ids
        self.__lookup_table_location = lookup_table_path
        self.__header = {}
        self.__detector = det
        self.__template = template
        self.__zenith_table_path = zenith_table_path

        pass


    def run(self, evt, station, det, debug = True, debugplots_path = None, fixed_depth = None, method = 'raytracing' ):

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

        if station.has_sim_station():
            for channel in station.get_sim_station().get_channels_by_channel_id(self.__channel_ids[0]):
                print("		channel id", channel.get_id())
                shower_id = channel.get_shower_id()


        receive_pickle, launch_pickle, solution_pickle, zenith_vertex_pickle = NuRadioReco.utilities.io_utilities.read_pickle(
            self.__zenith_table_path)

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
        if self.__lookup_table_location is not None:
            self.__lookup_table = {}
            for channel_id in self.__channel_ids:
                channel_z = abs(det.get_relative_position(station_id, channel_id)[2])
                if channel_z not in self.__lookup_table.keys():
                    f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                    self.__header[int(channel_z)] = f['header']
                    self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]

        self.__sampling_rate = station.get_channel(0).get_sampling_rate()

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


        def f(x, popt):
            return x*popt[0] + popt[1]


        def likelihood(param, sim = False, rec = False, minimize = True):
            zen, az, depth = param
           # print("zenith...", zen)
           # print("depth", depth)
            if depth  >= 600:
                popt = [0,0]
            if (depth >= 500)&(depth < 600):
                popt = [-0.05557555,  4.93865018]
            if (depth < 500)&(depth >= 400):
                popt = [-0.09456427, 8.18511759]
            if (depth < 400)&(depth >= 300):
                popt = [-0.15659163, 13.31646513]
            if (depth < 300)&(depth >= 200):
                popt = [-0.28464472, 23.55759437]
            if (depth < 200):
                popt = [-0.62995541, 49.91834848]
        
            offset = f(np.rad2deg(receive_zenith), popt)
            zen = zen# - offset
            cart = hp.spherical_to_cartesian(np.deg2rad(zen), np.deg2rad(az))
            R = -1*depth/cart[2]
            x1 = cart * R
            vertex = x1
            #print("depth {}, offset {}".format(depth, offset))
            #print("sim: {}, zenith vertex rec...{}, offset {}".format(np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[0]), np.rad2deg(hp.cartesian_to_spherical(*vertex)[0]), offset))
            if (np.sqrt(x1[0]**2 + x1[1]**2 + x1[2]**2) > 4000):
                if minimize:
                    return np.inf
             #   if not minimize:
                    

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
                index_1 = list(self.__channel_ids).index(ch_pair[0])
                index_2 = list(self.__channel_ids).index(ch_pair[1])

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
            if corr > 0:
                return -1*likelihood#/len(corrs)
            else:
                return 0


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
            zen_sim, az_sim = hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])
            depth_sim  = evt.get_sim_shower(shower_id)[shp.vertex][2] 
            print("	minimization value for simulated values:", likelihood([zen_sim, az_sim, depth_sim], sim = True))
            print("	simulated vertex:", evt.get_sim_shower(shower_id)[shp.vertex])
            print("	azimuth angle of sim vertex:", np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[1]))
            print("	zenith angle of sim vertex:", np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[0]))
            print("	zenith from pkl file:", zenith_vertex)


        print("		reconstructed planewave azimuth:", np.rad2deg(station[stnp.planewave_azimuth]))
        print("		reconstructed planewave zenith:", np.rad2deg(station[stnp.planewave_zenith]))

        print("		fixed depth", fixed_depth)
        

        print("start minimization....")
        delta_angle = 10
        diff_angle = .2
        diff_depth = 10
        zen_start, zen_end = zenith_vertex - delta_angle, zenith_vertex + delta_angle 
        az_start, az_end = np.rad2deg(station[stnp.planewave_azimuth]) - delta_angle, np.rad2deg(station[stnp.planewave_azimuth]) + delta_angle
        depth_start = 200#200
        depth_end = 2500

        ll, fval, xgrid, fgrid = opt.brute(
	likelihood, ranges=(
               slice(zen_start, zen_end, diff_angle),
               slice(az_start, az_end, diff_angle),
	slice(depth_start, depth_end,diff_depth)
           ) , full_output = True)#finish = opt.fmin,  full_output=True)       




        if debug:
            print("creating debug plot for vertex reconstruction....")
            extent = (
                zen_start,#xgrid[0,0,0, 0], #x0
                zen_end,#xgrid[0,-1, 0,0], #x1
                az_start,#xgrid[1,0,0,0], #y0
                az_end#xgrid[1,0,0,-1], #y1
            )

            fig1 = plt.figure()
            depths = np.arange(depth_start, depth_end, diff_depth)
            mask1 = (xgrid[2] == depths[np.argmin(abs(ll[2]-depths))])
            Lvalue = fgrid[mask1]#mask = np.ma.masked_where(xgrid[2] == ll[2], xgrid[1], copy=True)#(extent[2] == ll[2])
            plt.imshow(Lvalue.reshape(len(np.arange(zen_start, zen_end,diff_angle)), len(np.arange(az_start, az_end, diff_angle))), aspect = 'auto',extent = extent,cmap = 'viridis', interpolation = 'nearest', vmax = 0)#, aspect='auto', origin='lower')
            plt.xlabel(r"vertex zenith $[^{\circ}]$")
            plt.ylabel(r"vertex azimuth $[^{\circ}]$")
            plt.axhline(np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[1]), color = 'orange')
            plt.axvline(np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[0]), color = 'orange', label = 'simulated values')
            plt.axhline(ll[1], color = 'lightblue')
            plt.axvline(ll[0], color = 'lightblue', label = 'reconstructed values')
            cbar = plt.colorbar()
            cbar.set_label('minimization value', rotation=270, labelpad = +20)

            plt.legend()
            fig1.tight_layout()

            fig2 = plt.figure()
            extent = (
                zen_start, 
                zen_end, 
                -depth_end, 
                -depth_start, 
            )
            azs = np.arange(az_start, az_end, diff_angle)
            mask1 = (np.round(xgrid[1],3) == np.round(azs[np.argmin(abs(ll[1]-azs))],3))
            Lvalue = fgrid[mask1]#mask = np.ma.masked_where(xgrid[2] == ll[2], xgrid[1], copy=True)#(extent[2] == ll[2])
            plt.imshow(Lvalue.reshape(len(np.arange(zen_start, zen_end,diff_angle)), len(depths)), aspect = 'auto',extent = extent,cmap = 'viridis', interpolation = 'nearest', vmax = 0)#, aspect='auto', origin='lower')
            plt.xlabel(r"vertex zenith $[^{\circ}]$")
            plt.ylabel(r"depth [m]")
            plt.axhline(evt.get_sim_shower(shower_id)[shp.vertex][2], color = 'orange')
            plt.axvline(np.rad2deg(hp.cartesian_to_spherical(*evt.get_sim_shower(shower_id)[shp.vertex])[0]), color = 'orange', label = 'simulated values')
            plt.axhline(-1*ll[2], color = 'lightblue')
            plt.axvline(ll[0], color = 'lightblue', label = 'reconstructed values')
            cbar = plt.colorbar()
            cbar.set_label('minimization value', rotation=270, labelpad = +20)

            plt.legend()
            fig2.tight_layout()


            if debugplots_path != None:
                fig1.savefig("{}/vertex_map_zen_az.png".format(debugplots_path))          
                fig2.savefig("{}/vertex_map_zen_depth.png".format(debugplots_path))   


        zen, az, depth = ll
        cart = hp.spherical_to_cartesian(np.deg2rad(zen), np.deg2rad(az))
        R = -1*depth/cart[2]
        station[stnp.nu_vertex] = cart * R

        print("		reconstructed vertex", station[stnp.nu_vertex])
        print("		simulated vertex", evt.get_sim_shower(shower_id)[shp.vertex])
        print("		reconstructed value", likelihood(ll, rec = True))
        print("		len reconstructed corrs", len(likelihood(ll, minimize = False)))

    def end(self):
            pass


