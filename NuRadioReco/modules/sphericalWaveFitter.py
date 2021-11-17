from NuRadioReco.utilities import geometryUtilities as geo_utl
import scipy.optimize as opt
import numpy as np
from radiotools import helper as hp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import scipy.signal
from scipy import constants
from mpl_toolkits.axes_grid1 import make_axes_locatable


class sphericalWaveFitter:
    " Fits position x,y, z of a source using spherical fit to channels "
    
    def __init__(self):
        pass
        
        
    def begin(self, channel_ids = [0, 3, 9, 10]):
        self.__channel_ids = channel_ids
        pass


    def run(self, evt, station, det, start_pulser_position, n_index = None, debug = True):
       
        print("channels used for this reconstruction:", self.__channel_ids)

        
        def get_distance(x, y):
            return np.sqrt((x[0] - y[0])**2+ (x[1] - y[1])**2 + (x[2] - y[2])**2)
        
        
        def get_time_delay_spherical_wave(position, ch_pair, n=n_index):
            T0 = get_distance(position, det.get_relative_position(station_id, ch_pair[0]))/(constants.c/n_index)*units.s
            T1 = get_distance(position , det.get_relative_position(station_id, ch_pair[1]))/(constants.c/n_index)*units.s
            return T1 - T0


        self.__channel_pairs = []
        self.__relative_positions = []
        station_id = station.get_id()
        for i in range(len(self.__channel_ids) - 1):
            for j in range(i + 1, len(self.__channel_ids)):
                relative_positions = det.get_relative_position(station_id, self.__channel_ids[i]) - det.get_relative_position(station_id, self.__channel_ids[j])
                self.__relative_positions.append(relative_positions)
                
                self.__channel_pairs.append([self.__channel_ids[i], self.__channel_ids[j]])
                
       
        self.__sampling_rate = station.get_channel(0).get_sampling_rate()
        if debug:
            fig, ax = plt.subplots( len(self.__channel_pairs), 2)


        def likelihood(pulser_position, x = None, y = None, debug_corr = False):
            if len(pulser_position) == 1:
                pulser_position = [x, y, pulser_position[0]]
            corr = 0
            if debug_corr:
                fig = plt.figure(figsize= (5, 1*len(self.__channel_pairs)))
            for ich, ch_pair in enumerate(self.__channel_pairs):
                positions = self.__relative_positions[ich]
                times = []
                tmp = -1*get_time_delay_spherical_wave(pulser_position, ch_pair, n=n_index)
                n_samples = -1*tmp * self.__sampling_rate
                pos = int(len(self.__correlation[ich]) / 2 - n_samples)
                corr += self.__correlation[ich, pos]
                if debug_corr:
                    ax = fig.add_subplot(len(self.__channel_pairs), 1, ich+1)
                    ax.plot(self.__correlation[ich])
                    ax.set_ylim((0, max(self.__correlation[ich])))
                    ax.axvline(pos, label = 'reconstruction', lw = 1, color = 'orange')
                    ax.axvline(self._pos_starting[ich], label = 'starting pos', lw = 1, color = 'green')
                    ax.set_title("channel pair {}".format( ch_pair), fontsize = 5) 
                    ax.legend(fontsize = 5)
 
            if debug_corr:
                fig.tight_layout()
                fig.savefig("debug.pdf")

            return -1*corr
            
            

        trace = np.copy(station.get_channel(self.__channel_pairs[0][0]).get_trace())
        self.__correlation = np.zeros((len(self.__channel_pairs), len(np.abs(scipy.signal.correlate(trace, trace))) ))

        for ich, ch_pair in enumerate(self.__channel_pairs):
            trace1 = np.copy(station.get_channel(self.__channel_pairs[ich][0]).get_trace())
            trace2 = np.copy(station.get_channel(self.__channel_pairs[ich][1]).get_trace())
       
            t_max1 = station.get_channel(self.__channel_pairs[ich][0]).get_times()[np.argmax(np.abs(trace1))]
            t_max2 = station.get_channel(self.__channel_pairs[ich][1]).get_times()[np.argmax(np.abs(trace2))]
            corr_range = 50 * units.ns
            snr1 = np.max(np.abs(station.get_channel(self.__channel_pairs[ich][0]).get_trace()))
            snr2 = np.max(np.abs(station.get_channel(self.__channel_pairs[ich][1]).get_trace()))
            if snr1 > snr2:
                trace1[np.abs(station.get_channel(self.__channel_pairs[ich][0]).get_times() - t_max1) > corr_range] = 0
            else:
                trace2[np.abs(station.get_channel(self.__channel_pairs[ich][1]).get_times() - t_max2) > corr_range] = 0
            self.__correlation[ich] = np.abs(scipy.signal.correlate(trace1, trace2))
          


        #### set positions for starting position ####
        self._pos_starting = np.zeros(len(self.__channel_pairs))
        for ich, ch_pair in enumerate(self.__channel_pairs):
            positions = self.__relative_positions[ich]
            times = []

            tmp = get_time_delay_spherical_wave(start_pulser_position, ch_pair, n=n_index)
            n_samples = tmp * self.__sampling_rate
            self._pos_starting[ich] = int(len(self.__correlation[ich]) / 2 - n_samples)


        #method = 'Nelder-Mead'
        x_start, y_start, z_start = start_pulser_position
        dx, dy, dz = [.1, .1, .1]
        #ll = opt.minimize(likelihood, x0 = (start_pulser_position[0]-10, start_pulser_position[1], start_pulser_position[2]),method = method)#
        ll = opt.brute(likelihood, ranges=(slice(x_start - 2, x_start + 2, dx), slice(y_start - 2, y_start + 2, dy), slice(z_start - 2, z_start + 2,dz)), finish = opt.fmin)
        print("start position: {}".format(start_pulser_position))
        print("reconstructed position: {}".format([ll[0], ll[1], ll[2]]))

        if debug:
            
            method = 'Nelder-Mead'
            x = np.arange(x_start -3, x_start +3, dx)
            y = np.arange(y_start - 3, y_start +3, dy)
            z = np.arange(z_start - 2, z_start + 2, dz)
            xx, yy = np.meshgrid(x, y)
            zz = np.zeros((len(x), len(y)))
            zz_values = np.zeros((len(x), len(y)))
            for ix, x_i in enumerate(x):
                for iy, y_i in enumerate(y):
                    c = opt.minimize(likelihood, x0 = (start_pulser_position[2]), args = (x_i, y_i, False), method = method)
                    zz[ix, iy] = c.fun
                    zz_values[ix, iy] = c.x[0]
            
            fig = plt.figure(figsize = (10, 5))
            ax1 = fig.add_subplot(121)
            pax1 = ax1.pcolor(xx, yy,  zz.T)
            ymin = np.matrix(zz).argmin()
            z_min = opt.minimize(likelihood, x0 = (start_pulser_position[2]), args = (xx.T[np.unravel_index(ymin, (len(x), len(y)))], yy.T[np.unravel_index(ymin, (len(x), len(y)))], False), method = method)
            likelihood([xx.T[np.unravel_index(ymin, (len(x), len(y)))], yy.T[np.unravel_index(ymin, (len(x), len(y)))], z_min.x[0]], debug_corr = True)# debug plot
            print("z position for minimum grid: {}".format(z_min.x[0]))
            ax1.axhline(start_pulser_position[1], color = 'orange', label = 'starting position')
            ax1.axvline(start_pulser_position[0], color = 'orange')
            ax1.axhline(yy.T[np.unravel_index(ymin, (len(x), len(y)))], color = 'grey', label = 'minimum grid')
            ax1.axvline(xx.T[np.unravel_index(ymin, (len(x), len(y)))], color = 'grey')
            ax1.set_title("z starting position {}, z at minimum: {}".format(start_pulser_position[2], np.round(z_min.x[0],2)))
            ax1.set_xlabel("x position [m]")
            ax1.set_ylabel("y  position [m]")
            ax1.legend()
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel("x position [m]")
            pax2 = ax2.pcolor(xx, yy, zz_values.T)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(pax1, cax=cax)
            cbar.set_label("minimization value", rotation=270, labelpad = 15)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(pax2, cax = cax)
            cbar.set_label("best fitting depth [m]", rotation = 270, labelpad = 15)
            fig.tight_layout()
            fig.savefig("minimization_map.pdf")


    def end(self):
        pass
        
