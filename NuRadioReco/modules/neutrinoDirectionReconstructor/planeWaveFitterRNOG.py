from NuRadioReco.utilities import geometryUtilities as geo_utl
import scipy.optimize as opt
import numpy as np
from radiotools import helper as hp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
from NuRadioReco.framework.parameters import showerParameters as shp
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
import scipy.signal
from NuRadioReco.framework.parameters import channelParameters as chp

class planeWaveFitterRNOG:
    " Fits the direction using plane wave fit to channels "

    def __init__(self):

        pass


    def begin(self, det, channel_ids = [0, 3, 9, 10], template = None):
        """
        Set the channel ids to be used in the plane wave fit.
        """
        self.__channel_ids = channel_ids
        pass


    def run(
        self, evt, station, det, n_index = 1.,
        template = None, mode = 'add', full_output=False,
        zenith_min=0, zenith_max=180*units.deg, zenith_step=0.01,
        azimuth_min=0, azimuth_max=360*units.deg, azimuth_step=0.01,
        debug = True, debugplots_path = './'):
        """
        Run the plane wave fitter

        Parameters
        ----------
        evt: Event
        station: station
            The station to use for reconstruction
        det: Detector object
            The detector that specifies the channel positions and cable delays
        n_index: float (default: 1.)
            The average index of refraction
        template: np.array or None
            If given, the timing difference is determined by correlation with
            the template instead of pair-wise channel correlation.
        mode: string
            How to maximize the channel-channel or channel-template
            correlations. Options:

            * 'add' (default): maximize the sum of all correlations
            * 'add_normalize': same as 'add', but normalize all traces first
            * 'add_normalize_correlation': same as 'add', but normalize
              the maximum correlation for each channel pair
            * 'multiply': maximize the product of all correlations
            * 'log': maximize the log of the product of all correlations

        full_output: bool, default False
            If True, also return the brute-force meshgrid and the function
            values at all points. Useful for debugging or fitting multiple events

        Returns
        -------
        (zenith_fit, azimuth_fit): tuple of floats
            The fit result
        xgrid: np.ndarray
            (Only if full_output=True)
            the brute-force meshgrid
        fgrid: np.ndarray
            (Only if full_output=True)
            the function values at each point of xgrid


        Other Parameters
        ----------------
        zenith_min: float (default: 0)
        zenith_max: float (default: 180*units.deg)
        zenith_step: float (default: 0.01)
        azimuth_min: float (default: 0)
        azimuth_max: float (default: 360*units.deg)
        azimuth_step: float (default: 0.01)
        debug: bool, default True
            If True, produce some debug plots and save them under
            debugplots_path
        debugplots_path: string
            Path to existing directory to save debug plots. Default: './'

        """
        if station.has_sim_station():
            for channel in station.iter_channels():
                if channel.get_id() == 0:
                    signal_zenith = channel[chp.signal_receiving_zeniths]
                    signal_azimuth = channel[chp.signal_receiving_azimuths]
        else:
            signal_zenith = np.nan
            signal_azimuth = np.nan
            print('No simulation available')


        print("channels used for this reconstruction:", self.__channel_ids)


        self.__channel_pairs = []
        self.__relative_positions = []
        self.__relative_delays = []
        station_id = station.get_id()
        for i in range(len(self.__channel_ids) - 1):
            for j in range(i + 1, len(self.__channel_ids)):
                id1, id2 = self.__channel_ids[i], self.__channel_ids[j]
                relative_positions = det.get_relative_position(station_id, id1) - det.get_relative_position(station_id, id2)
                self.__relative_positions.append(relative_positions)
                self.__relative_delays.append(
                    station.get_channel(id1).get_trace_start_time()
                    - station.get_channel(id2).get_trace_start_time()
                )
                self.__channel_pairs.append([self.__channel_ids[i], self.__channel_ids[j]])


        self.__sampling_rate = station.get_channel(0).get_sampling_rate()
        self.__template = template

        if debug:
            fig, ax = plt.subplots( len(self.__channel_pairs), 2, figsize = (10, 1.5*len(self.__channel_pairs)+2), sharex='col')
            fig.subplots_adjust(hspace=0)


        def likelihood(angles, mode='add', sim = False, rec = False):#, debug = False):#, station):
            zenith, azimuth = angles
            if 'add' in mode:
                corr = 0
            else:
                corr = 1

            for ich, ch_pair in enumerate(self.__channel_pairs):
                positions = self.__relative_positions[ich]
                relative_delay = self.__relative_delays[ich]

                tmp = geo_utl.get_time_delay_from_direction(zenith, azimuth, positions, n=n_index)#,
                tmp -= relative_delay
                n_samples = -1*tmp * self.__sampling_rate

                pos = int(len(self.__correlation[ich]) / 2 - n_samples)
                if 'add' in mode:
                    corr += self.__correlation[ich, pos]
                else:
                    corr *= self.__correlation[ich, pos]

                if sim:
                    ax[ ich, 0].plot(self.__correlation[ich], color = 'blue')
                    ax[ich, 0].axvline(pos, alpha = .5, color = 'orange', lw = 1, label = 'sim')#self.__correlation[ich, pos])
                    #print("correlation[ich]", self.__correlation[ich, pos])
                    #ax[ich, 0].set_xlim((5000, 10000))
                    #ax[ich,0].legend()
                    #ax[ich, 0].set_title("channel pair {}- {}".format( ch_pair[0], ch_pair[1]))
                if rec:
                    ax[ ich, 0].plot(self.__correlation[ich])
                    ax[ich, 0].set_ylim((0, max(self.__correlation[ich])))
                    ax[ich, 0].axvline(pos, alpha = .5, color = 'red', lw = 1, label= 'rec')
                    ax[ich, 1].plot(station.get_channel(ch_pair[0]).get_times(), station.get_channel(ch_pair[0]).get_trace(), color = 'green', label = 'ch {}'.format(ch_pair[0]))
                  #  print("plot cannels", ch_pair)
                    ax[ich, 1].plot(station.get_channel(ch_pair[1]).get_times(), station.get_channel(ch_pair[1]).get_trace(), color = 'red', label = 'ch {}'.format(ch_pair[1]))

                    ax[ich, 1].legend()
                    ax[ich, 0].legend()
                if sim:
                    for channel in station.get_sim_station().get_channels_by_channel_id(ch_pair[0]):
                        ax[ich,1].plot(channel.get_times(), channel.get_trace(), color = 'orange', zorder = 100)
                    for channel in station.get_sim_station().get_channels_by_channel_id(ch_pair[1]):
                        ax[ich,1].plot(channel.get_times(), channel.get_trace(), color = 'orange', zorder = 100)
                    # ax[ich,1].plot(station.get_sim_station().get_channel(ch_pair[0]).get_times(), station.get_sim_station().get_channel(ch_pair[0]).get_trace(), color = 'orange')
    #                fig.tight_layout()
                    #ax[ich, 1].set_title("channel pair {}- {}".format( ch_pair[0], ch_pair[1]))
            if rec:
                ax[-1, 1].set_xlabel("timing [ns]")
                fig.tight_layout(h_pad=0)
                if debugplots_path is not None:
                    fig.savefig("{}/planewave_corr.pdf".format(debugplots_path))
                    plt.close()
                else:
                    plt.show()
           # print(stop)
            ### calculate timing shift due to plane wave
            ### get value in correlation due to timing

            if mode == 'log':
                corr = np.log(corr)
           # print("likelihood", likelihood)
            return -1*corr



        trace = np.copy(station.get_channel(self.__channel_pairs[0][0]).get_trace())
        if self.__template is None:
            self.__correlation = np.zeros((len(self.__channel_pairs), len(np.abs(scipy.signal.correlate(trace, trace))) ))

        else:

            self.__correlation = np.zeros((len(self.__channel_pairs), len(hp.get_normalized_xcorr(trace, self.__template))) )
        #if not self.__template: self.__correlation = np.zeros((len(self.__channel_pairs), len(np.abs(scipy.signal.correlate(trace, trace))) ))
        for ich, ch_pair in enumerate(self.__channel_pairs):
           # print("self.__channel_pairs[0]", self.__channel_pairs[ich][0])
            trace1 = np.copy(station.get_channel(self.__channel_pairs[ich][0]).get_trace())
            #print("channel", station.get_channel(4).get_trace())
            trace2 =np.copy(station.get_channel(self.__channel_pairs[ich][1]).get_trace())

            if self.__template is not None:
        #        self.__correlation = np.zeros((len(self.__channel_pairs), len(hp.get_normalized_xcorr(trace, self.__template))) )

                corr_1 = hp.get_normalized_xcorr(trace1, self.__template)
                #print("corr 1", corr_1)
                corr_2 = hp.get_normalized_xcorr(trace2, self.__template)
                #print("len corr1", len(corr_1))
               # print("corr 2", corr_2)
                #print(stop)
                #print("len channel pairs", len(self.__channel_pairs))
               # self.__correlation = np.zeros((len(self.__channel_pairs), len(corr_1)))
                sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
                for i_shift, shift_sample in enumerate(sample_shifts):
                  #  print("corr 2", corr_2)
                    if (np.isnan(corr_2).any()):# or (not corr_2): ### with noise this should not be needed
                        self.__correlation[ich, i_shift] = 0#np.zeros(len(corr_2))
                    #    print("self correlation",np.zeros(len(corr_2)))
                    elif (np.isnan(corr_1).any()):
                        self.__correlation[ich, i_shift] = 0#np.zeros(len(corr_2))

                    else:
                        self.__correlation[ich, i_shift] = np.max(corr_1 * np.roll(corr_2, shift_sample))
            else:
                t_max1 = station.get_channel(self.__channel_pairs[ich][0]).get_times()[np.argmax(np.abs(trace1))]
                t_max2 = station.get_channel(self.__channel_pairs[ich][1]).get_times()[np.argmax(np.abs(trace2))]
                corr_range = 50 * units.ns
                snr1 = np.max(np.abs(station.get_channel(self.__channel_pairs[ich][0]).get_trace()))
                snr2 = np.max(np.abs(station.get_channel(self.__channel_pairs[ich][1]).get_trace()))
                if snr1 > snr2:
                    trace1[np.abs(station.get_channel(self.__channel_pairs[ich][0]).get_times() - t_max1) > corr_range] = 0
                else:
                    trace2[np.abs(station.get_channel(self.__channel_pairs[ich][1]).get_times() - t_max2) > corr_range] = 0
                if mode == 'add_normalize':
                    self.__correlation[ich] = np.abs(scipy.signal.correlate(trace1/np.max(np.abs(trace1)), trace2/np.max(np.abs(trace2))))
                else:
                    self.__correlation[ich] = np.abs(scipy.signal.correlate(trace1, trace2))
                if mode == 'add_normalize_correlation':
                    self.__correlation[ich] /= np.max(self.__correlation[ich])

        if debug & station.has_sim_station():
            print(
                "Likelihood simulation",
                likelihood([signal_zenith, signal_azimuth], sim = True, mode=mode)
            )

        ll, fval, xgrid, fgrid = opt.brute(
            likelihood, ranges=(
                slice(zenith_min, zenith_max, zenith_step),
                slice(azimuth_min, azimuth_max, azimuth_step)
            ), args=(mode,), finish = opt.fmin, full_output=True)

        rec_zenith = ll[0]
        rec_azimuth = ll[1]

        if debug:
            print("creating debug plot for planwavefiter.....")
            extent = (
                xgrid[0,0,0] / units.deg,
                xgrid[0,-1,0] / units.deg,
                xgrid[1,0,0] / units.deg,
                xgrid[1,0,-1] / units.deg,
            )
            # zens = np.arange(np.rad2deg(signal_zenith) - 10, np.rad2deg(signal_zenith) + 10, .5)
            # azs = np.arange(np.rad2deg(signal_azimuth) - 10, np.rad2deg(signal_azimuth) + 10, .5)
            # xx, yy = np.meshgrid(zens, azs)
            # zz = np.zeros((len(zens), len(azs)))
            # for iz, z in enumerate(zens):
            #     for ia, a in enumerate(azs):
            #         c = likelihood([np.deg2rad(z), np.deg2rad(a)], mode=mode)
            #         zz[iz, ia] = c

            fig1 = plt.figure()
            # plt.pcolor(zens, azs,  zz.T)
            plt.imshow(fgrid.T, extent=extent, aspect='auto', origin='lower')
            plt.xlabel(r"zenith $[^{\circ}]$")
            plt.ylabel(r"azimuth $[^{\circ}]$")
            plt.axhline(np.rad2deg(signal_azimuth), color = 'orange')
            plt.axvline(np.rad2deg(signal_zenith), color = 'orange', label = 'simulated values')
            plt.axhline(np.rad2deg(rec_azimuth), color = 'white')
            plt.axvline(np.rad2deg(rec_zenith), color = 'white', label = 'reconstructed values')
            cbar = plt.colorbar()
            cbar.set_label('minimization value', rotation=270, labelpad = +20)

            plt.legend()
            fig1.tight_layout()
            if debugplots_path != None:
                fig1.savefig("{}/planewave_map.pdf".format(debugplots_path))
                plt.close()
            else:
                plt.show()
        ##### run with reconstructed values
        if debug: print("likelihood reconstruction", likelihood(ll, rec = True))
        print("simulated zenith {} and reconstructed zenith {}".format(np.rad2deg(signal_zenith), np.rad2deg(rec_zenith)))
        print("simulated azimuth {} and reconstructed azimuth {}".format(np.rad2deg(signal_azimuth), np.rad2deg(rec_azimuth)))

        station[stnp.planewave_zenith] = rec_zenith
        station[stnp.planewave_azimuth] = rec_azimuth

        if full_output:
            return (rec_zenith, rec_azimuth), xgrid, fgrid

        return (rec_zenith, rec_azimuth)


    def end(self):
        pass
