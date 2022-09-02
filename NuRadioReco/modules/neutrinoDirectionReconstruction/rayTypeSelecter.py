from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import channelParameters as chp
import math
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import scipy


class rayTypeSelecter:


    def __init__(self):
        self.begin()

    def begin(self):
        pass

    def run(
            self, event, station, det, use_channels=[9, 14], noise_rms = 10,
            template = None, shower_id = None,
            icemodel = 'greenland_simple', raytracing_method = 'analytic',
            att_model = 'GL1', sim = False,
            debug_plots = True, debugplots_path = None):
        """
        Finds the pulse position of the triggered pulse in the phased array and returns the raytype of the pulse

        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the Module on
        det: Detector
            The detector description
        use_channels: list
            List of phased array channels used for the pulse selection
        template: array
            Neutrino voltage template. Default = None
        shower_id: int
            For simulated event, the shower id is needed for vertex selection. Default = None
        icemodel: str
            Icemodel used for the propagation. Default = 'greenland_simple'
        raytracing_method: str
            Method used for the raytracing. Default = 'analytic'
        sim: Boolean
            True if simulated event is used. Default = False.
        debug_plots: Boolean
            Default = True.
        """


        ice = medium.get_ice_model(icemodel)
        prop = propagation.get_propagation_module(raytracing_method)
        sampling_rate = station.get_channel(0).get_sampling_rate() ## assume same for all channels
        station_id = station.get_id()

        if sim:
            vertex = event.get_sim_shower(shower_id)[shp.vertex]
            print("	simulated vertex:", vertex)
        else:
            vertex = station[stnp.nu_vertex]



        if debug_plots:
            fig, axs = plt.subplots(3, figsize = (10, 10))
            iax = 0
        #### determine position of pulse
        T_ref = np.zeros(3)
        max_totalcorr= np.zeros(3)
        pos_max = np.zeros(3)
        for raytype in [1,2,3]:
            type_exist = 0
            total_trace = np.zeros(len(station.get_channel(0).get_trace()))
            traces = dict()
            time_shifts = dict()
            if template is not None:
                # pad template if it is too short (probably not needed?)
                if (len(template) < len(total_trace)):
                    template = np.pad(template, (0, len(total_trace) - len(template)))
            corr_total = np.zeros(len(total_trace) + len(template) - 1)
            for channel in station.iter_channels():
                if channel.get_id() in use_channels:##[0,1,2,3]:
                   # channel = station.get_channel(channelid) # if channel in phased array
                    channel_id = channel.get_id()
                    x2 = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
                    r = prop( ice, att_model)
                    r.set_start_and_end_point(vertex, x2)
                    r.find_solutions()
                    for iS in range(r.get_number_of_solutions()):
                        if r.get_solution_type(iS) == raytype:
                            type_exist= 1
                            T = r.get_travel_time(iS)
                            if channel_id == use_channels[0]:
                                T_ref[iS] = T
                                trace_start_time_ref = channel.get_trace_start_time()

                            dt = T - T_ref[iS] - (channel.get_trace_start_time() - trace_start_time_ref)
                            dn_samples = -1*dt * sampling_rate
                            dn_samples = math.ceil(dn_samples)
                            cp_trace = np.copy(channel.get_trace())
                            cp_trace_roll = np.roll(cp_trace, dn_samples)
                            corr = scipy.signal.correlate(cp_trace_roll*(1/(max(cp_trace_roll))), template*(1/(max(template))))
                            corr_total += abs(corr)

                            time_shifts[channel_id] = dn_samples
                            traces[channel_id] = cp_trace

            if not type_exist:
                continue # no solutions for this ray type

            # we determine the approximate position of the pulse max
            # by summing the shifted data traces
            dt = np.argmax(corr_total) - len(corr_total)/2 + 1
            template_roll = np.roll(template, int(dt))
            position_max = np.argmax(template_roll)
            for channel_id in traces.keys():
                cp_trace = traces[channel_id]
                dn_samples = time_shifts[channel_id]
                # restrict to a 50 ns window around the expected pulse to avoid accidental maxima
                cp_trace[np.arange(len(cp_trace)) < (position_max - 20 * sampling_rate)] = 0
                cp_trace[np.arange(len(cp_trace)) > (position_max + 30 * sampling_rate)] = 0
                trace = np.roll(cp_trace, dn_samples)
                total_trace += trace

                if debug_plots:
                    axs[iax].plot(trace, color = 'darkgrey', lw =2, alpha=.75)

            hilbert_envelope = np.abs(scipy.signal.hilbert(total_trace))
            pos_max[raytype-1] = np.argmax(hilbert_envelope)
            if debug_plots and type_exist:
                axs[iax].plot(total_trace, lw = 2, color = 'darkgreen', label= 'combined trace')
                axs[iax].plot(hilbert_envelope, lw = 2, color = 'darkgreen', ls =':')

                axs[iax].legend(loc = 1, fontsize= 20)
                for tick in axs[iax].yaxis.get_majorticklabels():
                    tick.set_fontsize(20)
                for tick in axs[iax].xaxis.get_majorticklabels():
                    tick.set_fontsize(20)
                for tick in axs[2].yaxis.get_majorticklabels():
                    tick.set_fontsize(20)
                for tick in axs[2].xaxis.get_majorticklabels():
                    tick.set_fontsize(20)

                axs[iax].set_title("raytype: {}".format(['direct', 'refracted', 'reflected'][raytype-1]), fontsize = 40)
                axs[iax].grid()
                axs[iax].set_xlim((position_max - 40*sampling_rate, position_max + 40*sampling_rate))
                axs[iax].set_xlabel("samples", fontsize = 25)
                iax += 1

                axs[raytype-1].set_title("raytype {}".format(['direct', 'refracted', 'reflected'][raytype-1]), fontsize = 30)
                axs[2].plot(corr_total, lw = 2,  label= '{}'.format(['direct', 'refracted', 'reflected'][raytype-1]))

                axs[2].legend(fontsize = 20)
                axs[2].grid()
                axs[2].set_title("correlation", fontsize = 30)

            max_totalcorr[raytype-1] = max(abs(corr_total))
            where_are_NaNs = np.isnan(max_totalcorr)

        if debug_plots:
            fig.tight_layout()
            run_number = event.get_run_number()
            event_id = event.get_id()
            fig.savefig("{}/{}_{}_pulse_selection.pdf".format(debugplots_path, run_number, event_id))

        ### store parameters
        reconstructed_raytype = ['direct', 'refracted', 'reflected'][np.argmax(max_totalcorr)]
        print("		reconstructed raytype:", reconstructed_raytype)
        if not sim: station.set_parameter(stnp.raytype, reconstructed_raytype)
        #print("CHECK")
        if sim: station.set_parameter(stnp.raytype_sim, reconstructed_raytype)
        print("		max_totalcorr", max_totalcorr)
        print("		pos_mas", pos_max)
        position_pulse = pos_max[np.argmax(max_totalcorr)]
        print("		position pulse", position_pulse)
        #print("time position pulse", station.get_channel(use_channels[0]).get_times()[position_pulse])
        if not sim: station.set_parameter(stnp.pulse_position, position_pulse)
        if sim: station.set_parameter(stnp.pulse_position_sim, position_pulse)

        if debug_plots:
            fig, axs = plt.subplots(station.get_number_of_channels(), sharex = True, figsize = (5, station.get_number_of_channels() * 1.25))

        #### use pulse position to find places in traces of the other channels to determine which traces have a SNR > 3.5
        channels_pulses = []

        x2 = det.get_relative_position(station_id, use_channels[0]) + det.get_absolute_position(station_id)
        trace_start_time_ref = station.get_channel(use_channels[0]).get_trace_start_time()
        r = prop(ice, att_model)
        r.set_start_and_end_point(vertex, x2)
        r.find_solutions()
        for iS in range(r.get_number_of_solutions()):
            if r.get_solution_type(iS) in [np.argmax(max_totalcorr)+1]:

                T_reference = r.get_travel_time(iS)

        for ich, channel in enumerate(station.iter_channels()):
           # channel = station.get_channel(channelid)
            channel_id = channel.get_id()
            x2 = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
            trace_start_time_channel = channel.get_trace_start_time()
            r = prop( ice, att_model)
            r.set_start_and_end_point(vertex, x2)
            simchannel = []
            r.find_solutions()
           # print("channel id {}, number of solutions {}".format(channel_id, r.get_number_of_solutions()))
            for iS in range(r.get_number_of_solutions()):
               # print("ray type", r.get_solution_type(iS))
                T = r.get_travel_time(iS)
                delta_T =  T - T_reference - (trace_start_time_channel - trace_start_time_ref)
                delta_toffset = delta_T * sampling_rate
                ### if channel is phased array channel, and pulse is triggered pulse, store signal zenith and azimuth
                if channel_id == use_channels[0]: # if channel is upper phased array channel
                   # print("	solution type", r.get_solution_type(iS))
                  #  print("selected type", np.argmax(max_totalcorr)+1)
                    if r.get_solution_type(iS) in [np.argmax(max_totalcorr)+1]: ## if solution type is triggered solution type
                        #print("		get receive vector...............>>")
                        receive_vector = r.get_receive_vector(iS)
                        receive_zenith, receive_azimuth = hp.cartesian_to_spherical(*receive_vector)
                        if sim == True:
                            channel.set_parameter(chp.signal_receiving_zeniths, receive_zenith)
                            channel.set_parameter(chp.signal_receiving_azimuths, receive_azimuth)
                            print("	receive zenith vertex, simulated vertex:", np.rad2deg(receive_zenith))
                        if not sim:
                            channel.set_parameter(chp.receive_zenith_vertex, receive_zenith)
                            print("	receive zenith vertex, reconstructed vertex:", np.rad2deg(receive_zenith))
                            channel.set_parameter(chp.receive_azimuth_vertex, receive_azimuth)
                            print("	receive azimuth vertex, reconstructed vertex:", np.rad2deg(receive_azimuth))
                    #print("zenith", channel[chp.signal_receiving_zenith])#print("channel id", channel_id)
                ### figuring out the time offset for specfic trace
                k = int(position_pulse + delta_toffset )
                k_start = np.max([k-300, 0])
                k_stop = np.max([k+500, 0]) # probably never happens...
                pulse_window = channel.get_trace()[k_start: k_stop]
                if len(pulse_window) == 0:
                    pulse_window = np.zeros(800)
                if debug_plots:
                    plot_pulse_time = trace_start_time_channel + k / sampling_rate
                    axs[ich].plot(channel.get_times(), channel.get_trace(), color = 'blue')
                    axs[ich].set_xlabel("time [ns]")
                    if sim:
                        for sim_ch in station.get_sim_station().get_channels_by_channel_id(channel_id):
                            axs[ich].plot(sim_ch.get_times(), sim_ch.get_trace(), color = 'orange')
                    axs[ich].axvline(plot_pulse_time -300 / sampling_rate, color = 'grey')
                    axs[ich].axvline(plot_pulse_time +500 / sampling_rate, color = 'grey')
                    if ((np.max(pulse_window) - np.min(pulse_window))/(2*noise_rms) > 3.5):
                        axs[ich].axvline(plot_pulse_time, color = 'green')
                    else:
                        axs[ich].axvline(plot_pulse_time, color = 'red')
                    axs[ich].set_title("channel {}".format(channel_id))
                if ((np.max(pulse_window) - np.min(pulse_window))/(2*noise_rms) > 3.5):
                    channels_pulses.append(channel.get_id())


        if debug_plots:
            fig.tight_layout()
            fig.savefig("{}/{}_{}_pulse_window.pdf".format(debugplots_path, run_number, event_id))
        station.set_parameter(stnp.channels_pulses, channels_pulses)

    def end(self):
        pass
