from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities import units
import math
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import scipy
import logging
logging.basicConfig()

logger = logging.getLogger('rayTypeSelecter')
logger.setLevel(logging.INFO)
class rayTypeSelecter:

    def __init__(self, **kwargs):
        self.begin(**kwargs)
        pass

    def begin(self, debug=False, debugplots_path='.'):
        """
        Set debug parameters

        Parameters
        ----------
        debug: bool, default False
            If True, produce debug plots
        debugplots_path: str, default '.'
            Path to store debug plots
        """
        self.__debug = debug
        self.__debugplots_path = debugplots_path
        pass

    def run(
            self, event, station, det, use_channels, vrms,
            template, ice_model = 'greenland_simple', raytracing_method = 'analytic',
            attenuation_model = 'GL1', sim = False, shower_id = None,):
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
        vrms: float | dict | None
            RMS of the noise. This is used to define the SNR of the pulses
            in different channels. Can either be given as a single value
            for all channels, or as a dictionary where the keys are the channel_ids

            Can be set to ``None``; in this case, the vrms is estimated from the detector amplifier response,
            assuming a noise temperature of 300 K.
        template: array
            Neutrino voltage template. This is used to find the pulse in the
            phased array by correlation
        ice_model: str, default: 'greenland_simple'
            Icemodel used for the propagation. Default = 'greenland_simple'
        raytracing_method: str, default 'analytic'
            Method used for the raytracing. Default = 'analytic'
        attenuation_model: str, default 'GL1'
            Attenuation model used for the raytracing
        sim: Boolean
            True if simulated event is used. Default = False.
        
        Other Parameters
        ----------------
        shower_id: None | int, default None
            If using the simulated vertex, and the event contains multiple showers,
            the shower_id of the shower from which to obtain the vertex can be
            specified. If ``sim = True`` and the shower_id is not specified,
            the first sim_shower in the event is used.

        """

        debug = self.__debug
        debugplots_path = self.__debugplots_path

        if isinstance(ice_model, str):
            ice = medium.get_ice_model(ice_model)
        else:
            ice = ice_model
        prop = propagation.get_propagation_module(raytracing_method)
        sampling_rate = station.get_channel(use_channels[0]).get_sampling_rate() ## assume same for all channels
        station_id = station.get_id()

        if vrms is None: # compute vrms from noise temperature
            vrms = {}
            ff = np.linspace(0, 5*units.GHz, 8192)
            for channel_id in station.get_channel_ids():
                amp_response = det.get_amplifier_response(station_id, channel_id=channel_id, frequencies=ff)
                effective_bandwidth = np.trapz(np.abs(amp_response)**2, ff)
                vrms[channel_id] = (300 * 50 * scipy.constants.k * effective_bandwidth / units.Hz) ** 0.5 # assuming a noise temperature of 300 K

        if not isinstance(vrms, dict):
            vrms = {channel_id : vrms for channel_id in station.get_channel_ids()}

        if sim:
            if shower_id is not None:
                vertex = event.get_sim_shower(shower_id)[shp.vertex]
            else:
                vertex = event.get_first_sim_shower()[shp.vertex]
            logger.debug(f"using simulated vertex: {vertex}")
        else:
            vertex = station[stnp.nu_vertex]

        if debug:
            fig, axs = plt.subplots(3, figsize = (10, 10))
            iax = 0
        
        #### determine position of pulse
        T_ref = np.zeros(2)
        max_totalcorr= np.zeros(2)
        pos_max = np.zeros(2)
        for raytype in [0,1]:
            type_exist = 0
            total_trace = np.zeros(len(station.get_channel(0).get_trace()))
            traces = dict()
            time_shifts = dict()
            if template is not None:
                # pad template if it is too short (probably not needed?)
                if (len(template) < len(total_trace)):
                    template = np.pad(template, (0, len(total_trace) - len(template)))
            corr_total = np.zeros(len(total_trace) + len(template) - 1)
            trace_start_time_ref = None
            for channel_id in use_channels:
                channel = station.get_channel(channel_id)
                x2 = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
                r = prop(ice, attenuation_model)
                r.set_start_and_end_point(vertex, x2)
                r.find_solutions()
                for iS in range(r.get_number_of_solutions()):
                    if iS == raytype:
                        type_exist= 1
                        T = r.get_travel_time(iS)
                        if trace_start_time_ref is None:
                            T_ref[iS] = T
                            trace_start_time_ref = channel.get_trace_start_time()
                            raytype_string = propagation.solution_types[r.get_solution_type(iS)]
                            if not channel_id == use_channels[0]:
                                logger.warning(
                                    f"No solution for reference channel {use_channels[0]}, using channel {channel_id} instead..."
                                    )

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

                if debug:
                    axs[iax].plot(trace, color = 'darkgrey', lw =2, alpha=.75)

            hilbert_envelope = np.abs(scipy.signal.hilbert(total_trace))
            pos_max[raytype] = np.argmax(hilbert_envelope)
            if debug and type_exist:
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

                axs[iax].set_title("raytype: {} ({})".format(raytype_string, raytype), fontsize = 40)
                axs[iax].grid()
                axs[iax].set_xlim((position_max - 40*sampling_rate, position_max + 40*sampling_rate))
                axs[iax].set_xlabel("samples", fontsize = 25)
                iax += 1

                axs[raytype].set_title("raytype {} ({})".format(raytype_string, raytype), fontsize = 30)
                axs[2].plot(corr_total, lw = 2,  label= '{} ({})'.format(raytype_string, raytype))

                axs[2].legend(fontsize = 20)
                axs[2].grid()
                axs[2].set_title("correlation", fontsize = 30)

            max_totalcorr[raytype] = max(abs(corr_total))
            where_are_NaNs = np.isnan(max_totalcorr)

        if debug:
            fig.tight_layout()
            run_number = event.get_run_number()
            event_id = event.get_id()
            fig.savefig("{}/{}_{}_pulse_selection.pdf".format(debugplots_path, run_number, event_id))
            plt.close()

        ### store parameters
        reconstructed_raytype = np.argmax(max_totalcorr)
        logger.info(f"reconstructed raytype: {reconstructed_raytype}")
        if not sim: 
            station.set_parameter(stnp.raytype, reconstructed_raytype)
        #print("CHECK")
        else: 
            station.set_parameter(stnp.raytype_sim, reconstructed_raytype)
        logger.debug(f"max_totalcorr {max_totalcorr}")
        logger.debug("pos_max {pos_max}")
        position_pulse = pos_max[np.argmax(max_totalcorr)]
        logger.debug(f"position pulse {position_pulse}")
        #print("time position pulse", station.get_channel(use_channels[0]).get_times()[position_pulse])
        if not sim:
            station.set_parameter(stnp.pulse_position, position_pulse)
        else:
            station.set_parameter(stnp.pulse_position_sim, position_pulse)

        if debug:
            fig, axs = plt.subplots(station.get_number_of_channels(), sharex = True, figsize = (5, station.get_number_of_channels() * 1.25))

        #### use pulse position to find places in traces of the other channels to determine which traces have a SNR > 3.5
        channels_pulses = []

        x2 = det.get_relative_position(station_id, use_channels[0]) + det.get_absolute_position(station_id)
        trace_start_time_ref = station.get_channel(use_channels[0]).get_trace_start_time()
        r = prop(ice, attenuation_model)
        r.set_start_and_end_point(vertex, x2)
        r.find_solutions()
        for iS in range(r.get_number_of_solutions()):
            if iS == np.argmax(max_totalcorr):

                T_reference = r.get_travel_time(iS)

        # for ich, channel in enumerate(station.iter_channels()):
        for ich, channel_id in enumerate(vrms.keys()): # only estimate / plot pulse positions for channels with known vrms
            channel = station.get_channel(channel_id)
            channel_id = channel.get_id()
            x2 = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
            trace_start_time_channel = channel.get_trace_start_time()
            r = prop( ice, attenuation_model)
            r.set_start_and_end_point(vertex, x2)
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
                    if iS == np.argmax(max_totalcorr): ## if solution type is triggered solution type
                        #print("		get receive vector...............>>")
                        receive_vector = r.get_receive_vector(iS)
                        receive_zenith, receive_azimuth = hp.cartesian_to_spherical(*receive_vector)
                        if sim == True:
                            channel.set_parameter(chp.signal_receiving_zenith, receive_zenith)
                            channel.set_parameter(chp.signal_receiving_azimuth, receive_azimuth)
                            logger.debug(f"receive zenith vertex, simulated vertex: {receive_zenith/units.deg:.2f} deg")
                        if not sim:
                            channel.set_parameter(chp.receive_zenith_vertex, receive_zenith)
                            logger.debug(f"receive zenith vertex, reconstructed vertex:  {receive_zenith/units.deg:.2f} deg")
                            channel.set_parameter(chp.receive_azimuth_vertex, receive_azimuth)
                            logger.debug(f"receive azimuth vertex, reconstructed vertex: {receive_azimuth/units.deg:.2f} deg")
                    #print("zenith", channel[chp.signal_receiving_zenith])#print("channel id", channel_id)
                
                ### figuring out the time offset for specfic trace
                k = int(position_pulse + delta_toffset )
                k_start = np.max([k-300, 0])
                k_stop = np.max([k+500, 0]) # probably never happens...
                pulse_window = channel.get_trace()[k_start: k_stop]
                if len(pulse_window) == 0:
                    pulse_window = np.zeros(800)
                if debug:
                    plot_pulse_time = trace_start_time_channel + k / sampling_rate
                    axs[ich].plot(channel.get_times(), channel.get_trace(), color = 'blue')
                    if sim:
                        for sim_ch in station.get_sim_station().get_channels_by_channel_id(channel_id):
                            axs[ich].plot(sim_ch.get_times(), sim_ch.get_trace(), color = 'orange')
                    axs[ich].axvline(plot_pulse_time -300 / sampling_rate, color = 'grey')
                    axs[ich].axvline(plot_pulse_time +500 / sampling_rate, color = 'grey')
                    if ((np.max(pulse_window) - np.min(pulse_window))/(2*vrms[channel_id]) > 3.5):
                        axs[ich].axvline(plot_pulse_time, color = 'green')
                    else:
                        axs[ich].axvline(plot_pulse_time, color = 'red')
                    axs[ich].set_title("channel {}".format(channel_id))

                if ((np.max(pulse_window) - np.min(pulse_window))/(2*vrms[channel_id]) > 3.5):
                    channels_pulses.append(channel.get_id())


        if debug:
            axs[-1].set_xlabel("time [ns]")
            fig.tight_layout()
            fig.savefig("{}/{}_{}_pulse_window.pdf".format(debugplots_path, run_number, event_id))
            plt.close(fig)
        station.set_parameter(stnp.channels_pulses, channels_pulses)

    def end(self):
        pass
