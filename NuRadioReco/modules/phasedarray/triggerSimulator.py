from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

from NuRadioReco.framework.trigger import SimplePhasedTrigger
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import logging
import scipy
import numpy as np
from scipy import constants
from scipy.signal import hilbert

logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = np.deg2rad(-55.0)
main_high_angle = -1.0 * main_low_angle
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))


class triggerSimulator:
    """
    Calculates the trigger for a phased array with a primary beam.
    
    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    See https://arxiv.org/pdf/1809.04573.pdf
    """

    def __init__(self, log_level=logging.WARNING):
        self.__t = 0
        self.__pre_trigger_time = None
        self.__debug = None
        logger.setLevel(log_level)
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    def get_antenna_positions(self, station, det, triggered_channels=None, component=2):
        """
        Calculates the vertical coordinates of the antennas of the detector

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        component: int
            Which cartesian coordinate to return

        Returns
        -------
        ant_pos: array of floatss
            Desired antenna position in requested coordinate

        """

        ant_pos = {}
        for channel in station.iter_channels(use_channels=triggered_channels):
            ant_pos[channel.get_id()] = det.get_relative_position(station.get_id(), channel.get_id())[component]

        return ant_pos

    def calculate_time_delays(self, station, det,
                              triggered_channels,
                              phasing_angles=default_angles,
                              ref_index=1.75,
                              sampling_frequency=None):

        """
        Calculates the delays needed for phasing the array.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float
            refractive index for beam forming
        sampling_frequency: float
            Rate of the ADC used

        Returns
        -------
        beam_rolls: array of dicts of keys=antenna and content=delay
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        time_step = 1. / sampling_frequency

        ant_z = self.get_antenna_positions(station, det, triggered_channels, 2)

        self.check_vertical_string(station, det, triggered_channels)
        ref_z = np.max(np.fromiter(ant_z.values(), dtype=float))

        # Need to add in delay for trigger delay
        cable_delays = {}
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):
            cable_delays[channel.get_id()] = det.get_cable_delay(station.get_id(), channel.get_id())

        beam_rolls = []
        for angle in phasing_angles:

            delays = []
            for key in ant_z:
                delays += [-(ant_z[key] - ref_z) / cspeed * ref_index * np.sin(angle) - cable_delays[key]]

            delays -= np.max(delays)

            roll = np.array(np.round(np.array(delays) / time_step)).astype(int)

            subbeam_rolls = dict(zip(triggered_channels, roll))

            # logger.debug("angle:", angle / units.deg)
            # logger.debug(subbeam_rolls)

            beam_rolls.append(subbeam_rolls)

        return beam_rolls

    def get_channel_trace_start_time(self, station, triggered_channels):
        """
        Finds the start time of the desired traces.
        Throws an error if all the channels dont have the same start time.

        Parameters
        ----------
        station: Station object
            Description of the current station
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken

        Returns
        -------
        channel_trace_start_time: float
            Channel start time
        """

        channel_trace_start_time = None
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):
            if channel_trace_start_time is None:
                channel_trace_start_time = channel.get_trace_start_time()
            elif channel_trace_start_time != channel.get_trace_start_time():
                error_msg = 'Phased array channels do not have matching trace start times. '
                error_msg += 'This module is not prepared for this case.'
                raise ValueError(error_msg)

        return channel_trace_start_time

    def check_vertical_string(self, station, det, triggered_channels):
        """
        Checks if the triggering antennas lie in a straight vertical line
        Throws error if not.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        """

        cut = 1.e-3 * units.m
        ant_x = np.fromiter(self.get_antenna_positions(station, det, triggered_channels, 0).values(), dtype=float)
        diff_x = np.abs(ant_x - ant_x[0])
        ant_y = np.fromiter(self.get_antenna_positions(station, det, triggered_channels, 1).values(), dtype=float)
        diff_y = np.abs(ant_y - ant_y[0])
        if (sum(diff_x) > cut or sum(diff_y) > cut):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def power_sum(self, coh_sum, window, step, adc_output='voltage',rnog_like=False):
        """
        Calculate power summed over a length defined by 'window', overlapping at intervals defined by 'step'

        Parameters
        ----------
        coh_sum: array of floats
            Phased signal to be integrated over
        window: int
            Power integral window
            Units of ADC time ticks
        step: int
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows
            Units of ADC time ticks.
        adc_output: string

            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        Returns
        -------
        power:
            Integrated power in each integration window
        num_frames
            Number of integration windows calculated

        """

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        num_frames = int(np.floor((len(coh_sum) - window) / step))

        if(adc_output == 'voltage'):
            coh_sum_squared = (coh_sum * coh_sum).astype(float)
        else: #(adc_output == 'counts'):
            if rnog_like:
                coh_sum[coh_sum>2**6-1]=2**6-1
                coh_sum[coh_sum<-2**6]=-2**6
            coh_sum_squared = (coh_sum * coh_sum).astype(int)

        coh_sum_windowed = np.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, window),
                                                           (coh_sum_squared.strides[0] * step, coh_sum_squared.strides[0]))

        power = np.sum(coh_sum_windowed, axis=1)
        return_power=power.astype(float) / window

        if adc_output=='counts': return_power=np.floor(return_power)

        return return_power, num_frames

    def hilbert_envelope(self,coh_sum,adc_output='voltage',coeff_gain=1,rnog_like=False):

        if rnog_like:
            coeff_gain=128

        #31 sample fir transformer
        #hil=[ -0.0424413 , 0. , -0.0489708 , 0. , -0.0578745 , 0. , -0.0707355 , 0. , -0.0909457 , 0. , -0.127324 , 0. 
        #       , -0.2122066 , 0. , -0.6366198 , 0., 0.6366198 , 0. , 0.2122066 , 0. , 0.127324 ,0. , 0.0909457 , 0. , .0707355  
        #       , 0. , 0.0578745 , 0. , 0.0489708 , 0. , 0.0424413 ]

        #middle 15 coefficients ^
        hil=np.array([ -0.0909457  , 0. , -0.127324 , 0. , -0.2122066 , 0. , -0.6366198 , 0. , 0.6366198 , 0. , 0.2122066 ,
                       0. , 0.127324 , 0. , 0.0909457 ])

        if coeff_gain!=1:
            hil=np.round(hil*coeff_gain)/coeff_gain

        imag_an=np.convolve(coh_sum,hil,mode='full')[len(hil)//2:len(coh_sum)+len(hil)//2]

        if adc_output:
            imag_an=np.round(imag_an)

        if rnog_like:
            envelope=np.max(np.array((coh_sum,imag_an)),axis=0)+3/8*min(np.array((coh_sum,imag_an)),axis=0)
        else:
            envelope=np.sqrt(coh_sum**2+imag_an**2)

        if adc_output=='counts':
            envelope=np.round(envelope)

        return envelope

    def phase_signals(self, traces, beam_rolls):
        """
        Phase signals together given the rolls

        Parameters
        ----------
        traces: 2D array of floats
            Signals from the antennas to be phased together.
        beam_rolls: 2D array of floats
            The amount to shift each signal before phasing the
            traces together

        Returns
        -------
        phased_traces: array of arrays
        """

        phased_traces = [[] for i in range(len(beam_rolls))]

        running_i = 0
        for subbeam_rolls in beam_rolls:
            phased_trace = np.zeros(len(list(traces.values())[0]))

            for channel_id in traces:

                trace = traces[channel_id]
                phased_trace += np.roll(trace, subbeam_rolls[channel_id])

            phased_traces[running_i] = phased_trace
            running_i += 1

        return phased_traces

    def phased_trigger(self, station, det,
                       Vrms=None,
                       threshold=60 * units.mV,
                       triggered_channels=None,
                       phasing_angles=default_angles,
                       ref_index=1.75,
                       trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
                       clock_offset=0,
                       adc_output='voltage',
                       trigger_filter=None,
                       upsampling_factor=1,
                       window=32,
                       step=16,
                       apply_digitization=False,
                       upsampling_method='fft',
                       coeff_gain=128,
                       rnog_like=False,
                       trig_type='power_integration'
                       ):
        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        Vrms: float
            RMS of the noise on a channel, used to automatically create the digitizer
            reference voltage. If set to None, tries to use reference voltage as defined
            int the detector description file.
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float (default 1.75)
            refractive index for beam forming
        trigger_adc: bool (default True)
            If True, uses the ADC settings from the trigger. It must be specified in the
            detector file. See analogToDigitalConverter module for information
            (see option `apply_digitization`)
        clock_offset: float (default 0)
            Overall clock offset, for adc clock jitter reasons (see `apply_digitization`)
        adc_output: string (default 'voltage')
            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts and apply integer based math
        trigger_filter: array floats (default None)
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer (default 1)
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        window: int (default 32)
            Power integral window
            Units of ADC time ticks
        step: int (default 16)
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks
        apply_digitization: bool (default True)
            Perform the quantization of the ADC. If set to true, should also set options
            `trigger_adc`, `adc_output`, `clock_offset`
        upsampling_method: str (default 'fft')
            Choose between FFT, FIR, or Linear Interpolaion based upsampling methods
        coeff_gain: int (default 1)
            If using the FIR upsampling, this will convert the floating point output of the 
            scipy filter to a fixed point value by multiplying by this factor and rounding to an
            int.
        rnog_like: bool (default False)
            If true, this will apply the RNO-G FLOWER based math/rounding done in firmware.
        trig_type: str (default "power_integration")
            - "power_integration" do the power integration for the given window size and 
                step length
            - "envelope" perform a hilbrt envelope threshold trigger on the beamformed
                traces

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        trigger_delays: dictionary
            the delays for the primary channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        trigger_time: float
            the earliest trigger time with respect to first interaction time.
        trigger_times: dictionary
            all time bins that fulfil the trigger condition per beam. The key is the beam number. Time with respect to first interaction time.
        maximum_amps: list of floats (length equal to that of `phasing_angles`)
            the maximum value of all the integration windows for each of the phased waveforms
        n_trigs: int
            total number of triggers that happened for all beams across the full traces
        triggered_beams: list
            list of bools for which beams triggered
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        ADC = analogToDigitalConverter()

        is_triggered = False
        trigger_delays = {}

        logger.debug(f"trigger channels: {triggered_channels}")

        traces = {}
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):
            channel_id = channel.get_id()

            trace = np.array(channel.get_trace())

            if apply_digitization:
                trace, adc_sampling_frequency = ADC.get_digital_trace(station, det, channel,
                                                                  Vrms=Vrms,
                                                                  trigger_adc=trigger_adc,
                                                                  clock_offset=clock_offset,
                                                                  return_sampling_frequency=True,
                                                                  adc_type='perfect_floor_comparator',
                                                                  adc_output=adc_output,
                                                                  trigger_filter=None)
            else:
                adc_sampling_frequency = channel.get_sampling_rate()

            if not isinstance(upsampling_factor, int):
                try:
                    upsampling_factor = int(upsampling_factor)
                except:
                    raise ValueError("Could not convert upsampling_factor to integer. Exiting.")

            if(upsampling_factor >= 2):

                new_len = len(trace) * upsampling_factor 
                cur_t=np.arange(0,1/adc_sampling_frequency*len(trace),1/adc_sampling_frequency)
                new_t=np.arange(0,1/adc_sampling_frequency*len(trace),1/adc_sampling_frequency/upsampling_factor)

                if upsampling_method=='fft':
                    upsampled_trace = scipy.signal.resample(trace, new_len)

                elif upsampling_method=='lin':
                    upsampled_trace = np.interp(new_t,cur_t,trace)

                elif upsampling_method=='fir':
                    if rnog_like:
                        up_filt=np.array([ 0.0, 0.0 , 0.0 , 0.0 , 0.0078125 , 0.0078125 , 0.0078125 , -0.0 , -0.015625 , 
                                    -0.0390625 , -0.03125 , 0.0 , 0.0703125 , 0.15625 , 0.2265625 , 0.25 , 0.2265625 , 0.15625 ,
                                    0.0703125 , 0.0 , -0.03125 , -0.0390625 , -0.015625 , 0.0 , 0.0078125 , 0.0078125 , 0.0078125 ,
                                    0.0 , 0.0 , 0.0 , 0.0 ])
                    else:
                        cutoff=.5
                        base_freq_filter_length=8
                        filter_length=base_freq_filter_length*upsampling_factor-1
                        up_filt=scipy.signal.firwin(filter_length,adc_sampling_frequency*cutoff,pass_zero='lowpass',
                                                    fs=adc_sampling_frequency*upsampling_factor)
                        if coeff_gain!=1:
                            up_filt=np.round(up_filt*coeff_gain)/coeff_gain

                    zero_pad=np.zeros(len(trace)*upsampling_factor)
                    zero_pad[::upsampling_factor]=trace[:]
                    upsampled_trace=np.convolve(zero_pad,up_filt,mode='full')[len(up_filt)//2:len(zero_pad)+len(up_filt)//2]*upsampling_factor

                else:
                    error_msg = 'Interpolation method must be lin, fft, fir, ...'
                    raise ValueError(error_msg)

                if adc_output=='counts' and rnog_like==True: upsampled_trace=np.trunc(upsampled_trace)

                #  If upsampled is performed, the final sampling frequency changes
                trace = upsampled_trace[:]

                if(len(trace) % 2 == 1):
                    trace = trace[:-1]

                adc_sampling_frequency *= upsampling_factor

            time_step = 1.0 / adc_sampling_frequency

            traces[channel_id] = trace[:]

        beam_rolls = self.calculate_time_delays(station, det,
                                                triggered_channels,
                                                phasing_angles,
                                                ref_index=ref_index,
                                                sampling_frequency=adc_sampling_frequency)
        phased_traces = self.phase_signals(traces, beam_rolls)

        trigger_time = None
        trigger_times = {}
        channel_trace_start_time = self.get_channel_trace_start_time(station, triggered_channels)

        trigger_delays = {}
        maximum_amps = np.zeros(len(phased_traces))
        n_trigs=0
        triggered_beams=[]

        for iTrace, phased_trace in enumerate(phased_traces):
            is_triggered=False

            if trig_type=='power_integration':
                squared_mean, num_frames = self.power_sum(coh_sum=phased_trace, window=window, step=step, adc_output=adc_output, rnog_like=rnog_like)
                maximum_amps[iTrace] = np.max(squared_mean)

                if True in (squared_mean > threshold):
                    n_trigs+=len(np.where((squared_mean > threshold)==True)[0])
                    trigger_delays[iTrace] = {}

                    for channel_id in beam_rolls[iTrace]:
                        trigger_delays[iTrace][channel_id] = beam_rolls[iTrace][channel_id] * time_step

                    triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(squared_mean > threshold)))
                    logger.debug(f"Station has triggered, at bins {triggered_bins}")
                    logger.debug(trigger_delays)
                    logger.debug(f"trigger_delays {trigger_delays[iTrace][triggered_channels[0]]}")
                    is_triggered = True
                    trigger_times[iTrace] = trigger_delays[iTrace][triggered_channels[0]] + triggered_bins * step * time_step + channel_trace_start_time
                    logger.debug(f"trigger times  = {trigger_times[iTrace]}")

            elif trig_type=='envelope':
                hilbert_env = self.hilbert_envelope(coh_sum=phased_trace,adc_output=adc_output,rnog_like=rnog_like)
                maximum_amps[iTrace] = np.max(hilbert_env)

                if True in (hilbert_env>threshold):
                    n_trigs+=len(np.where((hilbert_env > threshold)==True)[0])
                    trigger_delays[iTrace] = {}
                    for channel_id in beam_rolls[iTrace]:
                        trigger_delays[iTrace][channel_id] = beam_rolls[iTrace][channel_id] * time_step
                    triggered_bins=np.atleast_1d(np.squeeze(np.argwhere(hilbert_env > threshold)))
                    is_triggered=True
                    trigger_times[iTrace] = trigger_delays[iTrace][triggered_channels[0]] + triggered_bins * step * time_step + channel_trace_start_time

            else:
                raise NotImplementedError("not a good trigger type: options (power_integration, envelope)")

            triggered_beams.append(is_triggered)

        is_triggered=np.any(triggered_beams)

        if is_triggered:
            logger.debug("Trigger condition satisfied!")
            logger.debug("all trigger times", trigger_times)
            trigger_time = min([x.min() for x in trigger_times.values()])
            logger.debug(f"minimum trigger time is {trigger_time:.0f}ns")

        return is_triggered, trigger_delays, trigger_time, trigger_times, maximum_amps, n_trigs, triggered_beams

    @register_run()
    def run(self, evt, station, det,
            Vrms=None,
            threshold=60 * units.mV,
            triggered_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles,
            set_not_triggered=False,
            ref_index=1.75,
            trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
            clock_offset=0,
            adc_output='voltage',
            trigger_filter=None,
            upsampling_factor=1,
            window=32,
            step=16,
            apply_digitization=True,
            upsampling_method='fft',
            coeff_gain=128,
            rnog_like=False,
            trig_type='power_integration',
            return_n_triggers=False
            ):

        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles.

        Parameters
        ----------
        evt: Event object
            Description of the current event
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        Vrms: float
            RMS of the noise on a channel, used to automatically create the digitizer
            reference voltage. If set to None, tries to use reference voltage as defined
            int the detector description file.
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        trigger_name: string
            name for the trigger
        phasing_angles: array of float
            pointing angles for the primary beam
        set_not_triggered: bool (default False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered
        ref_index: float (default 1.75)
            refractive index for beam forming
        trigger_adc: bool, (default True)
            If True, uses the ADC settings from the trigger. It must be specified in the
            detector file. See analogToDigitalConverter module for information
        clock_offset: float (default 0)
            Overall clock offset, for adc clock jitter reasons
        adc_output: string (default 'voltage')

            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        trigger_filter: array floats (default None)
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer (default 1)
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        window: int (default 32)
            Power integral window
            Units of ADC time ticks
        step: int (default 16)
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks
        apply_digitization: bool (default True)
            Perform the quantization of the ADC. If set to true, should also set options
            `trigger_adc`, `adc_output`, `clock_offset`
        upsampling_method: str (default 'fft')
            Choose between FFT, FIR, or Linear Interpolaion based upsampling methods
        coeff_gain: int (default 1)
            If using the FIR upsampling, this will convert the floating point output of the 
            scipy filter to a fixed point value by multiplying by this factor and rounding to an
            int.
        rnog_like: bool (default False)
            If true, this will apply the RNO-G FLOWER based math/rounding done in firmware.
        trig_type: str (default "power_integration")
            - "power_integration" do the power integration for the given window size and 
                step length
            - "envelope" perform a hilbrt envelope threshold trigger on the beamformed
                traces
        return_n_triggers: bool (default False)
            To better estimate simulated thresholds one should count the total triggers
            in the entire trace for each beam. If true, this return the total trigger number.

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        n_triggers: int (Optional)
            Count of the total number of triggers in all beamformed traces
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        is_triggered = False
        trigger_delays = {}

        if(set_not_triggered):
            is_triggered = False
            trigger_delays = {}
            triggered_beams = []
        else:
            is_triggered, trigger_delays, trigger_time, trigger_times,\
                  maximum_amps, n_triggers, triggered_beams = self.phased_trigger(station=station,
                                                                                    det=det,
                                                                                    Vrms=Vrms,
                                                                                    threshold=threshold,
                                                                                    triggered_channels=triggered_channels,
                                                                                    phasing_angles=phasing_angles,
                                                                                    ref_index=ref_index,
                                                                                    trigger_adc=trigger_adc,
                                                                                    clock_offset=clock_offset,
                                                                                    adc_output=adc_output,
                                                                                    trigger_filter=trigger_filter,
                                                                                    upsampling_factor=upsampling_factor,
                                                                                    window=window,
                                                                                    step=step,
                                                                                    apply_digitization=apply_digitization,
                                                                                    upsampling_method=upsampling_method,
                                                                                    coeff_gain=coeff_gain,
                                                                                    rnog_like=rnog_like,
                                                                                    trig_type=trig_type
                                                                                    )

        # Create a trigger object to be returned to the station
        if trig_type=='power_integration':
            trigger = SimplePhasedTrigger(
                trigger_name, 
                threshold, 
                channels=triggered_channels,
                primary_angles=phasing_angles, 
                trigger_delays=trigger_delays,
                window_size=window, 
                step_size=step,
                maximum_amps=maximum_amps
            )

        elif trig_type=='envelope':
            trigger = SimplePhasedTrigger(
                trigger_name, 
                threshold, 
                channels=triggered_channels,
                primary_angles=phasing_angles, 
                trigger_delays=trigger_delays,
                maximum_amps=maximum_amps
            )

        else:
            raise NotImplementedError("invalid phased trigger type")

        trigger.set_triggered(is_triggered)

        if is_triggered:
            #trigger_time(s)= time(s) from start of trace + start time of trace with respect to moment of first interaction = trigger time from moment of first interaction; time offset to interaction time (channel_trace_start_time) already recognized in self.phased_trigger
            trigger.set_trigger_time(trigger_time)# 
            trigger.set_trigger_times(trigger_times)
        else:
            trigger.set_trigger_time(None)

        station.set_trigger(trigger)

        if return_n_triggers:
            return is_triggered,n_triggers
        else:
            return is_triggered

    def end(self):
        pass
