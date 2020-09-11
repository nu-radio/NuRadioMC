from NuRadioReco.modules.base.module import register_run
import numpy as np
import fractions
from decimal import Decimal
from NuRadioReco.utilities import units
from scipy import signal
from radiotools import helper as hp
from NuRadioReco.utilities import templates
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('channelTemplateCorrelation')


class channelTemplateCorrelation:
    """
    Calculates correlation of waveform with neutrino/cr templates
    """

    def __init__(self, template_directory):
        self.__max_upsampling_factor = 5000
        self.__templates = templates.Templates(template_directory)
        self.__cr_templates = None
        self.__ref_cr_template = None
        self.__debug = None
        self.begin()

    def begin(self, debug=False):
        self.__cr_templates = {}
        self.__ref_cr_template = {}
        self.__debug = debug

    def match_sampling(self, ref_template, resampling_factor):
        if(resampling_factor.numerator != 1):
            ref_template_resampled = signal.resample(ref_template, resampling_factor.numerator * len(ref_template))
        else:
            ref_template_resampled = ref_template
        if(resampling_factor.denominator != 1):
            ref_template_resampled = signal.resample(ref_template_resampled, len(ref_template_resampled) / resampling_factor.denominator)
        return ref_template_resampled

    @register_run()
    def run(self, evt, station, det, channels_to_use=None, cosmic_ray=False,
            n_templates=1):
        """
        Parameters
        -----------
        evt: Event
            Event to run the module on
        station: Station
            Station to run the module on
        det: Detector
            The detector description
        channels_to_use: List of int (default: [0, 1, 2, 3])
            List of channel IDs for which the template correlation shall be calculated
        cosmic_ray: bool
            Switch for cosmic ray and neutrino analysis. Default is neutrino templates.
        n_templates: int
            default is 1: use just on standard template for all channels
            if n_templates is larger than one multiple templates are used and the
            average cross correlation for all templates is calculated. The set
            of templates contains several coreas input pulses and different
            incoming directions. The index first loops first over 6 different
            coreas pulses with different frequency content
            and then over azimuth angles of 0, 22.5 and 45 degree
            and then over zenith angles of 60, 50 and 70 degree
        """
        if channels_to_use is None:
            channels_to_use = [0, 1, 2, 3]
        station_id = station.get_id()
        event_id = int(evt.get_id())

        if n_templates == 1:

            if cosmic_ray:
                ref_template = self.__templates.get_cr_ref_template(station_id)
                ref_str = 'cr'
            else:
                ref_template = self.__templates.get_nu_ref_template(station_id)
                ref_str = 'nu'
        else:
            logger.debug("Using average of correlation over many templates")
            if cosmic_ray:
                ref_templates = self.__templates.get_set_of_cr_templates_full(station_id, n=n_templates)
                ref_str = 'cr'
            else:
                ref_templates = self.__templates.get_set_of_nu_templates_full(station_id, n=n_templates)
                ref_str = 'nu'

        xcorrs = []
        xcorrs_max = []

        for iCh, channel in enumerate(station.iter_channels()):
            channel_id = channel.get_id()
            xcorrs_ch = []
            xcorrpos_ch = []
            xcorrelations = {}

            orig_binning = 1. / det.get_sampling_frequency(station_id, channel_id)
            target_binning = 1. / channel.get_sampling_rate()
            resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning)).limit_denominator(self.__max_upsampling_factor)

            times = channel.get_times()
            trace = channel.get_trace()
            dt = times[1] - times[0]

            if n_templates == 1:

                ref_template_resampled = self.match_sampling(ref_template, resampling_factor)
                xcorr_trace = hp.get_normalized_xcorr(trace, ref_template_resampled)

                xcorrpos = np.argmax(np.abs(xcorr_trace))
                xcorr = xcorr_trace[xcorrpos]
                flip = np.sign(xcorr)
                xcorrelations['{}_ref_xcorr'.format(ref_str)] = xcorr
                xcorrs.append(xcorr)
                xcorrelations['{}_ref_xcorr_time'.format(ref_str)] = xcorrpos * dt

                if self.__debug:
                    if(xcorr > 0.1):
                        fig, (ax, ax2) = plt.subplots(2, 1)
                        ax.set_title('channel {}, xcorr = {:.2f}'.format(channel_id, xcorr))
                        ax.plot(times, trace, label='measurement')
                        tttemp = np.arange(0, len(ref_template_resampled) * dt, dt)
                        ax.plot(tttemp, flip * np.roll(ref_template_resampled * np.abs(trace).max(), xcorrpos), '--', label='template')
                        argmax = np.argmax(trace)
                        ax.set_xlim(argmax * dt - 64 * units.ns, argmax * dt + 64 * units.ns)
                        ax.legend()
                        ax2.plot(hp.get_normalized_xcorr(trace, ref_template_resampled))
                        ax2.set_ylim(-1, 1)
                        plt.tight_layout()
                        plt.show()

            else:
                template_key = []

                for key in ref_templates:

                    ref_template = ref_templates[key][channel.get_id()]
                    ref_template_resampled = self.match_sampling(ref_template, resampling_factor)

                    xcorr_trace = hp.get_normalized_xcorr(trace, ref_template_resampled)
                    xcorrpos = np.argmax(np.abs(xcorr_trace))
                    xcorr = np.abs(xcorr_trace[xcorrpos])

                    xcorrpos_ch.append(xcorrpos)
                    xcorrs_ch.append(xcorr)
                    template_key.append(key)

                if self.__debug:
                    print(event_id)
                    plt.figure()
                    plt.hist(xcorrs_ch, range=(0, 1), bins=50)
                    plt.axvline(np.mean(np.abs(xcorrs_ch)))
                    plt.axvline(np.max(np.abs(xcorrs_ch)))
                    print(np.mean(np.abs(xcorrs_ch)), np.max(np.abs(xcorrs_ch)),
                          channel[chp.maximum_amplitude] / units.mV)

                xcorrelations['{}_ref_xcorr'.format(ref_str)] = np.abs(xcorrs_ch).mean()
                xcorrelations['{}_ref_xcorr_all'.format(ref_str)] = np.abs(xcorrs_ch)
                xcorrelations['{}_ref_xcorr_max'.format(ref_str)] = np.abs(xcorrs_ch[np.argmax(np.abs(xcorrs_ch))])
                xcorrelations['{}_ref_xcorr_time'.format(ref_str)] = np.mean(xcorrpos_ch[np.argmax(np.abs(xcorrs_ch))]) * dt
                xcorrelations['{}_ref_xcorr_template'.format(ref_str)] = template_key[np.argmax(np.abs(xcorrs_ch))]

                logger.debug("average xcorr over all templates {:.2f} +- {:.2f}, \
                             best template is {} at position {:.2f}".format(np.abs(xcorrs_ch).mean(),
                             np.abs(xcorrs_ch).std(),
                             xcorrelations['{}_ref_xcorr_template'.format(ref_str)],
                             xcorrelations['{}_ref_xcorr_time'.format(ref_str)]))

                xcorrs.append(np.nanmean(np.abs(xcorrs_ch)))
                xcorrs_max.append(np.nanmax(np.abs(xcorrs_ch)))

            if self.__debug:
                print("per channel", len(xcorrs_ch))
                plt.hist(xcorrs_ch, range=(0, 1), bins=50)
                plt.show()
                print(xcorrs)
            # Writing information to channel
            if cosmic_ray:
                channel[chp.cr_xcorrelations] = xcorrelations
            else:
                channel[chp.nu_xcorrelations] = xcorrelations

        xcorrs = np.array(xcorrs)
        xcorrs_max = np.array(xcorrs_max)

        if n_templates == 1:

            xcorrelations_station = {'number_of_templates': n_templates,
                                     '{}_max_xcorr'.format(ref_str): np.abs(xcorrs).max()}
            ref_mask = np.array([channel.get_id() in channels_to_use for channel in station.iter_channels()])
            xcorrelations_station['{0}_max_xcorr_{0}channels'.format(ref_str)] = np.abs(xcorrs[ref_mask]).max()
            xcorrelations_station['{0}_avg_xcorr_{0}channels'.format(ref_str)] = np.abs(xcorrs[ref_mask]).mean()
        else:
            xcorrelations_station = {'number_of_templates': n_templates,
                                     '{}_max_xcorr'.format(ref_str): xcorrs_max.max()}
            ref_mask = np.array([channel.get_id() in channels_to_use for channel in station.iter_channels()])
            xcorrelations_station['{0}_max_xcorr_{0}channels'.format(ref_str)] = xcorrs_max[ref_mask].max()
            xcorrelations_station['{0}_avg_xcorr_{0}channels'.format(ref_str)] = xcorrs[ref_mask].max()

        # calculate average xcorr in parallel channels
        parallel_channels = det.get_parallel_channels(station_id)
        max_xcorr_parallel = 0
        for pair in parallel_channels:
            mask = np.in1d(pair, channels_to_use)  # we use only the specified channels to calculate the pair averages
            if(np.sum(mask)):
                ref_mask = np.array([channel.get_id() in pair[mask] for channel in station.iter_channels()])
                tmp = np.abs(xcorrs[ref_mask]).mean()
                logger.debug("calculating average xcorr for parallel channels {} = {:.2f}".format(pair[mask], tmp))
                max_xcorr_parallel = max(max_xcorr_parallel, tmp)

        xcorrelations_station['{0}_avg_xcorr_parallel_{0}channels'.format(ref_str)] = max_xcorr_parallel
        logger.debug("best average {0} correlation of parallel {0} channels is\
                     {1:.02f}".format(ref_str,
                     xcorrelations_station['{0}_avg_xcorr_parallel_{0}channels'.format(ref_str)]))

        # Writing information to station
        if cosmic_ray:
            station[stnp.cr_xcorrelations] = xcorrelations_station
        else:
            station[stnp.nu_xcorrelations] = xcorrelations_station

    def end(self):
        pass
