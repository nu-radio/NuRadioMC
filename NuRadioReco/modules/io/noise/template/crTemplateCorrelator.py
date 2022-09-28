import os
import sys
import datetime
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
import timeit
from numpy import linalg as LA
from scipy.signal import correlate
import logging

logger = logging.getLogger('crTemplateCorrelator')

class crTemplateCorrelator:

    def __init__(self):
        self.__inputfiles_data = None
        self.__inputfiles_template = None

    def begin(self, data_input_files, template_input_files, logger_level=logging.NOTSET):
        """
        begin method

        initialize readCoREASShower module

        Parameters
        ----------
        data_input_files: input file
            list of data input .nur file
        template_input_files: input file
            list of template input .nur file
        logger_level: string or logging variable
            Set verbosity level for logger (default: logging.NOTSET)
        """

        self.__inputfiles_template = template_input_files
        self.__inputfiles_data = data_input_files
        logger.setLevel(logger_level)

    def open_data_files(self):
        """
        opens list of data .nur files

        Returns
        ----------
        list of event objects
        """

        logger.info("Start opening the list of data input files")
        evt = []
        for filename in self.__inputfiles_data:
            evt.append(NuRadioRecoio.NuRadioRecoio(filename).get_event_i(0))  # each file only has one event: event number =0
        logger.info("Finished opening the list of data input files")
        return evt

    def open_data_file_i(self, file_number):
        """
        opens only the specified data .nur file

        Returns
        ----------
        event object
        """

        logger.info("Start opening the data input file")
        evt = NuRadioRecoio.NuRadioRecoio(self.__inputfiles_data[file_number]).get_event_i(
            0)  # each file only has one event: event number =0
        logger.info("Finished opening the data input file")

        return evt

    def open_template_files(self):
        """
        opens list of template .nur files

        Returns
        ----------
        list of event objects
        """

        logger.info("Start opening the list of template input files")
        evt = []
        for filename in self.__inputfiles_template:
            evt.append(NuRadioRecoio.NuRadioRecoio(filename).get_event_i(0)) #each file only has one event: event number =0
        logger.info("Finished opening the list of template input files")

        return evt

    def open_template_file_i(self, file_number):
        """
        opens only the specified template .nur file

        Returns
        ----------
        event object
        """

        logger.info("Start opening the template input file")
        evt = NuRadioRecoio.NuRadioRecoio(self.__inputfiles_template[file_number]).get_event_i(0)  # each file only has one event: event number =0
        logger.info("Finished opening the template input file")

        return evt

    def correlation_scan_single_spacing(self, evt_data, evt_template, channel_num, station_num_data, station_num_temp, showPlot=False):
        """
        calculate the correlation between to traces using for-loops and a single scan spacing

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            template event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        showPlot: bool
            if True, shows a plot of the correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()

        # loading the station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the traces
        dataTrace = sta_data.get_channel(channel_num).get_trace()
        templateTrace = sta_temp.get_channel(channel_num).get_trace()
        templateTrace = preparingTrace(dataTrace, templateTrace)

        # only use the window where at least some of the signal is visible
        max_amp = max(abs(templateTrace))
        help = np.where(templateTrace >= 0.1 * max_amp)[0]
        lower_bound = help[0] - len(dataTrace)
        upper_bound = help[len(help) - 1] + len(dataTrace)
        templateTrace = templateTrace[lower_bound:upper_bound]

        # run the correlation
        #length_for_loop = 2*len(dataTrace)
        length_for_loop = len(templateTrace) - len(dataTrace)
        length_data_trace = len(dataTrace)
        running_correlation_parameter = np.zeros(length_for_loop)
        for i in range(length_for_loop):
            running_correlation_parameter[i] = correlation_koeff(dataTrace, templateTrace[i:length_data_trace + i])

        stop = timeit.default_timer()
        logger.info(f'run time of the correlation scan: {stop-start} s')
        logger.info(f'maximal correlation: {max(abs(running_correlation_parameter[1:]))}')

        if showPlot:
            plt.plot(running_correlation_parameter)
            plt.plot(np.array([np.where(abs(running_correlation_parameter[1:])==max(abs(running_correlation_parameter[1:])))[0][0]]),np.array([0]), marker="x", markersize=12, color='tab:red')
            plt.ylabel(r"$\chi$")
            plt.ylim(-1.1, 1.1)
            plt.show()

        return running_correlation_parameter

    def correlation_scan_mixed_spacing(self, evt_data, evt_template, channel_num, station_num_data, station_num_temp, scanSpacing):
        """
        calculate the correlation between to traces using for-loops and a coarse and fine scan

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            template event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        scanSpacing: int
            interval used for the coarse correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()
        # loading the event and station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the data
        dataTrace = sta_data.get_channel(channel_num).get_trace()
        templateTrace = sta_temp.get_channel(channel_num).get_trace()
        templateTrace = preparingTrace(dataTrace, templateTrace)

        # only use the window where at least some of the signal is visible
        max_amp = max(abs(templateTrace))
        help = np.where(templateTrace >= 0.1 * max_amp)[0]
        lower_bound = help[0] - len(dataTrace)
        upper_bound = help[len(help) - 1] + len(dataTrace)
        templateTrace = templateTrace[lower_bound:upper_bound]

        # run the correlation
        length_for_loop = len(templateTrace) - len(dataTrace)
        scan_correlation_koeff = np.zeros(int(length_for_loop / scanSpacing))
        # coarse spacing
        for i in range(int(length_for_loop / scanSpacing)):
            scan_correlation_koeff[i] = correlation_koeff(dataTrace, templateTrace[i * scanSpacing:len(dataTrace) + (i * scanSpacing)])

        max_coarse_scan = max(abs(scan_correlation_koeff[1:]))
        max_coarse_scan_i = np.where(abs(scan_correlation_koeff) == max_coarse_scan)[0][0]


        # fine spacing
        search_factor = 5
        scan_correlation_koeff = np.zeros(2*search_factor*scanSpacing)
        start_low_mask = (max_coarse_scan_i * scanSpacing) - (search_factor * scanSpacing)
        if start_low_mask < 0:
            start_low_mask = 0
        if start_low_mask + (2*search_factor*scanSpacing) > (len(templateTrace) - len(dataTrace)):
            start_low_mask = (len(templateTrace) - len(dataTrace)) - (2*search_factor*scanSpacing)
        start_up_mask = start_low_mask + len(dataTrace)
        for i in range(int(2 * search_factor * scanSpacing)):
            scan_correlation_koeff[i] = correlation_koeff(dataTrace, templateTrace[start_low_mask +i:start_up_mask+i])
        max_fine_scan = max(abs(scan_correlation_koeff))
        max_fine_scan_i = np.where(abs(scan_correlation_koeff) == max_fine_scan)[0][0]
        stop = timeit.default_timer()

        logger.info(f'run time of the correlation scan: {stop - start} s')
        logger.info(f'max correlation (coarse scan): {max_coarse_scan}')
        logger.info(f'max correlation (fine scan): {max_fine_scan}')
        logger.info(f'max template deviation: {((max_coarse_scan_i * scanSpacing - search_factor * scanSpacing) + max_fine_scan_i) - (len(dataTrace)-lower_bound)}')

        return scan_correlation_koeff

    def correlation_scan_single_spacing_matrix(self, evt_data, evt_template, channel_num, station_num_data, station_num_temp, showPlot=True):
        """
        calculate the correlation between to traces using matrix multiplication

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            template event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        showPlot: bool
            if True, shows a plot of the correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()

        # loading the station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the traces
        dataTrace = sta_data.get_channel(channel_num).get_trace()
        templateTrace = sta_temp.get_channel(channel_num).get_trace()

        templateTrace = np.append(np.zeros(len(dataTrace) - 1), templateTrace)
        templateTrace = np.append(templateTrace, np.zeros(len(dataTrace) - 1))

        dataTrace = np.float32(dataTrace)
        templateTrace = np.float32(templateTrace)

        # only use the window where at least some of the signal is visible
        max_amp = max(abs(templateTrace))
        help = np.where(templateTrace >= 0.1 * max_amp)[0]
        lower_bound = help[0]-len(dataTrace)
        upper_bound = help[len(help) - 1] + len(dataTrace)
        templateTrace = templateTrace[lower_bound:upper_bound]

        # run the correlation
        templateMatrix = np.lib.stride_tricks.sliding_window_view(templateTrace, len(dataTrace))
        corr_numerator = templateMatrix.dot(dataTrace)
        norm_templateMatrix = LA.norm(templateMatrix, axis=1)
        norm_dataTrace = LA.norm(dataTrace)
        corr_denominator = norm_templateMatrix * norm_dataTrace
        correlation = corr_numerator / corr_denominator

        max_correlation = max(abs(correlation))

        stop = timeit.default_timer()
        logger.info(f'total run time: {stop - start} s')
        logger.info(f'max correlation: {max_correlation}')

        if showPlot:
            plt.plot(correlation)
            plt.plot(np.array([np.where(abs(correlation) == max(abs(correlation)))[0][0]]), np.array([max_correlation]), marker="x", markersize=12, color='tab:red')
            plt.ylim(-1.1, 1.1)
            plt.ylabel(r"$\chi$")
            plt.show()

        return correlation

    def correlation_scan_mixed_spacing_matrix(self, evt_data, evt_template, channel_num, station_num_data, station_num_temp, scanSpacing):
        """
        calculate the correlation between to traces using matrix multiplication and a coarse and fine scan

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            emplate event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        scanSpacing: int
            interval used for the coarse correlation scan
        showPlot: bool
            if True, shows a plot of the correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()

        # loading the event and station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the data
        dataTrace = sta_data.get_channel(channel_num).get_trace()
        templateTrace = sta_temp.get_channel(channel_num).get_trace()

        templateTrace = np.append(np.zeros(len(dataTrace) - 1), templateTrace)
        templateTrace = np.append(templateTrace, np.zeros(len(dataTrace) - 1))
        dataTrace = np.float32(dataTrace)
        templateTrace = np.float32(templateTrace)

        # only use the window where at least some of the signal is visible
        max_amp = max(abs(templateTrace))
        help = np.where(templateTrace >= 0.1 * max_amp)[0]
        lower_bound = help[0] - len(dataTrace)
        upper_bound = help[len(help) - 1] + len(dataTrace)
        templateTrace = templateTrace[lower_bound:upper_bound]

        # coarse scan
        templateMatrix_coarse = np.lib.stride_tricks.sliding_window_view(templateTrace, len(dataTrace))[::scanSpacing, :]
        numerator_coarse = templateMatrix_coarse.dot(dataTrace)
        norm_templateMatrix_coarse = LA.norm(templateMatrix_coarse, axis=1)
        norm_dataTrace_coarse = LA.norm(dataTrace)
        denominator_coarse = norm_templateMatrix_coarse * norm_dataTrace_coarse
        correlation_coarse = numerator_coarse / denominator_coarse

        max_corr_coarse = max(abs(correlation_coarse))
        max_corr_coarse_i = np.where(abs(correlation_coarse) == max_corr_coarse)[0][0]

        #fine scan
        search_factor = 5
        # check to avoid problems at the boundaries
        low_mask = max_corr_coarse_i * scanSpacing - search_factor * scanSpacing
        up_mask = max_corr_coarse_i * scanSpacing + len(dataTrace) + search_factor * scanSpacing
        if low_mask < 0:
            low_mask = 0
        if up_mask > len(templateTrace):
            up_mask = len(templateTrace)

        templateTrace = templateTrace[low_mask:up_mask]
        temp_matrix_fine = np.lib.stride_tricks.sliding_window_view(templateTrace, len(dataTrace))
        numerator_fine = temp_matrix_fine.dot(dataTrace)
        norm_temp_fine = LA.norm(temp_matrix_fine, axis=1)
        norm_data_fine = LA.norm(dataTrace)
        denominator_fine = norm_temp_fine * norm_data_fine
        correlation_fine = numerator_fine / denominator_fine

        max_corr_fine = max(abs(correlation_fine))
        max_corr_fine_i = np.where(abs(correlation_fine)==max_corr_fine)[0][0]

        stop = timeit.default_timer()

        logger.info(f'run time of the correlation scan: {stop - start} s')
        logger.info(f'max correlation (coarse scan): {max_corr_coarse}')
        logger.info(f'max correlation (fine scan): {max_corr_fine}')
        logger.info(f'max template deviation: {((max_corr_coarse_i * scanSpacing - search_factor * scanSpacing) + max_corr_fine_i) - (len(dataTrace)-lower_bound)}')

        return correlation_fine

    def correlation_scipy(self, evt_data, evt_template, channel_num_data, channel_num_temp, station_num_data, station_num_temp, showPlot=True):
        """
                calculate the correlation between to traces using a package from scipy

                Parameters
                ----------
                evt_data: event object
                    data event which contains the trace for the correlation
                evt_template: event object
                    template event which contains the trace for the correlation
                channel_num: int
                    number of the data channel for which the correlation is calculated
                channel_num_temp: int
                    number of the temp channel for which the correlation is calculated
                station_num_data: int
                    Number of the data station for which the correlation is calculated
                station_num_temp: int
                    Number of the template station for which the correlation is calculated
                showPlot: bool
                    if True, shows a plot of the correlation scan

                Returns
                ----------
                list with correlation values
                """
        start = timeit.default_timer()

        # loading the station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading the traces
        dataTrace = sta_data.get_channel(channel_num_data).get_trace()
        templateTrace = sta_temp.get_channel(channel_num_temp).get_trace()

        # run the correlation
        correlation = correlate(dataTrace, templateTrace, mode='full', method='auto') / (np.sum(dataTrace ** 2) * np.sum(templateTrace ** 2)) ** 0.5 # the latter part if to normalise the values to 1

        max_correlation = max(abs(correlation))

        stop = timeit.default_timer()
        logger.info(f'total run time: {stop - start} s')
        logger.info(f'max correlation: {max_correlation}')

        if showPlot:
            plt.plot(correlation)
            plt.plot(np.array([np.where(abs(correlation) == max(abs(correlation)))[0][0]]), np.array([max_correlation]), marker="x", markersize=12, color='tab:red')
            plt.ylim(-1.1, 1.1)
            plt.ylabel(r"$\chi$")
            plt.show()

        return correlation

    def correlation_scan_single_spacing_matrix_variable_window(self, evt_data, evt_template, channel_num_data, channel_num_temp, station_num_data, station_num_temp, window_size, return_time_difference=False,showPlot=True):
        """
        calculate the correlation between to traces using matrix multiplication

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            template event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        return_time_difference: bool
            if true, the time difference (for the maximal correlation value) between the starting of the data trace and the starting of the (cut) template trace is returned
        showPlot: bool
            if True, shows a plot of the correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()

        # loading the station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the traces
        dataTrace = sta_data.get_channel(channel_num_data).get_trace()
        templateTrace = sta_temp.get_channel(channel_num_temp).get_trace()

        dataTrace = np.float32(dataTrace)
        templateTrace = np.float32(templateTrace)

        # create the template window
        sampling_rate = sta_temp.get_channel(channel_num_temp).get_sampling_rate()
        window_steps = window_size * (sampling_rate * units.GHz)

        max_amp = max(abs(templateTrace))
        max_amp_i = np.where(abs(templateTrace) == max_amp)[0][0]
        lower_bound = int(max_amp_i - window_steps / 3)
        upper_bound = int(max_amp_i + 2 * window_steps / 3)
        templateTrace = templateTrace[lower_bound:upper_bound]

        dataTrace = np.append(np.zeros(len(templateTrace) - 1), dataTrace)
        dataTrace = np.append(dataTrace, np.zeros(len(templateTrace) - 1))

        # only use the window where at least some of the signal is visible
        plot_data_trace = dataTrace
        max_amp_data = max(abs(dataTrace))
        help_val = np.where(abs(dataTrace) >= 0.1 * max_amp_data)[0]
        lower_bound_data = help_val[0] - (len(templateTrace)-1)
        upper_bound_data = help_val[len(help_val) - 1] + (len(templateTrace)-1)
        dataTrace = dataTrace[lower_bound_data:upper_bound_data]

        # run the correlation
        # templateMatrix = np.lib.stride_tricks.sliding_window_view(templateTrace, len(dataTrace))
        dataMatrix = np.lib.stride_tricks.sliding_window_view(dataTrace, len(templateTrace))
        # corr_numerator = templateMatrix.dot(dataTrace)
        corr_numerator = dataMatrix.dot(templateTrace)
        norm_dataMatrix = LA.norm(dataMatrix, axis=1)
        # norm_dataTrace = LA.norm(dataTrace)
        norm_templateTrace = LA.norm(templateTrace)
        # corr_denominator = norm_templateMatrix * norm_dataTrace
        corr_denominator = norm_dataMatrix * norm_templateTrace
        correlation = corr_numerator / corr_denominator

        max_correlation = max(abs(correlation))

        max_corr_i = np.where(abs(np.asarray(correlation)) == max_correlation)[0][0]
        # time difference between the beginning of the template and data trace for the largest correlation value
        # time difference is given in ns
        time_diff = (max_corr_i + (lower_bound_data - len(templateTrace))) / sampling_rate

        stop = timeit.default_timer()
        logger.info(f'total run time: {stop - start} s')
        logger.info(f'max correlation: {max_correlation}')

        if showPlot:
            print(max_corr_i)
            fig, axs = plt.subplots(2)
            axs[0].plot(correlation)
            axs[0].plot(np.array([np.where(abs(correlation) == max(abs(correlation)))[0][0]]), np.array([max_correlation]), marker="x", markersize=12, color='tab:red')
            axs[0].set_ylim(-1.1, 1.1)
            axs[0].set_ylabel(r"$\chi$")
            axs[0].set_xlabel('N')
            axs[1].plot(plot_data_trace, label='complete data trace')
            x_data = np.arange(0, len(dataTrace), 1)
            x_data = x_data + lower_bound_data
            axs[1].plot(x_data, dataTrace, label='scanned data trace')
            x_template = np.arange(0,len(templateTrace),1)
            x_template = x_template + max_corr_i + lower_bound_data
            # x_template = x_template + 4886
            axs[1].plot(x_template,templateTrace, label='template')
            axs[1].set_xlabel('time')
            axs[1].set_ylabel('amplitude')
            plt.legend()
            plt.show()

        if return_time_difference:
            return correlation, time_diff
        else:
            return correlation

    def correlation_scan_mixed_spacing_matrix_variable_window(self, evt_data, evt_template, channel_num_data, channel_num_temp, station_num_data, station_num_temp, scanSpacing, window_size):
        """
        calculate the correlation between to traces using matrix multiplication and a coarse and fine scan

        Parameters
        ----------
        evt_data: event object
            data event which contains the trace for the correlation
        evt_template: event object
            emplate event which contains the trace for the correlation
        channel_num: int
            number of the channel for which the correlation is calculated
        station_num_data: int
            Number of the data station for which the correlation is calculated
        station_num_temp: int
            Number of the template station for which the correlation is calculated
        scanSpacing: int
            interval used for the coarse correlation scan
        showPlot: bool
            if True, shows a plot of the correlation scan

        Returns
        ----------
        event object
        """
        start = timeit.default_timer()

        # loading the event and station
        sta_temp = evt_template.get_station(station_num_temp)
        sta_data = evt_data.get_station(station_num_data)

        # loading and preparing the data
        dataTrace = sta_data.get_channel(channel_num_data).get_trace()
        templateTrace = sta_temp.get_channel(channel_num_temp).get_trace()

        dataTrace = np.float32(dataTrace)
        templateTrace = np.float32(templateTrace)

        # create the template window
        sampling_rate = sta_temp.get_channel(channel_num_temp).get_sampling_rate()
        window_steps = window_size * (sampling_rate * units.GHz)

        max_amp = max(abs(templateTrace))
        max_amp_i = np.where(abs(templateTrace) == max_amp)[0][0]
        lower_bound = int(max_amp_i - window_steps / 3)
        upper_bound = int(max_amp_i + 2 * window_steps / 3)
        templateTrace = templateTrace[lower_bound:upper_bound]

        dataTrace = np.append(np.zeros(len(templateTrace) - 1), dataTrace)
        dataTrace = np.append(dataTrace, np.zeros(len(templateTrace) - 1))

        # only use the window where at least some of the signal is visible
        max_amp_data = max(abs(dataTrace))
        help_val = np.where(abs(dataTrace) >= 0.1 * max_amp_data)[0]
        lower_bound_data = help_val[0] - (len(templateTrace) - 1)
        upper_bound_data = help_val[len(help_val) - 1] + (len(templateTrace) - 1)
        dataTrace = dataTrace[lower_bound_data:upper_bound_data]

        # coarse scan
        dataMatrix_coarse = np.lib.stride_tricks.sliding_window_view(dataTrace, len(templateTrace))[::scanSpacing, :]
        numerator_coarse = dataMatrix_coarse.dot(templateTrace)
        norm_dataMatrix_coarse = LA.norm(dataMatrix_coarse, axis=1)
        norm_templateTrace_coarse = LA.norm(templateTrace)
        denominator_coarse = norm_dataMatrix_coarse * norm_templateTrace_coarse
        correlation_coarse = numerator_coarse / denominator_coarse

        max_corr_coarse = max(abs(correlation_coarse))
        max_corr_coarse_i = np.where(abs(correlation_coarse) == max_corr_coarse)[0][0]

        #fine scan
        search_factor = 5
        # check to avoid problems at the boundaries
        low_mask = max_corr_coarse_i * scanSpacing - search_factor * scanSpacing
        up_mask = max_corr_coarse_i * scanSpacing + len(templateTrace) + search_factor * scanSpacing
        if low_mask < 0:
            low_mask = 0
        if up_mask > len(dataTrace):
            up_mask = len(dataTrace)

        dataTrace = dataTrace[low_mask:up_mask]
        data_matrix_fine = np.lib.stride_tricks.sliding_window_view(dataTrace, len(templateTrace))
        numerator_fine = data_matrix_fine.dot(templateTrace)
        norm_data_fine = LA.norm(data_matrix_fine, axis=1)
        norm_template_fine = LA.norm(templateTrace)
        denominator_fine = norm_data_fine * norm_template_fine
        correlation_fine = numerator_fine / denominator_fine

        max_corr_fine = max(abs(correlation_fine))
        max_corr_fine_i = np.where(abs(correlation_fine)==max_corr_fine)[0][0]

        stop = timeit.default_timer()

        logger.info(f'run time of the correlation scan: {stop - start} s')
        logger.info(f'max correlation (coarse scan): {max_corr_coarse}')
        logger.info(f'max correlation (fine scan): {max_corr_fine}')
        logger.info(f'max template deviation: {((max_corr_coarse_i * scanSpacing - search_factor * scanSpacing) + max_corr_fine_i) - (len(templateTrace)-lower_bound)}')

        return correlation_fine

def correlation_koeff(x1,x2):
    """
    Function to calculate the correlation between two arrays
    """
    return np.sum(x1*x2)/(np.sqrt(np.sum(x1**2)*np.sum(x2**2)))

def preparingTrace(dataTrace, templateTrace):
    """
    Function to prepare the template Trace for the correlation scan.
    """
    templateTrace = np.append(np.zeros(len(dataTrace)), templateTrace)
    templateTrace = np.append(templateTrace, np.zeros(len(dataTrace)))
    return templateTrace