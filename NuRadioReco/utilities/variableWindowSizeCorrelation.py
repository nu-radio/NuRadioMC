from NuRadioReco.utilities import units
import numpy as np
import timeit
from numpy import linalg as LA
import logging


class variableWindowSizeCorrelation:
    """
        Module that calculates the correlation between a data trace and a template trace with variable window size
    """

    def __init__(self):
        self.__debug = None
        self.logger = logging.getLogger('NuRadioReco.utilities.variableWindowSizeCorrelation')
        self.begin()

    def begin(self, debug=False, logger_level=logging.NOTSET):
        """
        begin method

        initialize variableWindowSizeCorrelation

        Parameters
        ----------
        debug: boolean
            if true, debug information and plots will be printed
        logger_level: string or logging variable
            Set verbosity level for logger (default: logging.NOTSET)
        """

        self.__debug = debug
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logger_level)

    def run(self, dataTrace, templateTrace, window_size, sampling_rate=3.2*units.GHz, return_time_difference=False):
        """
        run method

        calculate the correlation between to traces using a variable window size and matrix multiplication

        Parameters
        ----------
        dataTrace: array
            full trace of the data event
        templateTrace: array
            full trace of the template
        window_size: int
            size of the template window, used for the correlation (should be given in units.ns)
        sampling_rate: float
            sampling rate of the data and template trace
        return_time_difference: boolean
            if true, the time difference (for the maximal correlation value) between the starting of the data trace and the starting of the (cut) template trace is returned (returned time is in units.ns)

        Returns
        ----------
        correlation (time_diff)
        """
        if self.__debug:
            start = timeit.default_timer()

        # preparing the traces
        dataTrace = np.float32(dataTrace)
        templateTrace = np.float32(templateTrace)

        # create the template window
        window_steps = window_size * (sampling_rate * units.GHz)

        max_amp = max(abs(templateTrace))
        max_amp_i = np.where(abs(templateTrace) == max_amp)[0][0]
        lower_bound = int(max_amp_i - window_steps / 3)
        upper_bound = int(max_amp_i + 2 * window_steps / 3)
        templateTrace = templateTrace[lower_bound:upper_bound]

        # zero padding on the data trace
        dataTrace = np.append(np.zeros(len(templateTrace) - 1), dataTrace)
        dataTrace = np.append(dataTrace, np.zeros(len(templateTrace) - 1))

        # only calculate the correlation of the part of the trace where at least 10% of the maximum is visible (fastens the calculation)
        plot_data_trace = dataTrace
        max_amp_data = max(abs(dataTrace))
        help_val = np.where(abs(dataTrace) >= 0.1 * max_amp_data)[0]
        lower_bound_data = help_val[0] - (len(templateTrace) - 1)
        upper_bound_data = help_val[len(help_val) - 1] + (len(templateTrace) - 1)
        dataTrace = dataTrace[lower_bound_data:upper_bound_data]

        # run the correlation using matrix multiplication
        dataMatrix = np.lib.stride_tricks.sliding_window_view(dataTrace, len(templateTrace))
        corr_numerator = dataMatrix.dot(templateTrace)
        norm_dataMatrix = LA.norm(dataMatrix, axis=1)
        norm_templateTrace = LA.norm(templateTrace)
        corr_denominator = norm_dataMatrix * norm_templateTrace
        correlation = corr_numerator / corr_denominator

        max_correlation = max(abs(correlation))
        max_corr_i = np.where(abs(np.asarray(correlation)) == max_correlation)[0][0]

        if return_time_difference:
            # calculate the time difference between the beginning of the template and data trace for the largest correlation value
            # time difference is given in ns
            time_diff = (max_corr_i + (lower_bound_data - len(templateTrace))) / sampling_rate

        if self.__debug:
            stop = timeit.default_timer()
            self.logger.debug(f'total run time: {stop - start} s')
            self.logger.debug(f'max correlation: {max_correlation}')
            if return_time_difference:
                self.logger.debug(f'time difference: {time_diff} ns')

        if self.__debug:
            import matplotlib.pyplot as plt
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
            x_template = np.arange(0, len(templateTrace), 1)
            x_template = x_template + max_corr_i + lower_bound_data
            axs[1].plot(x_template, templateTrace, label='template')
            axs[1].set_xlabel('time')
            axs[1].set_ylabel('amplitude')
            plt.legend()
            plt.show()

        if return_time_difference:
            return correlation, time_diff
        else:
            return correlation
