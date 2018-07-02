from __future__ import absolute_import, division, print_function
import cPickle as pickle


class Trigger:

    def __init__(self):
        self.__threshold_high = None
        self.__threshold_low = None
        self.__high_low_window = None
        self.__coinc_window = None
        self.__number_concidences = None
        self.__triggered = False

    def set_trigger_settings(self, threshold_high, threshold_low,
                             high_low_window, coinc_window, number_concidences):
        self.__threshold_high = threshold_high
        self.__threshold_low = threshold_low
        self.__high_low_window = high_low_window
        self.__coinc_window = coinc_window
        self.__number_concidences = number_concidences

    def get_trigger_settings(self):
        return {'threshold_high': self.__threshold_high,
                'threshold_low': self.__threshold_low,
                'high_low_window': self.__high_low_window,
                'coinc_window': self.__coinc_window,
                'number_concidences': self.__number_concidences}

    def has_triggered(self):
        return self.__triggered

    def set_triggered(self, triggered=True):
        self.__triggered = triggered

    def serialize(self):
        data = {'__threshold_high': self.__threshold_high,
                '__threshold_low': self.__threshold_low,
                '__high_low_window': self.__high_low_window,
                '__coinc_window': self.__coinc_window,
                '__number_concidences': self.__number_concidences,
                '__triggered': self.__triggered}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self.__threshold_high = data['__threshold_high']
        self.__threshold_low = data['__threshold_low']
        self.__high_low_window = data['__high_low_window']
        self.__coinc_window = data['__coinc_window']
        self.__number_concidences = data['__number_concidences']
        self.__triggered = data['__triggered']
